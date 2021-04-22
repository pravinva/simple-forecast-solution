import os
import sys
import warnings
import numpy as np
import pandas as pd

from pyspark.sql import functions as F
from pyspark.sql import SparkSession

from pyspark.sql.types import \
    StructType, FloatType, TimestampType, IntegerType, StringType, StructField
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.context import SparkContext

from awsglue.context import GlueContext
from awsglue.utils import getResolvedOptions
from awsglue.job import Job

args = getResolvedOptions(sys.argv,
        ["JOB_NAME", "s3_input_path", "freq", "horiz"])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

#spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
spark.conf.set("spark.executorEnv.ARROW_PRE_0_15_IPC_FORMAT", "1")
spark.conf.set("spark.yarn.appMasterEnv.ARROW_PRE_0_15_IPC_FORMAT", "1")

job = Job(glueContext)
job.init(args["JOB_NAME"], args)

# parse input arguments
s3_input_path = args["s3_input_path"]
freq = str(args.get("freq", "MS"))
horiz = int(args.get("horiz", 6))

schema = StructType([
    StructField("timestamp", StringType()),
    StructField("channel", StringType()),
    StructField("category", StringType()),
    StructField("item_id", StringType()),
    StructField("demand", FloatType()),
    StructField("category_sum", FloatType())
])

@pandas_udf(schema, functionType=PandasUDFType.GROUPED_MAP)
def udf_preprocess(dd):
    """
    """
    import os
    import numpy as np
    import pandas as pd

    os.environ["ARROW_PRE_0_15_IPC_FORMAT"] = "1"
    
    def fill_dt_spine(dd, freq, horiz):
        df_spine = pd.DataFrame()
        dt_start = dd["timestamp"].min()

        xs_timestamp = \
            pd.date_range(dt_start, dd["timestamp"].max(), freq=freq) \
              .strftime("%Y-%m-%d")

        min_len = np.ceil(horiz * 1.5).astype(int)

        if horiz is not None:
            if xs_timestamp.shape[0] < min_len:
                periods = min_len - xs_timestamp.shape[0]
                xs_timestamp_xtra = \
                    pd.date_range(end=dt_start, freq=freq, periods=periods, 
                                  closed="left") \
                      .strftime("%Y-%m-%d")
                xs_timestamp = xs_timestamp.append(xs_timestamp_xtra)

        df_spine["timestamp"] = xs_timestamp

        dd = df_spine.merge(dd, on=["timestamp"], how="left")

        xs_demand = dd["demand"]

        dd.drop(["demand"], axis=1, inplace=True)
        dd = dd.ffill().bfill()
        dd["demand"] = xs_demand

        dd["channel"] = dd["channel"].ffill()
        dd["category"] = dd["category"].ffill()
        dd["item_id"] = dd["item_id"].ffill()
        
        return dd
    
    # ---
    
    def resample(df, freq, horiz):
        # resample time series frequency
        df2 = \
        df.groupby([pd.Grouper(key="timestamp", freq=freq),
                    "channel", "category", "item_id"]) \
          .agg({"demand": lambda xs: xs.sum(min_count=1)}) \
          .reset_index() ;

        df2["timestamp"] = df2["timestamp"].dt.strftime("%Y-%m-%d")

        # fill in missing timestamps
        df2 = df2.groupby(["channel", "category", "item_id"]) \
                 .apply(lambda dd: fill_dt_spine(dd, freq, horiz)) \
                 .reset_index(drop=True)

        assert(df2.shape[0] > horiz)

        return df2
    
    # ---
    
    def preprocess(df, freq, horiz):
        """Preprocess a "raw" demand dataframe of multiple time-series, resampling
        demand according a specified frequency (`freq`).

        """

        # resample demand to `freq`
        if "category" not in df:
            df.rename({"family": "category"}, axis=1, inplace=True)

        df.loc[:,"timestamp"] = pd.DatetimeIndex(df["timestamp"])

        df2 = resample(df, freq, horiz)

        df2["timestamp"] = pd.DatetimeIndex(df2["timestamp"]) \
                             .strftime("%Y-%m-%d")

        # add category-level sums
        df_cat_sums = df2.groupby(["category", "timestamp"]) \
                         .agg({"demand": sum}) \
                         .reset_index() \
                         .rename({"demand": "category_sum"}, axis=1)

        df2 = df2.merge(df_cat_sums, on=["timestamp", "category"], how="left")

#         df2["timestamp"] = pd.DatetimeIndex(df2["timestamp"])

        df2.sort_values(by=["channel", "category", "item_id", "timestamp"],
                        inplace=True)

#         df2.set_index("timestamp", inplace=True)

        # DEBUG
#        _dd = \
#        df2.groupby(["channel", "category", "item_id"]) \
#           .agg({"timestamp": "count"}) \
#           .rename({"timestamp": "count"}, axis=1)
#
#        assert(_dd["count"].min() >= horiz)
        # DEBUG

        return df2    
    
    dd = preprocess(dd, freq, horiz)
    
    return dd

# read from s3 csv file
sdf = spark.read.parquet(s3_input_path.replace('s3:', 's3a:')) \
           .select(F.col("timestamp").cast("string"),
                   F.col("channel").cast("string"),
                   F.col("family").cast("string"),
                   F.col("item_id").cast("string"),
                   F.col("demand").cast("float"))

#sdf = spark.createDataFrame(pd.read_parquet(s3_input_path))
sdf.printSchema()

sdf2 = sdf.groupBy(["channel", "family", "item_id"]) \
          .apply(udf_preprocess)

sdf2.printSchema()

#df2 = sdf2.toPandas()
#df2.info()

#df2.to_parquet(f"{s3_input_path}.resample", index=False)

sdf2.write \
    .mode("overwrite") \
    .parquet(f"{s3_input_path.replace('s3:', 's3a:')}.resample")

job.commit()

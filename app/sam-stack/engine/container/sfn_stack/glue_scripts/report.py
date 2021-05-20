import os
import sys
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.sql.types import \
    StructType, FloatType, TimestampType, IntegerType, StringType, StructField
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.context import SparkContext

from awsglue.context import GlueContext
from awsglue.utils import getResolvedOptions
from awsglue.job import Job

args = getResolvedOptions(sys.argv,
        ["JOB_NAME", "s3_input_path", "s3_parquet_paths"])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

# Needed to reliably work between parquet, spark, and pandas dataframes.
spark.conf.set("spark.executorEnv.ARROW_PRE_0_15_IPC_FORMAT", "1")
spark.conf.set("spark.yarn.appMasterEnv.ARROW_PRE_0_15_IPC_FORMAT", "1")

job = Job(glueContext)
job.init(args["JOB_NAME"], args)

s3_input_path = args["s3_input_path"]
s3_parquet_paths = args["s3_parquet_paths"]

# ~~~

import time
import warnings
import awswrangler as wr
import voyager as vr

from awswrangler._utils import pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from tqdm.auto import tqdm

s3_paths_csv = f"{s3_parquet_paths.replace('s3a:', 's3:')}"
df_paths = wr.s3.read_csv([s3_paths_csv])

dfs = []

print(s3_paths_csv)
print(df_paths)

for pth in tqdm(df_paths["s3_parquet_path"]):
    df = wr.s3.read_parquet(f"{pth}.resample/")
    dfs.append(df)
    
df = pd.concat(dfs)

from joblib import Parallel, delayed

calls = (delayed(vr.calc_series_metrics)(dd)
         for _, dd in df.groupby(["channel", "item_id"]))
pool = Parallel(n_jobs=-1, verbose=2)
results = pool(calls)

df_report = pd.DataFrame(results)
wr.s3.to_csv(df_report, f"{s3_input_path}.report", index=False)

job.commit()

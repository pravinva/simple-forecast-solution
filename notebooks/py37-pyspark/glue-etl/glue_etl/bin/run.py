import os
import sys
import warnings
import numpy as np
import pandas as pd

from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.context import SparkContext

from awsglue.context import GlueContext
from awsglue.utils import getResolvedOptions
from awsglue.job import Job

args = getResolvedOptions(sys.argv,
        ["JOB_NAME", "s3_input_path", "freq", "horiz"])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

spark.conf.set("spark.executorEnv.ARROW_PRE_0_15_IPC_FORMAT", "1")
spark.conf.set("spark.yarn.appMasterEnv.ARROW_PRE_0_15_IPC_FORMAT", "1")

job = Job(glueContext)
job.init(args["JOB_NAME"], args)

#
# define main job here
#

# parse input arguments
s3_input_path = args["s3_input_path"]
freq = str(args.get("freq", "MS"))
horiz = int(args.get("horiz", 6))

# read from s3 csv file
n_series_per_file = 10000

df = pd.read_csv(s3_input_path,
                 dtype={"channel": str, "item_id": str, "family": str},
                 parse_dates=["timestamp"],
                 low_memory=True)

# split the data into smaller (parquet) files
df_groups = df[["channel", "family", "item_id"]].drop_duplicates()

n_groups = np.ceil(df_groups.shape[0] / n_series_per_file).astype(int)
xs_grp = pd.Series(np.arange(df_groups.shape[0], dtype=int) % n_groups) \
           .sample(frac=1.0, random_state=12345) \
           .values

df_groups["grp"] = xs_grp

df2 = df.merge(df_groups, on=["channel", "family", "item_id"], how="left")

del df

grp_vals = df_groups["grp"].unique()

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from tqdm.autonotebook import tqdm
        grp_vals = tqdm(grp_vals)
except ImportError:
    pass

output_paths = []

# write the splits to s3
for grp in grp_vals:
    s3_output_path = f"{s3_input_path.replace('s3a:', 's3:')}.parquet.{grp:03d}"
    df2[df2["grp"] == grp].drop(["grp"], axis=1) \
                          .to_parquet(s3_output_path, index=False)
    output_paths.append(s3_output_path)

pd.DataFrame({"s3_output_path": output_paths}) \
  .to_csv(f"{s3_input_path}.paths.csv", index=False)

job.commit()

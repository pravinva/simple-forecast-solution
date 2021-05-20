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
        ["JOB_NAME", "s3_input_path", "freq", "horiz", "lambda_arn"])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

# Needed to reliably work between parquet, spark, and pandas dataframes.
spark.conf.set("spark.executorEnv.ARROW_PRE_0_15_IPC_FORMAT", "1")
spark.conf.set("spark.yarn.appMasterEnv.ARROW_PRE_0_15_IPC_FORMAT", "1")

job = Job(glueContext)
job.init(args["JOB_NAME"], args)

s3_input_path = args["s3_input_path"]
freq = str(args.get("freq", "MS"))
horiz = int(args.get("horiz", 6))
lambda_arn = str(args["lambda_arn"])

# ~~~

import time
import json
import base64
import cloudpickle
import boto3
import botocore
import warnings
import awswrangler as wr
import voyager as vr

from awswrangler._utils import pd

from concurrent import futures
from toolz import partition_all

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from tqdm.auto import tqdm


class LambdaFunction:
    """
    """
    
    def __init__(self, func, client, lambda_arn):
        """
        """
        
        self.func = func
        self.client = client
        self._lambda_arn = lambda_arn
        
        return
    
    def __call__(self, *args, **kwargs):
        """
        """
        
        payload = {
            "func": self.func,
            "args": args,
            "kwargs": kwargs
        }
        
        return self.invoke_handler(payload)
    
    def invoke_handler(self, payload):
        """
        """
        
        client = self.client
        payload = base64.b64encode(cloudpickle.dumps(payload)).decode("ascii")
        payload = json.dumps(payload)
        
        resp = client.invoke(
            FunctionName=self._lambda_arn,
            InvocationType="RequestResponse",
            Payload=payload
        )
        
        resp_bytes = resp["Payload"].read()
        
        if "FunctionError" in resp:
            result = resp_bytes
        else:
            result = cloudpickle.loads(resp_bytes)
        
        return result

    
class LambdaExecutor:
    """
    """
    
    def __init__(self, max_workers, lambda_arn):
        """
        """
        
        lambda_config = botocore.config.Config(
            retries={'max_attempts': 128},
            connect_timeout=60*10,
            read_timeout=60*10,
            max_pool_connections=10000
        )
        
        self._client = boto3.client("lambda", config=lambda_config)
        self._max_workers = max_workers
        self._executor = futures.ThreadPoolExecutor(max_workers=max_workers)
        self._lambda_arn = lambda_arn
        
        return
    
    def map(self, func, payloads, local_mode=False):
        """
        """
        
        from tqdm.auto import tqdm
        
        if local_mode:
            f = func
        else:
            f = LambdaFunction(func, self._client, self._lambda_arn)
        
        ex = self._executor
        wait_for = [ex.submit(f, *p["args"], **p["kwargs"]) for p in payloads]
        tbar = tqdm(total=len(wait_for))
        prev_n_done = 0
        n_done = sum(f.done() for f in wait_for)
        
        while n_done != len(wait_for):
            tbar.update(n_done - prev_n_done)
            prev_n_done = n_done
            n_done = sum(f.done() for f in wait_for)
            time.sleep(0.5)
            
        tbar.update(n_done - prev_n_done)   
        tbar.close()
            
        results = [f.result() for f in futures.as_completed(wait_for)]
        return results


def make_cfgs(df, freq, horiz):
    """
    """

    min_len = None

    if "category" not in df:
        df.rename({"family": "category"}, axis=1, inplace=True)

    cfgs = []
    groupby_cols = ["channel", "category", "item_id"]

    groups = df.groupby(groupby_cols)

    print("Number of time series", groups.ngroups)
    print("Number of cfgs", df.item_id.nunique())
    print("Number of channels", df.channel.nunique())
    print("Number of category", df.category.nunique())

    for (channel, category, item_id), dd in groups:
        demand = dd.demand
        item_name = dd.item_id.iloc[0]
        channel = dd.channel.iloc[0]
        category = dd.category.iloc[0]
        category_sum = dd.category_sum

        if demand.sum() >= 0:
            cfg ={}
            cfg["demand"] = np.nan_to_num(np.array(demand.values))
            cfg["demand_p90"] = np.nan_to_num(np.array(demand.rolling(5, min_periods=1).quantile(.9, interpolation='midpoint').values))
            cfg["demand_p10"] = np.nan_to_num(np.array(demand.rolling(5, min_periods=1).quantile(.1, interpolation='midpoint').values))

            cfg["category_sum"]= np.array(category_sum.values)
            cfg["item_name"] =str(item_name)
            cfg["channel_name"] =str(channel)
            cfg["category_name"] =str(category)
            cfg["first_date"]=demand.index.min()
            cfg["last_date"]=demand.index.max()
            cfg["backtest_date"]=demand.iloc[:-horiz].index.max()

            cfg["horizon"]=horiz
            cfg["forecast_frequency"]= freq

            if min_len is None:
                required_min_length = 8 + horiz + 20
                #required_min_length = 8+3*self.horizon
                #required_min_length = 8 * self.horizon
            else:
                required_min_length = min_len

            if cfg["demand"].shape[0] <= required_min_length:

                padding = np.ones(required_min_length)*0.1

                padding[-cfg["demand"].shape[0]:] = cfg["demand"]

                cfg["demand"] = padding

                padding = np.ones(required_min_length)*0.1

                padding[-cfg["demand_p90"].shape[0]:] = cfg["demand_p90"]

                cfg["demand_p90"] = padding

                padding = np.ones(required_min_length)*0.1

                padding[-cfg["demand_p10"].shape[0]:] = cfg["demand_p10"]

                cfg["demand_p10"] = padding

                padding = np.ones(required_min_length)*0.1

                padding[-cfg["category_sum"].shape[0]:] = cfg["category_sum"]

                cfg["category_sum"] = padding

            #print(type(seasonalityFactor))
            cfgs.append(cfg)
        else:
            pass

    return cfgs

# ~~~

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


s3_paths_csv = f"{s3_input_path.replace('s3a:', 's3:')}"
s3_forecast_csv = s3_paths_csv.replace(".paths.csv", "") + ".forecast.csv"

df_paths = wr.s3.read_csv([s3_paths_csv])
all_preds = []

for pth in tqdm(df_paths["s3_parquet_path"]):
    df = wr.s3.read_parquet(f"{pth}.resample/")
    pred_list = vr.make_forecast(df, freq, horiz, lambda_arn, 5000, 1000)
    all_preds.extend(pred_list)

s3_json_path = f"{s3_forecast_csv.replace('.forecast.csv', '.results.json')}"
local_json_path = os.path.join(".", os.path.basename(s3_json_path))

with open(local_json_path, "w") as f:
    json.dump(all_preds, f, cls=NumpyEncoder)

wr.s3.upload(local_json_path, s3_json_path)
df_preds = pd.concat(vr.forecast_df(d) for d in all_preds)
wr.s3.to_csv(df_preds, s3_forecast_csv, index=False)

job.commit()

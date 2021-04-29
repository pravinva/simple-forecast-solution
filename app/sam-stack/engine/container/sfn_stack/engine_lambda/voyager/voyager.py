# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import os
import sys
import base64
import time
import re
import itertools
import json
import warnings
import argparse
import numpy as np
import pandas as pd
import boto3
import botocore
import cloudpickle
import awswrangler as wr

from concurrent import futures
from toolz import partition_all
from utilities import *

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from tqdm.auto import tqdm

KEY_SEP = "@@"
MIN_HORIZON = 8

# Forecast numeric precision
ROUND = 4
SFS_ERROR_COL = "Voyager_Error_MAPE"
NAIVE_ERROR_COL = "Naive_Error_MAPE"


def check_expected_bucket_owner(bucket, account_id=None):
    """Check that the bucket owner is as expected. This function will raise
    a `ClientError` exception for an unexpected bucket owner.

    Parameters
    ----------
    bucket : str
    account_id : str (optional)
        The AWS account ID of the expeted bucket owner, it will be the AWS
        account ID of the account that calls this function by default.

    """
    bucket = re.sub(r"^s3://", "", bucket).rstrip("/")
    client = boto3.client("s3")

    if account_id is None:
        account_id = boto3.client("sts").get_caller_identity()["Account"]

    try:
        client.get_bucket_acl(Bucket=bucket, ExpectedBucketOwner=account_id)
    except ClientError as e:
        # http_status -> 403 if bucket owner is not as expected
        # http_status = e.response["ResponseMetadata"]["HTTPStatusCode"]
        raise(e)

    return


def read_csv_s3(bucket, s3_key, *args, **kwargs):
    """

    Parameters
    ----------

    Returns
    -------

    """

    check_expected_bucket_owner(bucket, account_id=None)

    s3 = boto3.client("s3")

    if s3_key.endswith(".parquet"):
        #df = wr.s3.read_parquet(f"s3://{bucket}/{s3_key}")
        local_path = f"/tmp/{os.path.basename(s3_key)}"
        with Timer(f"downloading {local_path}"):
            s3.download_file(bucket, s3_key, local_path)
        df = pd.read_parquet(local_path)
    else:
        df = wr.s3.read_csv(f"s3://{bucket}/{s3_key}", *args, **kwargs)

    return df


def write_csv_s3(bucket, s3_key, *args, **kwargs):
    """
    """

    check_expected_bucket_owner(bucket, account_id=None)
    df = pd.to_csv(f"s3://{bucket}/{s3_key}", *args, **kwargs)

    return


def grouper(n, iterable):
    """
    >>> list(grouper(3, 'ABCDEFG'))
    [['A', 'B', 'C'], ['D', 'E', 'F'], ['G']]
    """
    iterable = iter(iterable)
    return iter(lambda: list(itertools.islice(iterable, n)), [])


def jsonify_wren_rows(wren_rows):
    """Convert the pywren output to a JSON-compatible format.

    """

    int_cols = ["demand", "demand_p90", "demand_p10",
                "forecast_horizon_backtest", "forecast", "forecast_p90",
                "forecast_p10"]

    for row in wren_rows:
        for col in int_cols:
            assert((row[col] < 0).sum() == 0)
            assert(np.isinf(row[col]).sum() == 0)

            row[col] = np.nan_to_num(row[col], posinf=0, neginf=0) \
                         .astype(int) \
                         .tolist()

    return wren_rows


def spectralEntropy(series):
    series = series.dropna()

    f, Pxx_den = signal.periodogram(series)
    psd_norm = np.divide(Pxx_den, Pxx_den.sum())
    psd_norm += 0.000001
    se = -np.multiply(psd_norm, np.log2(psd_norm)).sum()
    return(se)


def lambdaBoxCox(series):
    xt, lam = stats.boxcox(series.clip(lower=0.1))
    return(lam)


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


class Timer(object):
    def __init__(self, description):
        self.description = description

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        self.end = time.time()
        print(f"{self.description}: {((self.end - self.start) / 60):.2f}mins")


class Engine:
    """
    """

    def __init__(self, s3_bucket=None, s3_filename=None, fc_freq=None,
        horizon=None, fn=None, include_naive=True, debug=False, verbose=False):
        """

        Parameters
        ----------
        fn : str, optional
            Path to the input CSV file.
        debug : bool, optional
            Toggle debugging mode for rapid (local) completion of a forecast,
            cloud resources are not used.

        """

        self._s3_bucket = s3_bucket
        self.s3_filename = s3_filename
        self.fn = fn
        self.fc_freq = remap_freq(fc_freq)
        self.horizon_orig = horizon
        self.horizon = max(MIN_HORIZON, horizon)
        self.include_naive = include_naive
        self.debug = debug

        bucket = self._s3_bucket
        s3_key = s3_filename 

        if fn is None:
            df = read_csv_s3(bucket, s3_key, dtype=INPUT_DTYPE)
        else:
            df = pd.read_csv(self.fn, dtype=INPUT_DTYPE)

        df['timestamp'] = pd.DatetimeIndex(df['timestamp'])

        if 'family' not in df:
            df['category'] = df['channel']
        else:
            df['family'] = df['family'].fillna("UNKNOWN")
            df['category'] = df['family']

        df["key"] = (
            df['item_id'].astype(str) + KEY_SEP +
            df['channel'].astype(str) + KEY_SEP + df['category'].astype(str)
        )

        df["key"] = df["key"].str.strip()

        itemSeriesTransient = \
            pd.pivot_table(df, values='demand', index=["timestamp"],
                           columns=['key'], aggfunc=np.sum)
        itemSeriesTransient= itemSeriesTransient.clip(0,np.inf)

        if fc_freq in ("MS",):
            df_resampled = itemSeriesTransient.resample(self.fc_freq).sum()
        else:
            df_resampled = itemSeriesTransient.resample(self.fc_freq,
                                label="right", closed="right").sum()

        df_resampled = df_resampled.fillna(0) \
                                   .unstack() \
                                   .reset_index() \
                                   .rename({0: "demand"}, axis=1)

        df_resampled["timestamp"] = pd.DatetimeIndex(df_resampled["timestamp"]) \
                                      .strftime("%Y-%m-%d")

        out = df_resampled
        out[['item_id','channel','category']] = out['key'].str.split(KEY_SEP,expand=True)
        out["key"] = out['item_id'].astype(str) + KEY_SEP + out['channel'].astype(str)

        out = out.reset_index(drop=True)
        out["category_sum"] = out.groupby(['category','timestamp']).demand.transform('sum')
        out["category_sum"] = out["category_sum"].clip(out["category_sum"].quantile(0.2)+1,np.inf)
        out.set_index('timestamp', inplace=True)

        self.resampled_data = out

        return

    @property
    def s3_bucket(self):
        check_expected_bucket_owner(self._s3_bucket)
        return self._s3_bucket

    @s3_bucket.setter
    def s3_bucket(self, value):
        self._s3_bucket = value
        return

    def calculate_product_metrics(self,df):
        """
        This code returns various KPIs on a SKU level. It is expected that you call this function from a groupby within the eda function.
        """
        from scipy import signal
        from scipy import stats
        from scipy.special import boxcox, inv_boxcox

        d = {}
        d['family'] = str(df.category.iloc[0])
        d['series_length'] = df.demand.values.size
        d['product_life_periods'] = np.trim_zeros(df.demand.values).size
        d['median_demand'] = np.nanmedian(np.trim_zeros(df.demand.values))
        d['tail_demand'] = df.demand.tail(8).mean()

        if d['tail_demand'] < 1:
            d["retired"] = True
        else:
            d["retired"] = False

        if d["retired"] and d['product_life_periods'] < d['series_length']/4:
                d['category'] = "Short"
        elif d["retired"] and d['product_life_periods'] > d['series_length']/4:
                d['category'] = "Medium"
        else:
            d['category'] = "Continuous"

        d["spectralEntropy"] = spectralEntropy(df.demand)

        if d['spectralEntropy'] >5:
            d["intermittent"] = True
        else:
            d["intermittent"] = False

        try:
            d["lamb"] = lambdaBoxCox(df.demand)

        except:
            d["lamb"] = np.nan

        return pd.Series(d)

    def eda(self):
        """
        This function generated the various KPIs on individual demand series.
        """
#       self.resampled_data \
#           .groupby(["item_id","channel"]) \
#           .apply(self.calculate_product_metrics) \
#           .reset_index() \
#           .to_csv("s3://" + self.s3_bucket + "/" + self.s3_filename
#                   + ".report", index=False)
        s3 = boto3.resource('s3')

        self.resampled_data \
            .groupby(["item_id","channel"]) \
            .apply(self.calculate_product_metrics) \
            .reset_index() \
            .to_csv("/tmp/data.report", index=False)

        s3.Bucket(self.s3_bucket) \
          .upload_file("/tmp/data.report", self.s3_filename + ".report")

        return

    def forecast(self, model_name=None, ignore_naive=False, quantile=None,
        min_len=None):
        """
        """
        items = []
        channel_group = self.resampled_data.groupby(['key'])

        print("Number of keys", self.resampled_data.key.nunique())
        print("Number of items", self.resampled_data.item_id.nunique())
        print("Number of channels", self.resampled_data.channel.nunique())
        print("Number of category", self.resampled_data.category.nunique())

        for channel_name, channel_item in channel_group:
            demand = channel_item.demand
            item_name = channel_item.item_id.iloc[0]
            channel_name = channel_item.channel.iloc[0]
            category_name = channel_item.category.iloc[0]
            category_sum = channel_item.category_sum
            if demand.sum() >= 0:
                item_dict ={}
                item_dict["demand"] = np.nan_to_num(np.array(demand.values))
                item_dict["demand_p90"] = np.nan_to_num(np.array(demand.rolling(5, min_periods=1).quantile(.9, interpolation='midpoint').values))
                item_dict["demand_p10"] = np.nan_to_num(np.array(demand.rolling(5, min_periods=1).quantile(.1, interpolation='midpoint').values))

                item_dict["category_sum"]= np.array(category_sum.values)
                item_dict["item_name"] =str(item_name)
                item_dict["channel_name"] =str(channel_name)
                item_dict["category_name"] =str(category_name)
                item_dict["first_date"]=demand.index.min()
                item_dict["last_date"]=demand.index.max()
                item_dict["backtest_date"]=demand.iloc[:-self.horizon].index.max()
                item_dict["horizon"]=self.horizon
                item_dict["forecast_frequency"]=self.fc_freq

                if min_len is None:
                    required_min_length = 8+self.horizon+20
                    #required_min_length = 8+3*self.horizon
                    #required_min_length = 8 * self.horizon
                else:
                    required_min_length = min_len

                if item_dict["demand"].shape[0] <= required_min_length:

                    padding = np.ones(required_min_length)*0.1

                    padding[-item_dict["demand"].shape[0]:] = item_dict["demand"]

                    item_dict["demand"] = padding

                    padding = np.ones(required_min_length)*0.1

                    padding[-item_dict["demand_p90"].shape[0]:] = item_dict["demand_p90"]

                    item_dict["demand_p90"] = padding

                    padding = np.ones(required_min_length)*0.1

                    padding[-item_dict["demand_p10"].shape[0]:] = item_dict["demand_p10"]

                    item_dict["demand_p10"] = padding

                    padding = np.ones(required_min_length)*0.1

                    padding[-item_dict["category_sum"].shape[0]:] = item_dict["category_sum"]

                    item_dict["category_sum"] = padding

                #print(type(seasonalityFactor))
                items.append(item_dict)
            else:
                pass

        seq = items

        print("Number of SKU-Channel pairs input ", len(items))

        wren_rows = []

        if self.debug:
            # one experiment -> one sku-channel combo
            list_of_experiments = list(grouper(1000, seq))

            for experiment_list in list_of_experiments[:1]:
                for e in experiment_list[:1]:
                    wren_rows.append(best_model_forecast(e,
                        ignore_naive=ignore_naive, debug=True))

            wren_rows = jsonify_wren_rows(wren_rows)

            with open("/tmp/results.json", "w") as f:
                json.dump(wren_rows, f)

            out = wren_rows
        else:
            executor = LambdaExecutor(
                max_workers=1000, # max. no. of concurrent lambdas
                lambda_arn="EngineMapStack-EngineMapFunction")

            check_expected_bucket_owner(self._s3_bucket)

            payloads = [{"args": tuple(), "kwargs": {"config": a}} for a in seq]
            results = executor.map(best_model_forecast, payloads)

            print("Number of SKU-Channel forecasts ", len(results))

#           with open("/tmp/results.json", "w") as f:
#               json.dump(results, f)

#           s3 = boto3.resource('s3')
#           s3.Bucket(self.s3_bucket) \
#             .upload_file("/tmp/results.json",
#                          self.s3_filename + ".results.json")

#       payloads = [{"args": (r,), "kwargs": {}} for r in results]
#       results = executor.map(best_model_forecast, payloads)

        with Timer("pd.concat of results"):
            out = pd.concat(results)

        #
        # Truncate forecast time-series to desired forecast horizon length
        #
        if self.fc_freq in ("D",):
            unit = "days"
        elif self.fc_freq in ("W", "W-MON",):
            unit = "weeks"
        elif self.fc_freq in ("M", "MS",):
            unit = "months"
        else:
            raise NotImplementedError

        offset = self.horizon_orig

        if unit == "months":
            out["index"] = \
                (pd.DatetimeIndex(out["index"]) + pd.offsets.MonthEnd(0) -
                 pd.offsets.MonthBegin(normalize=True))

            assert(out["index"].isnull().sum() == 0)
            offset += 1

        dfs = []

        with Timer("trimming each forecast"):
            for _, dd in out.groupby(["item_id", "channel", "family"],
                    as_index=False, sort=False):

                dt_start = pd.Timestamp(dd[~dd["demand"].isnull()]["index"].max())
                dt_stop = dt_start + pd.DateOffset(**{unit: offset})
                dt_stop = dt_stop.strftime("%Y-%m-%d")

                dfs.append(dd[dd["index"] <= dt_stop])

        dt_min = self.resampled_data.index.min()

        start = time.time()

        out = pd.concat(dfs).query(f"index >= '{dt_min}'")

        elapsed = (time.time() - start) / 60.

        print(f"1 elapsed: {elapsed:.2f}mins")

        if self.debug:
            #print(out.query("not forecast_backtest.isnull().values"))
            print(out.tail(24))
        else:

            if self.s3_filename.endswith(".parquet"):
                s3_output_path = \
                    f"s3://{self.s3_bucket}/{self.s3_filename}.forecast.parquet"
                wr.s3.to_parquet(out, s3_output_path, index=False)
            else:
                s3_output_path = \
                    f"s3://{self.s3_bucket}/{self.s3_filename}.forecast.csv"
                wr.s3.to_csv(out, s3_output_path, index=False)

#           pd.pivot_table(out.reset_index(),
#                          values=['demand','forecast'],
#                          index=["item_id"],
#                          columns=['index']) \
#             .to_csv("/tmp/pivot_results.csv")

#           s3.Bucket(self.s3_bucket) \
#             .upload_file("/tmp/pivot_results.csv",
#                          self.s3_filename + ".pivot_forecast.csv")

#           print(f"COMPLETED: "
#                   f"s3://{self.s3_bucket}/{self.s3_filename}.forecast.csv")
            print(s3_output_path)


# --- SFS v1.1

class Timer:
    def __init__(self, func=time.perf_counter):
        self.elapsed = 0.0
        self._func = func
        self._start = None

    def start(self):
        if self._start is not None:
            raise RuntimeError('Already started')
        self._start = self._func()

    def stop(self):
        if self._start is None:
            raise RuntimeError('Not started')
        end = self._func()
        self.elapsed += end - self._start
        self._start = None

    def reset(self):
        self.elapsed = 0.0

    @property
    def running(self):
        return self._start is not None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def fill_dt_spine(dd, freq, horiz=None):
    """Fill in the missing timestamps with NaN demand.

    """
    df_spine = pd.DataFrame()
    dt_start = dd["timestamp"].min()
    
    xs_timestamp = \
        pd.date_range(dt_start, dd["timestamp"].max(), freq=freq) \
          .strftime("%Y-%m-%d")

    min_len = np.ceil(horiz * 1.5).astype(int)
    
    if horiz is not None:
        if xs_timestamp.shape[0] < min_len:
            periods = min_len - xs_timestamp.size
            xs_timestamp_xtra = \
                pd.date_range(end=dt_start, freq=freq, periods=periods, closed="left") \
                  .strftime("%Y-%m-%d")
            xs_timestamp = xs_timestamp.append(xs_timestamp_xtra)

#   if min_len is not None:
#       if xs_timestamp.size < min_len:
#           diff_len = min_len - df_spine.size
#           xs_timestamp = \
#               pd.date_range(end=xs_timestamp[0], periods=diff_len, freq=freq)
#           assert(xs_timestamp.size == min_len)

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
    

def resample(df, freq, horiz=None):
    """Resample a dataframe of demand to a desired frequency.

    """

    if "category" not in df:
        df.rename({"family": "category"}, axis=1, inplace=True)

    df.loc[:,"timestamp"] = pd.DatetimeIndex(df["timestamp"])
    
    # resample time series frequency
    df2 = \
    df.groupby([pd.Grouper(key="timestamp", freq=freq),
                "channel", "category", "item_id"]) \
      .agg({"demand": lambda xs: xs.sum(min_count=1)}) \
      .reset_index() ;
    
    df2.loc[:,"timestamp"] = df2["timestamp"].dt.strftime("%Y-%m-%d")

    # fill in missing timestamps
    df2 = df2.groupby(["channel", "category", "item_id"]) \
             .apply(lambda dd: fill_dt_spine(dd, freq, horiz)) \
             .reset_index(drop=True)

    return df2


def preprocess(df, freq, horiz=None, batch_size=1000):
    """Preprocess a "raw" demand dataframe of multiple time-series, resampling
    demand according a specified frequency (`freq`).

    """

    # resample demand to `freq`

#   df2 = resample(df, freq, horiz).copy()

    # resample in batches
    groups = df.groupby(["channel", "category", "item_id"])

    max_workers = 1000
#   batch_size = 5000
    n_batches = int(np.ceil(groups.ngroups / batch_size))

    executor = LambdaExecutor(
        max_workers=max_workers, # max. no. of concurrent lambdas
        lambda_arn="EngineMapStack-EngineMapFunction")

    resamples = []

    for grp_batch in tqdm(partition_all(batch_size, groups), total=n_batches):
        payloads = [{"args": (dd, freq, horiz), "kwargs": {}}
                    for _, dd in grp_batch]
        results = executor.map(resample, payloads)
        resamples.extend(results)

    df2 = pd.concat(resamples)

    df2.loc[:,"timestamp"] = \
        pd.DatetimeIndex(df2["timestamp"]) \
          .strftime("%Y-%m-%d")

    # add category-level sums
    df_cat_sums = df2.groupby(["category", "timestamp"]) \
                     .agg({"demand": sum}) \
                     .reset_index() \
                     .rename({"demand": "category_sum"}, axis=1)

    df2 = df2.merge(df_cat_sums, on=["timestamp", "category"], how="left")

    df2.loc[:,"timestamp"] = pd.DatetimeIndex(df2["timestamp"])

    df2.sort_values(by=["channel", "category", "item_id", "timestamp"],
                    inplace=True)

    df2.set_index("timestamp", inplace=True)

    return df2


from scipy import signal
from scipy import stats
from scipy.special import boxcox, inv_boxcox


def calc_series_metrics(df):
    """
    """

    d = {}
    d['family'] = str(df.category.iloc[0])
    d['series_length'] = df.demand.values.size
    d['product_life_periods'] = np.trim_zeros(df.demand.values).size
    d['median_demand'] = np.nanmedian(np.trim_zeros(df.demand.values))
    d['tail_demand'] = df.demand.tail(8).mean()

    if d['tail_demand'] < 1:
        d["retired"] = True
    else:
        d["retired"] = False

    if d["retired"] and d['product_life_periods'] < d['series_length']/4:
            d['category'] = "Short"
    elif d["retired"] and d['product_life_periods'] > d['series_length']/4:
            d['category'] = "Medium"
    else:
        d['category'] = "Continuous"

    d["spectralEntropy"] = spectralEntropy(df.demand)

    if d['spectralEntropy'] >5:
        d["intermittent"] = True
    else:
        d["intermittent"] = False

    try:
        d["lamb"] = lambdaBoxCox(df.demand)

    except:
        d["lamb"] = np.nan

    return pd.Series(d)


def make_report(df):
    """
    """

    df_report = \
        df.groupby(["item_id", "channel"]) \
          .apply(calc_series_metrics) \
          .reset_index()

    return df_report


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

    for (channel, category, item_id), dd in tqdm(groups):
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

            assert(not pd.isnull(cfg["backtest_date"]))

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


def make_forecast(df, freq, horiz, lambda_arn, batch_size=1000,
        max_workers=1000, dev_mode=False):
    """
    """

    assert("timestamp" in df)

    df.loc[:,"timestamp"] = pd.DatetimeIndex(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    n_series = df[["channel", "category", "item_id"]] \
                    .drop_duplicates().shape[0]
    n_batches = int(np.ceil(n_series / batch_size))

    executor = LambdaExecutor(
        max_workers=max_workers, # max. no. of concurrent lambdas
        lambda_arn=lambda_arn)

    # generate payload(s)
    payloads_iter = ({"args": tuple(), "kwargs": {"config": c}}
                     for c in make_cfgs(df, freq, horiz))

    # store the forecasts as a list of dataframes
    pred_list = []

    # partition payloads into chunks of 1000 
    for batch in tqdm(partition_all(batch_size, payloads_iter), total=n_batches):
        batch_results = executor.map(best_model_forecast, batch)
        pred_list.extend(batch_results)

    return pred_list

# ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bucket", action="store", dest="bucket", required=True)
    parser.add_argument("-k", "--key", action="store", dest="key", required=True)
    parser.add_argument("-f", "--freq", action="store", dest="freq", required=True)
    parser.add_argument("-H", "--horizon", action="store", dest="horizon",
            type=int, required=True)

    args = parser.parse_args()

    event = {"queryStringParameters": {}}

    event["queryStringParameters"]["bucket"] = args.bucket
    event["queryStringParameters"]["file"] = args.key
    event["queryStringParameters"]["frequency"] = args.freq
    event["queryStringParameters"]["horizon"] = args.horizon
    event["queryStringParameters"]["include_naive"] = False
    event["queryStringParameters"]["debug"] = True

    engine = Engine(s3_bucket=args.bucket, s3_filename=args.key, 
            horizon=args.horizon, fc_freq=args.freq)

    engine.forecast()

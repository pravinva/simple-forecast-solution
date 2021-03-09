# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import os
import re
import sys
import numpy as np
import pandas as pd
import pywren
import itertools
import json
import boto3

from utilities import *

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
    df = pd.read_csv(f"s3://{bucket}/{s3_key}", *args, **kwargs)

    return df


def write_csv_s3(bucket, s3_key, *args, **kwargs):
    """
    """

    check_expected_bucket_owner(bucket, account_id=None)
    df = pd.to_csv(f"s3://{bucket}/{s3_key}", *args, **kwargs)

    return


def forecast_df(row):
    """
    """
    item_id = row["item_name"]
    channel = row["channel_name"]
    family = row["category_name"]
    forecast = row["forecast"]
    backtest_date = row["backtest_date"]
    forecast_horizon_backtest= row["forecast_horizon_backtest"]
    forecast_p90 = row["forecast_p90"]
    forecast_p10 = row["forecast_p10"]
    start_date = row["last_date"]
    frequency = row["forecast_frequency"]
    forecast_periods = len(forecast)
    demand = row["demand"]
    demand_periods = len(demand)
    model_name = row["model_name"]

    assert(isinstance(model_name, str))

    forecast_dates = pd.date_range(start = start_date, periods=forecast_periods+1, freq=frequency)
    forecast_series = pd.Series(forecast, index=forecast_dates[1:])
    backtest_dates= pd.date_range(start = backtest_date, periods=forecast_periods+1, freq=frequency)
    forecast_backtest = pd.Series(forecast_horizon_backtest, index=backtest_dates[1:])
    forecast_p90_series = pd.Series(forecast_p90, index=forecast_dates[1:])
    forecast_p10_series = pd.Series(forecast_p10, index=forecast_dates[1:])

    demand_dates = pd.date_range(end = start_date, periods=demand_periods, freq=frequency)
    demand_series = pd.Series(demand, index=demand_dates)

    df_sku = pd.DataFrame({
        "demand": demand_series,
        "forecast": forecast_series,
        "forecast_backtest": forecast_backtest, 
        "forecast_p90": forecast_p90_series, 
        "forecast_p10": forecast_p10_series
    })

    df_sku["model_name"] = model_name

    sku_sfs_error= row[SFS_ERROR_COL]
    sku_naive_error = row[NAIVE_ERROR_COL]

    df_sku["item_id"] = item_id
    df_sku["channel"] = channel
    df_sku["family"] = family

    # backtest metrics
    ix_backtest = ~df_sku["forecast_backtest"].isnull()
    ys_backtest = df_sku[ix_backtest]["demand"]
    yp_backtest = df_sku[ix_backtest]["forecast_backtest"]

    ape_backtest = calc_ape(ys_backtest, yp_backtest)
    aape_backtest = calc_aape(ys_backtest, yp_backtest)

    df_sku.loc[ix_backtest,"sfs_bt_aape"] = aape_backtest.round(ROUND)
    df_sku.loc[ix_backtest,"sfs_bt_ape"] = ape_backtest.round(ROUND)
    df_sku.loc[ix_backtest,"sfs_bt_acc"] = 1.0 - ape_backtest.round(ROUND)

    # historic (temporal) cross-validation metrics
    df_sku["naive_cv_error"] = sku_naive_error.round(ROUND)
    df_sku["naive_cv_acc"] = 1 - df_sku["naive_cv_error"].round(ROUND)

    df_sku["sfs_cv_error"] = sku_sfs_error.round(ROUND)
    df_sku["sfs_cv_acc"] = 1 - df_sku["sfs_cv_error"].round(ROUND)

    df_sku.reset_index(inplace=True)
    return df_sku


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


def read_csv_s3(bucket, s3_key, *args, **kwargs):
    """

    Parameters
    ----------

    Returns
    -------

    """

    check_expected_bucket_owner(bucket, account_id=None)
    df = pd.read_csv(f"s3://{bucket}/{s3_key}", *args, **kwargs)

    return df


def write_csv_s3(bucket, s3_key, *args, **kwargs):
    """
    """

    check_expected_bucket_owner(bucket, account_id=None)
    df = pd.to_csv(f"s3://{bucket}/{s3_key}", *args, **kwargs)

    return


class Engine:
    """
    """

    def __init__(self, s3_bucket=None, s3_filename=None, fc_freq=None,
        horizon=None, fn=None, debug=False, verbose=False):
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
        self.debug = debug

        bucket = self._s3_bucket
        s3_key = s3_filename 

        #
        #
        #
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

        def spectralEntropy(series):
            series = series.dropna()

            f, Pxx_den = signal.periodogram(series)
            psd_norm = np.divide(Pxx_den, Pxx_den.sum())
            psd_norm += 0.000001
            se = -np.multiply(psd_norm, np.log2(psd_norm)).sum()

            return(se)

        d["spectralEntropy"] = spectralEntropy(df.demand)

        if d['spectralEntropy'] >5:
            d["intermittent"] = True
        else:
            d["intermittent"] = False

        def lambdaBoxCox(series):

            xt, lam = stats.boxcox(series.clip(lower=0.1))

            return(lam)

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

    def forecast(self):
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

                required_min_length = 8+self.horizon+20
                #required_min_length = 8+3*self.horizon
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

        # one experiment -> one sku-channel combo
        list_of_experiments = list(grouper(1000, seq))

        wren_rows = []

        if self.debug:
            for experiment_list in list_of_experiments[:1]:
                for e in experiment_list[:1]:
                    wren_rows.append(best_model_forcast(e, debug=True))

            wren_rows = jsonify_wren_rows(wren_rows)

            with open("/tmp/results.json", "w") as f:
                json.dump(wren_rows, f)

            out = wren_rows
        else:
            wrenexec = pywren.default_executor()
            check_expected_bucket_owner(self._s3_bucket)
            for experiment_list in list_of_experiments:
                futures = wrenexec.map(best_model_forcast, experiment_list)
                wren_rows.extend(pywren.get_all_results(futures))

            wren_rows = jsonify_wren_rows(wren_rows)

            print("Number of SKU-Channel forecasts ",len(wren_rows))

            with open("/tmp/results.json", "w") as f:
                json.dump(wren_rows, f)

            s3 = boto3.resource('s3')
            s3.Bucket(self.s3_bucket) \
              .upload_file("/tmp/results.json",
                           self.s3_filename + ".results.json")

        out = pd.concat([forecast_df(row) for row in wren_rows])

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

        for _, dd in out.groupby(["item_id", "channel", "family"],
                as_index=False, sort=False):

            dt_start = pd.Timestamp(dd[~dd["demand"].isnull()]["index"].max())
            dt_stop = dt_start + pd.DateOffset(**{unit: offset})
            dt_stop = dt_stop.strftime("%Y-%m-%d")

            dfs.append(dd[dd["index"] <= dt_stop])

        dt_min = self.resampled_data.index.min()

        out = pd.concat(dfs).query(f"index >= '{dt_min}'")
        out.to_csv("/tmp/forecast.csv")

        if self.debug:
            #print(out.query("not forecast_backtest.isnull().values"))
            print(out.tail(24))
        else:
            # Use S3
            s3.Bucket(self.s3_bucket) \
              .upload_file("/tmp/forecast.csv",
                           self.s3_filename + ".forecast.csv")

            pd.pivot_table(out.reset_index(),
                           values=['demand','forecast'],
                           index=["item_id"],
                           columns=['index']) \
              .to_csv("/tmp/pivot_results.csv")

            s3.Bucket(self.s3_bucket) \
              .upload_file("/tmp/pivot_results.csv",
                           self.s3_filename + ".pivot_forecast.csv")

            print(f"COMPLETED: "
                    f"s3://{self.s3_bucket}/{self.s3_filename}.forecast.csv")


if __name__ == "__main__":
    PWD = os.path.dirname(os.path.realpath(__file__))

#   fn = os.path.join(PWD, "..", "refdata", "sample-demand-01-small.csv")

#   engine = Engine(fn=fn, fc_freq="W-MON", horizon=12, debug=True)
#   engine.forecast()

    horizon = 12
    fc_freq = "W"
    s3_bucket = sys.argv[1]
#   s3_filename = "sample-demand-01-small.csv"
#   s3_filename = "favorita_monthly_modified.csv"
    s3_filename = sys.argv[2]

    v = Engine(s3_bucket, s3_filename, fc_freq, horizon, debug=False)
#   v = Engine(fn="~/notebooks/data/aptiv-actuals.csv", fc_freq=fc_freq,
#           horizon=horizon, debug=True)
    v.eda()
    v.forecast()

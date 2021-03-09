# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import os
import re
import urllib.parse
import json
import boto3
import awswrangler as wr

from awswrangler._utils import pd
from botocore.exceptions import ClientError


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


def lambda_handler(event=None, context=None):
    """
    """
    # Extract AF forecast path components
    bucket = urllib.parse.unquote(event["Records"][0]["s3"]["bucket"]["name"])
    key = urllib.parse.unquote(event["Records"][0]["s3"]["object"]["key"])
    
    # Read the AF forecast parameters JSON
    s3 = boto3.resource("s3")
    
    s3_af_params_key = \
        os.path.join(os.path.dirname(key),
                     f"{os.path.basename(key).split('.')[0]}.af-params.json")
    
    check_expected_bucket_owner(bucket)
    obj = s3.Object(bucket, s3_af_params_key)
    
    af_params = json.load(obj.get()["Body"])
    fc_horizon = af_params["Predictor"]["ForecastHorizon"]
    fc_freq = af_params["Predictor"]["FeaturizationConfig"]["ForecastFrequency"]

    # Read the original AF forecast file from S3
    s3_af_fcast_path = f"s3://{os.path.join(bucket, key)}"
    
    check_expected_bucket_owner(bucket)
    df = wr.s3.read_csv(s3_af_fcast_path)
    
    # Resample the AF forecast file to meet the forecast frequency
    agg_funcs = {
        "Unnamed: 0": "first",
        "demand": sum,
        "forecast": "first",
        "forecast_backtest": "first",
        "forecast_p90": "first",
        "forecast_p10": "first"
    }
    
    col_order = df.columns.tolist()

    if fc_freq in ("W", "W-SUN",):
        fc_freq = "W-MON"
    
    df_hist_resampled = \
        df.assign(index=pd.DatetimeIndex(df["index"])) \
          .query("forecast.isnull() and ~demand.isnull()", engine="python") \
          .groupby([pd.Grouper(key="index", freq=fc_freq), "item_id", "channel", "family"], sort=False) \
          .agg(agg_funcs) \
          .reset_index() \
          .loc[:,col_order] \
          .sort_values(by=["channel", "family", "item_id", "index"], ascending=True)

    print(df_hist_resampled)

    if fc_freq == 'M':
        df_hist_resampled["index"] = \
            (pd.DatetimeIndex(df_hist_resampled["index"]) +
             pd.offsets.MonthEnd(0) -
             pd.offsets.MonthBegin(normalize=True)).strftime("%Y-%m-%d")
    else:
        df_hist_resampled["index"] = \
            pd.DatetimeIndex(df_hist_resampled["index"]).strftime("%Y-%m-%d")
    
    df_fcast = df.query("~forecast.isnull() and demand.isnull()")
    df_resampled = pd.concat([df_hist_resampled, df_fcast])
    
    # Generate the output filename
    s3_output = \
        os.path.join(os.path.dirname(s3_af_fcast_path),
                    f"{os.path.basename(s3_af_fcast_path).split('.')[0]}.af-resampled.csv")
    
    # Write the resampled AF forecast file back to S3
    check_expected_bucket_owner(bucket)
    wr.s3.to_csv(df_resampled, s3_output)
    
    return

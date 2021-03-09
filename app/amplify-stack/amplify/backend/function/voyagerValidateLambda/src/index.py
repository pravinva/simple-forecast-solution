# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import os
import re
import datetime
import json
import logging
import boto3
import awswrangler as wr

from botocore.exceptions import ClientError
from awswrangler._utils import pd

CORS_HEADERS = {
    'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token,X-Amz-User-Agent',
    'Access-Control-Allow-Methods': 'DELETE,GET,HEAD,OPTIONS,PATCH,POST,PUT',
    'Access-Control-Allow-Origin': '*',
}

# Logging defaults
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


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
    bucket = re.sub(r"^s3://", "", bucket)
    client = boto3.client("s3")

    LOGGER.info(bucket)

    if account_id is None:
        account_id = boto3.client("sts").get_caller_identity()["Account"]

    try:
        client.get_bucket_acl(Bucket=bucket, ExpectedBucketOwner=account_id)
    except ClientError as e:
        # http_status -> 403 if bucket owner is not as expected
        # http_status = e.response["ResponseMetadata"]["HTTPStatusCode"]
        raise(e)

    return


def handler(event, context):
    if "bucket" in event:
        s3_bucket = event["bucket"]
    else:
        s3_bucket = os.environ.get("STORAGE_VOYAGERFRONTENDAUTH_BUCKETNAME")

    body = json.loads(event["body"]) if 'body' in event else event
    fileS3Key = body["fileS3Key"]
    horizonAmount = int(body["horizonAmount"])
    horizonUnit = body["horizonUnit"]
    frequencyUnit = body["frequencyUnit"]
    s3_key = fileS3Key.lstrip("/")

    check_expected_bucket_owner(s3_bucket)

    full_s3_path = f"s3://{s3_bucket.rstrip('/')}/{s3_key}"

    try:
        df = wr.s3.read_csv(full_s3_path,
                dtype={"item_id": str, "channel": str, "family": str,
                       "timestamp": str})
        validation_payload = \
            validate(df, horizonAmount, horizonUnit, frequencyUnit)
        status_code = 200

        LOGGER.info(f"{full_s3_path} passed validation")
    except:
        validation_payload = { "status": None, "validations": None }
        status_code = 500

        # Delete the file if it doesn't pass validation.
        s3 = boto3.resource("s3")
        s3.Object(f"{s3_bucket}", f"{s3_key}").delete()

        LOGGER.info(f"{full_s3_path} failed validation")
        LOGGER.info(f"{full_s3_path} deleted")

    return {
        "isBase64Encoded": False,
        "statusCode": status_code,
        "headers": CORS_HEADERS,
        "body": json.dumps(validation_payload)
    }


def validate(df, horizonAmount, horizonUnit, frequencyUnit):
    """Validate a dataframe of historical demand data based on the required
    forecast parameters.

    """
    errors = []
    warnings = []
    exp_cols = ["timestamp", "channel", "item_id", "demand", "family"]

    for col in exp_cols:
        if col not in df:
            errors.append({"type": "error",
                           "message": f"Missing column '{col}'"})

    # timestamp format check
    try:
        df["timestamp"] = df["timestamp"].astype(str)
        df["timestamp"] = \
                df["timestamp"].apply(
                    lambda x: datetime.datetime.strptime(x, "%Y%m%d"))
        df["timestamp"] = pd.DatetimeIndex(df["timestamp"])
    except:
        logging.exception("Timestamp format check")
        errors.append({"type": "error",
            "message":
                f"Invalid 'timestamp' format, please reformat the "
                 "timestamps in the format: 'YYYYMMDD'"})

    if len(errors) == 0:
        #
        # Check minimum req. time periods for each sku
        #
        freq = horizonUnit

        if horizonUnit in ("M", "MS"):
            horizon_unit_human = "months"
        elif horizonUnit in ("W", "W-SUN"):
            horizon_unit_human = "weeks"
            freq = "W-MON"
        elif horizonUnit in ("D"):
            horizon_unit_human = "days"
        else:
            raise NotImplementedError

        for (channel, item_id), dd in \
            df.groupby(["channel", "item_id"], as_index=False):
            ts_min = dd["timestamp"].min()
            ts_max = dd["timestamp"].max()

            if horizon_unit_human == "months":
                dt_range = \
                    pd.date_range(
                        *(pd.to_datetime([ts_min, ts_max])
                            + pd.offsets.MonthEnd()), freq=freq)
            else:
                dt_range = pd.date_range(ts_min, ts_max, freq=freq)

            if len(dt_range) < horizonAmount:
                warnings.append({
                    "type": "warning",
                    "message":
                        f"item_id={item_id} channel={channel} has less than "
                        f"{horizonAmount} {horizon_unit_human} of "
                        "historical demand"})

    # demand values check
    n_neg_demand = (df["demand"] < 0).sum()

    if n_neg_demand > 0:
        warnings.append({"type": "warning",
            "message": f"{n_neg_demand} rows contain negative demand values"})

    # check for extraneous columns (warnings)
    extra_cols = set(df.columns.tolist()) - set(exp_cols)

    if len(extra_cols) > 0:
        for col in extra_cols:
            warnings.append({
                "type": "warning",
                "message": f"Unknown column '{col}'"})

    if len(errors) > 0:
        status = "failed"
    else:
        status = "passed"

    return {
        "status": status,
        "validations": errors + warnings
    }

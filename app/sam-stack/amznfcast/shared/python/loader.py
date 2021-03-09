# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import re
import logging
import boto3

from botocore.exceptions import ClientError
from boto3 import client


class Loader: 
    def __init__(self): 
        self.forecast_cli = client('forecast') 
        self.logger = logging.getLogger() 
        self.logger.setLevel(logging.INFO) 


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

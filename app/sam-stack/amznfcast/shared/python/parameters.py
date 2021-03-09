# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
from json import loads
from boto3 import client

def get_params(bucket_name, key_name):
    S3 = client('s3') 
    return loads(
        client('s3').get_object(
            Bucket=bucket_name, Key=key_name
        )['Body'].read().decode('utf-8')
    )

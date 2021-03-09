# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import os
import json
import pandas as pd
import s3fs

from boto3 import resource
from urllib.parse import unquote
from loader import check_expected_bucket_owner

fs = s3fs.S3FileSystem(anon=False)


def shallow_rename(df, *args, **kwargs) -> pd.DataFrame:
    df2 = df.copy(deep=False)
    df2.rename(*args, inplace=True, **kwargs)
    return df2


def lambda_handler(event, context):
    bucket_name = event['Records'][0]['s3']['bucket']['name']
    file_name = unquote(event['Records'][0]['s3']['object']['key'])
    fcast_bkt = os.environ['FORECAST_BUCKET']
    
    check_expected_bucket_owner(bucket_name)

    csv_path = "s3://{bucket}/{file_name}".format(bucket=bucket_name, file_name=file_name)
    df = pd.read_csv(csv_path, dtype={"timestamp": str}, low_memory=False) \
           .rename(columns={"item_id": "actual_box_barcode"})
    df['item_id'] = df.apply(lambda x: "%s-%s-%s"%(x["actual_box_barcode"], x["channel"], x["family"]), axis=1)
    df['timestamp'] = pd.DatetimeIndex(df["timestamp"]) \
                        .strftime("%Y-%m-%d")
    schema = ['item_id', 'timestamp', 'channel', 'actual_box_barcode', 'family', 'demand'] 

    check_expected_bucket_owner(fcast_bkt)

    df[schema].to_csv("s3://{fcast_bkt}/00-dwp-landing/{file_name}".format( fcast_bkt=fcast_bkt, file_name=file_name.split('/')[-1]), index=False)

    check_expected_bucket_owner(bucket_name)

    with fs.open(f"s3://{bucket_name}/{file_name.replace('.input.csv', '.af-params.json')}", "r") as f:
        params = json.load(f)

    meta_data = {"bucket": bucket_name, "object": file_name,
                 "params": params}
                 
    check_expected_bucket_owner(fcast_bkt)

    with fs.open("s3://{fcast_bkt}/00-dwp-landing/meta_data.json".format(fcast_bkt=fcast_bkt), "w") as f:
        json.dump(meta_data, f)

    return

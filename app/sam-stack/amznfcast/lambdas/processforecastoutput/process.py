# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import numpy as np
import pandas as pd

from typing import Tuple
from boto3 import client
from loader import check_expected_bucket_owner

S3_CLI = client('s3')
AFOUT_USECOLS = ("actual_box_barcode", "channel", "family", "date", "p10", "p50", "p90")
AFIN_USECOLS = ("timestamp", "channel", "actual_box_barcode", "family", "demand")
VOYAGER_COLS = (
    "index",
    "demand",
    "forecast",
    "forecast_backtest",
    "forecast_p90",
    "forecast_p10",
    "item_id",
    "channel",
    "family",
)


def shallow_rename(df, *args, **kwargs) -> pd.DataFrame:
    df2 = df.copy(deep=False)
    df2.rename(*args, inplace=True, **kwargs)
    return df2


def shallow_reset_index(df, *args, **kwargs) -> pd.DataFrame:
    df2 = df.copy(deep=False)
    df2.reset_index(*args, inplace=True, **kwargs)
    return df2


def shallow_replace(df, *args, **kwargs) -> pd.DataFrame:
    df2 = df.copy(deep=False)
    df2.replace(*args, inplace=True, **kwargs)
    return df2


def afout2voyager(event, ipath=".", to_replace=None, usecols=AFOUT_USECOLS) -> pd.DataFrame:
    check_expected_bucket_owner(event["bucket"])
    
    forecast_files = S3_CLI.list_objects_v2(
        Bucket=event['bucket'],
        Prefix='01-amazon-forecast-pipeline/{job_id}/010-fcast-export'.format(job_id=event['job_id'])
    )
    print(len(forecast_files['Contents']))
    #print(forecast_files['Contents']['Key'])

    dfs = [
            pd.read_csv('s3://{bucket}/{key}'.format(bucket=event['bucket'], key=p['Key']), low_memory=False, memory_map=True, usecols=usecols)
        for p in forecast_files['Contents'] if ((p['Key'].endswith('csv')) & (p['Size']>0))
    ]



    df = pd.concat(dfs, axis=0)
    df[["forecast_backtest", "demand"]] = (np.nan, np.nan)

    df = shallow_rename(
        df,
        columns={
            "date": "index",
            "p50": "forecast",
            "p90": "forecast_p90",
            "p10": "forecast_p10",
            "actual_box_barcode": "item_id",
        },
    )
    df["index"] = df["index"].str.slice(stop=len("yyyy-mm-dd"))
    df = df.loc[:, VOYAGER_COLS]

    df = shallow_replace(df, to_replace=to_replace)
    return df


def afin2voyager(fname, usecols=AFIN_USECOLS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(fname, low_memory=False, memory_map=True, usecols=usecols)
    df[["forecast", "forecast_backtest", "forecast_p90", "forecast_p10"]] = (
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    )
    df = shallow_rename(df, columns={"timestamp": "index", "actual_box_barcode": "item_id"})
    df["index"] = df["index"].str.slice(stop=len("yyyy-mm-dd"))
    df = df.loc[:, VOYAGER_COLS]
    #df.to_csv('s3://af-stack-forecastbucket-14po9nccy6opj/temp.csv')
    # Amazon Forecast lower-cases these values, and we need to revert these
    # lower-cased values back to their original.
    lookup_table = {
        colname: {v.lower(): v for v in df[colname] if isinstance(v, str) } for colname in ("item_id", "channel", "family")
    }

    df = df.sort_values(by=["channel", "family", "item_id", "index"], ascending=True)
    return df, lookup_table


def lambda_handler(event, context):
    """
    """
    check_expected_bucket_owner(event["bucket"])
    print(event)
    
    fcast_input_path = 's3://{bucket}/01-amazon-forecast-pipeline/{job_id}/000-fcast-input/input.csv'.format(bucket=event['bucket'], job_id=event['job_id'])
                                    
    df_hist, lookup_table = afin2voyager(fcast_input_path)
    df_fcast = afout2voyager(event, to_replace=lookup_table)

    # Must sort ascendingly by timestamp for Voyager UI to render correctly.
    # Otherwise, Voyager chart will shows criss-cross back-and-forth.
    df = shallow_reset_index(
        pd.concat((df_hist, df_fcast), axis=0).sort_values("index"),
        drop=True
    )

    output_path = 's3://{bucket}/01-amazon-forecast-pipeline/{job_id}/020-vfmt'.format(bucket=event['bucket'], job_id=event['job_id'])
    qs_path = 's3://{bucket}/02-amazon-quicksight-pipeline'.format(bucket=event['bucket'])
    
    check_expected_bucket_owner(event["bucket"])
    df.to_csv('{path}/output.csv'.format(path=output_path), index=True)
    df.to_csv('{path}/latest/data.csv'.format(path=qs_path), index=True)
    df.to_csv('{path}/history/{job_id}/data.csv'.format(path=qs_path, job_id=event['job_id']), index=True)
    return event

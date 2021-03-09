# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import os

from json import loads, dumps
from datetime import datetime
from boto3 import client
from boto3 import resource
from schema import SCHEMA_DEF

from datetime import datetime
from random import randrange
from loader import check_expected_bucket_owner

STEP_FUNCTIONS_CLI = client('stepfunctions')


def get_params(bucket_name, key_name, job_id):
    check_expected_bucket_owner(bucket_name)
    params = loads(
        client('s3').get_object(Bucket=bucket_name,
                                Key=key_name)['Body'].read().decode('utf-8')
    )
    # validate(params, SCHEMA_DEF)
    #append jobid to names
    for dataset in params["Datasets"]:
        dataset["DatasetName"] = dataset["DatasetName"] + job_id
    params["DatasetGroup"]["DatasetGroupName"] += job_id
    params["Predictor"]["PredictorName"] += job_id
    params["Forecast"]["ForecastName"] += job_id
    return params


def get_voyager_params(bucket_name, key_name, job_id):
    check_expected_bucket_owner(bucket_name)
    params = loads(
            client('s3').get_object(Bucket=bucket_name,
                                    Key=key_name)['Body'].read().decode('utf-8')
    )["params"]
    for dataset in params["Datasets"]:
        dataset["DatasetName"] = dataset["DatasetName"] + job_id
    params["DatasetGroup"]["DatasetGroupName"] += job_id
    params["Predictor"]["PredictorName"] += job_id
    params["Forecast"]["ForecastName"] += job_id
    return params


def get_voyager_info(bucket_name, key_name):
    check_expected_bucket_owner(bucket_name)
    voyager_source = loads(
            client('s3').get_object(Bucket=bucket_name,
                                    Key=key_name)['Body'].read().decode('utf-8')
    )
    return voyager_source


def lambda_handler(event, context):
    bucket_name = event['Records'][0]['s3']['bucket']['name']
    file_name = event['Records'][0]['s3']['object']['key']
    job_id = "job_%s_%s" %(datetime.utcnow().strftime('%Y%m%d_%H%M%S'), randrange(0x7fffffff))
    check_expected_bucket_owner(bucket_name)
    resource('s3').Object(bucket_name, 
                        "01-amazon-forecast-pipeline/{job_id}/000-fcast-input/input.csv".format(job_id=job_id)
                        ).copy_from(CopySource="{bucket}/{file_name}".format(bucket=bucket_name,
                                                                            file_name=file_name)
                                )
    
    check_expected_bucket_owner(bucket_name)
    return dumps(
        STEP_FUNCTIONS_CLI.start_execution(
            stateMachineArn=os.environ['STEP_FUNCTIONS_ARN'],
            name=datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
            input=dumps(
                {
                    'bucket': bucket_name,
                    'filename': file_name,
                    'job_id': job_id,
                    'currentDate': datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
                    'params':
                        get_voyager_params(bucket_name, '%s/meta_data.json'%(file_name.split('/')[0]), job_id),
                    'voyagerBucket':
                        get_voyager_info(bucket_name, '%s/meta_data.json'%(file_name.split('/')[0]))['bucket'],
                    'voyagerKey':
                        get_voyager_info(bucket_name, '%s/meta_data.json'%(file_name.split('/')[0]))['object']
                }
            )
        ),
        default=str
    )

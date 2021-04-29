# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import os
import re
import argparse
import json
import urllib.error
import urllib.request
import uuid
import datetime
import boto3
import socket
import logging
import traceback

from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

CORS_HEADERS = {
    'access-control-allow-headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token,X-Amz-User-Agent',
    'access-control-allow-methods': 'DELETE,GET,HEAD,OPTIONS,PATCH,POST,PUT',
    'access-control-allow-origin': '*',
}


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

    if account_id is None:
        account_id = boto3.client("sts").get_caller_identity()["Account"]

    try:
        client.get_bucket_acl(Bucket=bucket, ExpectedBucketOwner=account_id)
    except ClientError as e:
        # http_status -> 403 if bucket owner is not as expected
        # http_status = e.response["ResponseMetadata"]["HTTPStatusCode"]
        raise(e)

    return


def make_af_input(event, src_bucket, src_s3_key):
    """Make the input file that creates an Amazon Forecast workflow.

       / \   _ __ ___   __ _ _______  _ __  
      / _ \ | '_ ` _ \ / _` |_  / _ \| '_ \ 
     / ___ \| | | | | | (_| |/ / (_) | | | |
    /_/   \_\_| |_| |_|\__,_/___\___/|_| |_|
     _____                            _     
    |  ___|__  _ __ ___  ___ __ _ ___| |_   
    | |_ / _ \| '__/ _ \/ __/ _` / __| __|  
    |  _| (_) | | |  __/ (_| (_| \__ \ |_   
    |_|  \___/|_|  \___|\___\__,_|___/\__|  

    """
    #
    # duplicate the raw input csv to have a `.input.csv` file extension
    #
    out_ext = ".input.csv"
    des_s3_key = f"{ os.path.splitext(src_s3_key)[0] }.{out_ext.lstrip('.')}"
    
    # Create .af-params.json file required by AF stack
    params = make_af_params(event)

    # Enable AUTO ML for Amazon Forecast
    # WARNING: the forecast job will take hours to complete
    params["Predictor"]["PerformAutoML"] = True
    params["Predictor"].pop("AlgorithmArn")

    des_s3_params_key = f"{ os.path.splitext(src_s3_key)[0] }.af-params.json"

    print("LOG: make_af_input", des_s3_key)
    print("LOG: make_af_input", des_s3_params_key)

    s3 = boto3.resource("s3")
    
    # Write the AF params JSON to S3
    check_expected_bucket_owner(src_bucket)
    s3_json_obj = s3.Object(src_bucket, des_s3_params_key)
    s3_json_obj.put(
        Body=bytes(json.dumps(params).encode("utf-8")),
        ContentType="application/json"
    )

    check_expected_bucket_owner(src_bucket)
    s3.meta.client.copy({ "Bucket": src_bucket, "Key": src_s3_key },
                        src_bucket, des_s3_key)

    return


def make_af_params(event):
    """
    """
    body = json.loads(event["body"]) if 'body' in event else event
    
    # Set default AF parameters
    params = {
      "DatasetGroup": {
        "DatasetGroupName":"AuDemand",
        "Domain": "RETAIL"
      },
      "Predictor": {
        "PredictorName": "AuDemand",
        "ForecastHorizon": 72,
        "FeaturizationConfig":{
          "ForecastFrequency":"D"
        },
        "PerformAutoML": False,
        "AlgorithmArn": "arn:aws:forecast:::algorithm/ETS"
      },
      "Forecast": {
        "ForecastName": "AuDemand",
        "ForecastTypes":[
          "0.10",
          "0.50",
          "0.90"
        ]
      },
      "TimestampFormat": "yyyy-MM-dd",
      "Datasets": [
        {
          "DatasetName": "AuDemand",
          "Domain": "RETAIL",
          "DatasetType": "TARGET_TIME_SERIES",
          "DataFrequency": "D",
          "Schema": {
            "Attributes": [
          {
            "AttributeName": "item_id",
                "AttributeType": "string"
          }, {
                "AttributeName": "timestamp",
                "AttributeType": "timestamp"
              }, {
            "AttributeName": "channel",
                "AttributeType": "string"
          }, {
            "AttributeName": "actual_box_barcode",
                "AttributeType": "string"
          }, {
            "AttributeName": "family",
                "AttributeType": "string"
          }, {
                  "AttributeName": "demand",
                  "AttributeType": "float"
              }
            ]
          }
        }
      ],
      "PerformDelete": False
    }
    horizonAmount = str(body["horizonAmount"])
    horizonUnit = body["horizonUnit"]
    frequencyUnit = body["frequencyUnit"]
    
    params["Datasets"][0]["DataFrequency"] = frequencyUnit
    params["Predictor"]["ForecastHorizon"] = int(horizonAmount)
    params["Predictor"]["FeaturizationConfig"]["ForecastFrequency"] = horizonUnit
    
    return params
    

def handler(event, context):
    """
    """
    src_bucket = \
        event.get("bucket",
          os.environ["STORAGE_VOYAGERFRONTENDAUTH_BUCKETNAME"])
    src_bucket = re.sub(r"^s3://", "", src_bucket)

    body = json.loads(event["body"]) if 'body' in event else event
    fileS3Key = body["fileS3Key"]
    horizonAmount = str(body["horizonAmount"])
    horizonUnit = body["horizonUnit"]
    frequencyUnit = body["frequencyUnit"]
    createdAt = datetime.datetime.now() \
                        .astimezone(datetime.timezone.utc) \
                        .isoformat()
    cognitoIdentityId = event["requestContext"]["identity"]["cognitoIdentityId"]
    sub = event["requestContext"]["identity"]["cognitoAuthenticationProvider"].split(":")[-1]

    isAfJob = body.get("isAfJob", False)

    try:
        if isAfJob:
            fileS3Key = f"private/{cognitoIdentityId}/{os.path.basename(fileS3Key)}"
            make_af_input(event, src_bucket, fileS3Key)
            http_code = 200
        else:
            s3_input_path = f"s3://{src_bucket}/{fileS3Key}",
            print(s3_input_path, s3_input_path)
            sfn = boto3.client("stepfunctions")
            response = sfn.start_execution(
                stateMachineArn=os.environ["VOYAGER_LAMBDA_FUNCTION_NAME"],
                input=json.dumps({
                    "s3_input_path": f"s3://{src_bucket}/{fileS3Key}",
                    "freq": horizonUnit,
                    "horiz": horizonAmount
                })
            )
            response["StatusCode"] = 202
            http_code = int(response["StatusCode"])
    except Exception as e:
        traceback.print_exc()
        http_code = 500

    report_s3_key = f"{fileS3Key}.report"
    results_s3_key = f"{fileS3Key}.results.json"
    forecast_s3_key = f"{fileS3Key}.forecast.csv"

    if http_code == 200:
        msg = "The forecast was successfully generated"
    elif http_code == 202:
        msg = "The forecast is being generated in the background and will require 5-45 minutes, depending on the input dataset size"
    else:
        msg = "Internal server error"
        report_s3_key = None
        results_s3_key = None
        forecast_s3_key = None

    body_dict = {
        "http_code": http_code,
        "msg": msg,
        "dataAnalysisS3Key": report_s3_key,
        "resultsS3Key": results_s3_key,
        "forecastS3Key": forecast_s3_key
    }

    #
    # Add forecast details to dynamodb, only if not run in headless mode.
    # 
    is_headless = event.get("headless", False)

    if not is_headless and http_code in (200, 202) and (not isAfJob):
        # store forecast in dynamodb
        dynamodb = boto3.client("dynamodb")
        forecast_id = str(uuid.uuid4())
        existing_forecasts = get_forecasts(cognitoIdentityId)

        if len(existing_forecasts) == 0:
            # this is the first forecast request
            identity = get_cognito_identity(cognitoIdentityId)

            # insert entry into the `voyagerCustomers` dynamodb table
            dynamodb = boto3.client("dynamodb")
            dynamodb.put_item(
                TableName=os.environ["STORAGE_VOYAGERCUSTOMERSDB_NAME"],
                Item={
                  "cognitoIdentityId": {"S": cognitoIdentityId},
                  "userCreatedAt": {"S": identity["CreationDate"].astimezone(datetime.timezone.utc).isoformat()},
                  "firstForecastId": {"S": forecast_id},
                  "reportsQuota": {"N": "10"},
                  "sub": {"S": sub}
                })
        else:
            pass

        item = {
            "id": {"S": forecast_id},
            "name": {"S": cognitoIdentityId},
            "horizonAmount": {"N": horizonAmount},
            "horizonUnit": {"S": horizonUnit},
            "frequencyUnit": {"S": frequencyUnit},
            "dataAnalysisS3Key": {"S": report_s3_key},
            "resultsS3Key": {"S": results_s3_key},
            "forecastS3Key": {"S": forecast_s3_key},
            "createdAt": {"S": createdAt},
            "sub": {"S": sub}
        }

        dynamodb.put_item(TableName=os.environ["STORAGE_VOYAGERDB_NAME"],
                          Item=item)
    else:
        forecast_id = None

    body_dict["id"] = forecast_id

    return {
        "isBase64Encoded": False,
        "statusCode": body_dict["http_code"],
        "headers": CORS_HEADERS,
        "body": json.dumps(body_dict)
    }


def get_forecasts(cognitoIdentityId):
    dynamodb = boto3.resource("dynamodb", region_name=os.environ["REGION"])
    table = dynamodb.Table(os.environ["STORAGE_VOYAGERDB_NAME"])
    rows = table.query(KeyConditionExpression=Key("name").eq(cognitoIdentityId))
    return rows["Items"]


def get_cognito_identity(cognitoIdentityId):
    client = boto3.client("cognito-identity")
    identity = client.describe_identity(IdentityId=cognitoIdentityId)
    return identity


def make_event(bucket, key, freq, freq_unit, horizon):
    """Make an event dictionary object from the forecast request components.
    This is to be used to invoke the lambda handler from a separate function.

    Parameters
    ----------
    bucket : str
    key : str
    freq : str
    freq_unit : str
    horizon : int

    Returns
    -------
    event : dict

    """

    event = {"queryStringParameters": {}, "headless": True}

    event["queryStringParameters"]["fileS3Key"] = key
    event["queryStringParameters"]["frequencyUnit"] = freq_unit
    event["queryStringParameters"]["horizonUnit"] = freq
    event["queryStringParameters"]["horizonAmount"] = horizon

    return event

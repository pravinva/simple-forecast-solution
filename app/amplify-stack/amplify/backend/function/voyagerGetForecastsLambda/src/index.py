# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import os
import json
import boto3

from boto3.dynamodb.conditions import Key

CORS_HEADERS = {
    'access-control-allow-headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token,X-Amz-User-Agent',
    'access-control-allow-methods': 'DELETE,GET,HEAD,OPTIONS,PATCH,POST,PUT',
    'access-control-allow-origin': '*',
}


def handler(event, context):
    try:
        row_id = event["queryStringParameters"]["id"]
    except (KeyError, TypeError):
        row_id = None

    cognitoIdentityId = \
        event["requestContext"]["identity"]["cognitoIdentityId"]
    dynamodb = boto3.resource("dynamodb", region_name=os.environ["REGION"])

    table = dynamodb.Table(os.environ["STORAGE_VOYAGERDB_NAME"])
    rows = table.query(KeyConditionExpression=Key("name").eq(cognitoIdentityId))

    fcast_records = []

    for r in rows["Items"]:
        if row_id is not None:
            if r["id"] == row_id:
                r["horizonAmount"] = int(r["horizonAmount"])
                fcast_records.append(r)
                break
        else:
            r["horizonAmount"] = int(r["horizonAmount"])
            fcast_records.append(r)

    return {
        "isBase64Encoded": False,
        "statusCode": 200,
        "headers": CORS_HEADERS,
        "body": json.dumps(fcast_records)
    }

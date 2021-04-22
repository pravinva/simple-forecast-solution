# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import argparse
import sys
import traceback
import json
import pywren
import numpy as np

from voyager import Engine


def lambda_handler(event, context):
    """
    """

    try:
        bucket = event["queryStringParameters"]["bucket"]
        file_name = event["queryStringParameters"]["file"]
        frequency = event["queryStringParameters"]["frequency"]
        horizon = event["queryStringParameters"]["horizon"]
        debug = event["queryStringParameters"].get("debug", False)
        model_name = event["queryStringParameters"].get("model_name", None)
        ignore_naive = event["queryStringParameters"].get("ignore_naive", False)
        min_len = event["queryStringParameters"].get("min_len", None)
        quantile = event["queryStringParameters"].get("quantile", None)
        engine = Engine(bucket, file_name, frequency, int(horizon), debug=debug)
    except Exception as e:
        exc_type, exc_value, exc_tb = sys.exc_info()

        traceback.print_exc()

        return {
            "statusCode": 500,
            "body": json.dumps({"message": "Cannot Load, Hint:" + str(e)}),
        }

    try:
        engine.eda()
    except Exception as e:
        exc_type, exc_value, exc_tb = sys.exc_info()
        traceback.print_exc()

        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "message": "Cannot Report, Hint:" + str(e),
                }
            ),
        }

    if min_len is not None:
        min_len = int(min_len)

    engine.forecast(model_name=model_name,
                           ignore_naive=ignore_naive,
                           min_len=min_len,
                           quantile=quantile)

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "message": "Finished Running",
            }
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bucket", action="store", dest="bucket")
    parser.add_argument("-k", "--key", action="store", dest="key")
    parser.add_argument("-f", "--freq", action="store", dest="freq")
    parser.add_argument("-H", "--horizon", action="store", dest="horizon", type=int)

    args = parser.parse_args()

    event = {"queryStringParameters": {}}

    event["queryStringParameters"]["bucket"] = args.bucket
    event["queryStringParameters"]["file"] = args.key
    event["queryStringParameters"]["frequency"] = args.freq
    event["queryStringParameters"]["horizon"] = args.horizon
    event["queryStringParameters"]["include_naive"] = False
    event["queryStringParameters"]["debug"] = True

    response = lambda_handler(event, None)

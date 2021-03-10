# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import argparse
import sys
import traceback
import json
import pywren
import numpy as np

from voyager import *
from utilities import remap_freq


def lambda_handler(event, context):
    """
    """

    try:
        bucket = event["queryStringParameters"]["bucket"]
        file_name = event["queryStringParameters"]["file"]
        frequency = remap_freq(event["queryStringParameters"]["frequency"])
        horizon = event["queryStringParameters"]["horizon"]
        debug = event["queryStringParameters"].get("debug", False)
        ignore_naive = event["queryStringParameters"].get("ignore_naive", False)
        voyager_class = Engine(bucket, file_name, frequency, int(horizon), debug=debug)
    except Exception as ex:
        exc_type, exc_value, exc_tb = sys.exc_info()

        traceback.print_exc()

        return {
            "statusCode": 500,
            "body": json.dumps({"message": "Cannot Load, Hint:" + str(ex)}),
        }

    try:
        voyager_class.eda()
    except Exception as ex:
        exc_type, exc_value, exc_tb = sys.exc_info()
        traceback.print_exc()

        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "message": "Cannot Report, Hint:" + str(ex),
                }
            ),
        }

    voyager_class.forecast(ignore_naive=ignore_naive)

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

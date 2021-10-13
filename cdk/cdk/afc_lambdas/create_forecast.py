import os
import datetime
import json
import boto3

from urllib.parse import urlparse

session = boto3.Session()
afc = session.client("forecast")
s3 = boto3.resource("s3")


def update_status_json(resp, state, path):
    """
    """

    parsed_url = urlparse(path, allow_fragments=False)
    bucket = parsed_url.netloc
    key = os.path.join(parsed_url.path.lstrip("/").rstrip("/"))

    status_dict = dict(resp)
    status_dict["PROGRESS"] = {
        "state": state,
        "timestamp": datetime.datetime.now().astimezone().isoformat()
    }

    s3obj = s3.Object(bucket, key)
    s3obj.put(Body=bytes(json.dumps(status_dict).encode("utf-8")))

    return


def create_forecast_handler(event, context):
    """
    """

    payload = event["input"]["Payload"]
    prefix = payload["prefix"]

    AFC_FORECAST_NAME = payload["PredictorName"]

    update_status_json(payload, "IN_PROGRESS:create_forecast",
        payload["StatusJsonS3Path"])

    create_forecast_resp = afc.create_forecast(
        ForecastName=AFC_FORECAST_NAME,
        PredictorArn=payload["PredictorArn"]
    )

    resp = payload
    resp["ForecastArn"] = create_forecast_resp["ForecastArn"]
    resp["ForecastName"] = AFC_FORECAST_NAME

    update_status_json(resp, "DONE:create_forecast",
        payload["StatusJsonS3Path"])

    return resp
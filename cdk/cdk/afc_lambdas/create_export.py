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


def create_forecast_export_handler(event, context):
    """
    """

    payload = event["input"]["Payload"]
    prefix = payload["prefix"]

    update_status_json(payload, "IN_PROGRESS:create_forecast_export",
        payload["StatusJsonS3Path"])

    AFC_FORECAST_EXPORT_JOB_NAME = f"{prefix}_ExportJob"

    resp = afc.create_forecast_export_job(
        ForecastExportJobName=AFC_FORECAST_EXPORT_JOB_NAME,
        ForecastArn=payload["ForecastArn"],
        Destination={
            "S3Config": {
                "Path": os.path.join(payload["s3_export_path"], prefix),
                "RoleArn": os.environ["AFC_ROLE_ARN"],
            }
        }
    )

    resp_out = payload
    resp_out["ForecastExportJobArn"] = resp["ForecastExportJobArn"]

    update_status_json(resp_out, "DONE:create_forecast_export",
        payload["StatusJsonS3Path"])

    return resp_out
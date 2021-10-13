import os
import datetime
import json
import boto3

from urllib.parse import urlparse

session = boto3.Session()
afc = session.client("forecast")
afcq = session.client("forecastquery")
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


def delete_afc_resources_handler(event, context):
    """
    """

    payload = event["input"]["Payload"]
    prefix = payload["prefix"]

    update_status_json(payload, "IN_PROGRESS:delete_afc_resources",
        payload["StatusJsonS3Path"])

    try:
        # Delete forecast export job
        afc.delete_forecast_export_job(
            ForecastExportJobArn=payload["ForecastExportJobArn"])
    except:
        pass

    try:
        # Delete forecast
        afc.delete_forecast(ForecastArn=payload["ForecastArn"])
    except:
        pass

    try:
        # Delete predictor
        afc.delete_predictor(PredictorArn=payload["PredictorArn"])
    except:
        pass

    try:
        # Delete dataset
        afc.delete_dataset(DatasetArn=payload["DatasetArn"])
    except:
        pass

    try:
        # Delete dataset import job
        afc.delete_dataset_import_job(
            DatasetImportJobArn=payload["DatasetImportJobArn"])
    except:
        pass

    try:
        # Delete dataset group
        afc.delete_dataset_group(DatasetGroupArn=payload["DatasetGroupArn"])
    except:
        pass

    update_status_json(payload, "DONE:delete_afc_resources",
        payload["StatusJsonS3Path"])

    return payload

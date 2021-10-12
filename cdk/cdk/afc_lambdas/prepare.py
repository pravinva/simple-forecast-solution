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


def prepare_handler(event, context):
    """
    """

    prefix = event["input"]["prefix"]
    data_freq = event["input"]["data_freq"]
    horiz = int(event["input"]["horiz"])
    freq = event["input"]["freq"]
    s3_path = event["input"]["s3_path"]
    s3_export_path = event["input"]["s3_export_path"]

    update_status_json(event["input"], "IN_PROGRESS:create_dataset_import",
        f'{s3_export_path}/{prefix}_status.json')

    assert(freq in ("D", "W", "M"))

    now_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    AFC_DATASET_DOMAIN = "RETAIL"
    AFC_DATASET_GROUP_NAME = f"{prefix}_DatasetGroup"
    AFC_DATASET_NAME = f"{prefix}_Dataset"
    AFC_DATASET_FREQUENCY = data_freq # "Y|M|W|D|H" (input frequency)
    AFC_DATASET_TYPE = "TARGET_TIME_SERIES"
    AFC_ROLE_ARN = os.environ["AFC_ROLE_ARN"]
    AFC_INPUT_S3_PATH = s3_path

    create_dataset_group_resp = afc.create_dataset_group(
        Domain=AFC_DATASET_DOMAIN,
        DatasetGroupName=AFC_DATASET_GROUP_NAME,
        DatasetArns=[])

    AFC_DATASET_GROUP_ARN = create_dataset_group_resp["DatasetGroupArn"]

    ts_schema = {
        "Attributes": [
            {"AttributeName": "timestamp",
             "AttributeType": "timestamp"},
            {"AttributeName": "demand",
             "AttributeType": "float"},
            {"AttributeName": "item_id",
             "AttributeType": "string"}
        ]
    }

    create_dataset_resp = afc.create_dataset(
        Domain=AFC_DATASET_DOMAIN,
        DatasetType=AFC_DATASET_TYPE,
        DatasetName=AFC_DATASET_NAME,
        DataFrequency=data_freq,
        Schema=ts_schema
    )

    AFC_DATASET_ARN = create_dataset_resp["DatasetArn"]

    afc.update_dataset_group(
        DatasetGroupArn=AFC_DATASET_GROUP_ARN,
        DatasetArns=[AFC_DATASET_ARN]
    )

    dataset_import_resp = afc.create_dataset_import_job(
        DatasetImportJobName=AFC_DATASET_GROUP_NAME,
        DatasetArn=AFC_DATASET_ARN,
        DataSource={
            "S3Config": {
                "Path": AFC_INPUT_S3_PATH,
                "RoleArn": AFC_ROLE_ARN
            }
        },
        TimestampFormat="yyyy-MM-dd"
    )

    AFC_DATASET_IMPORT_JOB_ARN = dataset_import_resp["DatasetImportJobArn"]

    status_json_s3_path = f'{s3_export_path}/{prefix}_status.json'

    resp_out = event["input"]
    resp_out["DatasetGroupArn"] = AFC_DATASET_GROUP_ARN
    resp_out["DatasetArn"] = AFC_DATASET_ARN
    resp_out["DatasetImportJobArn"] = dataset_import_resp["DatasetImportJobArn"]
    resp_out["StatusJsonS3Path"] = status_json_s3_path

    update_status_json(resp_out, "DONE:create_dataset_import",
        status_json_s3_path)
        
    return resp_out
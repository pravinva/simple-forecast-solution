import os
import datetime
import json
import boto3

session = boto3.Session()
afc = session.client("forecast")
afcq = session.client("forecastquery")


def prepare_handler(event, context):
    """
    """

    print(type(event))
    print(event)

    prefix = event["input"]["prefix"]
    data_frq = event["input"]["data_freq"]
    horiz = int(event["input"]["horiz"])
    freq = event["input"]["freq"]
    s3_path = event["input"]["s3_path"]
    s3_export_path = event["input"]["s3_export_path"]

    assert(freq in ("D", "W", "M"))

    now_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    AFC_DATASET_DOMAIN = "RETAIL"
    AFC_DATASET_GROUP_NAME = f"{prefix}_DatasetGroup"
    AFC_DATASET_NAME = f"{prefix}_Dataset"
    AFC_DATASET_FREQUENCY = freq # "Y|M|W|D|H" (input frequency)
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
        DataFrequency=AFC_DATASET_FREQUENCY,
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

    resp_out = event["input"]
    resp_out["AFC_DATASET_GROUP_ARN"] = AFC_DATASET_GROUP_ARN
    resp_out["AFC_DATASET_ARN"] = AFC_DATASET_ARN
    resp_out["AFC_DATASET_IMPORT_JOB_ARN"] = AFC_DATASET_IMPORT_JOB_ARN

    return resp_out


def create_predictor_handler(event, context):
    """
    """

    PREFIX = event["input"]["Payload"]["prefix"]
    AFC_DATASET_GROUP_ARN = event["input"]["Payload"]["AFC_DATASET_GROUP_ARN"]
    AFC_FORECAST_HORIZON = event["input"]["Payload"]["horiz"]
    AFC_FORECAST_FREQUENCY = event["input"]["Payload"]["freq"]
    AFC_ALGORITHM_NAME = "NPTS"
    AFC_ALGORITHM_ARN = "arn:aws:forecast:::algorithm/NPTS"
    AFC_PREDICTOR_NAME = f"{PREFIX}_{AFC_ALGORITHM_NAME}"

    create_predictor_resp = afc.create_predictor(
        PredictorName=AFC_PREDICTOR_NAME,
        ForecastHorizon=AFC_FORECAST_HORIZON,
        AlgorithmArn=AFC_ALGORITHM_ARN, # TODO: delete this when ready
        #PerformAutoML=True, # TODO: Uncomment this when ready
        #PerformHPO=False,
        InputDataConfig={
            "DatasetGroupArn": AFC_DATASET_GROUP_ARN
        },
        FeaturizationConfig={
            "ForecastFrequency": AFC_FORECAST_FREQUENCY,
            "Featurizations": [
                {
                    "AttributeName": "demand",
                    "FeaturizationPipeline": [
                        {
                            "FeaturizationMethodName": "filling",
                            "FeaturizationMethodParameters": {
                                "aggregation": "sum",
                                "frontfill": "none",
                                "middlefill": "zero",
                                "backfill": "zero"
                            }
                        }
                    ]
                }
            ]
        }
    )

    resp = event["input"]["Payload"]
    resp["AFC_PREDICTOR_ARN"] = create_predictor_resp["PredictorArn"]
    resp["AFC_PREDICTOR_NAME"] = AFC_PREDICTOR_NAME

    return resp


def create_forecast_handler(event, context):
    """
    """

    AFC_FORECAST_NAME = event["input"]["Payload"]["AFC_PREDICTOR_NAME"]

    create_forecast_resp = afc.create_forecast(
        ForecastName=event["input"]["Payload"]["AFC_PREDICTOR_NAME"],
        PredictorArn=event["input"]["Payload"]["AFC_PREDICTOR_ARN"]
    )

    resp = event["input"]["Payload"]
    resp["AFC_FORECAST_ARN"] = create_forecast_resp["ForecastArn"]
    resp["AFC_FORECAST_NAME"] = AFC_FORECAST_NAME

    return resp


def create_forecast_export_handler(event, context):
    """
    """

    payload = event["input"]["Payload"]
    prefix = payload["prefix"]

    AFC_FORECAST_EXPORT_JOB_NAME = f"{prefix}_ExportJob"

    resp = afc.create_forecast_export_job(
        ForecastExportJobName=AFC_FORECAST_EXPORT_JOB_NAME,
        ForecastArn=payload["AFC_FORECAST_ARN"],
        Destination={
            "S3Config": {
                "Path": payload["s3_export_path"],
                "RoleArn": os.environ["AFC_ROLE_ARN"],
            }
        }
    )

    resp_out = payload
    resp_out["ForecastExportJobArn"] = resp["ForecastExportJobArn"]
    resp_out["Destination"] = resp["Destination"]

    return resp_out

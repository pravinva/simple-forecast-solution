# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import json
import actions

from os import environ
from boto3 import client
from boto3 import resource
from loader import Loader, check_expected_bucket_owner

s3 = resource('s3')
CLOUDWATCH_CLI = client('cloudwatch')
ARN = 'arn:aws:forecast:{region}:{account}:forecast/{name}'
JOB_ARN = 'arn:aws:forecast:{region}:{account}:forecast-export-job/' \
          '{name}/{name}_{date}'
LOADER = Loader()


def post_metric(metrics, event=None):
    best_mape = None
    best_algo = None
    for metric in metrics['PredictorEvaluationResults']:
        CLOUDWATCH_CLI.put_metric_data(
            Namespace='FORECAST',
            MetricData=[
                {
                    'Dimensions':
                        [
                            {
                                'Name': 'Algorithm',
                                'Value': metric['AlgorithmArn']
                            }, {
                                'Name': 'Quantile',
                                'Value': str(quantile['Quantile'])
                            }
                        ],
                    'MetricName': 'WQL',
                    'Unit': 'None',
                    'Value': quantile['LossValue']
                } for quantile in metric['TestWindows'][0]['Metrics']
                ['WeightedQuantileLosses']
            ] + [
                {
                    'Dimensions':
                        [
                            {
                                'Name': 'Algorithm',
                                'Value': metric['AlgorithmArn']
                            }
                        ],
                    'MetricName': 'RMSE',
                    'Unit': 'None',
                    'Value': metric['TestWindows'][0]['Metrics']['RMSE']
                }
            ]
        )
    for metric in metrics['PredictorEvaluationResults']:
        for quantile in metric['TestWindows'][0]['Metrics']['WeightedQuantileLosses']:
            if quantile['Quantile']==0.5:
                if best_mape is None:
                    best_mape = quantile['LossValue']
                    best_algo = metric['AlgorithmArn']
                elif quantile['LossValue'] < best_mape:
                    best_mape = quantile['LossValue']
                    best_algo = metric['AlgorithmArn']
    
    if event:
        check_expected_bucket_owner(event["bucket"])
        s3object = s3.Object(event['bucket'], '01-amazon-forecast-pipeline/{job_id}/020-vfmt/metrics.json'.format(job_id=event['job_id']))
        metric_data = {"automl": 1, "mape": best_mape, "selected_algo": best_algo}
        s3object.put(
            Body=(bytes(json.dumps(metric_data).encode('UTF-8')))
        )


def lambda_handler(event, context):
    forecast = event['params']['Forecast']
    status = None
    event['ForecastArn'] = ARN.format(
        account=event['AccountID'],
        name=forecast['ForecastName'],
        region=environ['AWS_REGION']
    )
    event['ForecastExportJobArn'] = JOB_ARN.format(
        account=event['AccountID'],
        name=forecast['ForecastName'],
        date=event['currentDate'],
        region=environ['AWS_REGION']
    )

    # Creates Forecast and export Predictor metrics if Forecast does not exist yet.
    # Will throw an exception while the forecast is being created.
    try:
        actions.take_action(
            LOADER.forecast_cli.describe_forecast(
                ForecastArn=event['ForecastArn']
            )['Status']
        )
    except LOADER.forecast_cli.exceptions.ResourceNotFoundException:
        post_metric(
            LOADER.forecast_cli.get_accuracy_metrics(
                PredictorArn=event['PredictorArn']
            ),
            event
        )
        LOADER.logger.info('Forecast not found. Creating new forecast.')
        LOADER.forecast_cli.create_forecast(
            **forecast, PredictorArn=event['PredictorArn']
        )
        actions.take_action(
            LOADER.forecast_cli.describe_forecast(
                ForecastArn=event['ForecastArn']
            )['Status']
        )

    # Creates forecast export job if it does not exist yet. Will trhow an exception
    # while the forecast export job is being created.
    try:
        status = LOADER.forecast_cli.describe_forecast_export_job(
            ForecastExportJobArn=event['ForecastExportJobArn']
        )
    except LOADER.forecast_cli.exceptions.ResourceNotFoundException:
        LOADER.logger.info('Forecast export not found. Creating new export.')
        check_expected_bucket_owner(event["bucket"])
        LOADER.forecast_cli.create_forecast_export_job(
            ForecastExportJobName='{name}_{date}'.format(
                name=forecast['ForecastName'], date=event['currentDate']
            ),
            ForecastArn=event['ForecastArn'],
            Destination={
                'S3Config':
                    {
                        'Path':
                            's3://{bucket}/01-amazon-forecast-pipeline/{job_id}/010-fcast-export/'.format(bucket=event['bucket'], job_id=event['job_id']),
                        'RoleArn':
                            environ['EXPORT_ROLE']
                    }
            }
        )
        post_metric(
            LOADER.forecast_cli.get_accuracy_metrics(
                PredictorArn=event['PredictorArn']
            ),
            event
        )
        status = LOADER.forecast_cli.describe_forecast_export_job(
            ForecastExportJobArn=event['ForecastExportJobArn']
        )

    actions.take_action(status['Status'])
    return event

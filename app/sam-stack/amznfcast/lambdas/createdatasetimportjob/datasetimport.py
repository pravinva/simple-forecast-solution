# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import actions

from os import environ
from loader import Loader, check_expected_bucket_owner

ARN = 'arn:aws:forecast:{region}:{account}:dataset-import-job/{name}/{name}_{date}'
LOADER = Loader()


def lambda_handler(event, context):
    params = event['params']
    status = None
    event['DatasetImportJobArn'] = ARN.format(
        account=event['AccountID'],
        date=event['currentDate'],
        name=params['Datasets'][0]['DatasetName'],
        region=environ['AWS_REGION']
    )
    try:
        status = LOADER.forecast_cli.describe_dataset_import_job(
            DatasetImportJobArn=event['DatasetImportJobArn']
        )

    except LOADER.forecast_cli.exceptions.ResourceNotFoundException:
        LOADER.logger.info(
            'Dataset import job not found! Will follow to create new job.'
        )
        check_expected_bucket_owner(event["bucket"])
        LOADER.forecast_cli.create_dataset_import_job(
            DatasetImportJobName='{name}_{date}'.format(
                name=params['Datasets'][0]['DatasetName'],
                date=event['currentDate']
            ),
            DatasetArn=event['DatasetArn'],
            DataSource={
                'S3Config':
                    {
                        'Path':
                            's3://{bucket}/01-amazon-forecast-pipeline/{job_id}/000-fcast-input/input.csv'.format(
                                bucket=event['bucket'],
                                job_id=event['job_id']
                            ),
                        'RoleArn':
                            environ['FORECAST_ROLE']
                    }
            },
            TimestampFormat=params['TimestampFormat']
        )
        status = LOADER.forecast_cli.describe_dataset_import_job(
            DatasetImportJobArn=event['DatasetImportJobArn']
        )

    actions.take_action(status['Status'])
    return event

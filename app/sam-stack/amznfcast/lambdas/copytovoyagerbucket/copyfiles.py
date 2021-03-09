# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
from boto3 import resource
from loader import check_expected_bucket_owner


def lambda_handler(event, context):
    bucket_name = event['voyagerBucket']
    file_name = event['voyagerKey']
    job_id = event['job_id']
    source_bucket = event['bucket']

    check_expected_bucket_owner(source_bucket)
    check_expected_bucket_owner(bucket_name)

    resource('s3').Object(bucket_name,
                        "{file_name}.forecast.csv.AF".format(file_name=file_name)
                        ).copy_from(CopySource="{bucket}/01-amazon-forecast-pipeline/{job_id}/020-vfmt/output.csv".format(bucket=source_bucket, job_id=job_id)
                                )
    resource('s3').Object(bucket_name,
                        "{file_name}.results.json.AF".format(file_name=file_name)
                        ).copy_from(CopySource="{bucket}/01-amazon-forecast-pipeline/{job_id}/020-vfmt/metrics.json".format(bucket=source_bucket, job_id=job_id)
                                )
    return event

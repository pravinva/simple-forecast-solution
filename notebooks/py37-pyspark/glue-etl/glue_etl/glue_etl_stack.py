import os

from aws_cdk import core
from aws_cdk import core as cdk
from aws_cdk import (
    aws_iam, aws_glue, aws_s3, aws_s3_assets, aws_s3_deployment,
    aws_lambda,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as tasks
)

PWD = os.path.dirname(os.path.realpath(__file__))
BIN_DIR = os.path.join(PWD, "bin")


class GlueEtlStack(cdk.Stack):
    def __init__(self, scope: cdk.Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # deploy S3 bucket to hold glue scripts
        bucket_name = kwargs.get("bucket_name", "EtlAssetsBucket")
        bucket = aws_s3.Bucket(self, bucket_name)

        # copy local glue scripts to s3 buckets
        bucket_deployment = \
            aws_s3_deployment.BucketDeployment(self, "EtlScripts",
                sources=[aws_s3_deployment.Source.asset(BIN_DIR)],
                destination_bucket=bucket,
                destination_key_prefix="bin"
            )

        # deploy glue iam role
        glue_role = \
            aws_iam.Role(self, "GlueEtlRole",
                role_name="GlueEtlRole",
                assumed_by=aws_iam.CompositePrincipal(
                    aws_iam.ServicePrincipal("glue.amazonaws.com"),
                    aws_iam.ServicePrincipal("lambda.amazonaws.com")
                ),
                managed_policies=[
                    aws_iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSGlueServiceRole"),
                    #aws_iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3ReadOnlyAccess")
                    aws_iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3FullAccess"),
                    aws_iam.ManagedPolicy.from_aws_managed_policy_name("AWSLambdaExecute")
                ]
            )

        # deploy glue etl job
        glue_etl_job = aws_glue.CfnJob(self, "GlueEtlJob",
            name="GlueEtlJob",
            glue_version="2.0",
            max_capacity=2,
            role=glue_role.role_name,
            command=aws_glue.CfnJob.JobCommandProperty(
                name="glueetl",
                python_version="3",
                script_location=bucket.s3_url_for_object("bin/run.py")
            )
        )

        # deploy glue resampling etl job
        glue_resample_job = aws_glue.CfnJob(self, "GlueResampleJob",
                name="GlueResampleJob",
                glue_version="2.0",
                max_capacity=4,
                execution_property=aws_glue.CfnJob.ExecutionPropertyProperty(
                    max_concurrent_runs=200,
                ),
                role=glue_role.role_name,
                command=aws_glue.CfnJob.JobCommandProperty(
                    name="glueetl",
                    python_version="3",
                    script_location=bucket.s3_url_for_object("bin/resample.py")
                )
            )

        # deploy lambda resampling function (async calls glue_resample_job)
        ecr_image = aws_lambda.EcrImageCode.from_asset_image(
                directory=os.path.join(PWD, "resample_lambda"))

        lambda_resample = aws_lambda.Function(self, 
            "ResampleLambda",
            description="ResampleLambda",
            code=ecr_image,
            handler=aws_lambda.Handler.FROM_IMAGE,
            runtime=aws_lambda.Runtime.FROM_IMAGE,
            memory_size=1024,
            timeout=core.Duration.minutes(15),
            role=glue_role
        )

        # deploy lambda forecast function (async calls make_forecast)

        """

        #
        # Example of how to call the sfn state-machine via lambdas
        #
        def lambda_handler(event, context):
            sfn = boto3.client("stepfunctions")
            sfn.start_execution(
                stateMachineArn=...,
                input=json.dumps({
                    "s3_input_path": "...",
                    "freq": "MS",
                    "horiz": "6"
                })
            )

        """

        # define sfn glue etl step
        glue_etl_task = tasks.GlueStartJobRun(self, "GlueEtlJobRun",
            glue_job_name=glue_etl_job.name,
            arguments=sfn.TaskInput.from_object({
                "--s3_input_path": sfn.JsonPath.string_at("$.s3_input_path"),
                "--freq": sfn.JsonPath.string_at("$.freq"),
                "--horiz": sfn.JsonPath.string_at("$.horiz")
            }),

            # req. to sync sfn job run states to glue job run, otherwise
            # the state will instantaneously success in sfn despite still 
            # running in glue
            integration_pattern=sfn.IntegrationPattern.RUN_JOB,
            timeout=core.Duration.minutes(20)
        )

        lambda_resample_task = \
            tasks.LambdaInvoke(self, "LambdaResample",
                lambda_function=lambda_resample,
                payload=sfn.TaskInput.from_object({
                    "arguments": sfn.JsonPath.string_at("$.Arguments"),
                    "glue_job_name": glue_resample_job.name
                })
            )

        # deploy etl state-machine
        definition = glue_etl_task.next(lambda_resample_task)

        sfn.StateMachine(self, "EtlStateMachine",
            definition=definition,
            timeout=core.Duration.minutes(45)
        )

        return

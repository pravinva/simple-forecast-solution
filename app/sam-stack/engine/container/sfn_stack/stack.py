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
BIN_DIR = os.path.join(PWD, "glue_scripts")


class SfnStack(cdk.Stack):
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
                destination_key_prefix="glue_scripts"
            )

        # deploy glue iam role
        glue_role = \
            aws_iam.Role(self, "GlueEtlRole",
                assumed_by=aws_iam.CompositePrincipal(
                    aws_iam.ServicePrincipal("glue.amazonaws.com"),
                    aws_iam.ServicePrincipal("lambda.amazonaws.com")
                ),
                managed_policies=[
                    aws_iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSGlueServiceRole"),
                    #aws_iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3ReadOnlyAccess")
                    aws_iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3FullAccess"),
                    aws_iam.ManagedPolicy.from_aws_managed_policy_name("AWSLambdaExecute"),
                    aws_iam.ManagedPolicy.from_aws_managed_policy_name("AWSLambda_FullAccess")

                ]
            )

        # deploy glue etl job
        glue_etl_job = aws_glue.CfnJob(self, "GlueEtlJob",
            name=f"{construct_id}-GlueEtlJob",
            glue_version="2.0",
            max_capacity=2,
            role=glue_role.role_name,
            command=aws_glue.CfnJob.JobCommandProperty(
                name="glueetl",
                python_version="3",
                script_location=bucket.s3_url_for_object("glue_scripts/split.py")
            )
        )
        name=f"{construct_id}-GlueResampleJob",

        # deploy glue resampling etl job
        glue_resample_job = aws_glue.CfnJob(self, "GlueResampleJob",
                name=f"{construct_id}-GlueResampleJob",
                glue_version="2.0",
                max_capacity=4,
                execution_property=aws_glue.CfnJob.ExecutionPropertyProperty(
                    max_concurrent_runs=200,
                ),
                role=glue_role.role_name,
                command=aws_glue.CfnJob.JobCommandProperty(
                    name="glueetl",
                    python_version="3",
                    script_location=bucket.s3_url_for_object("glue_scripts/resample.py")
                )
            )

        # deploy lambda resampling function (async calls glue_resample_job)
        resample_lambda_ecr_img = aws_lambda.EcrImageCode.from_asset_image(
                directory=os.path.join(PWD, "resample_lambda"))

        lambda_resample = aws_lambda.Function(self, 
            "ResampleLambda",
            description="ResampleLambda",
            code=resample_lambda_ecr_img,
            handler=aws_lambda.Handler.FROM_IMAGE,
            runtime=aws_lambda.Runtime.FROM_IMAGE,
            memory_size=256,
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

        # deploy the engine lambda ecr image
        engine_lambda_ecr_image = \
            aws_lambda.EcrImageCode.from_asset_image(
                    directory=os.path.join(PWD, "engine_lambda"))

        # deploy the engine lambda function
        engine_lambda_function = aws_lambda.Function(self, 
            "EngineMapFunction",
            description="EngineMapFunction",
            code=engine_lambda_ecr_image,
            handler= aws_lambda.Handler.FROM_IMAGE,
            runtime= aws_lambda.Runtime.FROM_IMAGE,
            memory_size=4096,
            timeout=core.Duration.minutes(15),
            role=glue_role
        )

        voyager_zip = aws_s3_assets.Asset(
            self, "PyAssets",
            path=os.path.join(PWD, "engine_lambda", "voyager"))

        # ~~~

        glue_pred_job = aws_glue.CfnJob(self, "PredictJob",
            name=f"{construct_id}-PredictJob",
            glue_version="2.0",
            max_capacity=2,
            role=glue_role.role_name,
            default_arguments={
                "--lambda_arn": engine_lambda_function.function_arn,
                "--additional-python-modules":
                    "pyarrow==2,cloudpickle==1.6.0,toolz==0.11.1,awswrangler",
                "--python-modules-installer-option": "--upgrade",
                "--extra-py-files": voyager_zip.s3_object_url,
                "--s3_input_path": "s3://BUCKET/PREFIX/",
                "--freq": "MS",
                "--horiz": "6"
            },
            command=aws_glue.CfnJob.JobCommandProperty(
                name="glueetl",
                python_version="3",
                script_location=bucket.s3_url_for_object("glue_scripts/predict.py")
            )
        )

        glue_pred_task = tasks.GlueStartJobRun(self, "PredictJobRun",
            glue_job_name=glue_pred_job.name,
            arguments=sfn.TaskInput.from_object({
                "--lambda_arn": engine_lambda_function.function_arn,
                "--additional-python-modules":
                    "pyarrow==2,cloudpickle==1.6.0,toolz==0.11.1,awswrangler,fsspec>=0.7.4",
                "--python-modules-installer-option": "--upgrade",
                "--extra-py-files": voyager_zip.s3_object_url,
                "--s3_input_path": sfn.JsonPath.string_at("$.Payload.s3_parquet_paths"),
                "--freq": sfn.JsonPath.string_at("$.Payload.freq"),
                "--horiz": sfn.JsonPath.string_at("$.Payload.horiz")
            }),

            # req. to sync sfn job run states to glue job run, otherwise
            # the state will instantaneously success in sfn despite still 
            # running in glue
            integration_pattern=sfn.IntegrationPattern.RUN_JOB,
            timeout=core.Duration.minutes(20)
        )

        # ~~~

        glue_report_job = aws_glue.CfnJob(self, "MakeReportJob",
            name=f"{construct_id}-MakeReportJob",
            glue_version="2.0",
            max_capacity=2,
            role=glue_role.role_name,
            command=aws_glue.CfnJob.JobCommandProperty(
                name="glueetl",
                python_version="3",
                script_location=bucket.s3_url_for_object("glue_scripts/report.py")
            )
        )

        glue_report_task = \
            tasks.GlueStartJobRun(self, "MakeReportTask",
            glue_job_name=glue_report_job.name,
            arguments=sfn.TaskInput.from_object({
                "--additional-python-modules":
                    ",".join(("pyarrow==2", "cloudpickle==1.6.0",
                              "toolz==0.11.1", "awswrangler", "fsspec>=0.7.4")),
                "--python-modules-installer-option": "--upgrade",
                "--extra-py-files": voyager_zip.s3_object_url,
                "--s3_input_path":
                    sfn.JsonPath.string_at("$.Payload.s3_input_path"),
                "--s3_parquet_paths":
                    sfn.JsonPath.string_at("$.Payload.s3_parquet_paths"),
            }),
            # req. to sync sfn job run states to glue job run, otherwise
            # the state will instantaneously success in sfn despite still 
            # running in glue
            integration_pattern=sfn.IntegrationPattern.RUN_JOB,
            timeout=core.Duration.minutes(20)
        )

        # ~~~

        parallel = sfn.Parallel(self, "ParallelTasks")
        parallel.branch(glue_pred_task)
        parallel.branch(glue_report_task)

        # deploy etl state-machine
        definition = glue_etl_task.next(lambda_resample_task) \
                                  .next(parallel)

        state_machine = \
            sfn.StateMachine(self, "SfsStateMachine",
                definition=definition,
                timeout=core.Duration.minutes(60)
            )

        cdk.CfnOutput(
            self,
            id="SfsStateMachine_ARN",
            value=state_machine.state_machine_arn
        )

        return

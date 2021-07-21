import os

from aws_cdk import core as cdk

from aws_cdk import (
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_s3 as s3,
    aws_sns as sns,
    aws_lambda as lambda_,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as tasks,
    aws_ssm as ssm,
    core
)

PWD = os.path.dirname(os.path.realpath(__file__))

# This is the lambda that sends the notification email to the user once
# the dashboard is deployed, it contains the URL to the landing page
# sagemaker notebook.
SNS_EMAIL_LAMBDA_INLINE = """import os
import re
import json
import boto3
import textwrap

def lambda_handler(event, context):
    payload = event["input"]["Payload"]

    # get s3 location of the exports
    afc_client = boto3.client("forecast")
    resp = afc_client.describe_forecast_export_job(
        ForecastExportJobArn=payload["ForecastExportJobArn"])
    s3_path = resp["Destination"]["S3Config"]["Path"]

    client = boto3.client("sns")

    response = client.publish(
        TopicArn=os.environ["TOPIC_ARN"],
        Subject="[Amazon SFS] Your Amazon Forecast job has completed!",
        Message=textwrap.dedent(f'''
        Hi!

        Your Amazon Forecast job has completed.

        The raw forecast files can be downloaded from the S3 path below:
        ‣ {s3_path}


        Sincerely,
        The Amazon SFS Team
        ‣ https://github.com/aws-samples/simple-forecast-solution
        '''))
    return response
"""


class AfcStack(cdk.Stack):
    def __init__(self, scope: cdk.Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        #
        # AFC/Lambda role
        #
        afc_role = iam.Role(
            self,
            f"{construct_id}-AfcRole",
            role_name=f"AfcRole",
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal("forecast.amazonaws.com"),
                iam.ServicePrincipal("lambda.amazonaws.com")
            ),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonForecastFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3FullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AWSLambda_FullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("CloudWatchLogsFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSNSFullAccess")
            ])

        fail_state = sfn.Fail(self, "Fail")
        succeed_state = sfn.Succeed(self, "Succeed")

        #
        # PREPARE DATA
        #

        prepare_lambda = \
            lambda_.Function(
                self,
                "PrepareLambda",
                runtime=lambda_.Runtime.PYTHON_3_8,
                handler="index.prepare_handler",
                code=lambda_.Code.from_asset(os.path.join(PWD, "afc_lambdas")),
                environment={
                        "AFC_ROLE_ARN": afc_role.role_arn
                    },
                role=afc_role,
                timeout=core.Duration.seconds(900))

        prepare_step = \
            tasks.LambdaInvoke(
                self,
                "PrepareDataStep",
                lambda_function=prepare_lambda,
                payload=sfn.TaskInput.from_object({
                    "input": sfn.JsonPath.string_at("$")
                })
            )

        #
        # CREATE PREDICTOR
        #
        create_predictor_lambda = \
            lambda_.Function(
                self,
                "CreatedPredictorLambda",
                runtime=lambda_.Runtime.PYTHON_3_8,
                handler="index.create_predictor_handler",
                code=lambda_.Code.from_asset(os.path.join(PWD, "afc_lambdas")),
                environment={
                    "AFC_ROLE_ARN": afc_role.role_arn
                },
                role=afc_role,
                timeout=core.Duration.seconds(900))

        create_predictor_step = \
            tasks.LambdaInvoke(
                self,
                "CreatePredictorStep",
                lambda_function=create_predictor_lambda,
                payload=sfn.TaskInput.from_object({
                    "input": sfn.JsonPath.string_at("$")
                })
            )

        create_predictor_step.add_retry(
                backoff_rate=1.05,
                interval=core.Duration.seconds(60),
                max_attempts=1000,
                errors=["ResourceNotFoundException",
                        "ResourceInUseException",
                        "ResourcePendingException"])

        #
        # CREATE FORECAST
        #
        create_forecast_lambda = \
            lambda_.Function(
                self,
                "CreatedforecastLambda",
                runtime=lambda_.Runtime.PYTHON_3_8,
                handler="index.create_forecast_handler",
                code=lambda_.Code.from_asset(os.path.join(PWD, "afc_lambdas")),
                role=afc_role,
                timeout=core.Duration.seconds(900))

        create_forecast_step = \
            tasks.LambdaInvoke(
                self,
                "CreateforecastStep",
                lambda_function=create_forecast_lambda,
                payload=sfn.TaskInput.from_object({
                    "input": sfn.JsonPath.string_at("$")
                })
            )

        create_forecast_step.add_retry(
            backoff_rate=1.1,
            interval=core.Duration.seconds(60),
            max_attempts=2000,
            errors=["ResourceNotFoundException",
                    "ResourceInUseException",
                    "ResourcePendingException"])

        #
        # CREATE FORECAST EXPORT
        #
        create_forecast_export_lambda = \
            lambda_.Function(
                self,
                "CreateExportLambda",
                runtime=lambda_.Runtime.PYTHON_3_8,
                handler="index.create_forecast_export_handler",
                code=lambda_.Code.from_asset(os.path.join(PWD, "afc_lambdas")),
                environment={
                    "AFC_ROLE_ARN": afc_role.role_arn
                },
                role=afc_role,
                timeout=core.Duration.seconds(900))

        create_forecast_export_step = \
            tasks.LambdaInvoke(
                self,
                "CreateExportStep",
                lambda_function=create_forecast_export_lambda,
                payload=sfn.TaskInput.from_object({
                    "input": sfn.JsonPath.string_at("$")
                })
            )

        create_forecast_export_step.add_retry(
            backoff_rate=1.1,
            interval=core.Duration.seconds(60),
            max_attempts=2000,
            errors=["ResourceInUseException",
                    "ResourcePendingException"])

        #
        # SNS EMAIL
        #
        sns_email_lambda = \
            lambda_.Function(self, f"{construct_id}-SnsEmailLambda",
            runtime=lambda_.Runtime.PYTHON_3_8,
            environment={"TOPIC_ARN": scope.topic.topic_arn},
            code=lambda_.Code.from_inline(SNS_EMAIL_LAMBDA_INLINE),
            handler="index.lambda_handler",
            role=afc_role)

        sns_email_step = \
            tasks.LambdaInvoke(
                self,
                "SnsAfcEmailStep",
                lambda_function=sns_email_lambda,
                payload=sfn.TaskInput.from_object({
                    "input": sfn.JsonPath.string_at("$")
                })
            )

        #
        # State machine
        #
        definition = prepare_step.next(create_predictor_step) \
                                 .next(create_forecast_step) \
                                 .next(create_forecast_export_step) \
                                 .next(sns_email_step)

        state_machine = sfn.StateMachine(self,
            "SfsSsmAfcStateMachine",
            state_machine_name=f"{construct_id}-AfcStateMachine",
            definition=definition,
            timeout=core.Duration.hours(24))

        ssm_state_machine_param = ssm.StringParameter(self,
            "SfsSsmAfcStateMachineArn",
            string_value=state_machine.state_machine_arn,
            parameter_name="SfsAfcStateMachineArn")

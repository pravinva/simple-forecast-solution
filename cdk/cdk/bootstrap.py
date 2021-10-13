#!/usr/bin/env python3
import os

from textwrap import dedent
from aws_cdk import (
    core,
    core as cdk,
    aws_lambda as lambda_,
    aws_iam as iam,
    aws_codebuild as codebuild,
)

AFA_REPO_URL = "https://github.com/aws-samples/simple-forecast-solution.git"
LAMBDAMAP_REPO_URL = "https://github.com/aws-samples/lambdamap.git"

LAMBDAMAP_STACK_NAME = "AfaLambdaMapStack"
LAMBDAMAP_FUNCTION_NAME = "AfaLambdaMapFunction"

# The lambda function to start a build of the codebuild project 
INLINE_CODEBUILD_LAMBDA = dedent("""
import os
import json
import boto3
import cfnresponse

def lambda_handler(event, context):
    client = boto3.client("codebuild")
    response = \
        client.start_build(projectName=os.environ["CODEBUILD_PROJECT_NAME"])
    response = json.loads(json.dumps(response, default=str))
    cfnresponse.send(event, context, cfnresponse.SUCCESS, response,
        "CustomResourcePhysicalID")
    return
""")


class BootstrapStack(core.Stack):
    def __init__(self, scope: cdk.Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id)

        email_address = core.CfnParameter(self, "emailAddress",
                allowed_pattern=".+",
                description="(Required) An e-mail address with which to receive "
                "deployment notifications.")

        instance_type = core.CfnParameter(self, "instanceType",
                default="ml.t2.medium",
                description="(Required) SageMaker Notebook instance type on which to host "
                "the AFA dashboard (e.g. ml.t2.medium, ml.t3.xlarge, ml.t3.2xlarge, ml.m4.4xlarge)")
        
        lambdamap_function_name = core.CfnParameter(self, "lambdamapFunctionName",
                default=LAMBDAMAP_FUNCTION_NAME)
        
        afa_branch = kwargs.get("afa_branch", "main")
        lambdamap_branch = kwargs.get("lambdamap_branch", "main")

        codebuild_project_id = "AfaCodeBuildProject"

        # Add any policies needed to deploy the main stack
        codebuild_role = iam.Role(
            self,
            f"CodeBuild",
            assumed_by=iam.ServicePrincipal("codebuild.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AWSCodeBuildDeveloperAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AWSCloudFormationFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEC2ContainerRegistryPowerUser"),
                iam.ManagedPolicy(
                    self, "CodeBuildManagedPolicy",
                    statements=[
                        iam.PolicyStatement(
                            effect=iam.Effect.ALLOW,
                            actions=[
                                "logs:*"
                            ],
                            resources=[
                                f"arn:aws:logs:{core.Aws.REGION}:{core.Aws.ACCOUNT_ID}:log-group:/aws/codebuild/{codebuild_project_id}*"
                            ]
                        )
                    ]
                ),
                iam.ManagedPolicy(
                    self, "S3ManagedPolicy",
                    statements=[
                        iam.PolicyStatement(
                            effect=iam.Effect.ALLOW,
                            actions=[
                                "s3:*"
                            ],
                            resources=[
                                f"arn:aws:s3:::cdktoolkit-stagingbucket-*",
                            ]
                        )
                    ]
                ),
                iam.ManagedPolicy(
                    self, "IamManagedPolicy",
                    statements=[
                        iam.PolicyStatement(
                            effect=iam.Effect.ALLOW,
                            actions=[
                                "iam:*"
                            ],
                            resources=[
                                f"*"
                            ]
                        )
                    ]
                ),
                iam.ManagedPolicy(
                    self, "CliManagedPolicy",
                    statements=[
                        iam.PolicyStatement(
                            effect=iam.Effect.ALLOW,
                            actions=[
                                "ec2:DescribeAvailabilityZones",
                                "sts:GetCallerIdentity",
                                "ecr:GetAuthorizationToken",
                                "ecr:BatchCheckLayerAvailability",
                                "ecr:GetDownloadUrlForLayer",
                                "ecr:GetRepositoryPolicy",
                                "ecr:DescribeRepositories",
                                "ecr:ListImages",
                                "ecr:DescribeImages",
                                "ecr:BatchGetImage",
                                "ecr:GetLifecyclePolicy",
                                "ecr:GetLifecyclePolicyPreview",
                                "ecr:ListTagsForResource",
                                "ecr:DescribeImageScanFindings",
                                "ecr:InitiateLayerUpload",
                                "ecr:UploadLayerPart",
                                "ecr:CompleteLayerUpload",
                                "ecr:PutImage",
                                "ecr:SetRepositoryPolicy"
                            ],
                            resources=[
                                f"*"
                            ]
                        )
                    ]
                ),
                iam.ManagedPolicy(
                    self, "LambdaManagedPolicy",
                    statements=[
                        iam.PolicyStatement(
                            effect=iam.Effect.ALLOW,
                            actions=[
                                "lambda:*",
                            ],
                            resources=[
                                f"arn:aws:lambda:{core.Aws.REGION}:{core.Aws.ACCOUNT_ID}:function:{lambdamap_function_name.value_as_string}"
                            ]
                        )
                    ]
                ),
            ],
        )

        codebuild_project = \
            codebuild.Project(self,
                codebuild_project_id,
                #project_name="RubixBootstrapProject",
                environment=codebuild.BuildEnvironment(
                    privileged=True,
                    build_image=codebuild.LinuxBuildImage.AMAZON_LINUX_2_3
                ),
                environment_variables={
                    "LAMBDAMAP_STACK_NAME": codebuild.BuildEnvironmentVariable(value=LAMBDAMAP_STACK_NAME),
                    "LAMBDAMAP_FUNCTION_NAME": codebuild.BuildEnvironmentVariable(value=LAMBDAMAP_FUNCTION_NAME),
                    "EMAIL": codebuild.BuildEnvironmentVariable(value=email_address.value_as_string),
                    "INSTANCE_TYPE": codebuild.BuildEnvironmentVariable(value=instance_type.value_as_string),
                },
                # define the deployment steps here
                build_spec=codebuild.BuildSpec.from_object(
                    {
                        "version": "0.2",
                        "phases": {
                            "install": {
                                "runtime-versions": {
                                    "python": "3.9",
                                    "nodejs": "12"
                                },
                                "commands": [
                                    "export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)",
                                    "npm i --silent --quiet --no-progress -g aws-cdk",
                                    "cdk bootstrap aws://$AWS_ACCOUNT_ID/$AWS_DEFAULT_REGION"
                                ]
                            },
                            "pre_build": {
                                "commands": [
                                ]
                            },
                            "build": {
                                "commands": [
                                    f"git clone {LAMBDAMAP_REPO_URL}",
                                    "cd lambdamap/",
                                    f"git checkout {lambdamap_branch}",
                                    "make deploy STACK_NAME=$LAMBDAMAP_STACK_NAME "
                                    "FUNCTION_NAME=$LAMBDAMAP_FUNCTION_NAME "
                                    f"EXTRA_CMDS='git clone {AFA_REPO_URL} ; cd ./simple-forecast-solution/ ; git checkout {afa_branch} ; pip install -e .'",
                                    "cd ..",
                                    f"git clone {AFA_REPO_URL}",
                                    "cd simple-forecast-solution/",
                                    "git checkout {afa_branch}",
                                    "make deploy-ui EMAIL=$EMAIL INSTANCE_TYPE=$INSTANCE_TYPE"
                                ]
                            },
                            "post_build": {
                                "commands": [
                                    # add post-deployment tests and checks here
                                    "echo 'Deploy Completed'"
                                ]
                            }
                        }
                    }
                ),
                role=codebuild_role
            )

        lambda_role = iam.Role(
            self, "LambdaRole",
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal("lambda.amazonaws.com")
            ),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AWSCodeBuildDeveloperAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
            ]
        )

        # this lambda function will trigger the cdk deployment via codebuild
        lambda_func = lambda_.Function(
            self, "LambdaFunction",
            runtime=lambda_.Runtime.PYTHON_3_8,
            code=lambda_.Code.from_inline(INLINE_CODEBUILD_LAMBDA),
            handler="index.lambda_handler",
            environment={
                "CODEBUILD_PROJECT_NAME": codebuild_project.project_name
            },
            role=lambda_role
        )

        cust_resource = core.CustomResource(self, "CustomResource",
            service_token=lambda_func.function_arn)
        cust_resource.node.add_dependency(codebuild_project)

        return


if __name__ == "__main__":
    app = core.App()
    BootstrapStack(app, "AfaBootstrapStack")
    app.synth()
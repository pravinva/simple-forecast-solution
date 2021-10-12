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

LAMBDAMAP_STACK_NAME = "AfaLambdaMapStack"
LAMBDAMAP_FUNCTION_NAME = "AfaLambdaMapFunction"

class BootstrapStack_OLD(cdk.Stack):
    def __init__(self, scope: cdk.Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        email_address = core.CfnParameter(self, "emailAddress",
                allowed_pattern=".+",
                description="(Required) An e-mail address with which to receive "
                "deployment notifications.")

        #   instance_type = core.CfnParameter(self, "instanceType",
        #           default="ml.t2.medium",
        #           description="(Required) SageMaker Notebook instance type on which to host "
        #           "the AFA dashboard (e.g. ml.t2.medium, ml.t3.xlarge, ml.t3.2xlarge, ml.m4.4xlarge)")
        instance_type = "ml.t2.medium"

        vpc = ec2.Vpc(self, f"BootstrapVpc", max_azs=1)

        ec2_role = iam.Role(
            self,
            f"AfaInstanceRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("IAMFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEC2ContainerRegistryFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMManagedInstanceCore"),
                iam.ManagedPolicy.from_aws_managed_policy_name("CloudWatchAgentServerPolicy")
            ])
        
        ec2_role.add_to_policy(iam.PolicyStatement(
            resources=[f"arn:aws:ec2:{self.region}:{self.account}:instance/*"],
            actions=["ec2:TerminateInstances"]
        ))
        
        ec2_role.add_to_policy(iam.PolicyStatement(
            resources=[f"arn:aws:cloudformation:{self.region}:{self.account}:*"],
            actions=["cloudformation:*"]
        ))

        ec2_role.add_to_policy(iam.PolicyStatement(
            resources=[f"arn:aws:logs:{self.region}:{self.account}:*"],
            actions=["logs:*"]
        ))

        ec2_role.add_to_policy(iam.PolicyStatement(
            resources=[
                f"arn:aws:s3:::cdktoolkit-stagingbucket-*",
                f"arn:aws:s3:::*-{self.account}-{self.region}",
            ],
            actions=["s3:*"]
        ))

        ec2_role.add_to_policy(iam.PolicyStatement(
            resources=[f"arn:aws:lambda:{self.region}:{self.account}:*"],
            actions=["lambda:*"]
        ))

        ec2_role.add_to_policy(iam.PolicyStatement(
            resources=[f"arn:aws:ssm:{self.region}:{self.account}:*"],
            actions=["ssm:*"]
        ))

        ec2_role.add_to_policy(iam.PolicyStatement(
            resources=[f"arn:aws:states:{self.region}:{self.account}:*"],
            actions=["states:*"]
        ))

        ec2_role.add_to_policy(iam.PolicyStatement(
            resources=[f"arn:aws:sns:{self.region}:{self.account}:*"],
            actions=["sns:*"]
        ))

        ami = ec2.MachineImage.generic_linux({
            "us-east-2": "ami-0443305dabd4be2bc",
            "us-west-1": "ami-04b6c97b14c54de18",
            "ap-southeast-2": "ami-0aab712d6363da7f9",
            "ap-southeast-1": "ami-0f511ead81ccde020",
            "eu-central-1": "ami-0453cb7b5f2b7fca2",
            "eu-west-1": "ami-02b4e72b17337d6c1"
        })

        user_data = ec2.UserData.for_linux()
        user_data.add_commands(
            dedent(f"""
            set -x

            # install git
            yum install -y git

            # install + start docker
            amazon-linux-extras install -y docker
            systemctl enable docker
            systemctl start docker
            usermod -aG docker ec2-user 
            setfacl --modify user:ec2-user:rw /var/run/docker.sock

            # install miniconda into ~/SageMaker/miniconda, which will make it
            # persistent
            CONDA_DIR=~/miniconda/
            mkdir -p "$CONDA_DIR"
            wget -q https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh \
                -O "$CONDA_DIR/miniconda.sh"
            bash "$CONDA_DIR/miniconda.sh" -b -u -p "$CONDA_DIR"
            rm -rf "$CONDA_DIR/miniconda.sh"

            # use local miniconda distro
            source "$CONDA_DIR/bin/activate"
            # install custom conda environment(s)
            conda create -y -q -n py39 python=3.9 nodejs=14
            conda activate py39
            # install the aws-cdk cli tool (req. for running `cdk deply ...`)

            npm i --silent --quiet --no-progress -g aws-cdk@1.116.0

            cd ~
            git clone https://github.com/aws-samples/lambdamap.git
            cd ./lambdamap/lambdamap_cdk/
            pip install -q -r ./requirements.txt
            cdk bootstrap aws://{self.account}/{self.region}
            nohup cdk deploy \
                --require-approval never \
                --context stack_name=AfaLambdaMapStack \
                --context function_name=AfaLambdaMapFunction \
                --context memory_size=256 \
                --context extra_cmds='git clone https://github.com/aws-samples/simple-forecast-solution.git ; cd ./simple-forecast-solution/ ; git checkout main ; pip install -e .'

            cd ~
            git clone https://github.com/aws-samples/simple-forecast-solution.git
            cd ./simple-forecast-solution/cdk
            pip install -q -r ./requirements.txt
            cdk bootstrap aws://{self.account}/{self.region}
            nohup cdk deploy AfaStack \
                --require-approval never \
                --parameters AfaStack:emailAddress={email_address.value_as_string} \
                --parameters AfaStack:instanceType={instance_type}

            sleep 10
            
            # self-terminate the bootstrap ec2 instance
            aws ec2 terminate-instances \
                --instance-ids $(curl -s http://169.254.169.254/latest/meta-data/instance-id) \
                --region {self.region}

            wait
            sleep 60

            shutdown
            """)
        )

        instance = ec2.Instance(
            self,
            f"Ec2Instance",
            instance_name=f"{construct_id}-Ec2Instance",
            instance_type=ec2.InstanceType("t2.micro"),
            block_devices=[{
                "deviceName": "/dev/xvda",
                "volume": ec2.BlockDeviceVolume.ebs(
                    volume_size=8,
                    delete_on_termination=True)
            }],
            machine_image=ami,
            user_data=user_data,
            role=ec2_role,
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC))


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
        super().__init__(scope, construct_id, **kwargs)

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
                    "LAMBDAMAP_FUNCTION_NAME": codebuild.BuildEnvironmentVariable(value=LAMBDAMAP_FUNCTION_NAME)
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
                                    "git clone https://github.com/aws-samples/lambdamap.git",
                                    "cd lambdamap",
                                    "git checkout develop", # TODO:
                                    "make bootstrap",
                                    "make deploy STACK_NAME=$LAMBDAMAP_STACK_NAME "
                                    "FUNCTION_NAME=$LAMBDAMAP_FUNCTION_NAME "
                                    "EXTRA_CMDS='git clone https://github.com/aws-samples/simple-forecast-solution.git ; cd ./simple-forecast-solution/ ; git checkout main ; pip install -e .'"
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
import os

from textwrap import dedent
from aws_cdk import (
    aws_ec2 as ec2,
    aws_iam as iam,
    core,
    core as cdk
)

PWD = os.path.dirname(os.path.realpath(__file__))


class BootstrapStack(cdk.Stack):
    def __init__(self, scope: cdk.Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        email_address = core.CfnParameter(self, "emailAddress").value_as_string
        instance_type = core.CfnParameter(self, "instanceType",
                default="ml.t3.xlarge").value_as_string

        vpc = ec2.Vpc(self, f"{construct_id}-Vpc", max_azs=1)

        ec2_role = iam.Role(
            self,
            "SfsInstanceRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AdministratorAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMManagedInstanceCore"),
                iam.ManagedPolicy.from_aws_managed_policy_name("CloudWatchAgentServerPolicy")
            ])

        ami = ec2.MachineImage.generic_linux({
            "ap-southeast-2": "ami-0aab712d6363da7f9"
        })

        user_data = ec2.UserData.for_linux(shebang="#!/bin/bash")
        user_data.add_commands(
            dedent(f"""
            whoami
            pwd

            yum install -y git

            time sudo -u ec2-user -i <<'EOF'
            #!/bin/bash
            unset SUDO_UID

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

            # install the aws-cdk cli tool (req. for running `cdk deploy ...`)
            npm i -g aws-cdk@1.116.0

            git clone https://github.com/aws-samples/simple-forecast-solution.git
            cd ./simple-forecast-solution
            git checkout develop
            cd ./cdk
            pip install -r ./requirements.txt

            cdk bootstrap
            cdk deploy SfsStack \
                --parameters SfsStack:emailAddress={email_address} \
                --parameters SfsStack:instance_type={instance_type} \
                --require-approval never

            which python
            EOF
            """)
        )

        instance = ec2.Instance(
            self,
            f"{construct_id}-Instance",
            instance_name=f"{construct_id}-Instance",
            instance_type=ec2.InstanceType("t2.micro"),
            block_devices=[{
                "deviceName": "/dev/xvda",
                "volume": ec2.BlockDeviceVolume.ebs(
                    volume_size=10,
                    delete_on_termination=True)
            }],
            machine_image=ami,
            user_data=user_data,
            role=ec2_role,
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC))

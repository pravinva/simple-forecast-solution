from aws_cdk import core as cdk

# For consistency with other languages, `cdk` is the preferred import name for
# the CDK's core module.  The following line also imports it as `core` for use
# with examples from the CDK Developer's Guide, which are in the process of
# being updated to use `cdk`.  You may delete this import if you don't need it.
from aws_cdk import (
    aws_ec2 as ec2,
#   aws_ecs as ecs,
#   aws_ecr as ecr,
#   aws_iam as iam,
#   aws_cognito as cognito,
#   aws_ecs_patterns as ecs_patterns,
#   aws_elasticloadbalancingv2 as elb,
#   aws_elasticloadbalancingv2_actions as elb_actions,
#   aws_certificatemanager as certificatemanager,
    aws_iam as iam,
    aws_sagemaker as sm,
    aws_sns as sns,
    aws
    core
)

#
# Lifecycle config raw strings
#
LCC_ONSTART_STR = r"""#!/bin/bash
set -e
initctl restart jupyter-server --no-wait

#
# Start SFS dashboard in the background
#
source /home/ec2-user/anaconda3/bin/activate sfs
streamlit hello &
"""

LCC_ONCREATE_STR = r"""#!/bin/bash
set -e
export LC_ALL=en_US.utf-8 && export LANG=en_US.utf-8
source /home/ec2-user/anaconda3/bin/activate JupyterSystemEnv

#
# Install SFS
#
/home/ec2-user/anaconda3/bin/conda create -q -n sfs python=3.8.10
source /home/ec2-user/anaconda3/bin/activate sfs

# Install the dashboard
git clone --recurse-submodules https://github.com/aws-samples/simple-forecast-solution.git
cd ./simple-forecast-solution
pip install -q -e .

# Install the lambdamap python library
cd ./sfs/lambdamap/
pip install -q -e .

# Install aws-cdk
curl -sL https://rpm.nodesource.com/setup_14.x | bash - \
    && yum install -y nodejs \
    && npm install -g aws-cdk@1.114.0

# Install the SfsLambdaMapStack
cd ./lambdamap_cdk/
pip install -q -r ./requirements.txt
cdk deploy --require-approval never \
    --context stack_name=SfsLambdaMapStack \
    --context function_name=SfsLambdaMapFunction \
    --context extra_cmds='pip install -q git+https://github.com/aws-samples/simple-forecast-solution.git#egg=sfs'

# Get the notebook URL
NOTEBOOK_URL=$(aws sagemaker describe-notebook-instance \
    --notebook-instance-name {construct_id}-NotebookInstance \
    --query "Url" \
    --output text)

# Get the instructions ipynb notebook URL (email to user)
INSTRUCTIONS_URL=$NOTEBOOK_URL/lab/tree/Instructions.ipynb

#
# Upgrade jupyter-server-proxy
#
source /home/ec2-user/anaconda3/bin/activate JupyterSystemEnv

pip uninstall -q --yes nbserverproxy || true
pip install -q --upgrade jupyter-server-proxy
"""


class BootstrapStack(cdk.Stack):
    def __init__(self, scope: cdk.Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        #
        # Notebook lifecycle configuration
        #
        lcc_onstart_obj = \
            sm.CfnNotebookInstanceLifecycleConfig \
              .NotebookInstanceLifecycleHookProperty(
                  content=core.Fn.base64(LCC_ONSTART_STR))

        lcc_oncreate_obj = \
            sm.CfnNotebookInstanceLifecycleConfig \
              .NotebookInstanceLifecycleHookProperty(
                  content=core.Fn.base64(LCC_ONCREATE_STR.format(construct_id=construct_id)))

        lcc = sm.CfnNotebookInstanceLifecycleConfig(
            self,
            f"{construct_id}-NotebookLifecycleConfig",
            on_create=[lcc_oncreate_obj],
            on_start=[lcc_onstart_obj])

        #
        # Notebook role
        #
        sm_role = iam.Role(
            self,
            f"{construct_id}-NotebookRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AWSCloudFormationFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AWSLambda_FullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("EC2InstanceProfileForImageBuilderECRContainerBuilds"),
                iam.ManagedPolicy.from_aws_managed_policy_name("IAMFullAccess")
            ])

        #
        # Notebook instance
        #
        sm.CfnNotebookInstance(
            self,
            f"{construct_id}-NotebookInstance",
            role_arn=sm_role.role_arn,
            instance_type="ml.t3.large",
            notebook_instance_name=f"{construct_id}-NotebookInstance",
            lifecycle_config_name=lcc.attr_notebook_instance_lifecycle_config_name)

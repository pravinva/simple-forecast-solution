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
conda activate sfs
# streamlit hello &
"""

LCC_ONCREATE_STR = r"""#!/bin/bash
set -e
export LC_ALL=en_US.utf-8 && export LANG=en_US.utf-8

#
# Install SFS
#
conda create -n sfs python=3.8.10 nodejs=14.17.3
conda activate sfs

# Install the dashboard
git clone https://github.com/aws-samples/simple-forecast-solution.git
cd ./simple-forecast-solution
pip install -e .

# Install the SfsLambdaMapStack
cd ./sfs/lambdamap/lambdamap_cdk
pip install -r ./requirements.txt

# The lambdamap function docker image needs sfs installed
cdk deploy \
    --context stack_name=SfsLambdaMapStack \
    --context function_name=SfsLambdaMapFunction \
    --context extra_cmds='pip install git+https://github.com/aws-samples/simple-forecast-solution.git#egg=sfs'

#
# Upgrade jupyter-server-proxy
#
source /home/ec2-user/anaconda3/bin/activate JupyterSystemEnv

pip uninstall --yes nbserverproxy || true
pip install --upgrade jupyter-server-proxy

# Clone your app's source code here
# git clone https://....
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
                  content=core.Fn.base64(LCC_ONCREATE_STR))

        lcc = sm.CfnNotebookInstanceLifecycleConfig(
            self,
            "SfsNotebookLifecycleConfig",
            on_create=[lcc_oncreate_obj],
            on_start=[lcc_onstart_obj])

        #
        # Notebook role
        #
        sm_role = iam.Role(
            self,
            "SfsNotebookRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess")
            ])

        #
        # Notebook instance
        #
        sm.CfnNotebookInstance(
            self,
            "SfsNotebookInstance",
            role_arn=sm_role.role_arn,
            instance_type="ml.t3.large",
            notebook_instance_name="SfsNotebookInstance",
            lifecycle_config_name=lcc.attr_notebook_instance_lifecycle_config_name)

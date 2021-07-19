#!/usr/bin/env python3
import os
import sys

from aws_cdk import core as cdk

# For consistency with TypeScript code, `cdk` is the preferred import name for
# the CDK's core module.  The following line also imports it as `core` for use
# with examples from the CDK Developer's Guide, which are in the process of
# being updated to use `cdk`.  You may delete this import if you don't need it.
from aws_cdk import core

from cdk.stack import BootstrapStack

PWD = os.path.dirname(os.path.realpath(__file__))
LAMBDAMAP_CDK_PATH = os.path.join(PWD, "../sfs/lambdamap/")

assert(os.path.exists(LAMBDAMAP_CDK_PATH))
sys.path.append(LAMBDAMAP_CDK_PATH)

from lambdamap_cdk.stack.stack import LambdaMapStack

app = core.App()
BootstrapStack(app, "SfsBootstrapStack")
LambdaMapStack(app, "SfsLambdaMapStack", function_name="SfsLambdaMapFunction")
app.synth()

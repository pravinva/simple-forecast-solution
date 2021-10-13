#!/usr/bin/env python3
import os
import sys

from aws_cdk import core
from aws_cdk import core as cdk

from cdk.stack import AfaStack
from cdk.bootstrap import BootstrapStack

PWD = os.path.dirname(os.path.realpath(__file__))

app = core.App()

stack_name = app.node.try_get_context("afa_stack_name")
boot_stack_name = app.node.try_get_context("boot_stack_name")
branch = app.node.try_get_context("branch")

if stack_name is None:
    stack_name = "AfaStack"

if boot_stack_name is None:
    boot_stack_name = "AfaBootstrapStack"

if branch is None:
    branch = "main"

stack = AfaStack(app, stack_name)
boot_stack = \
    BootstrapStack(app, boot_stack_name, lambdamap_branch=branch,
                   afa_branch=branch)
app.synth()

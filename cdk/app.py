#!/usr/bin/env python3
import os
import sys

from aws_cdk import core as cdk

# For consistency with TypeScript code, `cdk` is the preferred import name for
# the CDK's core module.  The following line also imports it as `core` for use
# with examples from the CDK Developer's Guide, which are in the process of
# being updated to use `cdk`.  You may delete this import if you don't need it.
from aws_cdk import core

from cdk.stack import SfsStack
from cdk.bootstrap import BootstrapStack

PWD = os.path.dirname(os.path.realpath(__file__))

app = core.App()

stack_name = app.node.try_get_context("sfs_stack_name")
boot_stack_name = app.node.try_get_context("boot_stack_name")

if stack_name is None:
    stack_name = "SfsStack"

if boot_stack_name is None:
    boot_stack_name = "SfsBootstrapStack"

stack = SfsStack(app, stack_name)
boot_stack = BootstrapStack(app, boot_stack_name)

app.synth()

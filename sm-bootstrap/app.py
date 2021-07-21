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
from cdk.afc_stack import AfcStack

PWD = os.path.dirname(os.path.realpath(__file__))

app = core.App()

boot_stack_name = app.node.try_get_context("sfs_stack_name")
afc_stack_name = app.node.try_get_context("afc_stack_name")

if boot_stack_name is None:
    boot_stack_name = "SfsStack"

if afc_stack_name is None:
    afc_stack_name = "Afc"

boot_stack = SfsStack(app, boot_stack_name)
AfcStack(boot_stack, afc_stack_name)

app.synth()

#!/usr/bin/env python3
from aws_cdk import core
from aws_cdk import core as cdk

from sfn_stack.stack import SfnStack


if __name__ == '__main__':
    app = core.App()
    app_id = app.node.try_get_context("app_id")
    sfn_stack = SfnStack(app, f"SfnStack-{app_id}")
    app.synth()

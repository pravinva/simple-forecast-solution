#!/usr/bin/env python3
import os

from aws_cdk import core
from aws_cdk import core as cdk

from glue_etl.glue_etl_stack import GlueEtlStack

app = core.App()

GlueEtlStack(app, "GlueEtlStack")

app.synth()

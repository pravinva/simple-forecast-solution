#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
aws cloudformation create-stack \
  --template-body file://template.yaml \
  --stack-name sfs-stack-01 \
  --capabilities CAPABILITY_IAM

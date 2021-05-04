#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
git clone https://github.com/aws-samples/simple-forecast-solution.git
cd simple-forecast-solution/
git checkout develop
( source ./install.sh ; install_prereqs )

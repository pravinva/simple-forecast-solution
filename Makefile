#
# ...
#
# Created:
#   Sat Jul 24 14:45:13 UTC 2021
#
# Usage:
#   make deploy EMAIL=<your@email.address> INSTANCE_TYPE=<ml.* ec2 instance type>
# 
export SHELL
SHELL:=/bin/bash
ROOT_DIR:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
EMAIL:=
INSTANCE_TYPE:=ml.t2.medium

# Deploy the SFS stack
deploy: SfsStack

template.yaml:
	( cd cdk ; cdk synth SfsBootstrapStack ) > $@

destroy:
	cd cdk ; cdk destroy --all

SfsStack:
	cd cdk ; \
		cdk deploy $@ --parameters $@:emailAddress=${EMAIL} \
		--parameters $@:instanceType=${INSTANCE_TYPE}

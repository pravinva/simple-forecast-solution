#
# ...
#
# Created:
#   Sat Jul 24 14:45:13 UTC 2021
#
# Usage:
#   make EMAIL=<your@email.address>
# 
export SHELL
SHELL:=/bin/bash
ROOT_DIR:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
EMAIL:=
INSTANCE_TYPE:=ml.t3.xlarge

# Deploy the SFS stack
deploy: SfsStack

destroy:
	cd sm-bootstrap ; cdk destroy --all

SfsStack:
	cd sm-bootstrap ; \
		cdk deploy --all --parameters $@:emailAddress=${EMAIL} --parameters $@:instanceType=${INSTANCE_TYPE}

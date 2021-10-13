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

STACK_NAME:=AfaBootstrapStack
S3_BUCKET:=

# SNS emails will be sent to this address to notify when the SageMaker Notebook
# instances are deployed and when the ML forecasting jobs are completed.
EMAIL:=
INSTANCE_TYPE:=ml.t2.medium

.PHONY: deploy

# create the virtual environment from which to run each target
.venv: requirements.txt
	python3 -B -m venv $@
	source $@/bin/activate ; pip install -r $<

destroy:
	cd cdk ; cdk destroy --all

# Generate the cloudformation YAML template
template.yaml: cdk/bootstrap.py .venv
	source $(word 2, $^)/bin/activate ; \
	cdk synth -a 'python3 -B $<' > $@

${STACK_NAME}: template.yaml
	aws cloudformation deploy \
		--template-file $< \
		--capabilities CAPABILITY_NAMED_IAM \
		--stack-name $@ \
		--parameter-overrides \
			emailAddress=${EMAIL} \
			instanceType=${INSTANCE_TYPE} \
			cdkRoleArn=abc

deploy: ${STACK_NAME}

#AfaStack: templ
#	cd cdk ; \
#		cdk deploy $@ --parameters $@:emailAddress=${EMAIL} \
#		--parameters $@:instanceType=${INSTANCE_TYPE}
#
#AfaBootstrapStack:
#	cd cdk ; \
#		cdk deploy $@ --parameters $@:emailAddress=${EMAIL}
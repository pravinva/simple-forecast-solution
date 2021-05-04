# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
export SHELL
SHELL:=/bin/bash

APP_NAME:=SFSDev

ENABLE_MFA:=1
ENABLE_SSE:=1
ENABLE_FRONTEND:=1

S3_BUCKET:=simple-forecast-service

docker_deploy_all:
	docker build -t sfs-deploy .
	docker run -it sfs-deploy:latest make deploy_all

deploy_no_frontend: ./install.sh
	source $< $(APP_NAME) $(ENABLE_MFA) $(ENABLE_SSE) 0 ; deploy_all

deploy_all: ./install.sh
	source $< $(APP_NAME) $(ENABLE_MFA) $(ENABLE_SSE) $(ENABLE_FRONTEND) ; $@

frontend: ./install.sh
	source $< $(APP_NAME) ; deploy_amplify_frontend

frontend_local:
	( cd app/amplify-stack ; npm run build ; npm run dev )

clean:
	rm -rf ./aws ./awscliv2.zip

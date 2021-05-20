# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# USAGE
#   ( source ./install.sh ; deploy_all )
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

PY37_VERSION=3.7.9
PY38_VERSION=3.8.6

AMPLIFY_DIR=$ROOT_DIR/app/amplify-stack/
SAM_DIR=$ROOT_DIR/app/sam-stack

AMPLIFY_APP_NAME=${1:-SFS2}
AMPLIFY_APP_ENV=delta
AMPLIFY_META_JSON=$AMPLIFY_DIR/amplify/backend/amplify-meta.json
AMPLIFY_PROVIDER_INFO=$AMPLIFY_DIR/amplify/team-provider-info.json
AMPLIFY_SNS_EMAIL=myforecast@amazon.com

FRONTEND_ZIP=/tmp/frontend-artifacts.zip
PYWREN_RUNTIME_BUCKET=pywren-runtimes-public-us-west-2

AWSWRANGLER_URL=https://github.com/awslabs/aws-data-wrangler/releases/download/2.0.0/awswrangler-layer-2.0.0-py3.8.zip
AWSWRANGLER_ZIP=/tmp/$(basename $AWSWRANGLER_URL)

ENABLE_MFA=${2:-1}
ENABLE_SSE=${3:-1}
ENABLE_FRONTEND=${4:-1}

#export AWS_IAM_ROLE=`curl -sL http://169.254.169.254/latest/meta-data/iam/security-credentials/`
#export AWS_ACCESS_KEY_ID=`curl -sL http://169.254.169.254/latest/meta-data/iam/security-credentials/$AWS_IAM_ROLE/ | jq -r '.AccessKeyId'`
#export AWS_SECRET_ACCESS_KEY=`curl -sL http://169.254.169.254/latest/meta-data/iam/security-credentials/$AWS_IAM_ROLE/ | jq -r '.SecretAccessKey'`
#export AWS_TOKEN=`curl -sL http://169.254.169.254/latest/meta-data/iam/security-credentials/$AWS_IAM_ROLE/ | jq -r '.Token'`
#export AWS_AZ=`curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone`
#export AWS_DEFAULT_REGION="`echo \"$AWS_AZ\" | sed -e 's:\([0-9][0-9]*\)[a-z]*\$:\\1:'`"


function get_aws_region() {
    EC2_AVAIL_ZONE=`curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone`
    echo $EC2_AVAIL_ZONE | sed 's/[a-z]$//'
}


function get_aws_account_id() {
    aws sts get-caller-identity --output text --query 'Account'
}


function get_amplify_stack_name() {
    jq -r '.'$AMPLIFY_APP_ENV'.awscloudformation.StackName' \
        $AMPLIFY_PROVIDER_INFO
}


function get_amplify_app_id() {
    aws amplify list-apps | \
        jq -r '.apps[] | select(.name=="'$AMPLIFY_APP_NAME'") .appId'
}


function get_deployment_bucket() {
    jq -r '.'$AMPLIFY_APP_ENV'.awscloudformation.DeploymentBucketName' \
        $AMPLIFY_PROVIDER_INFO
}


function get_amplify_data_bucket() {
    jq -r '.aws_user_files_s3_bucket' $AMPLIFY_DIR/amplify-out/aws-exports.json
}


function write_custom_http_yaml() {
    # Generate the customHttp.yml file required for custom HTTP headers
    (\
    cd $AMPLIFY_DIR

    cat > >(cut -c 5- | python - > ./customHttp.yml) <<-EOF
    print(
        open("./customHttp.yml.template", "r")
           .read()
           .replace("AMPLIFY_BRANCH", "$AMPLIFY_APP_ENV")
           .replace("AMPLIFY_DATA_BUCKET", "$(get_amplify_data_bucket)")
           .replace("AMPLIFY_APP_ID", "$(get_amplify_app_id)")
           .replace("AWS_REGION", "$(get_aws_region)")
    )
	EOF
    )
}


function install_prereqs() {
    set +x

    cat > >(cut -c 5-) <<-EOF
    ---------------------------
    INSTALLING SYSTEM LIBRARIES
    ---------------------------
	EOF

    export NEXT_TELEMETRY_DISABLED=1

    cd $ROOT_DIR

    sudo yum update -y
    sudo yum install -y shadow-utils.x86_64 sudo git zip unzip wget which tar jq \
        gcc-c++ make patch openssl-devel zlib-devel readline-devel sqlite-devel \
        bzip2-devel libffi-devel
    sudo yum clean all

    [[ -d ./aws/ ]] && ( rm -rf ./aws )

    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip -q awscliv2.zip
    sudo ./aws/install --update

    if [[ ! -d ~/.pyenv ]]
    then
        . ~/.bashrc
        git clone git://github.com/yyuu/pyenv.git ~/.pyenv

	echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
	echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
	echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
	echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    fi

    . ~/.bashrc
    [[ -d ~/.pyenv/versions/$PY37_VERSION ]] || ( pyenv install --force $PY37_VERSION )
    [[ -d ~/.pyenv/versions/$PY38_VERSION ]] || ( pyenv install --force $PY38_VERSION )

    pyenv global $PY38_VERSION
    pip install pipenv==2020.11.15
    pip install --force-reinstall \
        aws-sam-cli==1.13.2 \
        aws-cdk.core==1.100.0 \
        aws-cdk.aws_lambda==1.100.0 \
        aws-cdk.aws_ecr==1.100.0 \
        aws-cdk.aws-iam==1.100.0 \
        aws-cdk.aws-glue==1.100.0 \
        aws-cdk.aws-s3==1.100.0 \
        aws-cdk.aws-s3-assets==1.100.0 \
        aws-cdk.aws-s3-deployment==1.100.0 \
        aws-cdk.aws-stepfunctions==1.100.0 \
        aws-cdk.aws-stepfunctions-tasks==1.100.0

    mkdir -p ~/.aws/
    REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone | sed 's/[a-z]$//')
    echo -e "[default]\noutput = json\nregion = $REGION" > ~/.aws/config
    echo -e "[default]" > ~/.aws/credentials

    . ~/.bashrc ; \
    echo 'export N_PREFIX=$HOME/.n' >> ~/.bashrc ; \
    echo 'export PATH=$N_PREFIX/bin:$PATH' >> ~/.bashrc ; \
    export N_PREFIX=$HOME/.n ; export PATH=$N_PREFIX/bin:$PATH ; \
    curl -L https://raw.githubusercontent.com/tj/n/master/bin/n -o n ; \
    bash n lts ; \
    npm install -g @aws-amplify/cli@4.45.0
    npm install -g aws-cdk

    set +x
}


function deploy_amplify_backend() {
    set +x
    set -e

    cat > >(cut -c 5-) <<-EOF
    -----------------------------
    DEPLOYING THE AMPLIFY BACKEND
    -----------------------------
	EOF

    export AWS_SDK_LOAD_CONFIG=1
    export NEXT_TELEMETRY_DISABLED=1

    source ~/.bashrc

    pyenv global $PY38_VERSION

    cd $AMPLIFY_DIR

    rm --force $AMPLIFY_PROVIDER_INFO
    npm install

    set -x

    pipenv --rm || true
    pipenv --python $PY38_VERSION

    aws amplify create-app --name $AMPLIFY_APP_NAME

    # Enable MFA if required
    AUTH_JSON=$AMPLIFY_DIR/amplify/backend/auth/voyagerfrontend849169bf/parameters.json
    AUTH_JSON_TEMPLATE=${AUTH_JSON}.template
    AUTH_JSON_TMP=/tmp/parameters.json
    AUTH_JSON_BAK=$AUTH_JSON.bak

    cp $AUTH_JSON $AUTH_JSON_BAK

    if (($ENABLE_MFA))
    then
        jq '(.mfaConfiguration) |= "ON"' $AUTH_JSON_TEMPLATE > $AUTH_JSON_TMP
    else
        echo "WARNING: MFA NOT ENABLED"
        jq '(.mfaConfiguration) |= "OFF"' $AUTH_JSON_TEMPLATE > $AUTH_JSON_TMP
    fi

    cp $AUTH_JSON_TMP $AUTH_JSON

    pipenv run amplify init \
        --appId $(get_amplify_app_id) \
        --envName $AMPLIFY_APP_ENV \
        --yes

    pipenv run amplify push --yes

    # Generate the latest src/aws-exports.js file
    # - i.e. to store the latest ARN and ids
    pipenv run amplify env checkout $AMPLIFY_APP_ENV

    cp src/aws-exports.js amplify-out/

    # Generate `amplify-out/aws-exports.json` (JSON) from `aws-exports.js`
    node amplify-out/generate-json.js

    set +x
    set +e
}


function deploy_amplify_frontend() {
    set -x

    cat > >(cut -c 5-) <<-EOF
    ------------------------------
    DEPLOYING THE AMPLIFY FRONTEND
    ------------------------------
	EOF

    export NEXT_TELEMETRY_DISABLED=1

    pyenv global $PY38_VERSION

    (\
        cd $AMPLIFY_DIR

        write_custom_http_yaml

        npm install
        INLINE_RUNTIME_CHUNK=false npm run build

        rm --force $FRONTEND_ZIP

        (\
            cp ./customHttp.yml ./out/
            cd out/
            zip -r $FRONTEND_ZIP *
            aws s3 cp $FRONTEND_ZIP s3://$(get_deployment_bucket)
        )

        # deploy frontend zip file to amplify
        aws amplify create-branch \
        --app-id $(get_amplify_app_id) \
        --branch-name $AMPLIFY_APP_ENV

        aws amplify start-deployment \
        --app-id $(get_amplify_app_id) \
        --branch-name $AMPLIFY_APP_ENV \
        --source-url s3://$(get_deployment_bucket)/$(basename $FRONTEND_ZIP)
    )

    set +x
}


function patch_pywren() {
    mkdir -p /tmp/pywren/
    (\
        cd /tmp/pywren
        rm -f v0.4.zip
        wget https://github.com/pywren/pywren/archive/v0.4.zip
        unzip -o v0.4.zip

        cp pywren-0.4/pywren/jobrunner/jobrunner.py \
            pywren-0.4/pywren/version.py \
            pywren-0.4/pywren/wren.py \
            pywren-0.4/pywren/wrenconfig.py \
            pywren-0.4/pywren/wrenhandler.py \
            pywren-0.4/pywren/wrenutil.py \
            pywren-0.4/pywren/scripts/pywrencli.py \
            $SAM_DIR/pywren/src/

        cd $SAM_DIR/pywren/src/
        patch -i $SAM_DIR/pywren/py3_7_9-pywren-lambda.patch
    )
}


function deploy_engine_stack() {
    cat > >(cut -c 5-) <<-EOF
    --------------------------
    DEPLOYING SFS ENGINE STACK
    --------------------------
	EOF

    export SAM_CLI_TELEMETRY=0

    source ~/.bashrc
    cd $SAM_DIR

    FCAST_STACK=sam-$AMPLIFY_APP_NAME-$(get_amplify_app_id)
    PYWREN_LAMBDA=PyWrenFunction-$(get_amplify_app_id)
    PYWREN_ROLE=PyWrenRole-$(get_amplify_app_id)
    PYWREN_BUCKET=$(get_amplify_data_bucket)

    #
    # Create the `.pywren_config` file for the `pywren` sub-stack
    #
    rm -f $SAM_DIR/engine/src/.pywren_config

    cat > >( cut -c 5- > $SAM_DIR/engine/src/.pywren_config ) <<-EOF
    account:
        aws_account_id: "$(get_aws_account_id)"
        aws_lambda_role: $PYWREN_ROLE
        aws_region: $(get_aws_region)

    lambda:
        memory : 3008
        timeout : 300
        function_name : $PYWREN_LAMBDA

    s3:
        bucket: $PYWREN_BUCKET
        pywren_prefix: pywren.jobs

    runtime:
        s3_bucket: $PYWREN_RUNTIME_BUCKET
        s3_key: pywren.runtimes/default_3.7.meta.json

    scheduler:
        map_item_limit: 10000

    standalone:
        ec2_instance_type: m4.large
        sqs_queue_name: pywren-jobs-1
        visibility: 10
        ec2_ssh_key : PYWREN_DEFAULT_KEY
        target_ami : ami-a0cfeed8
        instance_name: pywren-standalone
        instance_profile_name: pywren-standalone
        max_idle_time: 60
        idle_terminate_granularity: 3600
	EOF

    patch_pywren

    #
    # Deploy the pywren stack
    #
    cd $SAM_DIR/pywren/

    PYWREN_STACK=sam-pywren-$(get_amplify_app_id)
    PYWREN_SAM_OVERRIDES="[\
        \"PyWrenFunctionName=$PYWREN_LAMBDA\",\
        \"PyWrenRoleName=$PYWREN_ROLE\",\
        \"PyWrenBucketName=$PYWREN_BUCKET\",\
        \"PyWrenRuntimeBucketName=$PYWREN_RUNTIME_BUCKET\"]"

    cat > >( cut -c 5- > $SAM_DIR/pywren/samconfig.toml ) <<-EOF
    version = 0.1
    [default]
    [default.deploy]
    [default.deploy.parameters]
    stack_name = "$PYWREN_STACK"
    s3_bucket = "$PYWREN_BUCKET"
    s3_prefix = "sam"
    region = "$(get_aws_region)"
    confirm_changeset = false
    capabilities = "CAPABILITY_NAMED_IAM"
    parameter_overrides = $PYWREN_SAM_OVERRIDES
	EOF

    pyenv global $PY37_VERSION

    pip install aws-sam-cli
    sam build --config-file $SAM_DIR/pywren/samconfig.toml
    sam deploy --config-file $SAM_DIR/pywren/samconfig.toml

    #
    # Deploy the engine stack
    #
    cd $SAM_DIR/engine/

    ENGINE_STACK=sam-engine-$(get_amplify_app_id)
    ENGINE_SAM_OVERRIDES="[\
        \"PyWrenBucketName=$PYWREN_BUCKET\",\
        \"DataBucketName=$PYWREN_BUCKET\"]"

    cat > >( cut -c 5- > $SAM_DIR/engine/samconfig.toml ) <<-EOF
    version = 0.1
    [default]
    [default.deploy]
    [default.deploy.parameters]
    stack_name = "$ENGINE_STACK"
    s3_bucket = "$PYWREN_BUCKET"
    s3_prefix = "sam"
    region = "$(get_aws_region)"
    confirm_changeset = false
    capabilities = "CAPABILITY_NAMED_IAM"
    parameter_overrides = $ENGINE_SAM_OVERRIDES
	EOF

    sam build --config-file $SAM_DIR/engine/samconfig.toml
    sam deploy --config-file $SAM_DIR/engine/samconfig.toml
}


function deploy_engine_stack2() {
    cat > >(cut -c 5-) <<-EOF
    --------------------------
    DEPLOYING SFS ENGINE STACK
    --------------------------
	EOF

    source ~/.bashrc
    cd $SAM_DIR/engine/container

    pyenv global $PY38_VERSION

    cdk bootstrap
    cdk deploy --require-approval never -O /tmp/engine-stack-outputs.json \
	    --context app_id=$(get_amplify_app_id)
}


function deploy_amznfcast_stack() {
    cat > >(cut -c 5-) <<-EOF
    -----------------------------------
    DEPLOYING THE AMAZON FORECAST STACK
    -----------------------------------
	EOF

    export SAM_CLI_TELEMETRY=0

    source ~/.bashrc
    cd $SAM_DIR/amznfcast

    AMZNFCAST_PIPELINE_BUCKET=amznfcast-$(get_amplify_app_id)-pipeline
    AMZNFCAST_STACK=sam-amznfcast-$(get_amplify_app_id)
    AMZNFCAST_SAM_OVERRIDES="[\"AmznFcastPipelineBucketName=$AMZNFCAST_PIPELINE_BUCKET\"]"

    cat > >( cut -c 5- > $SAM_DIR/amznfcast/samconfig.toml ) <<-EOF
    version = 0.1
    [default]
    [default.deploy]
    [default.deploy.parameters]
    stack_name = "$AMZNFCAST_STACK"
    s3_bucket = "$(get_amplify_data_bucket)"
    s3_prefix = "sam"
    region = "$(get_aws_region)"
    confirm_changeset = false
    capabilities = "CAPABILITY_NAMED_IAM"
    parameter_overrides = $AMZNFCAST_SAM_OVERRIDES
	EOF

    pyenv global $PY37_VERSION

    pip install aws-sam-cli

    sam build --config-file $SAM_DIR/amznfcast/samconfig.toml
    sam deploy --force-upload --config-file $SAM_DIR/amznfcast/samconfig.toml
}


function deploy_aux_stack() {
    cat > >(cut -c 5-) <<-EOF
    ------------------------------
    DEPLOYING THE AUXILLIARY STACK
    ------------------------------
	EOF

    set -x
    set -e

    source ~/.bashrc
    export SAM_CLI_TELEMETRY=0
    cd $SAM_DIR/

    AUX_STACK=sam-aux-$(get_amplify_app_id)
    AMZNFCAST_STACK=sam-amznfcast-$(get_amplify_app_id)

    # Upload the lambda layer zip files to the deployment bucket
    if [[ ! -f "$AWSWRANGLER_ZIP" ]]
    then
        wget -O $AWSWRANGLER_ZIP $AWSWRANGLER_URL
    fi

    aws s3 cp $AWSWRANGLER_ZIP s3://$(get_deployment_bucket)/

    AUX_SAM_OVERRIDES="[\
        \"AppId=$(get_amplify_app_id)\",\
        \"DataBucketName=$(get_amplify_data_bucket)\",\
        \"DeploymentBucketName=$(get_deployment_bucket)\"]"

    cat > >( cut -c 5- > $SAM_DIR/samconfig.toml ) <<-EOF
    version = 0.1
    [default]
    [default.deploy]
    [default.deploy.parameters]
    stack_name = "$AUX_STACK"
    s3_bucket = "$(get_amplify_data_bucket)"
    s3_prefix = "sam"
    region = "$(get_aws_region)"
    confirm_changeset = false
    capabilities = "CAPABILITY_IAM"
    parameter_overrides = $AUX_SAM_OVERRIDES
	EOF

    pyenv global $PY38_VERSION

    pip install aws-sam-cli

    sam build --config-file $SAM_DIR/samconfig.toml
    sam deploy --force-upload --config-file $SAM_DIR/samconfig.toml

    # Attach the layer to the validation function
    VALIDATE_LAMBDA=voyagerValidateLambda-${AMPLIFY_APP_ENV}
    LAYER_NAME=AwsDataWranglerLayer-$(get_amplify_app_id)
    LAYER_ARN=$(\
      aws lambda list-layer-versions \
      --layer-name $LAYER_NAME \
      --region $(get_aws_region) \
      --output text \
      --query 'max_by(LayerVersions, &Version).LayerVersionArn')

    aws lambda update-function-configuration \
      --layers '["'$LAYER_ARN'"]' \
      --function-name $VALIDATE_LAMBDA

    FCAST_LAMBDA=voyagerForecastLambda-${AMPLIFY_APP_ENV}
    ENGINE_STACK=sam-engine-$(get_amplify_app_id)
#   ENGINE_LAMBDA=$(aws cloudformation list-stack-resources \
#       --stack-name $ENGINE_STACK \
#       --query "StackResourceSummaries[?LogicalResourceId=='EngineForecastFunction'].PhysicalResourceId" \
#       --output text
#   )
    ENGINE_LAMBDA=$( cat /tmp/engine-stack-outputs.json | jq -r '."SfnStack-'$(get_amplify_app_id)'".SfsStateMachineARN' )

    rm -rf /tmp/tmp_json

    aws lambda update-function-configuration \
      --function-name $FCAST_LAMBDA \
        | jq '(.Environment.Variables
                | .VOYAGER_LAMBDA_FUNCTION_NAME) |= "'$ENGINE_LAMBDA'"' \
        | cat > /tmp/tmp_json

    aws lambda update-function-configuration \
      --function-name $FCAST_LAMBDA \
      --environment "$( cat /tmp/tmp_json | jq '.Environment' )"

    rm -f /tmp/tmp_json

    #
    # CopyFromVoyager Lambda trigger
    #
    AF_LAMBDA_ID=$(\
        aws cloudformation describe-stack-resources \
            --stack-name $AMZNFCAST_STACK \
            --logical-resource-id CopyFromVoyager \
            | jq -r '.StackResources[0].PhysicalResourceId')
    AF_STATEMENT_ID=$AMZNFCAST_STACK-copy-from-voyager
    AF_LAMBDA_ARN=arn:aws:lambda:$(get_aws_region):$(get_aws_account_id):function:$AF_LAMBDA_ID

    #
    # AmazonForecast output resampling Lambda trigger
    #
    RS_STATEMENT_ID=$AUX_STACK-statement-id
    RS_LAMBDA_ID=$(\
        aws cloudformation describe-stack-resources \
            --stack-name $AUX_STACK \
            --logical-resource-id ResampleFunction \
            | jq -r '.StackResources[0].PhysicalResourceId'
    )
    RS_LAMBDA_ARN=arn:aws:lambda:$(get_aws_region):$(get_aws_account_id):function:$RS_LAMBDA_ID

    #
    # Add Lambda permissions
    #
    aws lambda add-permission \
        --function-name $AF_LAMBDA_ID \
        --statement-id $AF_STATEMENT_ID \
        --action "lambda:InvokeFunction" \
        --principal "s3.amazonaws.com" \
        --source-arn "arn:aws:s3:::$(get_amplify_data_bucket)" \
        --source-account $(get_aws_account_id)

    aws lambda add-permission \
        --function-name $RS_LAMBDA_ID \
        --statement-id $RS_STATEMENT_ID \
        --action lambda:InvokeFunction \
        --principal s3.amazonaws.com \
        --source-arn arn:aws:s3:::$(get_amplify_data_bucket) \
        --source-account $(get_aws_account_id)

    #
    # Combine the Lambda trigger configurations, since they both read from the
    # same S3 bucket
    #
    cat > /tmp/lambda-cfg.json <<-EOF
    {
    "LambdaFunctionConfigurations": [
        {
          "Id": "$AF_STATEMENT_ID",
          "LambdaFunctionArn": "$AF_LAMBDA_ARN",
          "Events": ["s3:ObjectCreated:*"],
          "Filter": {
            "Key": {
              "FilterRules": [
                {
                  "Name": "prefix",
                  "Value": "private/"
                },
                {
                  "Name": "suffix",
                  "Value": ".input.csv"
                }
              ]
            }
          }
        },
        {
          "Id": "$RS_STATEMENT_ID",
          "LambdaFunctionArn": "$RS_LAMBDA_ARN",
          "Events": ["s3:ObjectCreated:*"],
          "Filter": {
            "Key": {
              "FilterRules": [
                {
                  "Name": "prefix",
                  "Value": "private/"
                },
                {
                  "Name": "suffix",
                  "Value": ".forecast.csv.AF"
                }
              ]
            }
          }
        }
      ]
    }
	EOF

    aws s3api put-bucket-notification-configuration \
        --bucket $(get_amplify_data_bucket) \
        --region $(get_aws_region) \
        --notification-configuration file:///tmp/lambda-cfg.json

    set +e
    set +x
}


function update_s3_sse() {
    cat > >(cut -c 5-) <<-EOF
    ----------------------------------
    APPLYING S3 SERVER-SIDE ENCRYPTION
    ----------------------------------
	EOF

    # Add S3 SSE to the amplify data bucket
    aws s3api put-bucket-encryption \
        --bucket $(get_amplify_data_bucket) \
        --server-side-encryption-configuration \
            '{"Rules": [{ "ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "aws:kms"}, "BucketKeyEnabled": true }]}'

    aws s3 cp s3://$(get_amplify_data_bucket) s3://$(get_amplify_data_bucket) --sse aws:kms --recursive

    aws s3api put-public-access-block \
        --bucket $(get_amplify_data_bucket) \
        --public-access-block-configuration \
            '{"BlockPublicAcls": true, "IgnorePublicAcls": true, "BlockPublicPolicy": true, "RestrictPublicBuckets": true}'

    # Add S3 SSE to the amplify deployment bucket
    aws s3api put-bucket-encryption \
        --bucket $(get_deployment_bucket) \
        --server-side-encryption-configuration \
            '{"Rules": [{ "ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "aws:kms"}, "BucketKeyEnabled": true }]}'

    aws s3 cp s3://$(get_deployment_bucket) s3://$(get_deployment_bucket) --sse aws:kms --recursive

    aws s3api put-public-access-block \
        --bucket $(get_deployment_bucket) \
        --public-access-block-configuration \
            '{"BlockPublicAcls": true, "IgnorePublicAcls": true, "BlockPublicPolicy": true, "RestrictPublicBuckets": true}'
}


function update_s3_versioning() {
    # Enable S3 bucket versioning
    aws s3api put-bucket-versioning \
        --bucket $(get_deployment_bucket) \
        --versioning-configuration Status=Enabled

    aws s3api put-bucket-versioning \
        --bucket $(get_amplify_data_bucket) \
        --versioning-configuration Status=Enabled
}


function update_dynamodb_sse() {
    aws dynamodb update-table \
        --table-name voyagerCustomers-$AMPLIFY_APP_ENV \
        --sse-specification 'Enabled=true,SSEType=KMS' > /dev/null

    aws dynamodb update-table \
        --table-name voyagerForecastJobs-$AMPLIFY_APP_ENV \
        --sse-specification 'Enabled=true,SSEType=KMS' > /dev/null
}


function update_policies() {
    # Make new policy document
    ROLE_NAME=voyagerfrontend849169bf-ExecutionRole-$AMPLIFY_APP_ENV

    aws iam get-role-policy \
        --role-name $ROLE_NAME \
        --policy-name UserGroupPassRolePolicy \
        | jq '(.PolicyDocument.Statement[0].Resource) |= "arn:aws:iam::'$(get_aws_account_id)':role/sam-amznfcast-'$(get_amplify_app_id)'-*"' \
        | jq '(.PolicyDocument)' > /tmp/UserGroupPassRolePolicy.json

    # Attach the new policy document
    aws iam delete-role-policy \
        --role-name $ROLE_NAME \
        --policy-name UserGroupPassRolePolicy

    aws iam put-role-policy \
        --role-name $ROLE_NAME \
        --policy-name UserGroupPassRolePolicy \
        --policy-document file:///tmp/UserGroupPassRolePolicy.json
}


function update_api_gw_throttling() {
    cat > >(cut -c 5-) <<-EOF
    -------------------------------
    UPDATING API GATEWAY THROTTLING
    -------------------------------
	EOF

    # References:
    # - https://docs.aws.amazon.com/apigateway/latest/developerguide/stages.html

    rate_limit=25
    burst_limit=50
    stage_name=$AMPLIFY_APP_ENV

    names=$(aws apigateway get-rest-apis | jq -r '.items[].name')

    for name in ${names[@]}
    do
        rest_api_id=$(aws apigateway get-rest-apis | \
            jq -r '.items[] | select(.name=="'$name'") .id')

        aws apigateway update-stage \
            --rest-api-id $rest_api_id \
            --stage-name $stage_name \
            --patch-operations \
                'op=replace,path=/*/*/throttling/rateLimit,value='$rate_limit''

        aws apigateway update-stage \
            --rest-api-id $rest_api_id \
            --stage-name $stage_name \
            --patch-operations \
                'op=replace,path=/*/*/throttling/burstLimit,value='$burst_limit''
    done
}


function update_s3_logging() {
    cat > >(cut -c 5-) <<-EOF
    -------------------
    UPDATING S3 LOGGING
    -------------------
	EOF

    aws s3api put-bucket-acl \
        --bucket $(get_deployment_bucket) \
        --grant-write URI=http://acs.amazonaws.com/groups/s3/LogDelivery \
        --grant-read-acp URI=http://acs.amazonaws.com/groups/s3/LogDelivery

    aws s3api put-bucket-acl \
        --bucket $(get_amplify_data_bucket) \
        --grant-write URI=http://acs.amazonaws.com/groups/s3/LogDelivery \
        --grant-read-acp URI=http://acs.amazonaws.com/groups/s3/LogDelivery

    aws s3api put-bucket-logging \
        --bucket $(get_deployment_bucket) \
        --bucket-logging-status '{"LoggingEnabled":{"TargetBucket":"'$(get_amplify_data_bucket)'","TargetPrefix":""}}'

    aws s3api put-bucket-logging \
        --bucket $(get_deployment_bucket) \
        --bucket-logging-status '{"LoggingEnabled":{"TargetBucket":"'$(get_deployment_bucket)'","TargetPrefix":""}}'
}


function deploy_all() {
    install_prereqs
    deploy_amplify_backend

    [[ $ENABLE_FRONTEND == 1 ]] && deploy_amplify_frontend

    deploy_engine_stack2
    deploy_amznfcast_stack
    deploy_aux_stack

    [[ $ENABLE_SSE == 1 ]] && update_s3_sse

    update_s3_versioning
    update_s3_logging

    update_api_gw_throttling
    update_dynamodb_sse
    update_policies
}

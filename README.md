# Amazon Simple Forecast Solution

![](https://img.shields.io/badge/license-MIT--0-green)
![](https://img.shields.io/github/workflow/status/aws-samples/simple-forecast-solution/pytest/main)


## One-Click Deployment

Region name | Region code | Launch
--- | --- | ---
US East (N. Virginia) | us-east-1 | [![Launch Stack](https://cdn.rawgit.com/buildkite/cloudformation-launch-stack-button-svg/master/launch-stack.svg)]()
US West (Oregon) | us-west-2 | [![Launch Stack](https://cdn.rawgit.com/buildkite/cloudformation-launch-stack-button-svg/master/launch-stack.svg)]()
Europe (Ireland) | eu-west-1 | [![Launch Stack](https://cdn.rawgit.com/buildkite/cloudformation-launch-stack-button-svg/master/launch-stack.svg)]()


# :construction: Currently Refactoring :construction:

Hi there! refactoring of SFS is nearly complete, please watch this space!

## Installation

### _Prerequisite_ – Install `npm` and `aws-cdk`

```bash
# Install npm
curl -L https://git.io/n-install | bash
source ~/.bashrc

# Install aws-cdk
npm i -g aws-cdk@1.114.0
```

### _Prerequisite_ – Install `lambdamap`

```bash
# Clone the lambdamap repository
git clone https://github.com/aws-samples/lambdamap.git
cd ./lambdamap/
pip install -q -e .

cd ./lambdamap_cdk/
pip install -q -r ./requirements.txt

# Deploy the lambdamap stack
cdk deploy \
    --context function_name=SfsLambdaMapFunction \
    --context extra_cmds='pip install -q git+https://github.com/aws-samples/simple-forecast-solution.git#egg=sfs'
```

### Install SFS

```bash
# Clone the SFS git repository
git clone https://github.com/aws-samples/simple-forecast-solution.git

# Install the SFS library
cd simple-forecast-solution
pip3 install -e .
```

## Run the app (locally)

The app will be accessible from your web browser at `http://localhost:8051`
```bash
cd ./sfs/app/
streamlit run ./app.py
```

## Run the app (via SageMaker Notebook instance)
```bash
# TODO
```

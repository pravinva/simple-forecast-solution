# Amazon Simple Forecast Solution

![](https://img.shields.io/badge/license-MIT--0-green)
![](https://img.shields.io/github/workflow/status/aws-samples/simple-forecast-solution/pytest/main)

## Installation

### _Prerequisite_ – Install `npm` and `aws-cdk`

```bash
# Install npm
curl -L https://git.io/n-install | bash
source ~/.bashrc

# Install aws-cdk
npm i -g aws-cdk
```

### _Prerequisite_ – Install `lambdamap`

```bash
# Install the lambdamap Python library
cd ./sfs/lambdamap
pip3 install -e .

# Deploy the lambdamap cloudformation stack
cd lambdamap/cdk
cdk deploy -c folder=../../container
```

### Install SFS

```bash
# Clone the SFS git repository
git clone https://github.com/aws-samples/simple-forecast-solution.git

# Install the SFS library
cd simple-forecast-solution
pip3 install -e .
```

## Run the app

The app will be accessible from your web browser at `http://localhost:8000`
```bash
cd ./sfs/app/
streamlit run ./app.py
```

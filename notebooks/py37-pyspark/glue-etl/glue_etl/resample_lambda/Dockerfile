FROM public.ecr.aws/lambda/python:3.7

RUN yum update -y \
    && yum install -y shadow-utils.x86_64 zip unzip \
        gcc-c++ make openssl-devel zlib-devel readline-devel git \ 
        blas blas-devel lapack lapack-devel

RUN python3.7 -m pip install --upgrade pip
RUN pip install awslambdaric

COPY requirements.txt ${LAMBDA_TASK_ROOT}/

RUN pip install -r ${LAMBDA_TASK_ROOT}/requirements.txt

COPY lambda.py ${LAMBDA_TASK_ROOT}/

CMD ["lambda.handler"]

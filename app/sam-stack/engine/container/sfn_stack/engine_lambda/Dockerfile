FROM public.ecr.aws/lambda/python:3.7

RUN yum update -y \
    && yum install -y shadow-utils.x86_64 zip unzip \
        gcc-c++ make openssl-devel zlib-devel readline-devel git \ 
        blas blas-devel lapack lapack-devel

COPY lambda.py ${LAMBDA_TASK_ROOT}/
COPY voyager/* ${LAMBDA_TASK_ROOT}/

RUN python3.7 -m pip install --upgrade pip
RUN pip install awslambdaric
RUN pip install -r ${LAMBDA_TASK_ROOT}/requirements.txt

CMD ["lambda.handler"]

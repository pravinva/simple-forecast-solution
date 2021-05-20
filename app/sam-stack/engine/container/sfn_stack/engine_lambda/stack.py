import os

from aws_cdk import core as cdk
from aws_cdk import core, aws_lambda, aws_ecr

PWD = os.path.dirname(os.path.realpath(__file__))


class EngineMapStack(cdk.Stack):
    def __init__(self, scope: cdk.Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        ecr_image = aws_lambda.EcrImageCode.from_asset_image(directory=PWD)

        memory_size = kwargs.get("memory_size", 4096)
        timeout_mins = kwargs.get("timeout_mins", 15)

        lambda_function = aws_lambda.Function(self, 
            id=f"{construct_id}-EngineMapFunction",
            description="EngineMapFunction",
            code=ecr_image,
            handler= aws_lambda.Handler.FROM_IMAGE,
            runtime= aws_lambda.Runtime.FROM_IMAGE,
            function_name = f"{construct_id}-EngineMapFunction",
            memory_size=memory_size,
            timeout=core.Duration.minutes(timeout_mins),
        )

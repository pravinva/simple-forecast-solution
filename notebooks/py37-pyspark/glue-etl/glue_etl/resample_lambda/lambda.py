import boto3

from awswrangler._utils import pd


def handler(event, context):
    """
    """

    parquet_paths = f"{event['arguments']['--s3_input_path']}.paths.csv"
    freq = event["arguments"]["--freq"]
    horiz = event["arguments"]["--horiz"]
    glue_job_name = event["glue_job_name"]

    df_parquet_paths = pd.read_csv(parquet_paths)

    glue = boto3.client(service_name="glue",
            region_name="ap-southeast-2",
            endpoint_url='https://glue.ap-southeast-2.amazonaws.com')

    responses = []

    for s3_parquet_path in df_parquet_paths["s3_output_path"]:
        resp = glue.start_job_run(
            JobName=glue_job_name,
            Arguments={
                "--s3_input_path": s3_parquet_path,
                "--freq": str(freq),
                "--horiz": str(horiz)
            }
        )
        responses.append(resp)

    print(responses)
    
    return

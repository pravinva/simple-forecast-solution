import time
import boto3

from awswrangler._utils import pd


def handler(event, context):
    """
    """

    s3_input_path = event['arguments']['--s3_input_path']
    parquet_paths = f"{event['arguments']['--s3_input_path']}.paths.csv"
    freq = event["arguments"]["--freq"]
    horiz = event["arguments"]["--horiz"]
    glue_job_name = event["glue_job_name"]

    df_parquet_paths = pd.read_csv(parquet_paths)

    glue = boto3.client(service_name="glue",
            region_name="ap-southeast-2",
            endpoint_url='https://glue.ap-southeast-2.amazonaws.com')

    responses = []

    for s3_parquet_path in df_parquet_paths["s3_parquet_path"]:
        resp = glue.start_job_run(
            JobName=glue_job_name,
            Arguments={
                "--s3_input_path": s3_parquet_path,
                "--freq": str(freq),
                "--horiz": str(horiz)
            }
        )
        responses.append(resp)

    # wait until all resampling jobs are complete
    while len(responses) > 0:
        for i, run in enumerate(responses):
            run_id = run["JobRunId"]
            resp = glue.get_job_run(JobName=glue_job_name, RunId=run_id)
            job_state = resp["JobRun"]["JobRunState"]

            assert(job_state != "STOPPED")
            
            if job_state == "SUCCEEDED":
                # generate forecast
                print(run_id, "COMPLETED...")
                responses.pop(i)

        time.sleep(1)

    return {"s3_input_path": s3_input_path,
            "s3_parquet_paths": parquet_paths,
            "freq": str(freq),
            "horiz": str(horiz)}

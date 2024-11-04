from pulumi_aws import s3


def create_buckets(project: str, stack: str):
    raw_bucket = s3.Bucket("raw-bucket", bucket=f"{project}-raw", force_destroy=True)

    processed_bucket = s3.Bucket(
        "processed-data-bucket",
        bucket=f"{project}-processed",
        force_destroy=True,
    )

    return raw_bucket, processed_bucket

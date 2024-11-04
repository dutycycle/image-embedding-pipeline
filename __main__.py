import pulumi
from resources.s3 import create_buckets
from resources.iam import create_lambda_role
from resources.lambdas import create_lambda
from resources.athena import create_analytics
from resources.dynamodb import create_lock_table

config = pulumi.Config()
stack = pulumi.get_stack()
project = pulumi.get_project()

raw_bucket, processed_bucket = create_buckets(project, stack)
lock_table = create_lock_table(project)
lambda_role = create_lambda_role(raw_bucket, processed_bucket, lock_table)
process_images_lambda = create_lambda(
    lambda_role, raw_bucket, processed_bucket, lock_table
)
athena_workgroup = create_analytics(project, processed_bucket)

pulumi.export("raw_bucket", raw_bucket.id)
pulumi.export("processed_bucket", processed_bucket.id)
pulumi.export("process_images_lambda_name", process_images_lambda.name)
pulumi.export("athena_workgroup_name", athena_workgroup.name)
pulumi.export("lock_table_name", lock_table.name)

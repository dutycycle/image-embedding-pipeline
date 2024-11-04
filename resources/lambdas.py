import pulumi
import pulumi_aws as aws
from pulumi_aws import lambda_, ecr
import pulumi_docker_build as docker_build


def create_lambda(lambda_role, raw_bucket, processed_bucket, lock_table):
    # Create ECR repositories for each Lambda
    images_repository = ecr.Repository(
        "images-repository",
        force_delete=True,
        image_scanning_configuration={"scanOnPush": True},
    )

    # Get ECR authorization token
    auth_token = aws.ecr.get_authorization_token_output(
        registry_id=images_repository.registry_id
    )

    # Build and push Docker images using pulumi_docker_build
    images_image = docker_build.Image(
        "process-images-image",
        dockerfile={
            "location": "./lambdas/process_image/Dockerfile",
        },
        context={
            "location": "./lambdas/process_image",
        },
        platforms=[docker_build.Platform.LINUX_AMD64],
        cache_from=[
            {
                "registry": {
                    "ref": images_repository.repository_url.apply(
                        lambda url: f"{url}:cache"
                    ),
                },
            }
        ],
        cache_to=[
            {
                "registry": {
                    "image_manifest": True,
                    "oci_media_types": True,
                    "ref": images_repository.repository_url.apply(
                        lambda url: f"{url}:cache"
                    ),
                },
            }
        ],
        push=True,
        registries=[
            {
                "address": images_repository.repository_url,
                "password": auth_token.password,
                "username": auth_token.user_name,
            }
        ],
        tags=[images_repository.repository_url.apply(lambda url: f"{url}:latest")],
    )

    # Create Lambda functions using the Docker images
    process_images_lambda = lambda_.Function(
        "process-images",
        package_type="Image",
        image_uri=pulumi.Output.concat(images_repository.repository_url, ":latest"),
        role=lambda_role.arn,
        architectures=["x86_64"],
        memory_size=2048,
        timeout=300,
        ephemeral_storage={"size": 4096},
        environment={
            "variables": {
                "PROCESSED_BUCKET": processed_bucket.id,
                "OUTPUT_PREFIX": "embeddings",
                "LOCK_TABLE": lock_table.name,
            }
        },
    )

    # add lambda permissions for s3 invocations
    images_lambda_permission = lambda_.Permission(
        "images-lambda-permission",
        action="lambda:InvokeFunction",
        function=process_images_lambda.name,
        principal="s3.amazonaws.com",
        source_arn=raw_bucket.arn,
    )

    s3_image_notification = aws.s3.BucketNotification(
        "image-notification",
        bucket=raw_bucket.id,
        lambda_functions=[
            {
                "lambda_function_arn": process_images_lambda.arn,
                "events": ["s3:ObjectCreated:*"],
                "filter_prefix": "images/",
            }
        ],
    )

    return process_images_lambda

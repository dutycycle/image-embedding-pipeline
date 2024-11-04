import pulumi
import pulumi_aws as aws
import json


def create_lambda_role(raw_bucket, processed_bucket, lock_table):
    # Get current AWS account ID
    current = aws.get_caller_identity()
    account_id = current.account_id

    lambda_role = aws.iam.Role(
        "lambda-role",
        assume_role_policy=json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Action": "sts:AssumeRole",
                        "Principal": {
                            "Service": ["lambda.amazonaws.com", "s3.amazonaws.com"]
                        },
                        "Effect": "Allow",
                    }
                ],
            }
        ),
    )

    lambda_policy = aws.iam.RolePolicy(
        "lambda-policy",
        role=lambda_role.id,
        policy=pulumi.Output.all(
            raw_bucket.arn, processed_bucket.arn, account_id, lock_table.arn
        ).apply(
            lambda args: json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Action": [
                                "s3:GetObject",
                                "s3:PutObject",
                                "s3:ListBucket",
                                "logs:CreateLogGroup",
                                "logs:CreateLogStream",
                                "logs:PutLogEvents",
                            ],
                            "Resource": [
                                args[0],
                                args[1],
                                f"{args[0]}/*",
                                f"{args[1]}/*",
                                "arn:aws:logs:*:*:*",
                            ],
                        },
                        {
                            "Effect": "Allow",
                            "Action": [
                                "ecr:GetDownloadUrlForLayer",
                                "ecr:BatchGetImage",
                                "ecr:BatchCheckLayerAvailability",
                            ],
                            "Resource": [f"arn:aws:ecr:*:{args[2]}:repository/*"],
                        },
                        {
                            "Effect": "Allow",
                            "Action": ["ecr:GetAuthorizationToken"],
                            "Resource": "*",  # this permission can't be scoped to a specific resource
                        },
                        {
                            "Effect": "Allow",
                            "Action": ["lambda:InvokeFunction"],
                            "Resource": f"arn:aws:lambda:*:{args[2]}:function:*",
                        },
                        {
                            "Effect": "Allow",
                            "Action": [
                                "s3:PutBucketNotification",
                                "s3:GetBucketNotification",
                            ],
                            "Resource": [
                                args[0],
                                args[1],
                            ],
                        },
                        {
                            "Effect": "Allow",
                            "Action": ["dynamodb:PutItem", "dynamodb:DeleteItem"],
                            "Resource": args[3],
                        },
                    ],
                }
            )
        ),
    )

    return lambda_role

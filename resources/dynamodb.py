from pulumi_aws import dynamodb


def create_lock_table(project: str):
    lock_table = dynamodb.Table(
        "parquet-file-locks",
        name=f"{project}-ParquetFileLocks",
        attributes=[dynamodb.TableAttributeArgs(name="lock_key", type="S")],
        hash_key="lock_key",
        billing_mode="PAY_PER_REQUEST",
        ttl=dynamodb.TableTtlArgs(attribute_name="expires_at", enabled=True),
    )

    return lock_table

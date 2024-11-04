from pulumi_aws import glue, athena


def create_analytics(project: str, processed_bucket):
    glue_database = glue.CatalogDatabase("analytics-database", name="analytics")

    def create_storage_descriptor(bucket_id: str, folder: str):
        return {
            "location": f"s3://{bucket_id}/{folder}",
            "input_format": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
            "output_format": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat",
            "ser_de_info": {
                "serialization_library": "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
            },
        }

    embeddings_table = glue.CatalogTable(
        "embeddings-table",
        name="embeddings",
        database_name=glue_database.name,
        table_type="EXTERNAL_TABLE",
        parameters={"classification": "parquet", "parquet.compression": "SNAPPY"},
        storage_descriptor=processed_bucket.id.apply(
            lambda bucket_id: {
                **create_storage_descriptor(bucket_id, "embeddings"),
                "columns": [
                    {"name": "image_path", "type": "string"},
                    {"name": "experiment_id", "type": "string"},
                    {"name": "embedding", "type": "array<double>"},
                    {"name": "processed_at", "type": "timestamp"},
                ],
            }
        ),
    )

    workgroup = athena.Workgroup(
        "analytics-workgroup",
        name=f"{project}-analytics",
        force_destroy=True,
        configuration={
            "enforce_workgroup_configuration": True,
            "result_configuration": {
                "output_location": processed_bucket.id.apply(
                    lambda bucket_id: f"s3://{bucket_id}/athena-results/"
                ),
                "encryption_configuration": {"encryption_option": "SSE_S3"},
            },
        },
    )

    return workgroup

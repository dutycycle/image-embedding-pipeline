# Serverless Image Embedding Pipeline

Proof of concept of generating image embeddings from uploads to object storage, using AWS serverless infrastructure. The entire infrastructure is defined in code using [Pulumi](https://www.pulumi.com/).

## Data Flow

Clients uploads one or more images to **s3://raw-bucket/images/[experiment_id]/**. The images can be in any format supported by [Pillow](https://pillow-wiredfool.readthedocs.io/en/latest/handbook/image-file-formats.html).

The raw bucket has a notification trigger which invokes the **process-image** Lambda function. The function uses [CLIP](https://github.com/openai/CLIP) to generate a 512 dimensional embedding of the image. CLIP was chosen because it's a popular and versatile embedding function, but could be easily replaced.

**process-image** will create or append to a Parquet file **s3://processed-bucket/embeddings/[experiment_id]/image_embeddings.parquet** with the following schema:

```
{"name": "image_path", "type": "string"},
{"name": "experiment_id", "type": "string"},
{"name": "embedding", "type": "array<double>"},
{"name": "processed_at", "type": "timestamp"},
```

A Dynamo DB table **ParquetFileLocks** is used to lock the Parquet file to in the case where are multiple images uploaded for the same experiment; the Lambda will use exponential backoff and wait for the lock to be released before appending.

A Glue Catalog Table and Athena Workgroup allow the user to query the **embeddings** table using Athena, backed by the Parquet files.

## Deployment

Deployment, including Docker image build, is handled entirely by Pulumi. After installing Pulumi and configuring a backend such as an S3 bucket, simply run `pulumi up` within a virtual environment.

## Tests

There are a few unit tests that can be run inside the Docker container. AWS resources are mocked so no credentials are needed. To run tests interactively:

```
docker build -t lambda-image .
docker run --entrypoint python3 lambda-image -m pytest /var/task/tests.py
````

By default the tests will mock CLIP for speed. If you want to test with CLIP actually generating embeddings you can use:

```
docker run --entrypoint python3 lambda-image -m pytest /var/task/tests.py --use-real-clip
```

Note that this slows down tests significantly.

## Design Notes and Future Improvements

* Lambda was chosen because of its integration with S3 notifications, which eliminates the need for a standalone message queue. Docker Lambda was chosen over Python Lambda since Python Lambdas have a maximum size of 250MB, which is too small to accomodate PyTorch and CLIP. Also, Docker dependencies are easier to manage.

* The process-images lambda was written with container reuse in mind, so that the CLIP model doesn't need to be redownloaded on each invocation. Cold start is slow at ~40s but subsequent invocations are <300ms. To further increase performance and reduce resource requirements a ONNX-based library such as [imgbeddings](https://github.com/minimaxir/imgbeddings) in place of PyTorch.

* Writing to a single Parquet file vs a separate file per image was done to reduce query time, since reading many small files is inefficient for Athena. with the tradeoff of requiring locking. If it's expected that there will be heavy concurrency per experiment, it would be better to write individual Parquet files and then periodically compact through a separate process.


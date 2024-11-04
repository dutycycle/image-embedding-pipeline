import datetime
import io
import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import boto3
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from botocore.exceptions import ClientError
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Configuration
@dataclass
class Config:
    LOCK_TTL_SECONDS: int = 300
    LOCK_MAX_RETRY: int = 5
    CACHE_DIR: Path = Path("/tmp")
    MODEL_NAME: str = "openai/clip-vit-base-patch32"
    DEVICE: str = "cpu"

    def __post_init__(self):
        # Set up cache directories
        os.environ.update({
            "TRANSFORMERS_CACHE": str(self.CACHE_DIR / "transformers_cache"),
            "TORCH_HOME": str(self.CACHE_DIR / "torch_home"),
            "HF_HOME": str(self.CACHE_DIR / "huggingface")
        })


class ModelManager:
    def __init__(self, config: Config):
        self.config = config
        self._model: Optional[CLIPModel] = None
        self._processor: Optional[CLIPProcessor] = None

    def _initialize_model(self) -> Tuple[CLIPModel, CLIPProcessor]:
        """Initialize the CLIP model and processor if not already initialized."""
        if self._model is None or self._processor is None:
            self._model = CLIPModel.from_pretrained(self.config.MODEL_NAME)
            self._processor = CLIPProcessor.from_pretrained(self.config.MODEL_NAME)
            self._model.to(self.config.DEVICE)
            self._model.eval()
        return self._model, self._processor

    def create_embedding(self, image_bytes: bytes) -> np.ndarray:
        """Create embedding from image bytes using CLIP model."""
        model, processor = self._initialize_model()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.config.DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            embedding = image_features / image_features.norm(dim=-1, keepdim=True)
            return embedding.cpu().numpy().flatten()


class LockManager:
    def __init__(self, config: Config, dynamodb_resource):
        self.config = config
        self.dynamodb = dynamodb_resource
        self.lock_owner = f"lambda-{datetime.datetime.now(datetime.UTC).isoformat()}-{uuid.uuid4()}"

    def acquire_lock(self, lock_table: str, lock_key: str) -> bool:
        """Attempt to acquire a lock in DynamoDB."""
        table = self.dynamodb.Table(lock_table)
        timestamp = int(time.time())
        expires_at = timestamp + self.config.LOCK_TTL_SECONDS

        try:
            table.put_item(
                Item={
                    "lock_key": lock_key,
                    "lock_owner": self.lock_owner,
                    "timestamp": timestamp,
                    "expires_at": expires_at,
                },
                ConditionExpression="attribute_not_exists(lock_key) OR expires_at < :now",
                ExpressionAttributeValues={":now": timestamp},
            )
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                return False
            raise

    def release_lock(self, lock_table: str, lock_key: str):
        """Release a previously acquired lock."""
        table = self.dynamodb.Table(lock_table)
        try:
            table.delete_item(
                Key={"lock_key": lock_key},
                ConditionExpression="lock_owner = :owner",
                ExpressionAttributeValues={":owner": self.lock_owner},
            )
        except ClientError:
            pass  # Lock might have expired or been taken by another process


class ParquetManager:
    def __init__(self, s3_client, config: Config):
        self.s3 = s3_client
        self.config = config

    def read_existing_parquet(self, bucket: str, key: str) -> Optional[pa.Table]:
        """Read existing parquet file from S3."""
        try:
            response = self.s3.get_object(Bucket=bucket, Key=key)
            with tempfile.NamedTemporaryFile() as tmp:
                tmp.write(response["Body"].read())
                tmp.flush()
                return pq.read_table(tmp.name)
        except self.s3.exceptions.NoSuchKey:
            return None

    def write_parquet_table(self, table: pa.Table, bucket: str, key: str):
        """Write parquet table to S3."""
        with tempfile.NamedTemporaryFile() as tmp:
            pq.write_table(table, tmp.name)
            tmp.flush()
            with open(tmp.name, "rb") as f:
                self.s3.put_object(Bucket=bucket, Key=key, Body=f.read())

    def update_with_lock(self, bucket: str, key: str, new_table: pa.Table, lock_manager: LockManager):
        """Update parquet file with locking mechanism."""
        lock_table = os.environ["LOCK_TABLE"]
        lock_key = f"{bucket}/{key}"

        for attempt in range(self.config.LOCK_MAX_RETRY):
            if not lock_manager.acquire_lock(lock_table, lock_key):
                time.sleep(2**attempt)  # exponential backoff
                continue

            try:
                existing_table = self.read_existing_parquet(bucket, key)
                combined_table = (
                    pa.concat_tables([existing_table, new_table])
                    if existing_table is not None
                    else new_table
                )
                self.write_parquet_table(combined_table, bucket, key)
                return True
            finally:
                lock_manager.release_lock(lock_table, lock_key)

        raise LockAcquisitionError(
            f"Failed to acquire lock after maximum retries ({self.config.LOCK_MAX_RETRY})"
        )


class ImageProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.s3 = boto3.client("s3")
        self.dynamodb = boto3.resource("dynamodb")
        self.model_manager = ModelManager(config)
        self.lock_manager = LockManager(config, self.dynamodb)
        self.parquet_manager = ParquetManager(self.s3, config)

    def process_image(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Process an image from an S3 event and generate embeddings."""
        # Extract event information
        record = event["Records"][0]["s3"]
        bucket = record["bucket"]["name"]
        key = record["object"]["key"]
        experiment_id = key.split("/")[1]

        # Process image
        response = self.s3.get_object(Bucket=bucket, Key=key)
        image_bytes = response["Body"].read()
        embedding = self.model_manager.create_embedding(image_bytes)

        # Prepare data
        new_data = {
            "image_path": [key],
            "experiment_id": [experiment_id],
            "embedding": [embedding.tolist()],
            "processed_at": [datetime.datetime.now(datetime.UTC)],
        }
        new_table = pa.Table.from_pydict(new_data)

        # Set up output location
        processed_bucket = os.environ["PROCESSED_BUCKET"]
        output_prefix = os.environ["OUTPUT_PREFIX"]
        output_key = f"{output_prefix}/experiment_id={experiment_id}/image_embeddings.parquet"

        # Update parquet file
        self.parquet_manager.update_with_lock(
            processed_bucket, output_key, new_table, self.lock_manager
        )

        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Image processed successfully",
                "output_location": f"s3://{processed_bucket}/{output_key}",
            }),
        }


class LockAcquisitionError(Exception):
    """Raised when unable to acquire a lock after maximum retries"""
    pass


# Initialize global configuration and processor
# This will reuse the model and processor across invocations, reducing startup time
config = Config()
image_processor = ImageProcessor(config)

def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Lambda handler function"""
    try:
        return image_processor.process_image(event)
    except LockAcquisitionError as e:
        print(f"Failed to acquire lock: {str(e)}")
        return {
            "statusCode": 429,
            "body": json.dumps("Too many concurrent updates, please retry"),
        }
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps(f"Error processing image: {str(e)}"),
        }
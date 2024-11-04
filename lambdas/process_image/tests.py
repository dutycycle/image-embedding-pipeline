import io
import json
import os
import uuid
from unittest.mock import Mock, patch

import boto3
import numpy as np
import pyarrow.parquet as pq
import pytest
from moto import mock_aws
from PIL import Image


# Configuration
class Config:
    SOURCE_BUCKET = "image-embedding-pipeline"
    PROCESSED_BUCKET = "image-embedding-pipeline"
    EXPERIMENT_ID = "0"
    LOCK_TABLE = "embedding-locks"
    AWS_REGION = "us-east-1"
    EMBEDDING_DIM = 512


class TestUtils:
    @staticmethod
    def generate_random_image(width=224, height=224):
        """Generate a random RGB test image"""
        random_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        image = Image.fromarray(random_array)
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="JPEG")
        img_byte_arr.seek(0)
        return img_byte_arr.getvalue()

    @staticmethod
    def create_s3_event(bucket: str, key: str) -> dict:
        """Create a mock S3 event"""
        return {
            "Records": [{
                "s3": {
                    "bucket": {"name": bucket},
                    "object": {"key": key}
                }
            }]
        }

    @staticmethod
    def get_parquet_data(s3_client, bucket: str, key: str) -> dict:
        """Retrieve and parse parquet data from S3"""
        response = s3_client.get_object(Bucket=bucket, Key=key)
        with io.BytesIO(response["Body"].read()) as bio:
            table = pq.read_table(bio)
            return table.to_pandas()


class AWSTestInfra:
    """Handles AWS test infrastructure setup"""
    
    @staticmethod
    def setup_s3(s3_client):
        """Create required S3 buckets"""
        s3_client.create_bucket(Bucket=Config.SOURCE_BUCKET)
        s3_client.create_bucket(Bucket=Config.PROCESSED_BUCKET)

    @staticmethod
    def setup_dynamodb(dynamodb):
        """Create required DynamoDB tables"""
        dynamodb.create_table(
            TableName=Config.LOCK_TABLE,
            KeySchema=[{"AttributeName": "lock_key", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "lock_key", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST"
        )

    @staticmethod
    def setup_environment():
        """Set required environment variables"""
        os.environ.update({
            "AWS_ACCESS_KEY_ID": "testing",
            "AWS_SECRET_ACCESS_KEY": "testing",
            "AWS_DEFAULT_REGION": Config.AWS_REGION,
            "LOCK_TABLE": Config.LOCK_TABLE,
            "PROCESSED_BUCKET": Config.PROCESSED_BUCKET,
            "OUTPUT_PREFIX": "embeddings"
        })


# Fixtures
@pytest.fixture(scope="session")
def use_real_clip(request):
    """Determine whether to use real CLIP model"""
    return request.config.getoption("--use-real-clip")


@pytest.fixture(autouse=True)
def aws_environment():
    """Set up AWS test environment"""
    AWSTestInfra.setup_environment()
    with mock_aws():
        boto3.setup_default_session(region_name=Config.AWS_REGION)
        s3_client = boto3.client("s3", region_name=Config.AWS_REGION)
        dynamodb = boto3.resource("dynamodb", region_name=Config.AWS_REGION)
        
        AWSTestInfra.setup_s3(s3_client)
        AWSTestInfra.setup_dynamodb(dynamodb)
        
        yield


@pytest.fixture
def mock_clip(monkeypatch, use_real_clip):
    """Mock CLIP model unless using real implementation"""
    if not use_real_clip:
        from process_image import ModelManager
        
        # Create a mock ModelManager
        mock_manager = Mock(spec=ModelManager)
        mock_manager.create_embedding.return_value = np.random.rand(Config.EMBEDDING_DIM)
        
        # Patch the ModelManager class to return our mock
        def mock_init(self, config):
            self.config = config
            self.create_embedding = mock_manager.create_embedding
        
        monkeypatch.setattr(ModelManager, "__init__", mock_init)
        monkeypatch.setattr(ModelManager, "create_embedding", mock_manager.create_embedding)


# Tests
class TestImageEmbeddingPipeline:
    def test_successful_processing(self, mock_clip, use_real_clip):
        """Test successful image processing through the pipeline"""
        from process_image import handler
        
        # Setup
        s3_client = boto3.client("s3", region_name=Config.AWS_REGION)
        image_bytes = TestUtils.generate_random_image()
        image_name = f"images/{Config.EXPERIMENT_ID}/{uuid.uuid4()}.jpg"
        
        # Upload test image
        s3_client.put_object(
            Bucket=Config.SOURCE_BUCKET,
            Key=image_name,
            Body=image_bytes
        )
        
        # Process image
        event = TestUtils.create_s3_event(Config.SOURCE_BUCKET, image_name)
        response = handler(event, None)
        
        # Verify response
        assert response["statusCode"] == 200
        response_body = json.loads(response["body"])
        assert response_body["message"] == "Image processed successfully"
        
        # Verify output
        output_key = f"embeddings/experiment_id={Config.EXPERIMENT_ID}/image_embeddings.parquet"
        df = TestUtils.get_parquet_data(s3_client, Config.PROCESSED_BUCKET, output_key)
        
        assert len(df) > 0
        assert image_name in df["image_path"].values
        assert Config.EXPERIMENT_ID in df["experiment_id"].values
        
        # Verify embedding
        embedding = df["embedding"].iloc[0]
        self._verify_embedding(embedding, use_real_clip)

    def test_concurrent_updates(self, mock_clip, monkeypatch):
        """Test concurrent updates to the same parquet file"""
        from process_image import handler
        
        # Setup
        s3_client = boto3.client("s3", region_name=Config.AWS_REGION)
        current_time = [1000]
        monkeypatch.setattr("time.time", lambda: current_time.pop(0))
        
        # Create test images
        image_names = []
        for _ in range(2):
            image_bytes = TestUtils.generate_random_image()
            image_name = f"images/{Config.EXPERIMENT_ID}/{uuid.uuid4()}.jpg"
            s3_client.put_object(
                Bucket=Config.SOURCE_BUCKET,
                Key=image_name,
                Body=image_bytes
            )
            image_names.append(image_name)
            current_time.append(current_time[-1] + 1)
        
        # Process images
        responses = [
            handler(TestUtils.create_s3_event(Config.SOURCE_BUCKET, name), None)
            for name in image_names
        ]
        
        # Verify responses
        assert all(r["statusCode"] == 200 for r in responses)
        
        # Verify output
        output_key = f"embeddings/experiment_id={Config.EXPERIMENT_ID}/image_embeddings.parquet"
        df = TestUtils.get_parquet_data(s3_client, Config.PROCESSED_BUCKET, output_key)
        
        assert len(df) == 2
        assert all(name in df["image_path"].values for name in image_names)
        assert all(len(emb) == Config.EMBEDDING_DIM for emb in df["embedding"])

    def test_invalid_image(self, mock_clip):
        """Test handling of invalid image data"""
        from process_image import ModelManager, handler
        
        # Setup
        s3_client = boto3.client("s3", region_name=Config.AWS_REGION)
        image_name = f"images/{Config.EXPERIMENT_ID}/{uuid.uuid4()}.jpg"
        
        # Mock the ModelManager to raise an error
        with patch.object(ModelManager, 'create_embedding', side_effect=ValueError("Invalid image data")):
            # Upload invalid image
            s3_client.put_object(
                Bucket=Config.SOURCE_BUCKET,
                Key=image_name,
                Body=b"not an image"
            )
            
            # Process image
            event = TestUtils.create_s3_event(Config.SOURCE_BUCKET, image_name)
            response = handler(event, None)
            
            # Verify error handling
            assert response["statusCode"] == 500
            assert "Error processing image" in response["body"]

    def _verify_embedding(self, embedding: np.ndarray, use_real_clip: bool):
        """Verify embedding properties based on CLIP implementation"""
        if use_real_clip:
            assert isinstance(embedding, np.ndarray)
            assert embedding.dtype in (np.float32, np.float64)
            assert embedding.shape == (Config.EMBEDDING_DIM,)
            assert not np.any(np.isnan(embedding))
            assert not np.any(np.isinf(embedding))
            
            # Verify L2 normalization
            norm = np.linalg.norm(embedding)
            assert abs(norm - 1.0) < 1e-6, f"Embedding should be normalized, got norm {norm}"
        else:
            assert len(embedding) == Config.EMBEDDING_DIM
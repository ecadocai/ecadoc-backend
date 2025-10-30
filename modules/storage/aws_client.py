"""
AWS S3 client for file storage operations
"""
import boto3
import os
from typing import Optional, BinaryIO
from modules.config.settings import settings

class AWSClient:
    """AWS S3 client for storage operations"""
    
    def __init__(self):
        self.aws_access_key = settings.AWS_ACCESS_KEY_ID
        self.aws_secret_key = settings.AWS_SECRET_ACCESS_KEY
        self.aws_region = settings.AWS_REGION
        self.s3_bucket = settings.S3_BUCKET_NAME
        
        # Initialize S3 client
        if all([self.aws_access_key, self.aws_secret_key, self.aws_region, self.s3_bucket]):
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.aws_region
            )
        else:
            self.s3_client = None
            print("⚠️ AWS credentials not configured - using local storage only")
    
    def upload_file(self, file_obj: BinaryIO, s3_key: str, content_type: str = None) -> bool:
        """Upload file to S3"""
        if not self.s3_client:
            return False  # Fallback to local storage

        try:
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type
            extra_args.setdefault('CacheControl', 'public, max-age=31536000, immutable')
            extra_args.setdefault('ACL', 'public-read')

            self.s3_client.upload_fileobj(
                file_obj,
                self.s3_bucket,
                s3_key,
                ExtraArgs=extra_args
            )
            return True
        except Exception as e:
            print(f"AWS upload error: {str(e)}")
            return False
    
    def download_file(self, s3_key: str, local_path: str) -> bool:
        """Download file from S3"""
        if not self.s3_client:
            return False
        
        try:
            self.s3_client.download_file(self.s3_bucket, s3_key, local_path)
            return True
        except Exception as e:
            print(f"AWS download error: {str(e)}")
            return False
    
    def delete_file(self, s3_key: str) -> bool:
        """Delete file from S3"""
        if not self.s3_client:
            return False
        
        try:
            self.s3_client.delete_object(Bucket=self.s3_bucket, Key=s3_key)
            return True
        except Exception as e:
            print(f"AWS delete error: {str(e)}")
            return False
    
    def get_file_url(self, s3_key: str, expires_in: int = 3600) -> Optional[str]:
        """Generate a stable URL for file access."""
        if not self.s3_client:
            return None

        try:
            if settings.CLOUDFRONT_DISTRIBUTION_DOMAIN:
                domain = settings.CLOUDFRONT_DISTRIBUTION_DOMAIN.rstrip('/')
                return f"https://{domain}/{s3_key}"

            if settings.S3_PUBLIC_BASE_URL:
                base = settings.S3_PUBLIC_BASE_URL.rstrip('/')
                return f"{base}/{s3_key}"

            region = self.aws_region or "us-east-1"
            if region == "us-east-1":
                return f"https://{self.s3_bucket}.s3.amazonaws.com/{s3_key}"
            return f"https://{self.s3_bucket}.s3.{region}.amazonaws.com/{s3_key}"
        except Exception as e:
            print(f"AWS URL generation error: {str(e)}")
            return None
    
    def get_file_size(self, s3_key: str) -> Optional[int]:
        """Get file size from S3"""
        if not self.s3_client:
            return None
        
        try:
            response = self.s3_client.head_object(Bucket=self.s3_bucket, Key=s3_key)
            return response['ContentLength']
        except Exception as e:
            print(f"AWS file size error: {str(e)}")
            return None
    
    def delete_user_files(self, user_id: int) -> bool:
        """Delete all files for a user"""
        if not self.s3_client:
            return False
        
        try:
            # List all objects with user prefix
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix=f"user_{user_id}/"
            )
            
            if 'Contents' in response:
                # Delete all objects
                objects = [{'Key': obj['Key']} for obj in response['Contents']]
                self.s3_client.delete_objects(
                    Bucket=self.s3_bucket,
                    Delete={'Objects': objects}
                )
            
            return True
        except Exception as e:
            print(f"AWS user files deletion error: {str(e)}")
            return False

# Global AWS client instance
aws_client = AWSClient()
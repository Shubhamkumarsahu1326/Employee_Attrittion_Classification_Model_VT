import os
import sys
import yaml
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

class S3StorageVault:
    def __init__(self):
        self.cloud_active = False
        
        # 1. Safely load config paths
        config_path = "config/config.yaml"
        if not os.path.exists(config_path):
            return
            
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
            
        self.s3_config = self.config["aws_storage"]
        self.bucket_name = str(self.s3_config["s3_bucket_name"]).strip()
        self.region = str(self.s3_config["aws_region"]).strip()

        # 2. Try to connect. If it fails, run silently in Local Sandbox Mode
        try:
            self.s3_client = boto3.client("s3", region_name=self.region)
            # A quick head bucket call to test if credentials work
            self.s3_client.list_objects_v2(Bucket=self.bucket_name, MaxKeys=1)
            self.cloud_active = True
            print("✅ AWS S3 Cloud Sync Enabled.")
        except Exception:
            print("⚠️ Running in Local Sandbox Mode (S3 uploads will bypass until deployed to EC2).")

    def upload_file_to_s3(self, local_file_path: str, s3_target_key: str) -> bool:
        if not self.cloud_active:
            return False  # Silently skip upload on local laptop
            
        try:
            self.s3_client.upload_file(local_file_path, self.bucket_name, s3_target_key)
            return True
        except Exception as e:
            print(f"⚠️ Cloud sync paused due to local signature variance: {str(e)}")
            return False


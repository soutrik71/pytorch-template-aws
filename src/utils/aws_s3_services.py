import boto3
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
load_dotenv(find_dotenv(".env"))


class S3Handler:
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION"),
        )

    def upload_folder(self, source_folder, dest_folder, filenames=None):
        """
        Upload specified files or all files from a local folder to an S3 folder.

        Args:
            source_folder (str): Local source folder path.
            dest_folder (str): Destination folder path in S3.
            filenames (list): List of filenames to upload (relative to source_folder). If None, uploads all files.
        """
        source_folder = Path(source_folder)

        # Select files based on filenames list or all files if filenames is None
        files_to_upload = (
            [source_folder / file for file in filenames]
            if filenames
            else list(source_folder.rglob("*"))
        )

        for file_path in files_to_upload:
            if file_path.is_file():
                s3_path = f"{dest_folder}/{file_path.relative_to(source_folder)}"
                self.s3.upload_file(str(file_path), self.bucket_name, s3_path)
                print(f"Uploaded: {file_path} to {s3_path}")
            else:
                print(f"File not found: {file_path}")

    def download_folder(self, s3_folder, dest_folder):
        """
        Download all files from an S3 folder to a local folder.

        Args:
            s3_folder (str): Source folder in S3.
            dest_folder (str): Local destination folder path.
        """
        dest_folder = Path(dest_folder)
        paginator = self.s3.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=s3_folder):
            for obj in page.get("Contents", []):
                s3_path = obj["Key"]
                local_path = dest_folder / Path(s3_path).relative_to(s3_folder)
                local_path.parent.mkdir(parents=True, exist_ok=True)
                self.s3.download_file(self.bucket_name, s3_path, str(local_path))
                print(f"Downloaded: {s3_path} to {local_path}")


# Usage Example
if __name__ == "__main__":
    # Initialize with bucket name
    s3_handler = S3Handler(bucket_name="deep-bucket-s3")

    # Upload specific files
    s3_handler.upload_folder(
        "checkpoints",
        "checkpoints_test",
    )

    # Download example
    s3_handler.download_folder("checkpoints_test", "checkpoints")

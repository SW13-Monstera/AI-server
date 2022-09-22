import os
import sys
from glob import glob
from typing import Tuple

import boto3
from botocore import UNSIGNED
from botocore.client import Config


class AwsS3Downloader:
    def __init__(self, aws_access_key_id=None, aws_secret_access_key=None) -> None:
        self.resource = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        ).resource("s3")
        self.client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            config=Config(signature_version=UNSIGNED),
        )

    def _split_url(self, url: str) -> Tuple[str, str]:
        if url.startswith("s3://"):
            url = url.replace("s3://", "")
        bucket, key = url.split("/", maxsplit=1)
        return bucket, key

    def download(self, url: str, local_dir: str) -> str:
        bucket, key = self._split_url(url)
        filename = os.path.basename(key)
        file_path = os.path.join(local_dir, filename)
        if glob(file_path):
            return file_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        meta_data = self.client.head_object(Bucket=bucket, Key=key)
        total_length = int(meta_data.get("ContentLength", 0))

        downloaded = 0

        def progress(chunk):
            nonlocal downloaded
            downloaded += chunk
            done = int(50 * downloaded / total_length)
            sys.stdout.write("\r{}[{}{}]".format(file_path, "█" * done, "." * (50 - done)))
            sys.stdout.flush()

        with open(file_path, "wb") as f:
            self.client.download_fileobj(bucket, key, f, Callback=progress)
        sys.stdout.write("\n")
        sys.stdout.flush()

        return file_path


if __name__ == "__main__":
    s3 = AwsS3Downloader()

    s3.download(url="s3://cs-broker-bucket/ai-models/2022-09-18/19-06-18/best_model.pt", local_dir=".cache")

import hashlib
import os
import re

from mlflow import data
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository


class PluginS3ArtifactRepository(S3ArtifactRepository):
    """S3ArtifactRepository provided through plugin system"""

    def __init__(self, artifact_uri):
        self.is_plugin = True
        return super(PluginS3ArtifactRepository, self).__init__(artifact_uri)

    def _get_s3_client(self):
        import boto3
        from botocore.client import Config

        s3_endpoint_url = os.environ.get("MLFLOW_S3_ENDPOINT_URL")
        ignore_tls = os.environ.get("MLFLOW_S3_IGNORE_TLS")

        verify = True
        if ignore_tls:
            verify = ignore_tls.lower() not in ["true", "yes", "1"]

        signature_version = os.environ.get("MLFLOW_EXPERIMENTAL_S3_SIGNATURE_VERSION", "s3v4")
        return boto3.client(
            "s3",
            config=Config(
                signature_version=signature_version, read_timeout=0.5, retries={'max_attempts': 120}
            ),
            endpoint_url=s3_endpoint_url,
            verify=verify,
        )

    def _calculate_md5(self, path):
        hash_md5 = hashlib.md5()
        with open(path, "rb") as file:
            for chunk in iter(lambda: file.read(2 ** 20), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _download_file(self, remote_file_path, local_path):
        # * Feature: Check MD5 before downloading
        if os.environ.get('MLFLOW_VALIDATE_FILES', '').lower() == 'checksum' and os.path.exists(
            local_path
        ):
            bucket, s3_root_path = data.parse_s3_uri(self.artifact_uri)
            s3_client = self._get_s3_client()
            s3_response = s3_client.head_object(
                Bucket=bucket, Key=s3_root_path + '/' + remote_file_path
            )
            s3obj_etag = s3_response['ETag'].strip('"')
            local_file_md5 = self._calculate_md5(local_path)
            if s3obj_etag == local_file_md5:
                print(f'Did not download {os.path.basename(local_path)}: MD5 identical')
                return

        return super(PluginS3ArtifactRepository, self)._download_file(remote_file_path, local_path)

    def log_artifacts(self, local_dir, artifact_path=None):
        # * Feature: ignore_files_for_upload
        if 'MLFLOW_IGNORE_PATH' in os.environ and os.path.exists(os.environ['MLFLOW_IGNORE_PATH']):
            with open(os.environ['MLFLOW_IGNORE_PATH']) as mlflow_ignore:
                self.ignore_files_for_upload = mlflow_ignore.readlines()

        return super(PluginS3ArtifactRepository, self).log_artifacts(local_dir, artifact_path)

    def _upload_file(self, s3_client, local_file, bucket, key):
        # * Feature: ignore_files_for_upload
        if hasattr(self, 'ignore_files_for_upload') and any(
            re.compile(pattern).search(local_file) for pattern in self.ignore_files_for_upload
        ):
            print('Ignored for upload:', local_file)
            return

        return super(PluginS3ArtifactRepository, self)._upload_file(
            s3_client, local_file, bucket, key
        )

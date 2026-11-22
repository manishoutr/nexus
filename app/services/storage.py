"""File storage behind one small interface.

Two backends:
  local  files under DATA_DIR (default, no setup)
  s3     any S3-compatible bucket via boto3 (AWS S3, MinIO, moto, ...)

Objects are stored under a generated key "<workspace_id>/<uuid>", never the
user's filename, so two uploads called "notes.pdf" can't overwrite each other
and a hostile filename can't touch the filesystem. The original name lives in
the documents table as metadata.
"""
import uuid
from pathlib import Path
from typing import Optional

from ..config import settings


def new_key(workspace_id: int) -> str:
    return f"{workspace_id}/{uuid.uuid4().hex}"


class LocalStorage:
    name = "local"

    def __init__(self, root: str) -> None:
        self.root = Path(root).resolve()

    def _path(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if self.root not in p.parents:  # defence against ../ keys
            raise ValueError("invalid storage key")
        return p

    def put(self, key: str, data: bytes) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)  # atomic: readers never see a half-written file

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def exists(self, key: str) -> bool:
        return self._path(key).exists()


class S3Storage:
    name = "s3"

    def __init__(self, bucket: str, endpoint_url: Optional[str] = None,
                 region: str = "us-east-1") -> None:
        import boto3  # imported lazily so local mode needs no boto3
        self.bucket = bucket
        self.client = boto3.client("s3", endpoint_url=endpoint_url or None,
                                   region_name=region)

    def ensure_bucket(self) -> None:
        from botocore.exceptions import ClientError
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError:
            self.client.create_bucket(Bucket=self.bucket)

    def put(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data)

    def get(self, key: str) -> bytes:
        from botocore.exceptions import ClientError
        try:
            return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                raise FileNotFoundError(key)
            raise

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)  # idempotent

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False


_storage = None


def get_storage():
    global _storage
    if _storage is None:
        if settings.storage_backend == "s3":
            s = S3Storage(settings.s3_bucket, settings.s3_endpoint_url,
                          settings.s3_region)
            s.ensure_bucket()
            _storage = s
        else:
            _storage = LocalStorage(settings.data_dir)
    return _storage


def reset_storage_for_tests() -> None:
    global _storage
    _storage = None

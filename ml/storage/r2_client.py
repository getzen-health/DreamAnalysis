"""Cloudflare R2 (S3-compatible) client for EEG session binary storage.

Usage:
    r2 = R2Client()
    if r2.available:
        r2.upload_bytes(b"...", "users/default/abc123.npz")
        data = r2.download_bytes("users/default/abc123.npz")
        r2.delete("users/default/abc123.npz")

Required environment variables (all optional — falls back to local disk if absent):
    R2_ACCOUNT_ID        Cloudflare account ID
    R2_ACCESS_KEY_ID     R2 API token key ID
    R2_SECRET_ACCESS_KEY R2 API token secret
    R2_BUCKET_NAME       Bucket name (default: neural-dream-sessions)
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class R2Client:
    """Thin wrapper around boto3 S3 client pointed at Cloudflare R2."""

    def __init__(self):
        self._client = None
        self._bucket = os.environ.get("R2_BUCKET_NAME", "neural-dream-sessions")
        self._setup()

    def _setup(self):
        account_id = os.environ.get("R2_ACCOUNT_ID", "")
        access_key = os.environ.get("R2_ACCESS_KEY_ID", "")
        secret_key = os.environ.get("R2_SECRET_ACCESS_KEY", "")

        if not all([account_id, access_key, secret_key]):
            logger.info("R2 credentials not set — session signals will be stored locally.")
            return

        try:
            import boto3
            self._client = boto3.client(
                "s3",
                endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name="auto",
            )
            logger.info(f"R2 client ready (bucket: {self._bucket})")
        except Exception as exc:
            logger.warning(f"R2 client init failed: {exc} — falling back to local disk.")
            self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def upload_bytes(self, data: bytes, key: str) -> bool:
        """Upload raw bytes to R2. Returns True on success."""
        if not self._client:
            return False
        try:
            self._client.put_object(Bucket=self._bucket, Key=key, Body=data)
            return True
        except Exception as exc:
            logger.error(f"R2 upload failed ({key}): {exc}")
            return False

    def download_bytes(self, key: str) -> Optional[bytes]:
        """Download object from R2. Returns None if not found or on error."""
        if not self._client:
            return None
        try:
            resp = self._client.get_object(Bucket=self._bucket, Key=key)
            return resp["Body"].read()
        except Exception as exc:
            logger.error(f"R2 download failed ({key}): {exc}")
            return None

    def delete(self, key: str) -> bool:
        """Delete an object from R2. Returns True on success."""
        if not self._client:
            return False
        try:
            self._client.delete_object(Bucket=self._bucket, Key=key)
            return True
        except Exception as exc:
            logger.error(f"R2 delete failed ({key}): {exc}")
            return False

    @staticmethod
    def session_key(user_id: str, session_id: str) -> str:
        """Canonical R2 key for a session's signal file."""
        return f"users/{user_id}/sessions/{session_id}.npz"


# Module-level singleton — imported once at startup
r2 = R2Client()

"""
Acceptance criteria:
- 404 from head_object → upload with md5 stored in Metadata.
- Matching md5 → no upload (zero R2 write ops).
- Mismatching md5 → upload with updated metadata.
- Non-404 ClientError → re-raised (do not swallow permission errors).
- Missing md5 key in existing object metadata → treated as mismatch, uploads.
- Stored md5 matches actual parquet bytes (not a stub).
"""

from __future__ import annotations

import hashlib
import io
from unittest.mock import MagicMock

import pandas as pd
import pytest
from botocore.exceptions import ClientError

from data.loader import upload_parquet_with_md5_dedup


def _make_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


def _md5_of_df(df: pd.DataFrame) -> str:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    return hashlib.md5(buf.getvalue()).hexdigest()


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": ""}}, "HeadObject")


class TestUploadParquetWithMd5Dedup:
    def test_uploads_on_404(self):
        df = _make_df()
        s3 = MagicMock()
        s3.head_object.side_effect = _client_error("404")

        upload_parquet_with_md5_dedup(df, "test.parquet", s3)

        s3.upload_fileobj.assert_called_once()

    def test_uploaded_md5_matches_bytes(self):
        df = _make_df()
        s3 = MagicMock()
        s3.head_object.side_effect = _client_error("404")

        upload_parquet_with_md5_dedup(df, "test.parquet", s3)

        extra_args = s3.upload_fileobj.call_args[1]["ExtraArgs"]
        assert extra_args["Metadata"]["md5"] == _md5_of_df(df)

    def test_skips_on_md5_match(self):
        df = _make_df()
        s3 = MagicMock()
        s3.head_object.return_value = {"Metadata": {"md5": _md5_of_df(df)}}

        upload_parquet_with_md5_dedup(df, "test.parquet", s3)

        s3.upload_fileobj.assert_not_called()

    def test_uploads_on_md5_mismatch(self):
        df = _make_df()
        s3 = MagicMock()
        s3.head_object.return_value = {"Metadata": {"md5": "stale_hash_abc"}}

        upload_parquet_with_md5_dedup(df, "test.parquet", s3)

        s3.upload_fileobj.assert_called_once()
        extra_args = s3.upload_fileobj.call_args[1]["ExtraArgs"]
        assert extra_args["Metadata"]["md5"] == _md5_of_df(df)

    def test_uploads_when_metadata_key_missing(self):
        df = _make_df()
        s3 = MagicMock()
        s3.head_object.return_value = {"Metadata": {}}

        upload_parquet_with_md5_dedup(df, "test.parquet", s3)

        s3.upload_fileobj.assert_called_once()

    def test_reraises_non_404_client_error(self):
        df = _make_df()
        s3 = MagicMock()
        s3.head_object.side_effect = _client_error("403")

        with pytest.raises(ClientError):
            upload_parquet_with_md5_dedup(df, "test.parquet", s3)
        s3.put_object.assert_not_called()

    def test_exactly_one_head_object_call_per_upload(self):
        df = _make_df()
        s3 = MagicMock()
        s3.head_object.side_effect = _client_error("404")

        upload_parquet_with_md5_dedup(df, "test.parquet", s3)

        assert s3.head_object.call_count == 1

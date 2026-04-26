from __future__ import annotations

import hashlib
import io
import os

import boto3
import pandas as pd
from botocore.exceptions import ClientError

R2_ENDPOINT = "https://a9e4828e0e2c14c92a0618cded4bf6b6.r2.cloudflarestorage.com"
R2_BUCKET = "arusto-skills"


def _get_r2_credentials() -> tuple[str, str]:
    key_id = os.environ.get("R2_ACCESS_KEY_ID")
    secret = os.environ.get("R2_SECRET_ACCESS_KEY")

    if key_id and secret:
        return key_id, secret

    try:
        import streamlit as st

        key_id = st.secrets.get("R2_ACCESS_KEY_ID")
        secret = st.secrets.get("R2_SECRET_ACCESS_KEY")

        if key_id and secret:
            return key_id, secret

    except (ImportError, FileNotFoundError, KeyError):
        pass

    raise OSError(
        "R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY must be set "
        "in environment variables or Streamlit secrets."
    )


def _get_s3_client() -> boto3.client:
    key_id, secret = _get_r2_credentials()
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        region_name="auto",
    )


def upload_parquet_with_md5_dedup(df: pd.DataFrame, key: str, s3: boto3.client) -> None:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    data = buf.getvalue()
    md5_hex = hashlib.md5(data).hexdigest()

    try:
        head = s3.head_object(Bucket=R2_BUCKET, Key=key)
        if head.get("Metadata", {}).get("md5") == md5_hex:
            print(f"Skipping {key} (md5 match)")
            return
    except ClientError as e:
        if e.response["Error"]["Code"] != "404":
            raise

    buf.seek(0)
    s3.upload_fileobj(buf, R2_BUCKET, key, ExtraArgs={"Metadata": {"md5": md5_hex}})
    print(f"Uploaded {key} ({len(df):,} rows)")


def download_kaggle_data(dest_dir: str) -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    # Prefer env vars; fall back to st.secrets; kaggle.json auto-detected by SDK.
    if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
        try:
            import streamlit as st

            os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
            os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]
        except (ImportError, FileNotFoundError, KeyError):
            pass

    api = KaggleApi()
    api.authenticate()
    os.makedirs(dest_dir, exist_ok=True)
    api.dataset_download_files(
        "asaniczka/1-3m-linkedin-jobs-and-skills-2024",
        path=dest_dir,
        unzip=True,
    )

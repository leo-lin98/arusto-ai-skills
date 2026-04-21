from __future__ import annotations

import io
import os

import boto3
import pandas as pd

R2_ENDPOINT = "https://a9e4828e0e2c14c92a0618cded4bf6b6.r2.cloudflarestorage.com"
R2_BUCKET = "arusto-skills"
R2_KEY = "merged.parquet"


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


def load_parquet_from_r2() -> pd.DataFrame:
    key_id, secret = _get_r2_credentials()
    storage_options = {
        "key": key_id,
        "secret": secret,
        "client_kwargs": {"endpoint_url": R2_ENDPOINT},
    }
    return pd.read_parquet(
        f"s3://{R2_BUCKET}/{R2_KEY}",
        storage_options=storage_options,
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


def upload_parquet_to_r2(df: pd.DataFrame, key: str, s3: boto3.client | None = None) -> None:
    client = s3 or _get_s3_client()
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    buf.seek(0)
    client.upload_fileobj(buf, R2_BUCKET, key)
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

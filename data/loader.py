import os
import urllib.request


def get_parquet_url() -> str | None:
    if "PARQUET_URL" in os.environ:
        return os.environ["PARQUET_URL"]
    try:
        import streamlit as st

        return st.secrets.get("PARQUET_URL")
    except (ImportError, FileNotFoundError, KeyError):
        return None


def download_parquet(url: str, dest_path: str) -> None:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp_path = dest_path + ".tmp"
    print("Downloading parquet from R2...")
    try:
        urllib.request.urlretrieve(url, tmp_path)
        os.replace(tmp_path, dest_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    print("Parquet downloaded.")

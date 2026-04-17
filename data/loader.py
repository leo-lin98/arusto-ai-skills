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


def download_kaggle_data(dest_dir: str) -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    # Prefer env vars; fall back to st.secrets (Streamlit runtime); ~/.kaggle/kaggle.json auto-detected by SDK.
    if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
        try:
            import streamlit as st

            os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
            os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]
        except Exception:
            pass

    api = KaggleApi()
    api.authenticate()
    os.makedirs(dest_dir, exist_ok=True)
    api.dataset_download_files(
        "asaniczka/1-3m-linkedin-jobs-and-skills-2024",
        path=dest_dir,
        unzip=True,
    )

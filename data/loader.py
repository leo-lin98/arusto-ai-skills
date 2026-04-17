import os


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

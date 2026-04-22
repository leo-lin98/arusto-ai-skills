import duckdb
import streamlit as st

from data.loader import _get_r2_credentials

_R2_ENDPOINT_HOST = "a9e4828e0e2c14c92a0618cded4bf6b6.r2.cloudflarestorage.com"
_R2_BUCKET = "arusto-skills"
PARQUET_S3_PATH = f"s3://{_R2_BUCKET}/merged.parquet"

_SESSION_KEY = "duckdb_conn"


def _create_connection() -> duckdb.DuckDBPyConnection:
    key_id, secret = _get_r2_credentials()
    conn = duckdb.connect()
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    conn.execute(
        f"""
        SET s3_endpoint='{_R2_ENDPOINT_HOST}';
        SET s3_access_key_id='{key_id.replace("'", "''")}';
        SET s3_secret_access_key='{secret.replace("'", "''")}';
        SET s3_region='auto';
        SET s3_use_ssl=true;
        SET s3_url_style='path';
        """
    )
    return conn


def get_db_connection() -> duckdb.DuckDBPyConnection:
    if _SESSION_KEY not in st.session_state:
        st.session_state[_SESSION_KEY] = _create_connection()
    return st.session_state[_SESSION_KEY]


def filter_conditions(company: str, location: str) -> tuple[list[str], list[str]]:
    conditions: list[str] = []
    params: list[str] = []
    if company != "All":
        conditions.append("company = ?")
        params.append(company)
    if location != "All":
        conditions.append("search_city = ?")
        params.append(location)
    return conditions, params

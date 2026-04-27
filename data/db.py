import os
from datetime import date

import duckdb
import streamlit as st

from data.loader import _get_r2_credentials

_R2_ENDPOINT_HOST = os.environ.get("R2_ENDPOINT")
_R2_BUCKET = os.environ.get("R2_BUCKET")
PARQUET_S3_PATH = f"s3://{_R2_BUCKET}/jobs.parquet"
TOPIC_RANKINGS_S3_PATH = f"s3://{_R2_BUCKET}/topic_rankings.parquet"
SKILL_THEME_MAP_S3_PATH = f"s3://{_R2_BUCKET}/skill_theme_map.parquet"

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


def filter_conditions(
    company: str,
    country: str,
    date_range: tuple[date, date] | None,
) -> tuple[list[str], list[str | date]]:
    conditions: list[str] = []
    params: list[str | date] = []
    if company != "All":
        conditions.append("company = ?")
        params.append(company)
    if country != "All":
        conditions.append("search_country = ?")
        params.append(country)
    if date_range is not None:
        conditions.append("first_seen >= ?")
        params.append(date_range[0])
        conditions.append("first_seen <= ?")
        params.append(date_range[1])
    return conditions, params

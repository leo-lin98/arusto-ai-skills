import duckdb
import streamlit as st

from data.db import PARQUET_S3_PATH, filter_conditions


@st.cache_data
def _filter_options(_conn: duckdb.DuckDBPyConnection) -> tuple[list[str], list[str]]:
    companies = (
        _conn.execute(
            f"""
            SELECT company FROM read_parquet('{PARQUET_S3_PATH}')
            GROUP BY company ORDER BY COUNT(*) DESC LIMIT 50
            """
        )
        .df()["company"]
        .tolist()
    )
    locations = (
        _conn.execute(
            f"""
            SELECT job_location FROM read_parquet('{PARQUET_S3_PATH}')
            GROUP BY job_location ORDER BY COUNT(*) DESC LIMIT 50
            """
        )
        .df()["job_location"]
        .tolist()
    )
    return companies, locations


def sidebar_filters(conn: duckdb.DuckDBPyConnection) -> tuple[str, str]:
    companies, locations = _filter_options(conn)

    st.sidebar.header("Filters")
    selected_company = st.sidebar.selectbox("Company", ["All"] + companies)
    selected_location = st.sidebar.selectbox("Location", ["All"] + locations)

    conditions, params = filter_conditions(selected_company, selected_location)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    count = conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{PARQUET_S3_PATH}') {where}", params
    ).fetchone()[0]
    st.sidebar.markdown(f"**{count:,}** postings shown")

    return selected_company, selected_location

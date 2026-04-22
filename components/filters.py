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
            SELECT search_city FROM read_parquet('{PARQUET_S3_PATH}')
            GROUP BY search_city ORDER BY COUNT(*) DESC LIMIT 50
            """
        )
        .df()["search_city"]
        .tolist()
    )
    return companies, locations


@st.cache_data
def _posting_count(
    _conn: duckdb.DuckDBPyConnection, company: str, location: str
) -> int:
    conditions, params = filter_conditions(company, location)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    row = _conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{PARQUET_S3_PATH}') {where}", params
    ).fetchone()
    return int(row[0]) if row else 0


def sidebar_filters(conn: duckdb.DuckDBPyConnection) -> tuple[str, str]:
    companies, locations = _filter_options(conn)

    st.sidebar.header("Filters")
    selected_company = st.sidebar.selectbox("Company", ["All"] + companies)
    selected_location = st.sidebar.selectbox("Location", ["All"] + locations)

    count = _posting_count(conn, selected_company, selected_location)
    st.sidebar.markdown(f"**{count:,}** postings shown")

    return selected_company, selected_location

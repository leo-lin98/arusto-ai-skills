import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from components.charts import top_companies_chart
from components.filters import sidebar_filters
from data.db import PARQUET_S3_PATH, filter_conditions, get_db_connection

st.set_page_config(page_title="Overview", layout="wide")
st.title("Overview")

conn = get_db_connection()
company, location = sidebar_filters(conn)


@st.cache_data
def get_metrics(_conn, company: str, location: str) -> tuple[int, int, int]:
    conditions, params = filter_conditions(company, location)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    return _conn.execute(
        f"""
        SELECT
            COUNT(*) AS total,
            COUNT(DISTINCT company) AS companies,
            COUNT(DISTINCT search_city) AS locations
        FROM read_parquet('{PARQUET_S3_PATH}')
        {where}
        """,
        params,
    ).fetchone()


@st.cache_data
def get_top_jobs(_conn, company: str, location: str) -> pd.DataFrame:
    conditions, params = filter_conditions(company, location)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    return (
        _conn.execute(
            f"""
            SELECT job_title, COUNT(*) AS "Listings"
            FROM read_parquet('{PARQUET_S3_PATH}')
            {where}
            GROUP BY job_title ORDER BY "Listings" DESC LIMIT 10
            """,
            params,
        )
        .df()
        .set_index("job_title")
    )


@st.cache_data
def get_day_of_week(_conn, company: str, location: str) -> pd.DataFrame:
    conditions, params = filter_conditions(company, location)
    time_where = f"WHERE {' AND '.join(conditions + ['first_seen IS NOT NULL'])}"
    return (
        _conn.execute(
            f"""
            SELECT dayname(first_seen) AS day_of_week, COUNT(*) AS "Postings"
            FROM read_parquet('{PARQUET_S3_PATH}')
            {time_where}
            GROUP BY day_of_week
            """,
            params,
        )
        .df()
        .set_index("day_of_week")
    )


@st.cache_data
def get_job_level(_conn, company: str, location: str) -> pd.DataFrame:
    conditions, params = filter_conditions(company, location)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    return (
        _conn.execute(
            f"""
            SELECT job_level, COUNT(*) AS "Postings"
            FROM read_parquet('{PARQUET_S3_PATH}')
            {where}
            GROUP BY job_level ORDER BY "Postings" DESC
            """,
            params,
        )
        .df()
        .set_index("job_level")
    )


@st.cache_data
def get_top_locations(_conn, company: str, location: str) -> pd.DataFrame:
    conditions, params = filter_conditions(company, location)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    return (
        _conn.execute(
            f"""
            SELECT search_city, COUNT(*) AS "Postings"
            FROM read_parquet('{PARQUET_S3_PATH}')
            {where}
            GROUP BY search_city ORDER BY "Postings" DESC LIMIT 10
            """,
            params,
        )
        .df()
        .set_index("search_city")
    )


@st.cache_data
def get_hourly(_conn, company: str, location: str) -> pd.DataFrame:
    conditions, params = filter_conditions(company, location)
    time_where = f"WHERE {' AND '.join(conditions + ['first_seen IS NOT NULL'])}"
    return (
        _conn.execute(
            f"""
            SELECT hour(first_seen) AS hour, COUNT(*) AS "Postings"
            FROM read_parquet('{PARQUET_S3_PATH}')
            {time_where}
            GROUP BY hour ORDER BY hour
            """,
            params,
        )
        .df()
        .set_index("hour")
    )


@st.cache_data
def get_day_of_month(_conn, company: str, location: str) -> pd.Series:
    conditions, params = filter_conditions(company, location)
    time_where = f"WHERE {' AND '.join(conditions + ['first_seen IS NOT NULL'])}"
    return (
        _conn.execute(
            f"""
            SELECT dayofmonth(first_seen) AS day
            FROM read_parquet('{PARQUET_S3_PATH}')
            {time_where}
            LIMIT 10000
            """,
            params,
        )
        .df()["day"]
        .dropna()
    )


@st.cache_data
def get_search_positions(_conn, company: str, location: str) -> pd.DataFrame:
    conditions, params = filter_conditions(company, location)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    return (
        _conn.execute(
            f"""
            SELECT search_position, COUNT(*) AS "Postings"
            FROM read_parquet('{PARQUET_S3_PATH}')
            {where}
            GROUP BY search_position ORDER BY "Postings" DESC LIMIT 20
            """,
            params,
        )
        .df()
        .set_index("search_position")
    )


day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

metrics = get_metrics(conn, company, location)
col1, col2, col3 = st.columns(3)
col1.metric("Total Postings", f"{metrics[0]:,}")
col2.metric("Unique Companies", f"{metrics[1]:,}")
col3.metric("Unique Locations", f"{metrics[2]:,}")

st.divider()

st.subheader("Top 10 Job Titles")
st.bar_chart(get_top_jobs(conn, company, location))

st.subheader("Job Postings by Day of Week")
st.bar_chart(
    get_day_of_week(conn, company, location)
    .reindex(day_order)
    .fillna(0)
)

st.subheader("Top 10 Companies by Job Count")
top_companies_chart(conn, 10, company, location)

st.subheader("Job Level Distribution")
st.bar_chart(get_job_level(conn, company, location))

st.subheader("Top 10 Job Locations")
st.bar_chart(get_top_locations(conn, company, location))

st.subheader("Job Postings by Hour of Day")
st.bar_chart(get_hourly(conn, company, location))

st.subheader("Job Openings by Day of Month")
days = get_day_of_month(conn, company, location)
if len(days) < 2:
    st.info("Not enough date data to render distribution.")
else:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.violinplot(days, vert=False)
    ax.set_xlabel("Day of Month")
    ax.set_title("Distribution of Job Openings by Day of Month")
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Search Position Distribution")
st.bar_chart(get_search_positions(conn, company, location))

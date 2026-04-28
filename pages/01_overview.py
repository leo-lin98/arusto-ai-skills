from datetime import date

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from components.charts import CATEGORICAL_SCHEME, SEQUENTIAL_SCHEME, top_companies_chart
from components.filters import sidebar_filters
from data.db import PARQUET_S3_PATH, filter_conditions, get_db_connection

st.set_page_config(page_title="Overview", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="metric-container"] {
        background: white;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 6px rgba(157,78,221,0.15);
        border-left: 4px solid #9D4EDD;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Overview")

conn = get_db_connection()
company, country, date_range = sidebar_filters(conn)


@st.cache_data
def get_metrics(
    _conn,
    company: str,
    country: str,
    date_range: tuple[date, date],
) -> tuple[int, int, int]:
    conditions, params = filter_conditions(company, country, date_range)
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
def get_top_jobs(
    _conn,
    company: str,
    country: str,
    date_range: tuple[date, date],
) -> pd.DataFrame:
    conditions, params = filter_conditions(company, country, date_range)
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
def get_day_of_week(
    _conn,
    company: str,
    country: str,
    date_range: tuple[date, date],
) -> pd.DataFrame:
    conditions, params = filter_conditions(company, country, date_range)
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
def get_job_level(
    _conn,
    company: str,
    country: str,
    date_range: tuple[date, date],
) -> pd.DataFrame:
    conditions, params = filter_conditions(company, country, date_range)
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
def get_top_locations(
    _conn,
    company: str,
    country: str,
    date_range: tuple[date, date],
) -> pd.DataFrame:
    conditions, params = filter_conditions(company, country, date_range)
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
def get_day_of_month(
    _conn,
    company: str,
    country: str,
    date_range: tuple[date, date],
) -> pd.Series:
    conditions, params = filter_conditions(company, country, date_range)
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
def get_search_positions(
    _conn,
    company: str,
    country: str,
    date_range: tuple[date, date],
) -> pd.DataFrame:
    conditions, params = filter_conditions(company, country, date_range)
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


day_order = [
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
]

metrics = get_metrics(conn, company, country, date_range)
col1, col2, col3 = st.columns(3)
col1.metric("Total Postings", f"{metrics[0]:,}")
col2.metric("Unique Companies", f"{metrics[1]:,}")
col3.metric("Unique Locations", f"{metrics[2]:,}")

st.divider()

st.subheader("Top 10 Job Titles")
_jobs_df = get_top_jobs(conn, company, country, date_range).reset_index()
st.altair_chart(
    alt.Chart(_jobs_df)
    .mark_bar()
    .encode(
        x=alt.X("job_title:N", sort=None, title="Job Title"),
        y=alt.Y("Listings:Q", title="Listings"),
        color=alt.Color(
            "Listings:Q",
            scale=alt.Scale(scheme=SEQUENTIAL_SCHEME),
            legend=None,
        ),
    ),
    width="stretch",
)

st.subheader("Job Postings by Day of Week")
_dow_df = (
    get_day_of_week(conn, company, country, date_range)
    .reindex(day_order)
    .fillna(0)
    .reset_index()
)
st.altair_chart(
    alt.Chart(_dow_df)
    .mark_bar()
    .encode(
        x=alt.X("day_of_week:N", sort=None, title="Day"),
        y=alt.Y("Postings:Q", title="Postings"),
        color=alt.Color(
            "day_of_week:N",
            scale=alt.Scale(scheme=CATEGORICAL_SCHEME),
            legend=None,
        ),
    ),
    width="stretch",
)

st.subheader("Top 10 Companies by Job Count")
top_companies_chart(conn, 10, company, country, date_range)

st.subheader("Job Level Distribution")
_level_df = get_job_level(conn, company, country, date_range).reset_index()
st.altair_chart(
    alt.Chart(_level_df)
    .mark_bar()
    .encode(
        x=alt.X("job_level:N", sort=None, title="Job Level"),
        y=alt.Y("Postings:Q", title="Postings"),
        color=alt.Color(
            "job_level:N",
            scale=alt.Scale(scheme=CATEGORICAL_SCHEME),
            legend=None,
        ),
    ),
    width="stretch",
)

st.subheader("Top 10 Job Locations")
_loc_df = get_top_locations(conn, company, country, date_range).reset_index()
st.altair_chart(
    alt.Chart(_loc_df)
    .mark_bar()
    .encode(
        x=alt.X("search_city:N", sort=None, title="City"),
        y=alt.Y("Postings:Q", title="Postings"),
        color=alt.Color(
            "Postings:Q",
            scale=alt.Scale(scheme=SEQUENTIAL_SCHEME),
            legend=None,
        ),
    ),
    width="stretch",
)

st.subheader("Job Openings by Day of Month")
days = get_day_of_month(conn, company, country, date_range)
if len(days) < 2:
    st.info("Not enough date data to render distribution.")
else:
    fig, ax = plt.subplots(figsize=(10, 4))
    parts = ax.violinplot(days, vert=False)
    for body in parts["bodies"]:
        body.set_facecolor("#9D4EDD")
        body.set_alpha(0.7)
    ax.set_xlabel("Day of Month")
    ax.set_title("Distribution of Job Openings by Day of Month")
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Search Position Distribution")
_pos_df = get_search_positions(conn, company, country, date_range).reset_index()
st.altair_chart(
    alt.Chart(_pos_df)
    .mark_bar()
    .encode(
        x=alt.X("search_position:N", sort=None, title="Search Position"),
        y=alt.Y("Postings:Q", title="Postings"),
        color=alt.Color(
            "Postings:Q",
            scale=alt.Scale(scheme=SEQUENTIAL_SCHEME),
            legend=None,
        ),
    ),
    width="stretch",
)

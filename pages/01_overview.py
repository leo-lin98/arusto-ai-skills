import matplotlib.pyplot as plt
import streamlit as st

from components.charts import top_companies_chart
from components.filters import sidebar_filters
from data.db import PARQUET_S3_PATH, filter_conditions, get_db_connection

st.set_page_config(page_title="Overview", layout="wide")
st.title("Overview")

conn = get_db_connection()
company, location = sidebar_filters(conn)

conditions, params = filter_conditions(company, location)
where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
time_where = f"WHERE {' AND '.join(conditions + ['first_seen IS NOT NULL'])}"

metrics = conn.execute(
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

col1, col2, col3 = st.columns(3)
col1.metric("Total Postings", f"{metrics[0]:,}")
col2.metric("Unique Companies", f"{metrics[1]:,}")
col3.metric("Unique Locations", f"{metrics[2]:,}")

st.divider()

st.subheader("Top 10 Job Titles")
top_jobs = (
    conn.execute(
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
st.bar_chart(top_jobs)

st.subheader("Job Processing Time by Day of Week")
day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
day_df = (
    conn.execute(
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
    .reindex(day_order)
    .dropna()
)
st.bar_chart(day_df)

st.subheader("Top 10 Companies by Job Count")
top_companies_chart(conn, 10, company, location)

st.subheader("Job Level Distribution")
level_df = (
    conn.execute(
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
st.bar_chart(level_df)

st.subheader("Top 10 Job Locations")
location_df = (
    conn.execute(
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
st.bar_chart(location_df)

st.subheader("Job Postings by Hour of Day")
hourly_df = (
    conn.execute(
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
st.bar_chart(hourly_df)

st.subheader("Job Openings by Day of Month")
days = (
    conn.execute(
        f"""
    SELECT dayofmonth(first_seen) AS day FROM read_parquet('{PARQUET_S3_PATH}')
    {time_where}
    LIMIT 10000
    """,
        params,
    )
    .df()["day"]
    .dropna()
)
fig, ax = plt.subplots(figsize=(10, 4))
ax.violinplot(days, vert=False)
ax.set_xlabel("Day of Month")
ax.set_title("Distribution of Job Openings by Day of Month")
st.pyplot(fig)
plt.close(fig)

st.subheader("Search Position Distribution")
search_df = (
    conn.execute(
        f"""
    SELECT search_position, COUNT(*) AS "Postings"
    FROM read_parquet('{PARQUET_S3_PATH}')
    {where}
    GROUP BY search_position ORDER BY search_position LIMIT 20
    """,
        params,
    )
    .df()
    .set_index("search_position")
)
st.bar_chart(search_df)

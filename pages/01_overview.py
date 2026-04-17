import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from components.charts import top_companies_chart
from components.filters import sidebar_filters
from data.processor import PARQUET_PATH, SAMPLE_N, get_merged

st.set_page_config(page_title="Overview", layout="wide")
st.title("Overview")


@st.cache_data
def load_data() -> pd.DataFrame:
    return get_merged(PARQUET_PATH, SAMPLE_N)


with st.status(
    "Loading data — this may take a few minutes on first run...", expanded=False
):
    df = load_data()

df = sidebar_filters(df)

col1, col2, col3 = st.columns(3)
col1.metric("Total Postings", f"{len(df):,}")
col2.metric("Unique Companies", f"{df['company'].nunique():,}")
col3.metric("Unique Locations", f"{df['job_location'].nunique():,}")

st.divider()

st.subheader("Top 10 Job Titles")
top_jobs = df["job_title"].value_counts().head(10).rename("Listings")
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
day_counts = (
    df["day_of_week"].value_counts().reindex(day_order).dropna().rename("Postings")
)
st.bar_chart(day_counts)

st.subheader("Top 10 Companies by Job Count")
top_companies_chart(df, 10)

st.subheader("Job Level Distribution")
level_counts = df["job_level"].value_counts().rename("Postings")
st.bar_chart(level_counts)

st.subheader("Top 10 Job Locations")
top_locations = df["job_location"].value_counts().head(10).rename("Postings")
st.bar_chart(top_locations)

st.subheader("Job Postings by Hour of Day")
hourly = df["hour"].value_counts().sort_index().rename("Postings")
st.bar_chart(hourly)

st.subheader("Job Openings by Day of Month")
fig, ax = plt.subplots(figsize=(10, 4))
ax.violinplot(df["day"].dropna().head(10000), vert=False)
ax.set_xlabel("Day of Month")
ax.set_title("Distribution of Job Openings by Day of Month")
st.pyplot(fig)
plt.close(fig)

st.subheader("Search Position Distribution")
search_counts = df.groupby("search_position").size().head(20).rename("Postings")
st.bar_chart(search_counts)

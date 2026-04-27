import altair as alt
import pandas as pd
import streamlit as st

from data.db import TOPIC_RANKINGS_S3_PATH, get_db_connection

st.set_page_config(page_title="Course Opportunities", layout="wide")
st.title("Course Opportunities")

conn = get_db_connection()

_TOP_N = 15


@st.cache_data
def get_all_topics(_conn: object) -> pd.DataFrame:
    return _conn.execute(
        f"""
        SELECT
            course_topic,
            course_opportunity_score,
            opportunity_label,
            volume,
            forecast_4w,
            forecast_12w,
            trend_label
        FROM read_parquet('{TOPIC_RANKINGS_S3_PATH}')
        ORDER BY course_opportunity_score DESC
        """
    ).df()


topics = get_all_topics(conn)
labels = topics["opportunity_label"]
total = len(topics)
high = int((labels == "High Opportunity").sum())
emerging = int((labels == "Emerging").sum())
saturated = int((labels == "Saturated").sum())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Topics", f"{total:,}")
col2.metric("High Opportunity", f"{high:,}")
col3.metric("Emerging", f"{emerging:,}")
col4.metric("Saturated", f"{saturated:,}")

st.divider()

search = st.text_input("Search topic", placeholder="e.g. Python, Data Science")
filtered = (
    topics[topics["course_topic"].str.contains(search, case=False, na=False)]
    if search
    else topics
)

st.subheader(f"Top {_TOP_N} Course Opportunities")
top15 = filtered.head(_TOP_N).reset_index(drop=True)
top15.index += 1
top15.index.name = "Rank"
st.dataframe(
    top15.rename(
        columns={
            "course_topic": "Topic",
            "course_opportunity_score": "Score",
            "opportunity_label": "Label",
            "volume": "Volume",
            "forecast_4w": "Forecast 4w",
            "forecast_12w": "Forecast 12w",
            "trend_label": "Trend",
        }
    ).style.format(
        {
            "Score": "{:.1f}",
            "Volume": "{:,.0f}",
            "Forecast 4w": "{:.0f}",
            "Forecast 12w": "{:.0f}",
        },
        na_rep="—",
    ),
    width="stretch",
)

st.divider()

st.subheader("Volume vs Opportunity Score")
scatter_df = filtered.dropna(subset=["volume", "course_opportunity_score"])
if scatter_df.empty:
    st.info("No data to display.")
else:
    chart = (
        alt.Chart(scatter_df)
        .mark_circle(size=80, opacity=0.7)
        .encode(
            x=alt.X("volume:Q", title="Volume (Job Postings)"),
            y=alt.Y("course_opportunity_score:Q", title="Opportunity Score"),
            color=alt.Color(
                "trend_label:N",
                title="Trend",
                scale=alt.Scale(scheme="category10"),
            ),
            tooltip=[
                alt.Tooltip("course_topic:N", title="Topic"),
                alt.Tooltip("course_opportunity_score:Q", title="Score", format=".1f"),
                alt.Tooltip("volume:Q", title="Volume", format=","),
                alt.Tooltip("trend_label:N", title="Trend"),
                alt.Tooltip("opportunity_label:N", title="Label"),
            ],
        )
        .properties(height=450)
    )
    st.altair_chart(chart, width="stretch")

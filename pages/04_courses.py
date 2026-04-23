from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from data.db import get_db_connection
from data.loader import R2_BUCKET

st.set_page_config(page_title="Course Opportunity", layout="wide")
st.title("Course Opportunity Dashboard")
st.caption('Anchor question: "What courses should institutions build next?"')


def _r2(key: str) -> str:
    return f"s3://{R2_BUCKET}/{key}"


@st.cache_data
def get_totals() -> dict[str, int]:
    con = get_db_connection()
    row = con.execute(f"""
        SELECT
            SUM(volume) AS total_jobs,
            COUNT(*) AS total_topics,
            COUNT(*) FILTER (WHERE opportunity_label = 'High Opportunity') AS high_opp,
            COUNT(*) FILTER (WHERE opportunity_label = 'Emerging') AS emerging,
            COUNT(*) FILTER (WHERE opportunity_label = 'Saturated') AS saturated
        FROM read_parquet('{_r2("topic_rankings.parquet")}')
    """).fetchone()
    return {
        "total_jobs": int(row[0] or 0),
        "total_topics": int(row[1] or 0),
        "high_opp": int(row[2] or 0),
        "emerging": int(row[3] or 0),
        "saturated": int(row[4] or 0),
    }


@st.cache_data
def get_top_skills(n: int) -> pd.DataFrame:
    con = get_db_connection()
    return con.execute(f"""
        SELECT skill, skill_count
        FROM read_parquet('{_r2("skill_theme_map.parquet")}')
        ORDER BY skill_count DESC
        LIMIT {n}
    """).df()


@st.cache_data
def get_topic_rankings(label: str, top_n: int) -> pd.DataFrame:
    con = get_db_connection()
    where = "" if label == "All" else "WHERE opportunity_label = ?"
    params: list[str] = [] if label == "All" else [label]
    return con.execute(f"""
        SELECT rank, course_topic, course_opportunity_score, opportunity_label,
               volume, salary_proxy, breadth_score, trend_30d
        FROM read_parquet('{_r2("topic_rankings.parquet")}')
        {where}
        ORDER BY rank
        LIMIT {top_n}
    """, params).df()


@st.cache_data
def get_top_topics_chart(top_n: int) -> pd.DataFrame:
    con = get_db_connection()
    return con.execute(f"""
        SELECT course_topic, course_opportunity_score, opportunity_label
        FROM read_parquet('{_r2("topic_rankings.parquet")}')
        ORDER BY course_opportunity_score DESC
        LIMIT {top_n}
    """).df()


@st.cache_data
def get_volume_vs_salary() -> pd.DataFrame:
    con = get_db_connection()
    return con.execute(f"""
        SELECT course_topic, volume, salary_proxy, breadth_score, opportunity_label
        FROM read_parquet('{_r2("topic_rankings.parquet")}')
        ORDER BY volume DESC
        LIMIT 100
    """).df()


@st.cache_data
def get_trend_top(top_n: int) -> pd.DataFrame:
    con = get_db_connection()
    return con.execute(f"""
        SELECT course_topic, trend_30d
        FROM read_parquet('{_r2("topic_rankings.parquet")}')
        ORDER BY trend_30d DESC
        LIMIT {top_n}
    """).df()


@st.cache_data
def get_job_explorer(search: str, label: str) -> pd.DataFrame:
    con = get_db_connection()
    conditions: list[str] = []
    params: list[str | float] = []
    if search:
        conditions.append("LOWER(course_topic) LIKE ?")
        params.append(f"%{search.lower()}%")
    if label != "All":
        conditions.append("opportunity_label = ?")
        params.append(label)
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    return con.execute(f"""
        SELECT rank, course_topic, course_opportunity_score, opportunity_label,
               volume, salary_proxy, breadth_score, trend_30d
        FROM read_parquet('{_r2("topic_rankings.parquet")}')
        {where}
        ORDER BY rank
        LIMIT 200
    """, params).df()


@st.cache_data
def get_label_rollup() -> pd.DataFrame:
    con = get_db_connection()
    return con.execute(f"""
        SELECT *
        FROM read_parquet('{_r2("label_rollup.parquet")}')
        ORDER BY avg_score DESC
    """).df()


@st.cache_data
def get_skill_themes(min_confidence: float) -> pd.DataFrame:
    con = get_db_connection()
    return con.execute(f"""
        SELECT skill, skill_count, ml_theme, ml_confidence
        FROM read_parquet('{_r2("skill_theme_map.parquet")}')
        WHERE ml_confidence >= ?
        ORDER BY skill_count DESC
        LIMIT 300
    """, [min_confidence]).df()


@st.cache_data
def get_theme_counts(min_confidence: float) -> pd.DataFrame:
    con = get_db_connection()
    return con.execute(f"""
        SELECT ml_theme,
               COUNT(*) AS n_skills,
               SUM(skill_count) AS total_mentions
        FROM read_parquet('{_r2("skill_theme_map.parquet")}')
        WHERE ml_confidence >= ?
        GROUP BY ml_theme
        ORDER BY total_mentions DESC
    """, [min_confidence]).df()


@st.cache_data
def get_skill_bundles() -> pd.DataFrame:
    con = get_db_connection()
    return con.execute(f"""
        SELECT skill_a, skill_b, cooccur_count
        FROM read_parquet('{_r2("skill_bundles.parquet")}')
        ORDER BY cooccur_count DESC
    """).df()


@st.cache_data
def get_cooccurrence_pivot(top_n: int) -> pd.DataFrame:
    bundles = get_skill_bundles()
    top_skills = (
        pd.concat([bundles["skill_a"], bundles["skill_b"]])
        .value_counts()
        .head(top_n)
        .index.tolist()
    )
    heat_df = bundles[
        bundles["skill_a"].isin(top_skills) & bundles["skill_b"].isin(top_skills)
    ]
    return heat_df.pivot_table(
        index="skill_a", columns="skill_b", values="cooccur_count", fill_value=0
    )


st.sidebar.header("Filters")
top_n = st.sidebar.slider("Show top N topics", min_value=10, max_value=100, value=50, step=5)
label_sel = st.sidebar.selectbox(
    "Opportunity label",
    ["All", "High Opportunity", "Emerging", "Saturated"],
)
min_conf = st.sidebar.slider("Min ML confidence (skills)", 0.0, 1.0, 0.6, 0.05)

tabs = st.tabs(["Ranked Courses", "Skill Themes (ML)", "Skill Bundles"])

with tabs[0]:
    totals = get_totals()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Job Postings", f"{totals['total_jobs']:,}")
    c2.metric("Total Topics", f"{totals['total_topics']:,}")
    c3.metric("High Opportunity", totals["high_opp"])
    c4.metric("Emerging", totals["emerging"])
    c5.metric("Saturated", totals["saturated"])

    st.divider()

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader(f"Top {top_n} most mentioned skills")
        skills_df = get_top_skills(top_n)
        st.altair_chart(
            alt.Chart(skills_df)
            .mark_bar()
            .encode(
                x=alt.X("skill:N", sort=None, title="Skill"),
                y=alt.Y("skill_count:Q", title="Count"),
            )
            .properties(height=400),
            use_container_width=True,
        )

    with col_r:
        st.subheader(f"Top {top_n} course topics by opportunity score")
        topics_df = get_top_topics_chart(top_n)
        st.altair_chart(
            alt.Chart(topics_df)
            .mark_bar()
            .encode(
                x=alt.X("course_topic:N", sort=None, title="Course Topic"),
                y=alt.Y("course_opportunity_score:Q", title="Opportunity Score"),
            )
            .properties(height=400),
            use_container_width=True,
        )

    st.divider()
    st.subheader("Volume vs salary proxy (top 100 topics)")
    st.scatter_chart(
        get_volume_vs_salary(),
        x="volume",
        y="salary_proxy",
        color="opportunity_label",
        size="breadth_score",
    )

    st.divider()
    st.subheader(f"30-day trend — top {top_n} fastest growing topics")
    st.caption("Growth rate vs prior 30-day window. Requires ≥60 days of posting history in dataset.")
    trend_df = get_trend_top(top_n)
    if trend_df.empty or trend_df["trend_30d"].eq(0).all():
        st.info("No trend signal — dataset lacks sufficient temporal spread for a 30-day comparison.")
    else:
        st.bar_chart(
            trend_df.set_index("course_topic")["trend_30d"].astype(float), height=300
        )

    st.divider()
    st.subheader("Job explorer")
    exp_col1, exp_col2 = st.columns([3, 1])
    search_text = exp_col1.text_input("Search topic", placeholder="e.g. data analyst")
    label_filter = exp_col2.selectbox(
        "Label filter",
        ["All", "High Opportunity", "Emerging", "Saturated"],
        key="explorer_label",
    )
    st.dataframe(
        get_job_explorer(search=search_text, label=label_filter),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("Ranked course topics")
    st.dataframe(
        get_topic_rankings(label=label_sel, top_n=top_n),
        use_container_width=True,
        hide_index=True,
    )
    st.subheader("Label rollup")
    st.dataframe(get_label_rollup(), use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("Skill theme breakdown — total mentions per theme")
    theme_counts = get_theme_counts(min_conf)
    st.bar_chart(
        theme_counts.set_index("ml_theme")["total_mentions"].astype(float),
        height=350,
    )

    st.divider()
    st.subheader("Skill → theme mapping (weak supervision + ML)")
    st.caption(
        "Seed keywords provide high-precision labels. "
        "Char-ngram classifier generalizes to unlabeled skills."
    )
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(get_skill_themes(min_conf), use_container_width=True, hide_index=True)
    with col2:
        st.dataframe(theme_counts, use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("Skill co-occurrence heatmap (top 15 skills)")
    st.dataframe(
        get_cooccurrence_pivot(15).style.background_gradient(cmap="Blues"),
        use_container_width=True,
    )

    st.divider()
    st.subheader("Skill bundling evidence (co-occurrence pairs)")
    st.dataframe(get_skill_bundles(), use_container_width=True, hide_index=True)
    st.caption(
        "Skills like communication, problem solving, and adaptability co-occur as bundles — "
        "supporting bundled course design."
    )

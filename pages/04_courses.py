from __future__ import annotations

import os

import duckdb
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

R2_ENDPOINT_HOST: str = (
    os.environ.get("R2_ENDPOINT_URL", "https://a9e4828e0e2c14c92a0618cded4bf6b6.r2.cloudflarestorage.com")
    .replace("https://", "")
)
R2_BUCKET: str = os.environ.get("R2_BUCKET_NAME", "arusto-skills")


def _get_r2_credentials() -> tuple[str, str]:
    key_id = os.environ.get("R2_ACCESS_KEY_ID") or st.secrets.get("R2_ACCESS_KEY_ID", "")
    secret = os.environ.get("R2_SECRET_ACCESS_KEY") or st.secrets.get("R2_SECRET_ACCESS_KEY", "")
    if not key_id or not secret:
        raise ValueError("R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY must be set in .env or Streamlit secrets")
    return key_id, secret


def _new_db_connection() -> duckdb.DuckDBPyConnection:
    key_id, secret = _get_r2_credentials()
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute(f"""
        SET s3_region='auto';
        SET s3_endpoint='{R2_ENDPOINT_HOST}';
        SET s3_access_key_id='{key_id}';
        SET s3_secret_access_key='{secret}';
        SET s3_url_style='path';
    """)
    return con


@st.cache_resource
def get_db() -> duckdb.DuckDBPyConnection:
    # one connection per Streamlit process; each @st.cache_data query runs sequentially
    # so this is safe — DuckDB connections are not thread-safe
    return _new_db_connection()


def _r2(key: str) -> str:
    return f"s3://{R2_BUCKET}/{key}"


@st.cache_data
def get_totals() -> dict[str, int]:
    con = get_db()
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
def get_top_skills(n: int = 20) -> pd.DataFrame:
    con = get_db()
    return con.execute(f"""
        SELECT skill, skill_count
        FROM read_parquet('{_r2("skill_theme_map.parquet")}')
        ORDER BY skill_count DESC
        LIMIT {n}
    """).df()


@st.cache_data
def get_topic_rankings(label: str = "All", top_n: int = 50) -> pd.DataFrame:
    con = get_db()
    if label == "All":
        return con.execute(f"""
            SELECT rank, course_topic, course_opportunity_score, opportunity_label,
                   volume, salary_proxy, breadth_score, trend_30d
            FROM read_parquet('{_r2("topic_rankings.parquet")}')
            ORDER BY rank
            LIMIT {top_n}
        """).df()
    return con.execute(f"""
        SELECT rank, course_topic, course_opportunity_score, opportunity_label,
               volume, salary_proxy, breadth_score, trend_30d
        FROM read_parquet('{_r2("topic_rankings.parquet")}')
        WHERE opportunity_label = ?
        ORDER BY rank
        LIMIT {top_n}
    """, [label]).df()


@st.cache_data
def get_top_topics_chart(top_n: int = 20) -> pd.DataFrame:
    con = get_db()
    return con.execute(f"""
        SELECT course_topic, course_opportunity_score, opportunity_label
        FROM read_parquet('{_r2("topic_rankings.parquet")}')
        ORDER BY course_opportunity_score DESC
        LIMIT {top_n}
    """).df()


@st.cache_data
def get_volume_vs_salary() -> pd.DataFrame:
    con = get_db()
    return con.execute(f"""
        SELECT course_topic, volume, salary_proxy, breadth_score, opportunity_label
        FROM read_parquet('{_r2("topic_rankings.parquet")}')
        ORDER BY volume DESC
        LIMIT 100
    """).df()


@st.cache_data
def get_trend_top(top_n: int = 15) -> pd.DataFrame:
    con = get_db()
    return con.execute(f"""
        SELECT course_topic, trend_30d
        FROM read_parquet('{_r2("topic_rankings.parquet")}')
        ORDER BY trend_30d DESC
        LIMIT {top_n}
    """).df()


@st.cache_data
def get_job_explorer(search: str = "", label: str = "All") -> pd.DataFrame:
    con = get_db()
    conditions: list[str] = []
    params: list[str] = []
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
    con = get_db()
    return con.execute(f"""
        SELECT *
        FROM read_parquet('{_r2("label_rollup.parquet")}')
        ORDER BY avg_score DESC
    """).df()


@st.cache_data
def get_skill_themes(min_confidence: float = 0.6) -> pd.DataFrame:
    con = get_db()
    return con.execute(f"""
        SELECT skill, skill_count, ml_theme, ml_confidence
        FROM read_parquet('{_r2("skill_theme_map.parquet")}')
        WHERE ml_confidence >= ?
        ORDER BY skill_count DESC
        LIMIT 300
    """, [min_confidence]).df()


@st.cache_data
def get_theme_counts(min_confidence: float = 0.6) -> pd.DataFrame:
    con = get_db()
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
    con = get_db()
    return con.execute(f"""
        SELECT skill_a, skill_b, cooccur_count
        FROM read_parquet('{_r2("skill_bundles.parquet")}')
        ORDER BY cooccur_count DESC
    """).df()


st.set_page_config(page_title="Course Opportunity", layout="wide")
st.title("Course Opportunity Dashboard")
st.caption('Anchor question: "What courses should institutions build next?"')

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
        st.subheader("Top 20 most mentioned skills")
        top_skills = get_top_skills(20)
        st.bar_chart(top_skills.set_index("skill")["skill_count"], height=400)

    with col_r:
        st.subheader("Top 20 course topics by opportunity score")
        top_topics = get_top_topics_chart(20)
        st.bar_chart(
            top_topics.set_index("course_topic")["course_opportunity_score"],
            height=400,
        )

    st.divider()
    st.subheader("Volume vs salary proxy (top 100 topics)")
    st.caption("Bubble size not supported in st.scatter_chart — point = 1 topic")
    scatter_df = get_volume_vs_salary()
    st.scatter_chart(
        scatter_df,
        x="volume",
        y="salary_proxy",
        color="opportunity_label",
        size="breadth_score",
    )

    st.divider()
    st.subheader("30-day trend — fastest growing topics")
    trend_df = get_trend_top(15)
    st.bar_chart(trend_df.set_index("course_topic")["trend_30d"], height=300)

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
        theme_counts.set_index("ml_theme")["total_mentions"],
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
    bundles = get_skill_bundles()
    top_skills_set = (
        pd.concat([bundles["skill_a"], bundles["skill_b"]])
        .value_counts()
        .head(15)
        .index.tolist()
    )
    heat_df = bundles[
        bundles["skill_a"].isin(top_skills_set) & bundles["skill_b"].isin(top_skills_set)
    ].copy()
    pivot = heat_df.pivot_table(
        index="skill_a", columns="skill_b", values="cooccur_count", fill_value=0
    )
    st.dataframe(
        pivot.style.background_gradient(cmap="Blues"),
        use_container_width=True,
    )

    st.divider()
    st.subheader("Skill bundling evidence (co-occurrence pairs)")
    st.dataframe(bundles, use_container_width=True, hide_index=True)
    st.caption(
        "Skills like communication, problem solving, and adaptability co-occur as bundles — "
        "supporting bundled course design."
    )

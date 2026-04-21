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


@st.cache_resource
def get_db() -> duckdb.DuckDBPyConnection:
    key_id = os.environ.get("R2_ACCESS_KEY_ID") or st.secrets.get("R2_ACCESS_KEY_ID", "")
    secret = os.environ.get("R2_SECRET_ACCESS_KEY") or st.secrets.get("R2_SECRET_ACCESS_KEY", "")
    if not key_id or not secret:
        raise ValueError("R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY must be set in .env or Streamlit secrets")
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


def _r2(key: str) -> str:
    return f"s3://{R2_BUCKET}/{key}"


@st.cache_data
def get_topic_rankings(label: str = "All", top_n: int = 50) -> pd.DataFrame:
    con = get_db()
    where = "" if label == "All" else f"WHERE opportunity_label = '{label}'"
    return con.execute(f"""
        SELECT rank, course_topic, course_opportunity_score, opportunity_label,
               volume, salary_proxy, breadth_score, trend_30d
        FROM read_parquet('{_r2("topic_rankings.parquet")}')
        {where}
        ORDER BY rank
        LIMIT {top_n}
    """).df()


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
        WHERE ml_confidence >= {min_confidence}
        ORDER BY skill_count DESC
        LIMIT 300
    """).df()


@st.cache_data
def get_theme_counts(min_confidence: float = 0.6) -> pd.DataFrame:
    con = get_db()
    return con.execute(f"""
        SELECT ml_theme,
               COUNT(*) AS n_skills,
               SUM(skill_count) AS total_mentions
        FROM read_parquet('{_r2("skill_theme_map.parquet")}')
        WHERE ml_confidence >= {min_confidence}
        GROUP BY ml_theme
        ORDER BY total_mentions DESC
    """).df()


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
    st.subheader("Ranked course topics")
    st.dataframe(
        get_topic_rankings(label=label_sel, top_n=top_n),
        use_container_width=True,
        hide_index=True,
    )
    st.subheader("Label rollup")
    st.dataframe(get_label_rollup(), use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("Skill → theme mapping (weak supervision + ML)")
    st.caption(
        "Seed keywords provide high-precision labels. "
        "Char-ngram classifier generalizes to unlabeled skills."
    )
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(get_skill_themes(min_conf), use_container_width=True, hide_index=True)
    with col2:
        st.dataframe(get_theme_counts(min_conf), use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("Skill bundling evidence (co-occurrence pairs)")
    st.dataframe(get_skill_bundles(), use_container_width=True, hide_index=True)
    st.caption(
        "Skills like communication, problem solving, and adaptability co-occur as bundles — "
        "supporting bundled course design."
    )

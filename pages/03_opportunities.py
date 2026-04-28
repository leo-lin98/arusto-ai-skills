from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from components.charts import CATEGORICAL_SCHEME, OPPORTUNITY_COLORS, SEQUENTIAL_SCHEME
from data.db import (
    PARQUET_S3_PATH,
    SKILL_THEME_MAP_S3_PATH,
    TOPIC_RANKINGS_S3_PATH,
    get_db_connection,
)

st.set_page_config(page_title="Course Opportunity", layout="wide")
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
st.title("Course Opportunity Dashboard")
st.caption('Anchor question: "What courses should institutions build next?"')


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
        FROM read_parquet('{TOPIC_RANKINGS_S3_PATH}')
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
        FROM read_parquet('{SKILL_THEME_MAP_S3_PATH}')
        ORDER BY skill_count DESC
        LIMIT {n}
    """).df()


@st.cache_data
def get_topic_rankings(label: str, top_n: int, country: str) -> pd.DataFrame:
    con = get_db_connection()
    conditions: list[str] = []
    params: list[str] = []
    if label != "All":
        conditions.append("r.opportunity_label = ?")
        params.append(label)
    if country != "All":
        conditions.append(
            f"r.course_topic IN ("
            f"SELECT DISTINCT search_position"
            f" FROM read_parquet('{PARQUET_S3_PATH}')"
            f" WHERE search_country = ?)"
        )
        params.append(country)
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    return con.execute(
        f"""
        SELECT r.rank, r.course_topic, r.course_opportunity_score, r.opportunity_label,
               r.volume, r.salary_proxy, r.breadth_score,
               r.trend_slope, r.forecast_4w, r.forecast_12w, r.trend_label
        FROM read_parquet('{TOPIC_RANKINGS_S3_PATH}') r
        {where}
        ORDER BY r.rank
        LIMIT {top_n}
    """,
        params,
    ).df()


@st.cache_data
def get_top_topics_chart(top_n: int) -> pd.DataFrame:
    con = get_db_connection()
    return con.execute(f"""
        SELECT course_topic, course_opportunity_score, opportunity_label
        FROM read_parquet('{TOPIC_RANKINGS_S3_PATH}')
        ORDER BY course_opportunity_score DESC
        LIMIT {top_n}
    """).df()


@st.cache_data
def get_volume_vs_salary() -> pd.DataFrame:
    con = get_db_connection()
    return con.execute(f"""
        SELECT course_topic, volume, salary_proxy, breadth_score, opportunity_label
        FROM read_parquet('{TOPIC_RANKINGS_S3_PATH}')
        ORDER BY volume DESC
        LIMIT 100
    """).df()


@st.cache_data
def get_trend_top(top_n: int) -> pd.DataFrame:
    con = get_db_connection()
    return con.execute(f"""
        SELECT course_topic, trend_slope
        FROM read_parquet('{TOPIC_RANKINGS_S3_PATH}')
        ORDER BY trend_slope DESC
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
    return con.execute(
        f"""
        SELECT rank, course_topic, course_opportunity_score, opportunity_label,
               volume, salary_proxy, breadth_score,
               trend_slope, forecast_4w, forecast_12w, trend_label
        FROM read_parquet('{TOPIC_RANKINGS_S3_PATH}')
        {where}
        ORDER BY rank
        LIMIT 200
    """,
        params,
    ).df()


@st.cache_data
def get_label_rollup() -> pd.DataFrame:
    con = get_db_connection()
    return con.execute(f"""
        SELECT
            opportunity_label,
            COUNT(*) AS n_topics,
            ROUND(AVG(course_opportunity_score), 1) AS avg_score,
            SUM(volume) AS total_postings
        FROM read_parquet('{TOPIC_RANKINGS_S3_PATH}')
        GROUP BY opportunity_label
        ORDER BY avg_score DESC
    """).df()


@st.cache_data
def get_skill_themes(min_confidence: float) -> pd.DataFrame:
    con = get_db_connection()
    return con.execute(
        f"""
        SELECT skill, skill_count, ml_theme, ml_confidence
        FROM read_parquet('{SKILL_THEME_MAP_S3_PATH}')
        WHERE ml_confidence >= ?
        ORDER BY skill_count DESC
        LIMIT 300
    """,
        [min_confidence],
    ).df()


@st.cache_data
def get_theme_counts(min_confidence: float) -> pd.DataFrame:
    con = get_db_connection()
    return con.execute(
        f"""
        SELECT ml_theme,
               COUNT(*) AS n_skills,
               SUM(skill_count) AS total_mentions
        FROM read_parquet('{SKILL_THEME_MAP_S3_PATH}')
        WHERE ml_confidence >= ?
        GROUP BY ml_theme
        ORDER BY total_mentions DESC
    """,
        [min_confidence],
    ).df()


@st.cache_data
def get_countries() -> list[str]:
    con = get_db_connection()
    rows = con.execute(f"""
        SELECT DISTINCT search_country
        FROM read_parquet('{PARQUET_S3_PATH}')
        WHERE search_country != 'Unknown'
        ORDER BY search_country
    """).fetchall()
    return ["All"] + [r[0] for r in rows]


@st.cache_data
def get_topic_theme_mix() -> pd.DataFrame:
    con = get_db_connection()
    return con.execute(f"""
        SELECT
            j.search_position AS course_topic,
            s.ml_theme,
            COUNT(*) AS mention_count
        FROM (
            SELECT search_position,
                   unnest(string_split(skills_norm, ',')) AS skill_raw
            FROM read_parquet('{PARQUET_S3_PATH}')
            WHERE search_position IN (
                SELECT course_topic FROM read_parquet('{TOPIC_RANKINGS_S3_PATH}')
            )
              AND skills_norm IS NOT NULL
              AND skills_norm != ''
        ) j
        JOIN read_parquet('{SKILL_THEME_MAP_S3_PATH}') s
          ON trim(lower(j.skill_raw)) = s.skill
        GROUP BY j.search_position, s.ml_theme
        ORDER BY j.search_position, mention_count DESC
    """).df()


st.sidebar.header("Filters")
top_n = st.sidebar.slider(
    "Show top N topics", min_value=10, max_value=100, value=50, step=5
)
label_sel = st.sidebar.selectbox(
    "Opportunity label",
    ["All", "High Opportunity", "Emerging", "Saturated"],
)
country_sel = st.sidebar.selectbox("Country", get_countries())
min_conf = st.sidebar.slider("Min ML confidence (skills)", 0.0, 1.0, 0.6, 0.05)

tabs = st.tabs(["Ranked Courses", "Skill Themes (ML)"])

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
                color=alt.Color(
                    "skill_count:Q",
                    scale=alt.Scale(scheme=SEQUENTIAL_SCHEME),
                    legend=None,
                ),
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
                color=alt.Color(
                    "opportunity_label:N",
                    scale=alt.Scale(
                        domain=OPPORTUNITY_COLORS["domain"],
                        range=OPPORTUNITY_COLORS["range"],
                    ),
                    title="Label",
                ),
            )
            .properties(height=400),
            use_container_width=True,
        )

    st.divider()
    st.subheader("Volume vs salary proxy (top 100 topics)")
    _scatter_df = get_volume_vs_salary()
    st.altair_chart(
        alt.Chart(_scatter_df)
        .mark_circle()
        .encode(
            x=alt.X("volume:Q", title="Volume"),
            y=alt.Y("salary_proxy:Q", title="Salary Proxy"),
            color=alt.Color(
                "opportunity_label:N",
                scale=alt.Scale(
                    domain=OPPORTUNITY_COLORS["domain"],
                    range=OPPORTUNITY_COLORS["range"],
                ),
                title="Label",
            ),
            size=alt.Size("breadth_score:Q", legend=None),
            tooltip=["course_topic", "volume", "salary_proxy", "opportunity_label"],
        ),
        width="stretch",
    )

    st.divider()
    st.subheader(f"OLS trend slope — top {top_n} fastest growing topics")
    st.caption(
        "Slope (postings/week) from OLS fit. Requires ≥4 distinct weeks in dataset."
    )
    trend_df = get_trend_top(top_n)
    if trend_df.empty or trend_df["trend_slope"].eq(0).all():
        st.info(
            "No trend signal — dataset lacks sufficient temporal spread for OLS fit."
        )
    else:
        st.altair_chart(
            alt.Chart(trend_df)
            .mark_bar()
            .encode(
                x=alt.X("course_topic:N", sort=None, title="Course Topic"),
                y=alt.Y("trend_slope:Q", title="Slope (postings/week)"),
                color=alt.Color(
                    "trend_slope:Q",
                    scale=alt.Scale(scheme=SEQUENTIAL_SCHEME),
                    legend=None,
                ),
            )
            .properties(height=300),
            width="stretch",
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
    ranked_df = get_topic_rankings(label=label_sel, top_n=top_n, country=country_sel)
    st.dataframe(ranked_df, use_container_width=True, hide_index=True)

    if not ranked_df.empty:
        st.subheader("Why these courses? (skill theme mix)")
        mix_df = get_topic_theme_mix()
        shown_topics = set(ranked_df["course_topic"].astype(str).tolist())
        mix_filtered = mix_df[mix_df["course_topic"].astype(str).isin(shown_topics)]
        st.dataframe(mix_filtered.head(300), use_container_width=True, hide_index=True)

    st.subheader("Label rollup")
    st.dataframe(get_label_rollup(), use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("Skill theme breakdown — total mentions per theme")
    theme_counts = get_theme_counts(min_conf)
    st.altair_chart(
        alt.Chart(theme_counts)
        .mark_bar()
        .encode(
            x=alt.X("ml_theme:N", sort=None, title="Theme"),
            y=alt.Y("total_mentions:Q", title="Total Mentions"),
            color=alt.Color(
                "ml_theme:N",
                scale=alt.Scale(scheme=CATEGORICAL_SCHEME),
                legend=None,
            ),
        )
        .properties(height=350),
        width="stretch",
    )

    st.divider()
    st.subheader("Skill → theme mapping (weak supervision + ML)")
    st.caption(
        "Seed keywords provide high-precision labels. "
        "Char-ngram classifier generalizes to unlabeled skills."
    )
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(
            get_skill_themes(min_conf), use_container_width=True, hide_index=True
        )
    with col2:
        st.dataframe(theme_counts, use_container_width=True, hide_index=True)

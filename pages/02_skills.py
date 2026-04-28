from collections import Counter
from itertools import combinations

import altair as alt
import pandas as pd
import streamlit as st

from components.charts import SEQUENTIAL_SCHEME, skills_frequency_chart
from data.db import PARQUET_S3_PATH, SKILL_THEME_MAP_S3_PATH, get_db_connection

st.set_page_config(page_title="Skills", layout="wide")
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
st.title("Skills Analysis")

conn = get_db_connection()

_COOCCURRENCE_SAMPLE = 50_000
_HEATMAP_SKILLS = 15
_THEME_HEATMAP_SAMPLE = 100_000


@st.cache_data
def get_categories(_conn: object) -> list[str]:
    return (
        _conn.execute(
            f"""
            SELECT DISTINCT category FROM read_parquet('{PARQUET_S3_PATH}')
            WHERE category IS NOT NULL ORDER BY category
            """
        )
        .df()["category"]
        .tolist()
    )


@st.cache_data
def get_top_cat_skills(_conn: object, category: str) -> pd.DataFrame:
    conditions = ["category IS NOT NULL"]
    params: list[str] = []
    if category != "All":
        conditions.append("category = ?")
        params.append(category)
    cat_where = f"WHERE {' AND '.join(conditions)}"
    return (
        _conn.execute(
            f"""
        SELECT skill, COUNT(*) AS "Count"
        FROM (
            SELECT TRIM(UNNEST(string_split(skills_norm, ','))) AS skill
            FROM read_parquet('{PARQUET_S3_PATH}')
            {cat_where}
        )
        WHERE skill IS NOT NULL AND TRIM(skill) != '' AND LOWER(TRIM(skill)) != 'nan'
        GROUP BY skill ORDER BY "Count" DESC LIMIT 20
        """,
            params,
        )
        .df()
        .set_index("skill")
    )


@st.cache_data
def get_cooccurrence(_conn: object) -> pd.DataFrame:
    skills_series = _conn.execute(
        f"""
        SELECT skills_norm FROM read_parquet('{PARQUET_S3_PATH}')
        WHERE category IS NOT NULL
        LIMIT {_COOCCURRENCE_SAMPLE}
        """,
    ).df()["skills_norm"]

    pair_counts: Counter = Counter()
    for xs in skills_series:
        if isinstance(xs, list):
            skills = [s.strip() for s in xs if s and s.strip() and s.strip() != "nan"]
        else:
            skills = [
                s.strip()
                for s in str(xs).split(",")
                if s.strip() and s.strip() != "nan"
            ]
        uniq = sorted(set(skills))[:40]
        for a, b in combinations(uniq, 2):
            pair_counts[(a, b)] += 1

    rows = [
        {"skill_a": a, "skill_b": b, "cooccur_count": c}
        for (a, b), c in pair_counts.most_common(100)
    ]
    if not rows:
        return pd.DataFrame(columns=["skill_a", "skill_b", "cooccur_count"])
    return pd.DataFrame(rows)


@st.cache_data
def get_skill_theme_matrix(_conn: object) -> pd.DataFrame:
    raw = _conn.execute(
        f"""
        WITH top_skills AS (
            SELECT skill FROM read_parquet('{SKILL_THEME_MAP_S3_PATH}')
            ORDER BY skill_count DESC LIMIT {_HEATMAP_SKILLS}
        ),
        skill_cat AS (
            SELECT
                TRIM(sk) AS skill,
                category,
                COUNT(*) AS cnt
            FROM (
                SELECT UNNEST(string_split(skills_norm, ',')) AS sk, category
                FROM read_parquet('{PARQUET_S3_PATH}')
                WHERE skills_norm IS NOT NULL AND category IS NOT NULL
                LIMIT {_THEME_HEATMAP_SAMPLE}
            )
            WHERE TRIM(sk) != '' AND LOWER(TRIM(sk)) != 'nan'
            GROUP BY skill, category
        )
        SELECT sc.skill, sc.category, sc.cnt
        FROM skill_cat sc
        JOIN top_skills ts ON sc.skill = ts.skill
        """
    ).df()
    if raw.empty:
        return pd.DataFrame()
    pivot = raw.pivot_table(
        index="skill", columns="category", values="cnt", fill_value=0
    )
    pivot.columns.name = None
    pivot.index.name = None
    return pivot


def _build_pivot(bundles: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if bundles.empty:
        return pd.DataFrame()
    top_skills = (
        pd.concat([bundles["skill_a"], bundles["skill_b"]])
        .value_counts()
        .head(top_n)
        .index.tolist()
    )
    filtered = bundles[
        bundles["skill_a"].isin(top_skills) & bundles["skill_b"].isin(top_skills)
    ]
    if filtered.empty:
        return pd.DataFrame()
    pivot = filtered.pivot_table(
        index="skill_a", columns="skill_b", values="cooccur_count", fill_value=0
    )
    pivot = pivot.add(pivot.T, fill_value=0)
    pivot.columns.name = None
    pivot.index.name = None
    return pivot


st.subheader("Top Skills by Frequency")
n = st.slider("Number of skills", min_value=10, max_value=50, value=25)
skills_frequency_chart(conn, n)

st.divider()

st.subheader("Skills Breakdown by Category")
categories = get_categories(conn)
selected_cat = st.selectbox("Category", ["All"] + categories)
_cat_df = get_top_cat_skills(conn, selected_cat).reset_index()
st.altair_chart(
    alt.Chart(_cat_df)
    .mark_bar()
    .encode(
        x=alt.X("skill:N", sort=None, title="Skill"),
        y=alt.Y("Count:Q", title="Count"),
        color=alt.Color(
            "Count:Q",
            scale=alt.Scale(scheme=SEQUENTIAL_SCHEME),
            legend=None,
        ),
    ),
    width="stretch",
)

st.divider()

st.subheader("Skill Co-occurrence Heatmap")
st.caption(
    f"Top {_HEATMAP_SKILLS} skills by co-occurrence — "
    f"sampled from up to {_COOCCURRENCE_SAMPLE:,} postings."
)
bundles = get_cooccurrence(conn)
pivot = _build_pivot(bundles, _HEATMAP_SKILLS)
if pivot.empty:
    st.info("Not enough co-occurrence data to render heatmap.")
else:
    st.dataframe(
        pivot.style.background_gradient(cmap="Purples"),
        width="stretch",
    )

st.divider()

st.subheader("Skill × Job Category Heatmap")
st.caption(
    f"Top {_HEATMAP_SKILLS} skills (by overall frequency) × job categories — "
    f"sampled from up to {_THEME_HEATMAP_SAMPLE:,} postings."
)
theme_pivot = get_skill_theme_matrix(conn)
if theme_pivot.empty:
    st.info("Not enough data to render skill × category heatmap.")
else:
    st.dataframe(
        theme_pivot.style.background_gradient(cmap="Oranges"),
        width="stretch",
    )

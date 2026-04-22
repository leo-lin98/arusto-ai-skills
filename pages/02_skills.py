from collections import Counter
from itertools import combinations

import pandas as pd
import streamlit as st

from components.charts import skills_frequency_chart
from components.filters import sidebar_filters
from data.db import PARQUET_S3_PATH, filter_conditions, get_db_connection
from data.loader import R2_BUCKET

st.set_page_config(page_title="Skills", layout="wide")
st.title("Skills Analysis")

conn = get_db_connection()
company, location = sidebar_filters(conn)

_SKILL_BUNDLES_PATH = f"s3://{R2_BUCKET}/skill_bundles.parquet"
_COOCCURRENCE_SAMPLE = 50_000


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
def get_top_cat_skills(
    _conn: object, company: str, location: str, category: str
) -> pd.DataFrame:
    conditions, params = filter_conditions(company, location)
    conditions.append("category IS NOT NULL")
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
def get_cooccurrence_precomputed(_conn: object) -> pd.DataFrame:
    rows = _conn.execute(f"""
        SELECT skill_a || ' + ' || skill_b AS skill_pair, cooccur_count AS count
        FROM read_parquet('{_SKILL_BUNDLES_PATH}')
        ORDER BY cooccur_count DESC
        LIMIT 15
    """).df()
    if rows.empty:
        return pd.DataFrame(columns=["count"])
    return rows.set_index("skill_pair")


@st.cache_data
def get_cooccurrence_filtered(
    _conn: object, company: str, location: str
) -> pd.DataFrame:
    conditions, params = filter_conditions(company, location)
    conditions.append("category IS NOT NULL")
    where = f"WHERE {' AND '.join(conditions)}"
    skills_series = _conn.execute(
        f"""
        SELECT skills_norm FROM read_parquet('{PARQUET_S3_PATH}')
        {where}
        LIMIT {_COOCCURRENCE_SAMPLE}
        """,
        params,
    ).df()["skills_norm"]

    pair_counts: Counter = Counter()
    for xs in skills_series:
        if isinstance(xs, list):
            skills = [s.strip() for s in xs if s and s.strip() and s.strip() != "nan"]
        else:
            skills = [
                s.strip() for s in str(xs).split(",")
                if s.strip() and s.strip() != "nan"
            ]
        uniq = sorted(set(skills))[:40]
        for a, b in combinations(uniq, 2):
            pair_counts[(a, b)] += 1

    rows = [
        {"skill_pair": f"{a} + {b}", "count": c}
        for (a, b), c in pair_counts.most_common(15)
    ]
    if not rows:
        return pd.DataFrame(columns=["count"])
    return pd.DataFrame(rows).set_index("skill_pair")


st.subheader("Top Skills by Frequency")
n = st.slider("Number of skills", min_value=10, max_value=50, value=25)
skills_frequency_chart(conn, n, company, location)

st.divider()

st.subheader("Skills Breakdown by Category")
categories = get_categories(conn)
selected_cat = st.selectbox("Category", ["All"] + categories)
st.bar_chart(get_top_cat_skills(conn, company, location, selected_cat))

st.divider()

st.subheader("Skill Co-occurrence Pairs")
if company == "All" and location == "All":
    st.caption("Top 15 pairs from precomputed dataset (1.3M postings).")
    st.bar_chart(get_cooccurrence_precomputed(conn))
else:
    st.caption(
        f"Top 15 pairs sampled from up to {_COOCCURRENCE_SAMPLE:,} filtered postings."
    )
    st.bar_chart(get_cooccurrence_filtered(conn, company, location))

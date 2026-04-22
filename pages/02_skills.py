from collections import Counter
from itertools import combinations

import pandas as pd
import streamlit as st

from components.charts import skills_frequency_chart
from components.filters import sidebar_filters
from data.db import PARQUET_S3_PATH, filter_conditions, get_db_connection

st.set_page_config(page_title="Skills", layout="wide")
st.title("Skills Analysis")

conn = get_db_connection()
company, location = sidebar_filters(conn)

conditions, params = filter_conditions(company, location)
conditions.append("category IS NOT NULL")
where = f"WHERE {' AND '.join(conditions)}"


@st.cache_data
def compute_cooccurrence(skills_series: pd.Series) -> pd.DataFrame:
    pair_counts: Counter = Counter()
    for xs in skills_series:
        skills = [s.strip() for s in str(xs).split(",") if s.strip()]
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
categories = (
    conn.execute(
        f"""
        SELECT DISTINCT category FROM read_parquet('{PARQUET_S3_PATH}')
        WHERE category IS NOT NULL ORDER BY category
        """
    )
    .df()["category"]
    .tolist()
)
selected_cat = st.selectbox("Category", ["All"] + categories)

cat_conditions = list(conditions)
cat_params = list(params)
if selected_cat != "All":
    cat_conditions.append("category = ?")
    cat_params.append(selected_cat)
cat_where = f"WHERE {' AND '.join(cat_conditions)}"

top_cat_skills = (
    conn.execute(
        f"""
    SELECT skill, COUNT(*) AS "Count"
    FROM (
        SELECT TRIM(UNNEST(string_split(skills_norm, ','))) AS skill
        FROM read_parquet('{PARQUET_S3_PATH}')
        {cat_where}
    )
    GROUP BY skill ORDER BY "Count" DESC LIMIT 20
    """,
        cat_params,
    )
    .df()
    .set_index("skill")
)
st.bar_chart(top_cat_skills)

st.divider()

st.subheader("Skill Co-occurrence Pairs")
skills_series = conn.execute(
    f"SELECT skills_norm FROM read_parquet('{PARQUET_S3_PATH}') {where}",
    params,
).df()["skills_norm"]
top_pairs = compute_cooccurrence(skills_series)
st.bar_chart(top_pairs)

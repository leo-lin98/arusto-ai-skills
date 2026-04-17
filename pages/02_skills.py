from collections import Counter
from itertools import combinations

import pandas as pd
import streamlit as st

from components.charts import skills_frequency_chart
from components.filters import sidebar_filters
from data.processor import PARQUET_PATH, SAMPLE_N, build_features, get_merged

st.set_page_config(page_title="Skills", layout="wide")
st.title("Skills Analysis")


@st.cache_data
def load_data() -> pd.DataFrame:
    df = get_merged(PARQUET_PATH, SAMPLE_N)
    return build_features(df)


@st.cache_data
def compute_cooccurrence(skills_series: pd.Series) -> pd.DataFrame:
    pair_counts: Counter = Counter()
    for xs in skills_series:
        uniq = sorted(set(x for x in xs if x))[:40]
        for a, b in combinations(uniq, 2):
            pair_counts[(a, b)] += 1
    return pd.DataFrame(
        [
            {"skill_pair": f"{a} + {b}", "count": c}
            for (a, b), c in pair_counts.most_common(15)
        ]
    ).set_index("skill_pair")


with st.status(
    "Loading data — this may take a few minutes on first run...", expanded=False
):
    df = load_data()

df = df[df["category"].notna()].copy()
df = sidebar_filters(df)

st.subheader("Top Skills by Frequency")
n = st.slider("Number of skills", min_value=10, max_value=50, value=25)
skills_frequency_chart(df, n)

st.divider()

st.subheader("Skills Breakdown by Category")
categories = sorted(df["category"].dropna().unique().tolist())
selected_cat = st.selectbox("Category", ["All"] + categories)

filtered = df if selected_cat == "All" else df[df["category"] == selected_cat]

all_skills = [s for xs in filtered["skills_norm"] for s in xs]
counts = Counter(all_skills)
top_cat_skills = pd.Series(dict(counts.most_common(20)), name="Count")
st.bar_chart(top_cat_skills)

st.divider()

st.subheader("Skill Co-occurrence Pairs")
top_pairs = compute_cooccurrence(df["skills_norm"])
st.bar_chart(top_pairs)

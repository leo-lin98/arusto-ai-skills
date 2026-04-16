import pandas as pd
import plotly.express as px
import streamlit as st

from app.skills import Skills

st.set_page_config(
    page_title="Arusto Skills Taxonomy",
    layout="wide",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv("data/skills_data.csv")


@st.cache_resource
def load_skills() -> Skills:
    return Skills()


def main() -> None:
    df = load_data()
    skills = load_skills()

    # ------------------------------------------------------------------
    # Sidebar filters — add dropdowns here once columns are known
    # ------------------------------------------------------------------
    st.sidebar.header("Filters")
    # example:
    # category = st.sidebar.selectbox("Category", df["category"].unique())

    # ------------------------------------------------------------------
    # Main dashboard
    # ------------------------------------------------------------------
    st.title("Skills Taxonomy Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Features", df.shape[1])
    col3.metric("Models Loaded", len(skills.available_models()))

    st.divider()

    # Add charts below as dashboard takes shape
    # example:
    # fig = px.bar(df, x="skill", y="demand")
    # st.plotly_chart(fig, use_container_width=True)

import streamlit as st

st.set_page_config(
    page_title="Skills in Demand",
    layout="wide",
)

st.title("Skills in Demand Dashboard")
st.markdown("""
Analyze 1.3M LinkedIn job postings to surface which skills are in demand.

Use the sidebar to navigate between pages:

- **01 Overview** — job titles, companies, locations, temporal trends
- **02 Skills** — top skills by frequency and category
- **03 Model** — job category predictor
""")

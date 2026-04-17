import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from data.processor import PARQUET_PATH, SAMPLE_N, build_features, get_merged
from models.predict import load_model, predict_category

st.set_page_config(page_title="Model", layout="wide")
st.title("Job Category Predictor")

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "assets", "models", "baseline.pkl"
)


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


@st.cache_data
def load_test_split() -> tuple[pd.Series, pd.Series]:
    df = get_merged(PARQUET_PATH, SAMPLE_N)
    jobs_eda = build_features(df)
    jobs_eda = jobs_eda[jobs_eda["category"].notna()].copy()
    X = jobs_eda["combined_text"].astype(str)
    y = jobs_eda["category"].astype(str)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test, y_test


if not os.path.exists(MODEL_PATH):
    st.warning(
        "No trained model found. Run `python models/train.py` locally"
        " to generate `assets/models/baseline.pkl`."
    )
    st.stop()

model = get_model()

st.subheader("Predict Job Category")
user_input = st.text_area(
    "Enter a job title or description:",
    placeholder="e.g. Senior Software Engineer with experience in Python and AWS",
    height=120,
)

if st.button("Predict"):
    if user_input.strip():
        prediction = predict_category(model, user_input)
        st.success(f"Predicted category: **{prediction}**")
    else:
        st.error("Please enter a job title or description.")

st.divider()

with st.expander("Model Evaluation", expanded=False):
    with st.spinner("Loading test split..."):
        X_test, y_test = load_test_split()

    final_pred = model.predict(X_test)

    st.markdown("**Classification Report**")
    report = classification_report(y_test, final_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).T.round(2))

    st.markdown("**Confusion Matrix**")
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, final_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cmap="BuPu",
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Best Model)")
    st.pyplot(fig)
    plt.close(fig)

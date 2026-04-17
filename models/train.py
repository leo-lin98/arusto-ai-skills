"""
Run locally to train and save the baseline model.

    python models/train.py

Output: assets/models/baseline.pkl
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from data.processor import PARQUET_PATH, SAMPLE_N, build_features, get_merged

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "assets", "models", "baseline.pkl"
)


def train() -> Pipeline:
    df = get_merged(PARQUET_PATH, SAMPLE_N)
    jobs_eda = build_features(df)
    jobs_eda = jobs_eda[jobs_eda["category"].notna()].copy().reset_index(drop=True)

    X = jobs_eda["combined_text"].astype(str)
    y = jobs_eda["category"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    baseline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.9,
                    max_features=50000,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="saga",
                ),
            ),
        ]
    )

    param_grid = {
        "tfidf__max_features": [20000, 50000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.5, 1.0, 2.0],
    }

    search = GridSearchCV(
        estimator=baseline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    final_pred = best_model.predict(X_test)

    print("Best params:", search.best_params_)
    print("Best CV f1_macro:", search.best_score_)
    print(classification_report(y_test, final_pred))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"Saved: {MODEL_PATH}")

    return best_model


if __name__ == "__main__":
    train()

from __future__ import annotations

import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

W_VOLUME, W_SALARY, W_BREADTH = 0.40, 0.35, 0.25
W_REMOTE, W_HYBRID, W_SENIOR = 0.40, 0.25, 0.35
W_CITY, W_COMPANY = 0.50, 0.50
LABEL_HIGH, LABEL_MID = 60.0, 40.0
MIN_VOLUME = 2_000

SEED_KEYWORDS: dict[str, list[str]] = {
    "Communication & Collaboration": [
        "communication", "written", "verbal", "presentation", "stakeholder",
        "teamwork", "collaboration", "interpersonal", "customer service",
        "client", "documentation",
    ],
    "Problem Solving & Critical Thinking": [
        "problem", "problem solving", "problemsolving", "troubleshooting",
        "debug", "analysis", "critical thinking", "root cause", "decision",
    ],
    "Adaptability & Learning Agility": [
        "adaptability", "flexible", "fast learner", "learning", "change",
    ],
    "Project & Program Management": [
        "project", "program", "agile", "scrum", "planning", "roadmap",
        "coordination", "schedule",
    ],
    "Data & Analytics": [
        "data", "analytics", "excel", "sql", "statistics", "dashboard",
        "power bi", "tableau", "reporting",
    ],
    "Software & Cloud": [
        "python", "java", "javascript", "react", "aws", "azure", "gcp",
        "cloud", "devops", "docker", "kubernetes",
    ],
    "Sales, Marketing & Customer": [
        "sales", "marketing", "business development", "crm", "lead generation",
        "account", "customer success",
    ],
    "Operations & Quality": [
        "operations", "logistics", "supply chain", "inventory", "warehouse",
        "quality", "safety", "compliance",
    ],
    "Leadership & People Management": [
        "leadership", "management", "coaching", "mentoring", "hiring",
        "recruiting", "performance",
    ],
}

THEMES: list[str] = list(SEED_KEYWORDS.keys()) + ["Domain / Other"]


def _norm_text(x: str) -> str:
    x = str(x).strip().lower()
    return re.sub(r"\s+", " ", x)


def parse_skill_list(cell: str) -> list[str]:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    return [s for s in (_norm_text(p) for p in str(cell).split(",")) if s]


def seed_label(skill: str) -> str:
    s = _norm_text(skill)
    for theme, kws in SEED_KEYWORDS.items():
        for kw in kws:
            if kw in s:
                return theme
    return "Domain / Other"


def minmax_norm(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - lo) / (hi - lo) * 100.0


def opportunity_label(score: float) -> str:
    if score >= LABEL_HIGH:
        return "High Opportunity"
    if score >= LABEL_MID:
        return "Emerging"
    return "Saturated"


def load_postings(data_dir: str) -> pd.DataFrame:
    df = pd.read_csv(
        f"{data_dir}/linkedin_job_postings.csv",
        usecols=[
            "job_link", "job_title", "job_type", "job_level",
            "search_position", "search_city", "search_country",
            "company", "first_seen",
        ],
        dtype=str,
    )
    df = df.dropna(subset=["job_link", "search_position"])
    for col in ["job_type", "job_level", "search_city", "search_country", "company", "first_seen", "job_title"]:
        df[col] = df[col].str.strip().fillna("Unknown")
    df["search_position"] = df["search_position"].str.strip()
    df["job_title"] = df["job_title"].str.lower().str.strip()
    df["first_seen"] = pd.to_datetime(df["first_seen"], errors="coerce")
    return df


def load_skills(data_dir: str, job_links: set[str]) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        f"{data_dir}/job_skills.csv",
        usecols=["job_link", "job_skills"],
        chunksize=200_000,
        dtype=str,
    ):
        filtered = chunk[chunk["job_link"].isin(job_links)]
        if not filtered.empty:
            chunks.append(filtered)
    if not chunks:
        return pd.DataFrame(columns=["job_link", "job_skills"])
    return pd.concat(chunks, ignore_index=True)


def aggregate_skills(skills_raw: pd.DataFrame) -> pd.DataFrame:
    # filter NaN at row level before groupby so agg can use the fast C-level ",".join
    clean = skills_raw.dropna(subset=["job_skills"]).copy()
    clean["job_skills"] = (
        clean["job_skills"].str.strip().str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )
    clean = clean[(clean["job_skills"].str.len() > 0) & (clean["job_skills"] != "nan")]
    return (
        clean.groupby("job_link")["job_skills"]
        .agg(",".join)
        .reset_index()
        .rename(columns={"job_skills": "skills_norm"})
    )


def load_summary(data_dir: str, job_links: set[str]) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        f"{data_dir}/job_summary.csv",
        usecols=["job_link", "job_summary"],
        chunksize=200_000,
        dtype=str,
    ):
        filtered = chunk[chunk["job_link"].isin(job_links)]
        if not filtered.empty:
            chunks.append(filtered)
    if not chunks:
        return pd.DataFrame(columns=["job_link", "job_summary"])
    return pd.concat(chunks, ignore_index=True)


def get_merged(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    postings = load_postings(data_dir)
    job_links = set(postings["job_link"])
    # load_skills and load_summary are independent chunked CSV reads — run concurrently
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_skills  = pool.submit(load_skills,  data_dir, job_links)
        fut_summary = pool.submit(load_summary, data_dir, job_links)
        skills_raw = fut_skills.result()
        summary    = fut_summary.result()
    skills_agg = aggregate_skills(skills_raw)
    merged = postings.merge(skills_agg, on="job_link", how="left")
    merged = merged.merge(summary, on="job_link", how="left")
    merged["skills_norm"] = merged["skills_norm"].fillna("")
    merged["job_summary"] = merged["job_summary"].fillna("")
    return merged, skills_raw


def train_skill_theme_model(
    skills_raw: pd.DataFrame, top_n: int = 8000
) -> tuple[TfidfVectorizer, SGDClassifier]:
    freq: Counter = Counter()
    for cell in skills_raw["job_skills"].tolist():
        for sk in parse_skill_list(cell):
            freq[sk] += 1

    rows = [{"skill": s, "skill_count": c} for s, c in freq.most_common(top_n)]
    skills_df = pd.DataFrame(rows)
    skills_df["seed_theme"] = skills_df["skill"].apply(seed_label)
    labeled = skills_df[skills_df["seed_theme"] != "Domain / Other"].copy()

    if len(labeled) < 200:
        raise ValueError(f"Only {len(labeled)} labeled samples — not enough to train. Check SEED_KEYWORDS.")

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)
    X = vec.fit_transform(labeled["skill"].astype(str).tolist())
    clf = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=2000, random_state=42)
    clf.fit(X, labeled["seed_theme"].tolist())
    print(f"Model trained on {len(labeled):,} labeled skills across {len(set(labeled['seed_theme']))} themes.")
    return vec, clf


def build_features(
    merged: pd.DataFrame, vec: TfidfVectorizer, clf: SGDClassifier
) -> pd.DataFrame:
    out = merged.copy()
    out["job_title_len"] = out["job_title"].astype(str).str.len()
    out["n_skills"] = out["skills_norm"].apply(lambda x: len(parse_skill_list(x)))
    out["combined_text"] = (
        out["job_title"].fillna("") + "\n"
        + out["job_summary"].fillna("") + "\nskills: "
        + out["skills_norm"].fillna("")
    )

    X_all = vec.transform(out["search_position"].astype(str).tolist())
    proba = clf.predict_proba(X_all)
    classes = list(clf.classes_)
    out["category"] = [classes[i] for i in proba.argmax(axis=1)]

    # seed labels override ML where available (high precision)
    seed_labels = out["search_position"].apply(seed_label)
    mask = seed_labels != "Domain / Other"
    out.loc[mask, "category"] = seed_labels[mask]

    return out


def score_topics(df: pd.DataFrame) -> pd.DataFrame:
    pos = (
        df.groupby("search_position")
        .agg(
            volume=("job_link", "count"),
            remote_count=("job_type", lambda x: (x == "Remote").sum()),
            hybrid_count=("job_type", lambda x: (x == "Hybrid").sum()),
            senior_count=("job_level", lambda x: (x == "Mid senior").sum()),
            city_count=("search_city", "nunique"),
            company_count=("company", "nunique"),
            country_count=("search_country", "nunique"),
            last_seen=("first_seen", "max"),
        )
        .reset_index()
    )
    pos = pos[pos["volume"] >= MIN_VOLUME].copy()

    pos["remote_rate"] = pos["remote_count"] / pos["volume"]
    pos["hybrid_rate"] = pos["hybrid_count"] / pos["volume"]
    pos["senior_rate"] = pos["senior_count"] / pos["volume"]
    pos["salary_proxy"] = (
        W_REMOTE * pos["remote_rate"]
        + W_HYBRID * pos["hybrid_rate"]
        + W_SENIOR * pos["senior_rate"]
    )

    # min-max scale each metric to [0, 100] before applying weights
    pos["volume_score"] = minmax_norm(pos["volume"])
    pos["salary_score"] = minmax_norm(pos["salary_proxy"])
    pos["breadth_score"] = (
        W_CITY * minmax_norm(pos["city_count"])
        + W_COMPANY * minmax_norm(pos["company_count"])
    )

    pos["course_opportunity_score"] = (
        W_VOLUME * pos["volume_score"]
        + W_SALARY * pos["salary_score"]
        + W_BREADTH * pos["breadth_score"]
    ).round(1)
    pos["opportunity_label"] = pos["course_opportunity_score"].apply(opportunity_label)

    if df["first_seen"].notna().any():
        max_dt = df["first_seen"].max()
        w1 = df[df["first_seen"] >= max_dt - pd.Timedelta(days=30)]
        w0 = df[
            (df["first_seen"] < max_dt - pd.Timedelta(days=30))
            & (df["first_seen"] >= max_dt - pd.Timedelta(days=60))
        ]
        c1 = w1.groupby("search_position")["job_link"].count()
        c0 = w0.groupby("search_position")["job_link"].count()
        # pct change relative to prior window; fillna(0) before ops so sparse Series
        # don't produce NaN for topics present in only one window; +1 avoids div-by-zero
        pct_change = (c1.fillna(0) - c0.fillna(0)) / (c0.fillna(0) + 1)
        pos["trend_30d"] = pos["search_position"].map(pct_change).fillna(0.0).astype(float)
    else:
        pos["trend_30d"] = 0.0

    ranked = pos.sort_values("course_opportunity_score", ascending=False).reset_index(drop=True)
    ranked.index += 1
    ranked.insert(0, "rank", ranked.index)
    return ranked.rename(columns={"search_position": "course_topic"})


def build_label_rollup(topic_rankings: pd.DataFrame) -> pd.DataFrame:
    return (
        topic_rankings.groupby("opportunity_label", as_index=False)
        .agg(
            n_topics=("course_topic", "count"),
            avg_score=("course_opportunity_score", "mean"),
            total_postings=("volume", "sum"),
        )
        .sort_values("avg_score", ascending=False)
        .round(1)
    )


def build_skill_theme_map(
    skills_raw: pd.DataFrame,
    vec: TfidfVectorizer,
    clf: SGDClassifier,
    top_n: int = 5000,
) -> pd.DataFrame:
    freq: Counter = Counter()
    for cell in skills_raw["job_skills"].tolist():
        for sk in parse_skill_list(cell):
            freq[sk] += 1

    rows = [{"skill": s, "skill_count": c} for s, c in freq.most_common(top_n)]
    df = pd.DataFrame(rows)

    X_all = vec.transform(df["skill"].astype(str).tolist())
    proba = clf.predict_proba(X_all)
    classes = list(clf.classes_)
    df["ml_theme"] = [classes[i] for i in proba.argmax(axis=1)]
    df["ml_confidence"] = proba.max(axis=1).round(4)

    seed_labels = df["skill"].apply(seed_label)
    mask = seed_labels != "Domain / Other"
    df.loc[mask, "ml_theme"] = seed_labels[mask]
    df.loc[mask, "ml_confidence"] = 1.0

    return df.sort_values("skill_count", ascending=False)


def build_skill_bundle_pairs(
    skills_raw: pd.DataFrame,
    sample_rows: int = 200_000,
    top_pairs: int = 250,
) -> pd.DataFrame:
    pair_counts: Counter = Counter()
    for i, cell in enumerate(skills_raw["job_skills"].tolist()):
        if i >= sample_rows:
            break
        skills = sorted(set(parse_skill_list(cell)))
        if len(skills) < 2:
            continue
        pair_counts.update(combinations(skills[:30], 2))
    return pd.DataFrame(
        [{"skill_a": a, "skill_b": b, "cooccur_count": c}
         for (a, b), c in pair_counts.most_common(top_pairs)]
    )

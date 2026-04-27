from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

IS_NOISE_SKILL_VERSION: int = (
    1  # bump when noise patterns change to force pipeline re-run
)

W_VOLUME, W_SALARY, W_BREADTH = 0.40, 0.35, 0.25
W_REMOTE, W_HYBRID, W_SENIOR = 0.40, 0.25, 0.35
W_CITY, W_COMPANY = 0.50, 0.50
LABEL_HIGH, LABEL_MID = 60.0, 40.0
MIN_VOLUME = 2_000
# TODO: workshop R2_THRESHOLD and SLOPE_THRESHOLD against real data percentiles
R2_THRESHOLD: float = 0.30
SLOPE_THRESHOLD: float = 1.0
MIN_TREND_WEEKS: int = 4

SEED_KEYWORDS: dict[str, list[str]] = {
    "Communication & Collaboration": [
        "communication",
        "written",
        "verbal",
        "presentation",
        "stakeholder",
        "teamwork",
        "collaboration",
        "interpersonal",
        "customer service",
        "client",
        "documentation",
    ],
    "Problem Solving & Critical Thinking": [
        "problem",
        "problem solving",
        "problemsolving",
        "troubleshooting",
        "debug",
        "analysis",
        "critical thinking",
        "root cause",
        "decision",
    ],
    "Adaptability & Learning Agility": [
        "adaptability",
        "flexible",
        "fast learner",
        "learning",
        "change",
    ],
    "Project & Program Management": [
        "project",
        "program",
        "agile",
        "scrum",
        "planning",
        "roadmap",
        "coordination",
        "schedule",
    ],
    "Data & Analytics": [
        "data",
        "analytics",
        "excel",
        "sql",
        "statistics",
        "dashboard",
        "power bi",
        "tableau",
        "reporting",
    ],
    "Software & Cloud": [
        "python",
        "java",
        "javascript",
        "react",
        "aws",
        "azure",
        "gcp",
        "cloud",
        "devops",
        "docker",
        "kubernetes",
    ],
    "Sales, Marketing & Customer": [
        "sales",
        "marketing",
        "business development",
        "crm",
        "lead generation",
        "account",
        "customer success",
    ],
    "Operations & Quality": [
        "operations",
        "logistics",
        "supply chain",
        "inventory",
        "warehouse",
        "quality",
        "safety",
        "compliance",
    ],
    "Leadership & People Management": [
        "leadership",
        "management",
        "coaching",
        "mentoring",
        "hiring",
        "recruiting",
        "performance",
    ],
}

THEMES: list[str] = list(SEED_KEYWORDS.keys()) + ["Domain / Other"]


def pipeline_config_hash() -> str:
    config = {
        "W_VOLUME": W_VOLUME,
        "W_SALARY": W_SALARY,
        "W_BREADTH": W_BREADTH,
        "W_REMOTE": W_REMOTE,
        "W_HYBRID": W_HYBRID,
        "W_SENIOR": W_SENIOR,
        "W_CITY": W_CITY,
        "W_COMPANY": W_COMPANY,
        "LABEL_HIGH": LABEL_HIGH,
        "LABEL_MID": LABEL_MID,
        "MIN_VOLUME": MIN_VOLUME,
        "R2_THRESHOLD": R2_THRESHOLD,
        "SLOPE_THRESHOLD": SLOPE_THRESHOLD,
        "MIN_TREND_WEEKS": MIN_TREND_WEEKS,
        "SEED_KEYWORDS": SEED_KEYWORDS,
        "IS_NOISE_SKILL_VERSION": IS_NOISE_SKILL_VERSION,
    }
    return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()


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


_BENEFITS_NOISE_PAT = re.compile(
    r"(\bpaid time off\b|\bpaid leave\b|\bleave(s)? of absence\b|\bunlimited pto\b|\bpto\b|\b401k\b|\b401\(k\)\b|"
    r"\bhealth insurance\b|\bhealth coverage\b|\bbenefits?\b|\bbenefits package\b|"
    r"\bdental\b|\bvision\b|\bvacation\b|\bholiday\b|\bholidays\b|\bsick( time)?\b|\bparental leave\b|"
    r"\bannual leave\b|\bdonor leave\b|"
    r"\bfamily leave\b|\bmedical leave\b|\bmaternity leave\b|\bpaternity leave\b|\bprofit\s*sharing\b|"
    r"\breferral bonus\b|\bsign[-\s]?on bonus\b|\bbonus\b|\bbonuses\b|\bcommission(s|ed)?\b|"
    r"\bpaid volunteering\b|\bvolunteer(ing)? days?\b|\btuition (reimbursement|assistance)\b|\beducation reimbursement\b|"
    r"\breimbursement\b|\ballowance\b|\bshift differentials?\b|\bovertime\b|\bcompetitive pay\b|\bhourly pay\b)",
    flags=re.IGNORECASE,
)

_DASHBOARD_NOISE_PAT = re.compile(
    r"(^oil\s*&\s*gas$|^oil\s+and\s+gas$|^blood banking$|trauma center|^heating$|^online application$|"
    r"uniforms?|24\s*/\s*7\s*/\s*365\s+support|^fun$|fun and enthusiastic personality)",
    flags=re.IGNORECASE,
)

_USER_REVIEWED_NOISE_PAT = re.compile(
    r"(^skills$|standard office equipment|employee discount program|^days$|^driven$|^dust$|"
    r"\bdiabetes\b|\bcovid[-\s]?19 vaccine\b|\bblood draws?\b|\busda accreditation\b|lessons learned|"
    r"\bchemical dependency\b|\bhazardous chemicals?\b|\btoxic chemicals?\b|\bexposure to irritant chemicals\b|"
    r"\birritant chemicals?\b|\bchemical\b|\bfirearms?\b|"
    r"long service awards?|\bonesite\b|^coagulation$)",
    flags=re.IGNORECASE,
)

_PAID_PERK_TOKENS: set[str] = {
    "housing",
    "relocation",
    "travel",
    "training",
    "orientation",
    "malpractice",
    "compliance",
    "cost",
    "costs",
    "license",
    "licensure",
    "certification",
    "tuition",
    "reimbursement",
    "coverage",
    "insurance",
    "meals",
    "uniform",
    "phone",
    "laptop",
}

_EQUIPMENT_OBJECT_KEYWORDS: set[str] = {
    "toaster",
    "toaster oven",
    "microwave",
    "blender",
    "dishwasher",
    "stove",
    "oven",
    "refrigerator",
    "freezer",
    "laptop",
    "iphone",
    "ipad",
    "android",
}

_EQUIPMENT_ACTION_KEYWORDS: set[str] = {
    "operate",
    "operation",
    "operating",
    "use",
    "using",
    "setup",
    "install",
    "installation",
    "configure",
    "configuration",
    "maintain",
    "maintenance",
    "repair",
    "troubleshoot",
    "calibrate",
    "clean",
    "cleaning",
    "sanitize",
    "sanitization",
}


def is_noise_skill(skill: str) -> bool:
    s = _norm_text(skill)
    if not s or len(s) <= 2:
        return True
    if "insurance" in s:
        return True
    if "coverage" in s and "test coverage" not in s:
        return True
    if "reimbursement" in s:
        return True
    if "leave of absence" in s or "leaves of absence" in s:
        return True
    if s.startswith("paid ") and any(
        t in s
        for t in [
            "leave",
            "time off",
            "holiday",
            "vacation",
            "volunteer",
            "coverage",
            "benefit",
        ]
    ):
        return True
    if s.startswith("paid "):
        tail = s.removeprefix("paid ").strip()
        if any(tok in tail for tok in _PAID_PERK_TOKENS):
            return True
        if any(
            tok in s
            for tok in [
                "cost",
                "costs",
                "housing",
                "travel",
                "training",
                "malpractice",
                "compliance",
            ]
        ):
            return True
    if any(k in s for k in _EQUIPMENT_OBJECT_KEYWORDS) and not any(
        a in s for a in _EQUIPMENT_ACTION_KEYWORDS
    ):
        return True
    if any(t in s for t in ["contract", "w2", "1099"]):
        return True
    if "shift" in s:
        return True
    if "schedule" in s:
        return True
    if "hourly wage" in s or "hourly wages" in s:
        return True
    if re.search(r"\b\d{1,2}\s*hours\s*per\s*week\b", s):
        return True
    if "citizenship" in s or "permanent residency" in s or "work authorization" in s:
        return True
    if "residency" in s:
        return True
    if "sexual orientation" in s or "gender identity" in s:
        return True
    if (
        re.search(r"\b\d+\+?\s*years?\s+(of\s+)?experience\b", s)
        or "year experience" in s
    ):
        return True
    if (
        re.search(r"\b\d+\s*years?\s+of\s+age\b", s)
        or re.search(r"\b\d+\s*years?\s+old\b", s)
        or "years of age" in s
    ):
        return True
    if (
        re.search(r"\b\d+\s*week\b", s)
        or re.search(r"\b\d+week\b", s)
        or "week duration" in s
    ):
        return True
    if any(t in s for t in ["referral", "relocation"]):
        return True
    if _DASHBOARD_NOISE_PAT.search(s):
        return True
    if _USER_REVIEWED_NOISE_PAT.search(s):
        return True
    if "clearance" in s or "fbi" in s:
        return True
    if s.startswith("adn "):
        return True
    if "safety requirements" in s:
        return True
    if "fleet program" in s:
        return True
    if s == "enthusiastic":
        return True
    if "cash control policies" in s:
        return True
    if s == "transfers":
        return True
    if re.search(
        r"\b("
        r"diversity|inclusion|equity and inclusion|dei|de&i|equal opportunity|eeo|affirmative action|"
        r"protected class|minority status|underrepresented|belonging\b|"
        r"sexual orientation|gender identity|lgbtq?\+?|lgbt\b|"
        r"race\b|ethnicity\b|national origin|religion\b|faith\b|"
        r"marriage\b|married\b|single\b|divorce\b|domestic partner\b|"
        r"pregnancy\b|disability\b|disabled\b|veteran status|military status|"
        r"height\b|weight\b|bmi\b|body mass|"
        r"discrimination\b|harassment\b"
        r")\b",
        s,
    ):
        return True
    return bool(_BENEFITS_NOISE_PAT.search(s))


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


def _linear_trend_for_position(
    dates: pd.Series,
    forecast_weeks: int,
    min_weeks: int,
    r2_threshold: float,
    slope_threshold: float,
) -> dict[str, object]:
    null_result: dict[str, object] = {
        "slope": 0.0,
        "r2": 0.0,
        "forecast": np.nan,
        "trend_label": "Insufficient data",
    }
    dates = dates.dropna()
    if len(dates) < min_weeks * 3:
        return null_result

    week_series = dates.dt.to_period("W").value_counts().sort_index()
    if len(week_series) < min_weeks:
        return null_result

    x = np.arange(len(week_series), dtype=float)
    y = week_series.values.astype(float)
    x_mean, y_mean = x.mean(), y.mean()
    ss_xx = ((x - x_mean) ** 2).sum()
    if ss_xx == 0:
        return null_result

    slope = ((x - x_mean) * (y - y_mean)).sum() / ss_xx
    intercept = y_mean - slope * x_mean
    y_pred = slope * x + intercept
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y_mean) ** 2).sum()
    r2 = float(np.clip(1 - ss_res / ss_tot if ss_tot > 0 else 0.0, 0.0, 1.0))

    last_x = x[-1]
    forecast = sum(
        max(0.0, slope * (last_x + k) + intercept) for k in range(1, forecast_weeks + 1)
    )

    if slope > slope_threshold and r2 >= r2_threshold:
        label = "Growing"
    elif slope < -slope_threshold and r2 >= r2_threshold:
        label = "Declining"
    else:
        label = "Stable"

    return {
        "slope": round(float(slope), 3),
        "r2": round(r2, 3),
        "forecast": round(forecast, 1),
        "trend_label": label,
    }


def add_trend_forecast(
    pos: pd.DataFrame,
    postings: pd.DataFrame,
    forecast_weeks: int,
    r2_threshold: float,
    slope_threshold: float,
) -> pd.DataFrame:
    forecast_col = f"forecast_{forecast_weeks}w"

    if not postings["first_seen"].notna().any():
        pos["trend_slope"] = 0.0
        pos["trend_r2"] = 0.0
        pos[forecast_col] = np.nan
        pos["trend_label"] = "Insufficient data"
        return pos

    qualifying = set(pos["search_position"].tolist())
    subset = postings[
        postings["search_position"].isin(qualifying) & postings["first_seen"].notna()
    ][["search_position", "first_seen"]]

    records = []
    for position, grp in subset.groupby("search_position"):
        result = _linear_trend_for_position(
            grp["first_seen"],
            forecast_weeks=forecast_weeks,
            min_weeks=MIN_TREND_WEEKS,
            r2_threshold=r2_threshold,
            slope_threshold=slope_threshold,
        )
        result["search_position"] = position
        records.append(result)

    if not records:
        pos["trend_slope"] = 0.0
        pos["trend_r2"] = 0.0
        pos[forecast_col] = np.nan
        pos["trend_label"] = "Insufficient data"
        return pos

    trend_df = pd.DataFrame(records).rename(
        columns={
            "slope": "trend_slope",
            "r2": "trend_r2",
            "forecast": forecast_col,
        }
    )
    pos = pos.merge(trend_df, on="search_position", how="left")
    pos["trend_slope"] = pos["trend_slope"].fillna(0.0)
    pos["trend_r2"] = pos["trend_r2"].fillna(0.0)
    pos["trend_label"] = pos["trend_label"].fillna("Insufficient data")
    return pos


def load_postings(data_dir: str) -> pd.DataFrame:
    df = pd.read_csv(
        f"{data_dir}/linkedin_job_postings.csv",
        usecols=[
            "job_link",
            "job_title",
            "job_type",
            "job_level",
            "search_position",
            "search_city",
            "search_country",
            "company",
            "first_seen",
        ],
        dtype=str,
    )
    df = df.dropna(subset=["job_link", "search_position"])
    for col in [
        "job_type",
        "job_level",
        "search_city",
        "search_country",
        "company",
        "first_seen",
        "job_title",
    ]:
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
        clean["job_skills"].str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    )
    clean = clean[(clean["job_skills"].str.len() > 0) & (clean["job_skills"] != "nan")]
    clean = clean[~clean["job_skills"].apply(is_noise_skill)]
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
        fut_skills = pool.submit(load_skills, data_dir, job_links)
        fut_summary = pool.submit(load_summary, data_dir, job_links)
        skills_raw = fut_skills.result()
        summary = fut_summary.result()
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
            if not is_noise_skill(sk):
                freq[sk] += 1

    rows = [{"skill": s, "skill_count": c} for s, c in freq.most_common(top_n)]
    skills_df = pd.DataFrame(rows)
    skills_df["seed_theme"] = skills_df["skill"].apply(seed_label)
    labeled = skills_df[skills_df["seed_theme"] != "Domain / Other"].copy()

    if len(labeled) < 200:
        raise ValueError(
            f"Only {len(labeled)} labeled samples — not enough to train. Check SEED_KEYWORDS."
        )

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)
    X = vec.fit_transform(labeled["skill"].astype(str).tolist())
    clf = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=2000, random_state=42)
    clf.fit(X, labeled["seed_theme"].tolist())
    print(
        f"Model trained on {len(labeled):,} labeled skills across {len(set(labeled['seed_theme']))} themes."
    )
    return vec, clf


def build_features(
    merged: pd.DataFrame, vec: TfidfVectorizer, clf: SGDClassifier
) -> pd.DataFrame:
    out = merged.copy()
    out["job_title_len"] = out["job_title"].astype(str).str.len()
    out["n_skills"] = out["skills_norm"].apply(lambda x: len(parse_skill_list(x)))
    out["combined_text"] = (
        out["job_title"].fillna("")
        + "\n"
        + out["job_summary"].fillna("")
        + "\nskills: "
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
    pos["log_volume"] = np.log1p(pos["volume"])
    pos["volume_score"] = minmax_norm(pos["log_volume"])
    pos["salary_score"] = minmax_norm(pos["salary_proxy"])
    pos["breadth_score"] = W_CITY * minmax_norm(
        pos["city_count"]
    ) + W_COMPANY * minmax_norm(pos["company_count"])

    pos["course_opportunity_score"] = (
        W_VOLUME * pos["volume_score"]
        + W_SALARY * pos["salary_score"]
        + W_BREADTH * pos["breadth_score"]
    ).round(1)
    pos["opportunity_label"] = pos["course_opportunity_score"].apply(opportunity_label)

    if not df["first_seen"].notna().any():
        pos = pos.assign(
            trend_slope=0.0,
            trend_r2=0.0,
            forecast_4w=np.nan,
            forecast_12w=np.nan,
            trend_label="Insufficient data",
        )
    else:
        qualifying = set(pos["search_position"].tolist())
        subset = df[df["search_position"].isin(qualifying) & df["first_seen"].notna()][
            ["search_position", "first_seen"]
        ]
        records = []
        for position, grp in subset.groupby("search_position"):
            r4 = _linear_trend_for_position(
                grp["first_seen"],
                forecast_weeks=4,
                min_weeks=MIN_TREND_WEEKS,
                r2_threshold=R2_THRESHOLD,
                slope_threshold=SLOPE_THRESHOLD,
            )
            r12 = _linear_trend_for_position(
                grp["first_seen"],
                forecast_weeks=12,
                min_weeks=MIN_TREND_WEEKS,
                r2_threshold=R2_THRESHOLD,
                slope_threshold=SLOPE_THRESHOLD,
            )
            records.append(
                {
                    "search_position": position,
                    "trend_slope": r12["slope"],
                    "trend_r2": r12["r2"],
                    "forecast_4w": r4["forecast"],
                    "forecast_12w": r12["forecast"],
                    "trend_label": r12["trend_label"],
                }
            )
        if records:
            trend_df = pd.DataFrame(records)
            pos = pos.merge(trend_df, on="search_position", how="left")
            pos["trend_slope"] = pos["trend_slope"].fillna(0.0)
            pos["trend_r2"] = pos["trend_r2"].fillna(0.0)
            pos["trend_label"] = pos["trend_label"].fillna("Insufficient data")
        else:
            pos = pos.assign(
                trend_slope=0.0,
                trend_r2=0.0,
                forecast_4w=np.nan,
                forecast_12w=np.nan,
                trend_label="Insufficient data",
            )

    ranked = pos.sort_values("course_opportunity_score", ascending=False).reset_index(
        drop=True
    )
    ranked.index += 1
    ranked.insert(0, "rank", ranked.index)
    return ranked.rename(columns={"search_position": "course_topic"})


def build_skill_theme_map(
    skills_raw: pd.DataFrame,
    vec: TfidfVectorizer,
    clf: SGDClassifier,
    top_n: int = 5000,
) -> pd.DataFrame:
    freq: Counter = Counter()
    for cell in skills_raw["job_skills"].tolist():
        for sk in parse_skill_list(cell):
            if not is_noise_skill(sk):
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

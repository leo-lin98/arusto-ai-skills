import re

import pandas as pd

DATA_DIR = "/tmp/data"
PARQUET_PATH = f"{DATA_DIR}/merged.parquet"
POSTINGS_PATH = f"{DATA_DIR}/linkedin_job_postings.csv"
SKILLS_PATH = f"{DATA_DIR}/job_skills.csv"
SUMMARY_PATH = f"{DATA_DIR}/job_summary.csv"
SAMPLE_N = 200_000

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "TECHNOLOGY": [
        "software engineer",
        "software developer",
        "backend",
        "frontend",
        "full stack",
        "fullstack",
        "devops",
        "cloud engineer",
        "platform engineer",
        "site reliability",
        "mobile developer",
        "ios developer",
        "android developer",
        "systems engineer",
        "network engineer",
        "security engineer",
        "cybersecurity",
        "infrastructure engineer",
        "embedded",
        "firmware",
        "it engineer",
        "it specialist",
        "programmer",
    ],
    "DATA-ANALYTICS": [
        "data analyst",
        "data scientist",
        "data engineer",
        "analytics engineer",
        "business analyst",
        "business intelligence",
        "bi developer",
        "reporting analyst",
        "quantitative analyst",
        "research analyst",
        "data architect",
        "machine learning engineer",
        "ml engineer",
        "ai engineer",
        "nlp engineer",
        "data warehouse",
    ],
    "MARKETING": [
        "marketing manager",
        "marketing specialist",
        "marketing coordinator",
        "seo",
        "sem",
        "content strategist",
        "content writer",
        "brand manager",
        "digital marketing",
        "social media manager",
        "growth manager",
        "campaign manager",
        "communications manager",
        "public relations",
        "copywriter",
        "creative director",
        "email marketing",
    ],
    "SALES": [
        "sales representative",
        "sales rep",
        "account executive",
        "business development representative",
        "bdr",
        "sdr",
        "sales development",
        "sales engineer",
        "solutions engineer",
        "inside sales",
        "outside sales",
        "territory manager",
        "regional sales manager",
        "sales manager",
        "sales director",
    ],
    "FINANCE": [
        "financial analyst",
        "finance analyst",
        "accountant",
        "accounting manager",
        "controller",
        "fp&a",
        "financial planning",
        "auditor",
        "tax analyst",
        "treasury analyst",
        "investment analyst",
        "portfolio manager",
        "risk analyst",
        "credit analyst",
        "actuary",
        "finance manager",
        "chief financial",
    ],
    "HR-OPERATIONS": [
        "human resources",
        "hr manager",
        "hr generalist",
        "hr business partner",
        "hr coordinator",
        "recruiter",
        "talent acquisition",
        "people operations",
        "operations manager",
        "operations analyst",
        "supply chain",
        "logistics manager",
        "procurement",
        "warehouse manager",
        "fulfillment manager",
        "compensation",
        "payroll",
    ],
    "PRODUCT-DESIGN": [
        "product manager",
        "product owner",
        "ux designer",
        "ui designer",
        "ux researcher",
        "product designer",
        "user experience",
        "user interface",
        "interaction designer",
        "visual designer",
        "graphic designer",
        "product lead",
        "head of product",
    ],
    "CUSTOMER-SUCCESS": [
        "customer success",
        "customer support",
        "customer service manager",
        "client success",
        "client services",
        "support engineer",
        "technical support",
        "help desk",
        "customer experience",
        "service desk",
        "cx manager",
    ],
}


def load_postings(path: str, sample_n: int) -> pd.DataFrame:
    # nrows = first N rows, not random — CSV is chronological (recency bias).
    df = pd.read_csv(path, nrows=sample_n)
    df["job_title"] = df["job_title"].astype(str).str.lower().str.strip()
    df["company"] = df["company"].fillna("Unknown Company")
    df["job_location"] = df["job_location"].fillna("Unknown Location")
    df = df.drop(columns=["got_summary", "got_ner", "is_being_worked"], errors="ignore")
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "job_link"])
    processed = pd.to_datetime(
        df["last_processed_time"], dayfirst=True, format="mixed", utc=True
    )
    df["date"] = processed.dt.date
    df["hour"] = processed.dt.hour
    df["day"] = processed.dt.day
    df["day_of_week"] = processed.dt.day_name()
    df = df.drop(columns=["last_processed_time"])
    return df


def aggregate_skills(path: str, job_links: set[str]) -> pd.DataFrame:
    chunks = []
    for chunk in pd.read_csv(path, chunksize=100_000):
        filtered = chunk[chunk["job_link"].isin(job_links)]
        if not filtered.empty:
            chunks.append(filtered)
    skills_raw = (
        pd.concat(chunks, ignore_index=True)
        if chunks
        else pd.DataFrame(columns=["job_link", "job_skills"])
    )
    return (
        skills_raw.groupby("job_link")["job_skills"]
        .apply(list)
        .reset_index()
        .rename(columns={"job_skills": "skills_list"})
    )


def load_summary(path: str, job_links: set[str]) -> pd.DataFrame:
    chunks = []
    for chunk in pd.read_csv(
        path, usecols=["job_link", "job_summary"], chunksize=100_000
    ):
        filtered = chunk[chunk["job_link"].isin(job_links)]
        if not filtered.empty:
            chunks.append(filtered)
    return (
        pd.concat(chunks, ignore_index=True)
        if chunks
        else pd.DataFrame(columns=["job_link", "job_summary"])
    )


def merge_datasets(
    postings: pd.DataFrame,
    skills_agg: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    df = postings.merge(skills_agg, on="job_link", how="left")
    df = df.merge(summary, on="job_link", how="left")
    df["skills_list"] = df["skills_list"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    df["job_summary"] = df["job_summary"].fillna("")
    return df


def derive_category(title: str) -> str | None:
    t = title.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            return category
    return None


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in ["job_title", "job_summary"]:
        out[f"{col}_len"] = out[col].astype(str).str.len()
        out[f"{col}_word_count"] = out[col].astype(str).str.split().str.len()

    out["n_skills"] = out["skills_list"].apply(len)

    skills_text = out["skills_list"].apply(
        lambda xs: " ".join(str(x).strip() for x in xs if str(x).strip())
    )
    out["combined_text"] = (
        out["job_title"].fillna("")
        + "\n"
        + out["job_summary"].fillna("")
        + "\nskills: "
        + skills_text
    )

    out["skills_norm"] = out["skills_list"].apply(
        lambda xs: [
            re.sub(r"\s+", " ", str(x).strip()).lower() for x in xs if str(x).strip()
        ]
    )

    out["category"] = out["job_title"].apply(derive_category)

    return out

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

IS_NOISE_SKILL_VERSION: int = (
    2  # bump when noise patterns change to force pipeline re-run
)

_NOISE_JOB_ROLES: set[str] = {
    "river",
    "extra",
    "orderly",
    "olderly",
    "change person",
    "principle",
    "model",
    "page",
    "stand-in",
    "stand in",
    "buyer",
    "consultant education",
    "manager benefits",
    "nurse infection control",
    "nurse office",
    "nurse staff community health",
    "manager records analysis",
    "nurse school",
}


def _norm_role(x: str) -> str:
    return re.sub(r"\s+", " ", str(x).strip().lower())


W_VOLUME, W_SALARY, W_BREADTH = 0.40, 0.35, 0.25
W_REMOTE, W_HYBRID, W_SENIOR = 0.40, 0.25, 0.35
W_CITY, W_COMPANY = 0.50, 0.50
LABEL_HIGH, LABEL_MID = 60.0, 40.0
MIN_VOLUME = 2_000

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
        "SEED_KEYWORDS": SEED_KEYWORDS,
        "IS_NOISE_SKILL_VERSION": IS_NOISE_SKILL_VERSION,
        "_NOISE_JOB_ROLES": sorted(_NOISE_JOB_ROLES),
    }
    return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()


def _norm_text(x: str) -> str:
    x = str(x).strip().lower()
    return re.sub(r"\s+", " ", x)


_US_STATE_ABBR: set[str] = {
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
    "DC",
}

_US_STATE_NAMES: set[str] = {
    "Alabama",
    "Alaska",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "Florida",
    "Georgia",
    "Hawaii",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming",
    "District of Columbia",
}

_CA_PROVINCES: set[str] = {
    "Alberta",
    "British Columbia",
    "Manitoba",
    "New Brunswick",
    "Newfoundland and Labrador",
    "Nova Scotia",
    "Ontario",
    "Prince Edward Island",
    "Quebec",
    "Saskatchewan",
    "Northwest Territories",
    "Nunavut",
    "Yukon",
}

_AU_STATES: set[str] = {
    "New South Wales",
    "Victoria",
    "Queensland",
    "South Australia",
    "Western Australia",
    "Tasmania",
    "Northern Territory",
    "Australian Capital Territory",
}

_UK_NATIONS: set[str] = {"England", "Scotland", "Wales", "Northern Ireland"}


def _clean_loc_token(x: object) -> str:
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none"}:
        return ""
    return re.sub(r"\s+", " ", s)


def parse_job_location(
    job_location: object, countries: set[str]
) -> tuple[str, str, str]:
    raw = _clean_loc_token(job_location)
    if not raw:
        return "", "", ""
    parts = [_clean_loc_token(p) for p in raw.split(",")]
    parts = [p for p in parts if p]
    if not parts:
        return "", "", ""
    country = ""
    if parts and parts[-1] in countries:
        country = parts.pop(-1)
    if not parts:
        return "", "", country
    if len(parts) == 1:
        tok = parts[0]
        if tok in _US_STATE_NAMES:
            if not country:
                country = "United States"
            return "", tok, country
        if tok in _CA_PROVINCES and not country:
            country = "Canada"
        if tok in _AU_STATES and not country:
            country = "Australia"
        if tok in _UK_NATIONS and not country:
            country = "United Kingdom"
        return tok, "", country
    city = parts[0]
    tail = parts[-1]
    if tail.upper() in _US_STATE_ABBR:
        if not country:
            country = "United States"
        return city, tail.upper(), country
    if tail in _US_STATE_NAMES and not country:
        country = "United States"
    if tail in _CA_PROVINCES and not country:
        country = "Canada"
    if tail in _AU_STATES and not country:
        country = "Australia"
    if tail in _UK_NATIONS and not country:
        country = "United Kingdom"
    return city, tail, country


_SKILL_CANON: dict[str, str] = {
    "problemsolving": "problem solving",
}


def parse_skill_list(cell: str) -> list[str]:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    out: list[str] = []
    for s in (_norm_text(p) for p in str(cell).split(",")):
        if not s or is_noise_skill(s):
            continue
        out.append(_SKILL_CANON.get(s, s))
    return out


def seed_label(skill: str) -> str:
    s = _norm_text(skill)
    for theme, kws in SEED_KEYWORDS.items():
        for kw in kws:
            if kw in s:
                return theme
    return "Domain / Other"


_BENEFITS_NOISE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bpaid time off\b"),
    re.compile(r"\bpaid leave\b"),
    re.compile(r"\bpto\b"),
    re.compile(r"\bunlimited pto\b"),
    re.compile(r"\bannual leave\b"),
    re.compile(r"\bdonor leave\b"),
    re.compile(r"\bvacation\b"),
    re.compile(r"\bholiday\b"),
    re.compile(r"\bholidays\b"),
    re.compile(r"\bsick( time)?\b"),
    re.compile(r"\bparental leave\b"),
    re.compile(r"\bfamily leave\b"),
    re.compile(r"\bmedical leave\b"),
    re.compile(r"\bmaternity leave\b"),
    re.compile(r"\bpaternity leave\b"),
    re.compile(r"\bbereavement\b"),
    re.compile(r"\bjury duty\b"),
    re.compile(r"\bleave(s)? of absence\b"),
    re.compile(r"^loa$"),
    re.compile(r"\bbenefits?\b"),
    re.compile(r"\bbenefits package\b"),
    re.compile(r"\bhealth insurance\b"),
    re.compile(r"\bhealth coverage\b"),
    re.compile(r"\bdental\b"),
    re.compile(r"^vision$"),
    re.compile(r"\blife insurance\b"),
    re.compile(r"\bdisability insurance\b"),
    re.compile(r"\bshort[-\s]?term disability\b"),
    re.compile(r"\blong[-\s]?term disability\b"),
    re.compile(r"\bwellness\b"),
    re.compile(r"\btelemedicine\b"),
    re.compile(r"\bemployer[-\s]?paid\b"),
    re.compile(r"\b401k\b"),
    re.compile(r"\b401\(k\)\b"),
    re.compile(r"\bretirement\b"),
    re.compile(r"\bpension\b"),
    re.compile(r"\bprofit\s*sharing\b"),
    re.compile(r"^matching$"),
    re.compile(r"\bstock purchase\b"),
    re.compile(r"\bsign[-\s]?on bonus\b"),
    re.compile(r"\breferral\b"),
    re.compile(r"\bbonuses\b"),
    re.compile(r"\bbonus\b"),
    re.compile(r"\bcommission(s|ed)?\b"),
    re.compile(r"\brsu\b"),
    re.compile(r"\bpaid volunteering\b"),
    re.compile(r"\bvolunteer(ing)? days?\b"),
    re.compile(r"\btuition (reimbursement|assistance)\b"),
    re.compile(r"\beducation reimbursement\b"),
    re.compile(r"\brelocation\b"),
    re.compile(r"\bemployee assistance program\b"),
    re.compile(r"\bgym reimbursement\b"),
    re.compile(r"\bcompany (car|phone|laptop)\b"),
    re.compile(r"\breimbursement\b"),
    re.compile(r"\ballowance\b"),
    re.compile(r"\bfull[-\s]?time\b"),
    re.compile(r"\bpart[-\s]?time\b"),
    re.compile(r"^remote$"),
    re.compile(r"\bhybrid\b"),
    re.compile(r"\bon[-\s]?site\b"),
    re.compile(r"\bflexible schedule\b"),
    re.compile(r"\bweekends?\b"),
    re.compile(r"\bshift differentials?\b"),
    re.compile(r"\bovertime\b"),
    re.compile(r"\bcompetitive pay\b"),
    re.compile(r"\bhourly pay\b"),
    re.compile(r"\b\d{1,2}\s*hours\s*per\s*week\b"),
    re.compile(r"\b(us|u\.s\.)\s*citizenship\b"),
    re.compile(r"\bcitizenship required\b"),
    re.compile(r"\bpermanent residency\b"),
    re.compile(r"\bwork authorization\b"),
    re.compile(r"\b(us|u\.s\.)\s*residency\b"),
    re.compile(r"\bresidency required\b"),
    re.compile(r"\bsexual orientation\b"),
    re.compile(r"\bgender identity\b"),
    re.compile(r"\bprotected class\b"),
    re.compile(r"\b\d+\+?\s*years?\s+(of\s+)?experience\b"),
    re.compile(r"\bminimum\s+\d+\s*years?\b"),
    re.compile(r"\b\d+\s*year\s+experience\b"),
    re.compile(r"\bentry[-\s]?level\b"),
    re.compile(r"\b\d+\s*years?\s+of\s+age\b"),
    re.compile(r"\b\d+\s*years?\s+old\b"),
    re.compile(r"\b(age|aged)\s+\d+\+?\b"),
    re.compile(r"\bmust be\s+\d+\+?\b"),
    re.compile(r"\b\d+\s*week(s)?\b"),
    re.compile(r"\b\d+\s*week\s+duration\b"),
    re.compile(r"^oil\s*&\s*gas$"),
    re.compile(r"^oil\s+and\s+gas$"),
    re.compile(r"^blood banking$"),
    re.compile(r"trauma center"),
    re.compile(r"^heating$"),
    re.compile(r"^online application$"),
    re.compile(r"uniforms?"),
    re.compile(r"24\s*/\s*7\s*/\s*365\s+support"),
    re.compile(r"^fun$"),
    re.compile(r"fun and enthusiastic personality"),
    re.compile(r"^skills$"),
    re.compile(r"standard office equipment"),
    re.compile(r"employee discount program"),
    re.compile(r"^days$"),
    re.compile(r"^driven$"),
    re.compile(r"^dust$"),
    re.compile(r"\bdiabetes\b"),
    re.compile(r"\bcovid[-\s]?19 vaccine\b"),
    re.compile(r"\bblood draws?\b"),
    re.compile(r"\bfirearms?\b"),
    re.compile(r"\busda accreditation\b"),
    re.compile(r"lessons learned"),
    re.compile(r"\bchemical dependency\b"),
    re.compile(r"\bhazardous chemicals?\b"),
    re.compile(r"\btoxic chemicals?\b"),
    re.compile(r"\bexposure to irritant chemicals\b"),
    re.compile(r"\birritant chemicals?\b"),
    re.compile(r"\bchemical (dependency|exposure|hazard)\b"),
    re.compile(r"long service awards?"),
    re.compile(r"\bonesite\b"),
    re.compile(r"\bdivorce\b"),
    re.compile(r"\bmarriage\b|\bmarried\b|\bdomestic partner\b"),
    re.compile(r"\bheight\b|\bweight\b|\bbmi\b|\bbody mass\b"),
    re.compile(r"\bethnicity\b|\breligion\b|\bfaith\b"),
    re.compile(r"\blgbtq?\+?\b|\blgbt\b"),
    re.compile(r"\bdei\b|\bde&i\b"),
    re.compile(
        r"\bdiversity\b|\binclusion\b|\bequal opportunity\b|\beeo\b|\baffirmative action\b"
    ),
    re.compile(r"\bdiversity statement\b|\bcommitment to diversity\b"),
    re.compile(r"^coagulation$"),
]

_EDUCATION_CREDENTIAL_RE = re.compile(
    r"(?:"
    r"\b(?:high\s+school\s+diploma|ged(?:\s+equivalency)?|general\s+(?:education\s+)?(?:development|equivalency)(?:\s+diploma)?)\b|"
    r"\b(?:bachelor'?s?\s+degree|bachelors?\s+degree|master'?s?\s+degree|masters?\s+degree)\b|"
    r"\b(?:associate'?s?\s+degree|associate\s+degree)\b|"
    r"\b(?:doctorate|doctoral\s+degree|doctor\s+of(?:\s+\w+)?|ph\.?\s*d\.?)\b|\bphd\b|"
    r"\b(?:mba|undergraduate\s+degree|graduate\s+degree|college\s+degree|university\s+degree)\b|"
    r"\b(?:aa|as|ba|bs|ma|ms)\s+degree\b|"
    r"\bdegree\s+(?:in|from|program)\b|"
    r"学士学位|硕士学位|博士学位|本科学位|本科学历|研究生学历|本科及以上学历|统招本科|"
    r"高中毕业|大专(?:学历|毕业)?|专科学历|专科毕业"
    r")",
    re.I,
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
    if "hourly wage" in s or "hourly wages" in s:
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
    if _EDUCATION_CREDENTIAL_RE.search(s):
        return True
    for pat in _BENEFITS_NOISE_PATTERNS:
        if pat.search(s):
            return True
    return False


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
            "job_link",
            "job_title",
            "job_type",
            "job_level",
            "job_location",
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
        "job_location",
        "search_city",
        "search_country",
        "company",
        "first_seen",
        "job_title",
    ]:
        df[col] = df[col].str.strip().fillna("Unknown")
    df["search_position"] = df["search_position"].str.strip()
    df = df[
        ~df["search_position"].map(lambda x: _norm_role(x) in _NOISE_JOB_ROLES)
    ].copy()
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

    if df["first_seen"].notna().any():
        max_dt = df["first_seen"].max()
        w1 = df[df["first_seen"] >= max_dt - pd.Timedelta(days=30)]
        w0 = df[
            (df["first_seen"] < max_dt - pd.Timedelta(days=30))
            & (df["first_seen"] >= max_dt - pd.Timedelta(days=60))
        ]
        c1 = w1.groupby("search_position")["job_link"].count()
        c0 = w0.groupby("search_position")["job_link"].count()
        pos["trend_30d"] = (
            pos["search_position"].map((c1 - c0).fillna(0)).fillna(0).astype(float)
        )
    else:
        pos["trend_30d"] = 0.0

    ranked = pos.sort_values("course_opportunity_score", ascending=False).reset_index(
        drop=True
    )
    ranked.index += 1
    ranked.insert(0, "rank", ranked.index)
    return ranked.rename(columns={"search_position": "job_role"})


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

    df["rule_theme"] = df["skill"].apply(seed_label)

    X_all = vec.transform(df["skill"].astype(str).tolist())
    proba = clf.predict_proba(X_all)
    classes = list(clf.classes_)
    df["ml_theme"] = [classes[i] for i in proba.argmax(axis=1)]
    df["ml_confidence"] = proba.max(axis=1).round(4)

    mask = df["rule_theme"] != "Domain / Other"
    df.loc[mask, "ml_theme"] = df.loc[mask, "rule_theme"]
    df.loc[mask, "ml_confidence"] = 1.0

    return df.sort_values("skill_count", ascending=False)


def topic_breakdowns(
    postings: pd.DataFrame, top_roles: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = postings[postings["search_position"].isin(top_roles)].copy()
    p = p.rename(columns={"search_position": "job_role"})
    by_country = (
        p.groupby(["job_role", "search_country"], as_index=False)
        .agg(volume=("job_link", "count"))
        .sort_values(["job_role", "volume"], ascending=[True, False])
    )
    by_type = (
        p.groupby(["job_role", "job_type"], as_index=False)
        .agg(volume=("job_link", "count"))
        .sort_values(["job_role", "volume"], ascending=[True, False])
    )
    by_level = (
        p.groupby(["job_role", "job_level"], as_index=False)
        .agg(volume=("job_link", "count"))
        .sort_values(["job_role", "volume"], ascending=[True, False])
    )
    return by_country, by_type, by_level


def topic_theme_mix(
    postings: pd.DataFrame,
    skills_raw: pd.DataFrame,
    top_roles: list[str],
    skill_to_theme: dict[str, str],
    max_links_per_topic: int = 2000,
) -> pd.DataFrame:
    p = postings[postings["search_position"].isin(top_roles)][
        ["job_link", "search_position"]
    ].copy()
    p = p.rename(columns={"search_position": "job_role"})
    p = p.sort_values(["job_role", "job_link"])
    p = p.groupby("job_role", as_index=False).head(max_links_per_topic)
    link_to_topic = dict(zip(p["job_link"].astype(str), p["job_role"].astype(str)))
    wanted_links = set(link_to_topic.keys())

    subset = skills_raw[skills_raw["job_link"].isin(wanted_links)]
    theme_counts: dict[str, Counter] = defaultdict(Counter)
    for jl, cell in zip(subset["job_link"].astype(str), subset["job_skills"]):
        topic = link_to_topic.get(jl)
        if not topic:
            continue
        for sk in parse_skill_list(str(cell)):
            theme = skill_to_theme.get(sk) or seed_label(sk)
            theme_counts[topic][theme] += 1

    rows = []
    for topic, c in theme_counts.items():
        total = sum(c.values()) or 1
        for theme, cnt in c.items():
            rows.append(
                {
                    "job_role": topic,
                    "skill_theme": theme,
                    "skill_mentions": cnt,
                    "share": round(cnt / total, 4),
                }
            )
    df = pd.DataFrame(rows)
    return df.sort_values(["job_role", "share"], ascending=[True, False])


def skill_bundle_pairs(skills_raw: pd.DataFrame, top_pairs: int = 250) -> pd.DataFrame:
    pair_counts: Counter = Counter()
    for cell in skills_raw["job_skills"]:
        skills = sorted(set(parse_skill_list(str(cell))))
        if len(skills) < 2:
            continue
        skills = skills[:30]
        for i in range(len(skills)):
            for j in range(i + 1, len(skills)):
                pair_counts[(skills[i], skills[j])] += 1
    return pd.DataFrame(
        [
            {"skill_a": a, "skill_b": b, "cooccur_count": c}
            for (a, b), c in pair_counts.most_common(top_pairs)
        ]
    )


def compute_location_toplists(postings: pd.DataFrame, top_k: int = 25) -> pd.DataFrame:
    rows = []
    for loc_type, col in [("city", "search_city"), ("country", "search_country")]:
        for name, count in postings[col].value_counts().head(top_k).items():
            if name and name != "Unknown":
                rows.append(
                    {"location_type": loc_type, "name": name, "volume": int(count)}
                )

    # state is not a structured column — requires parsing job_location
    countries = set(postings["search_country"].dropna().unique())
    c_state: Counter = Counter()
    for loc in postings["job_location"]:
        _, state, _ = parse_job_location(loc, countries)
        if state:
            c_state[state] += 1
    for name, count in c_state.most_common(top_k):
        rows.append({"location_type": "state", "name": name, "volume": count})

    return pd.DataFrame(rows)

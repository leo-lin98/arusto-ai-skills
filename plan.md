# Plan: Arusto Skills Taxonomy Dashboard

## Context

We are building a Streamlit dashboard that surfaces labor market intelligence from job posting data. The raw data is a static CSV with two columns — `job_links` and `job_skills` (an array of skills per posting). The pipeline parses and normalizes those skill arrays, feeds them into pre-trained models, and renders the results as an interactive single-page dashboard.

Two pre-trained models are shipped as `.pkl` files:
1. **Text classifier** — predicts job category from job title, description, and normalized skills.
2. **Salary regression** — estimates salary from job attributes (experience level, company size).

The dashboard has both predictive outputs (model inference) and non-predictive analytics (skill rankings, co-occurrence, course tracks). Models are trained offline and dropped into `/models/` — no training happens inside the app.

Multiple team members work in parallel. Interfaces between layers (data, models, dashboard) are defined here so work can proceed independently.

---

## Directory Structure

```
arusto-skills-taxonomy/
├── plan.md
├── CLAUDE.md
├── README.md
│
├── data/
│   └── skills_data.csv           # Raw data: job_links (str), job_skills (array)
│
├── models/
│   ├── text_classifier.pkl       # Job category classifier — drop here manually
│   └── salary_regressor.pkl      # Salary linear regression — drop here manually
│
├── scripts/
│   └── parse_skills.py           # Offline ETL: reads skills_data.csv, normalizes
│                                 # job_skills arrays, writes processed_skills.csv
│
├── app/
│   ├── __init__.py
│   ├── main.py                   # Streamlit entry point — all UI and layout
│   ├── skills.py                 # Skills class — loads .pkl models, exposes
│                                 # predict_category() and predict_salary()
│   └── analytics.py              # Pure functions over processed DataFrame:
│                                 # skill_ranking(), in_demand_skills(),
│                                 # co_occurring_pairs(), course_opportunity_score(),
│                                 # course_tracks()
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml                # Ruff config
└── .github/
    └── workflows/
        └── ci.yml                # Lint + Docker build on push
```

---

## Data Flow

```
skills_data.csv
  └── [job_links, job_skills (raw array)]
        │
        ▼
scripts/parse_skills.py          ← run once offline
        │
        ▼
processed_skills.csv             ← normalized, one skill per row or exploded
  └── [job_link, skill, category, experience_level, company_size, salary]
        │
        ├──▶ app/analytics.py    (non-predictive dashboard sections)
        └──▶ app/skills.py       (model inference)
```

`parse_skills.py` is an offline script, not part of the Streamlit runtime. Its output (`processed_skills.csv`) is what the app loads.

---

## Component Responsibilities

### `scripts/parse_skills.py`
- Read `skills_data.csv`
- Parse `job_skills` arrays into individual normalized skill strings (lowercase, strip whitespace, deduplicate per posting)
- Output `data/processed_skills.csv` with one row per job–skill pair (or one row per job with a clean skills list — agree on format as a team before writing)
- **Owner:** assign to one person; everyone else unblocked once schema is agreed

### `app/skills.py` — `Skills` class
- Auto-loads every `.pkl` in `/models/` on init
- Exposes typed methods so `main.py` never calls `joblib` directly:
  - `predict_category(X: pd.DataFrame) -> np.ndarray` — runs text classifier
  - `predict_salary(X: pd.DataFrame) -> np.ndarray` — runs salary regressor
- Adding a new model = drop `.pkl` + add one method. No other files change.

### `app/analytics.py`
All functions are pure (take a DataFrame, return a DataFrame or Series). No Streamlit imports.

| Function | Returns | Dashboard section |
|---|---|---|
| `skill_ranking(df)` | Series: skill → count | Skill Ranking |
| `in_demand_skills(df, top_n)` | Series: skill → demand score | In-Demand Skills |
| `co_occurring_pairs(df, top_n)` | DataFrame: skill_a, skill_b, count | Co-Occurring Skill Pairs |
| `course_opportunity_score(df)` | Series: course → score | Course Opportunity Score |
| `course_tracks(df)` | DataFrame: track, skills | Course Tracks |

### `app/main.py`
- Loads `processed_skills.csv` via `@st.cache_data`
- Loads `Skills()` via `@st.cache_resource`
- Calls `analytics.py` functions and `Skills` methods
- Renders all charts with Plotly, all metrics with `st.metric`
- Sidebar holds all filter dropdowns (skill category, experience level, company size)
- **No business logic here** — only layout and rendering

---

## Dashboard Sections

| Section | Type | Source |
|---|---|---|
| Course Opportunity Score | Predictive + analytic composite | `analytics.course_opportunity_score()` |
| Skill Ranking | Non-predictive | `analytics.skill_ranking()` |
| In-Demand Skills | Non-predictive | `analytics.in_demand_skills()` |
| Co-Occurring Skill Pairs | Non-predictive | `analytics.co_occurring_pairs()` |
| Course Tracks | Non-predictive | `analytics.course_tracks()` |

Charts are Plotly. Exact chart types (bar, heatmap, scatter) to be decided during dashboard build — `analytics.py` only cares about returning the right data shape.

---

## Parallel Work Streams

| Stream | Files | Depends on |
|---|---|---|
| Data / ETL | `scripts/parse_skills.py` | Raw CSV schema (known) |
| Models | Train + export `.pkl` files | Processed CSV schema |
| Analytics | `app/analytics.py` | Processed CSV schema |
| Dashboard UI | `app/main.py` | `analytics.py` function signatures, `Skills` method signatures |
| Infra | `Dockerfile`, `docker-compose.yml`, `ci.yml` | Nothing |

**Blocking dependency:** the schema of `processed_skills.csv` must be agreed before analytics and model work begins. Everything else can proceed in parallel.

# Skills in Demand Dashboard

Analytics in Practice, Spring 2026

Streamlit dashboard surfacing labor market intelligence from 1.3M LinkedIn job postings. Answers: **"What courses should institutions build next?"**

## Pages

| Page                    | Description                                                                                            |
| ----------------------- | ------------------------------------------------------------------------------------------------------ |
| 01 Overview             | Job titles, companies, locations, temporal trends. Sidebar filters: Company, Country, Date Range       |
| 02 Skills               | Top skills by frequency, category breakdown, skill × skill and skill × category co-occurrence heatmaps |
| 03 Course Opportunities | Course opportunity rankings, scatter (volume vs score), skill theme mix, trend forecast                |

## Data

Source: [`asaniczka/1-3m-linkedin-jobs-and-skills-2024`](https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024) on Kaggle — full 1.3M postings.

Pre-processed parquets live in Cloudflare R2. The app reads them directly via DuckDB HTTP range requests — no full file download, no Kaggle credentials needed at runtime.

| Parquet                   | Contents                                                          |
| ------------------------- | ----------------------------------------------------------------- |
| `jobs.parquet`            | Full merged + featured dataset (1.3M rows, source of truth)       |
| `topic_rankings.parquet`  | Ranked course topics with opportunity scores + OLS trend forecast |
| `skill_theme_map.parquet` | Top 5k skills → ML-predicted theme + confidence                   |

## Architecture

```
Kaggle API (offline, pipeline only)
        │  download CSVs into memory
        ▼
data/pipeline.py — merge, feature engineering, ML scoring, OLS trend
        │  stream 3 parquets (MD5-dedup)
        ▼
Cloudflare R2  (arusto-skills/)
        │
        ▼
DuckDB (httpfs) — columnar SQL, HTTP range requests, no full download
        │
        ▼
Streamlit pages — @st.cache_data queries → charts and dataframes
```

DuckDB connection is shared across the app via `@st.cache_resource`. Query results are cached per parameter set via `@st.cache_data`.

## ML Pipeline (offline)

Weakly supervised classifier built inside `data/pipeline.py`:

1. **Seed keywords** assign high-precision labels for 9 skill themes.
2. **TF-IDF char-ngram (3–5) + SGD** generalizes labels to unlabeled skills.
3. `build_features()` assigns a `category` to every job posting via `search_position`.
4. `score_topics()` aggregates per topic: volume, salary proxy, breadth score → `course_opportunity_score`.
5. `build_skill_theme_map()` predicts theme for the top 5k skills.

Course opportunity score: `0.40 × log_volume + 0.35 × salary_proxy + 0.25 × breadth` (all Min-Max scaled to [0, 100]).

Salary proxy: `0.40 × remote_rate + 0.25 × hybrid_rate + 0.35 × senior_rate`.

Topics with fewer than 2,000 postings are excluded from rankings.

## Setup

### Local (Docker)

```bash
docker compose up
```

### Local (bare)

```bash
pyenv local 3.14
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Credentials go in `.env` (local) or Streamlit secrets (production):

```
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
KAGGLE_USERNAME=...   # pipeline only
KAGGLE_KEY=...        # pipeline only
```

## Pipeline

Run to re-download the dataset, rebuild all parquets, and upload to R2:

```bash
python -m data.pipeline
```

Requires Kaggle and R2 credentials. Uploads are MD5-deduplicated — unchanged parquets are skipped.

## Development

```bash
# Lint
ruff check .
ruff format .

# Tests
pytest

# Build image
docker build -t arusto-skills-taxonomy .
```

CI runs lint then Docker build on every push.

## Structure

```
streamlit_app.py              # Entry point — multipage nav
pages/
  01_overview.py              # Job market overview (filters: Company, Country, Date Range)
  02_skills.py                # Skills frequency + two co-occurrence heatmaps
  03_opportunities.py         # Course opportunity ranking + skill theme mix
data/
  pipeline.py                 # CLI: Kaggle → jobs.parquet → R2 (MD5 dedup)
  processor.py                # get_merged → build_features → score_topics → build_skill_theme_map
  loader.py                   # R2 creds, MD5-dedup uploads, Kaggle download
  db.py                       # Cached DuckDB conn + filter helpers
components/
  charts.py                   # Shared chart helpers
  filters.py                  # sidebar_filters() — Company + Country + Date Range (page 1 only)
tests/
  test_processor.py           # score_topics, trend forecast, noise filter
  test_loader.py              # MD5 dedup upload logic
  test_db.py                  # filter condition builder
  test_trend.py               # OLS trend labels and slope computation
```

## Team

Muxi Chen, Leo Lin, Sirui Wang

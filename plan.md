# Plan: Arusto Skills Taxonomy Dashboard

## Context

Streamlit dashboard surfacing labor-market intelligence from the full 1.3M LinkedIn job postings (`asaniczka/1-3m-linkedin-jobs-and-skills-2024` on Kaggle). Audience is dean / curriculum committee. Anchor question: **"What courses should institutions build next?"**

Pipeline downloads raw CSVs, merges into one source-of-truth parquet, derives skill themes via weak supervision, scores course topics, and uploads three parquets to Cloudflare R2 with MD5-based dedup. The app reads R2 directly via DuckDB `httpfs` — no full download.

Three pages:
1. **Overview** — dataset shape: titles, companies, locations, time, level.
2. **Skills** — category dropdown → ranked skills + two stacked heatmaps.
3. **Course Opportunities** — top-15 ranked course topics with OLS-based growth forecast.

---

## Directory Structure

```
arusto-skills-taxonomy/
├── streamlit_app.py              # Entry point — multipage nav
│
├── pages/
│   ├── 01_overview.py            # Job market overview + Country/Date filters
│   ├── 02_skills.py              # Skills frequency, category breakdown, two heatmaps
│   └── 03_opportunities.py       # Course-opportunity ranking (top 15)
│
├── data/
│   ├── pipeline.py               # Kaggle → jobs.parquet → R2 (CLI only, MD5 dedup)
│   ├── processor.py              # get_merged → build_features → score_topics → trend_forecast
│   ├── loader.py                 # R2 creds, MD5-dedup upload, Kaggle download
│   └── db.py                     # cached DuckDB conn + filter helpers
│
├── components/
│   ├── charts.py                 # Shared chart helpers
│   └── filters.py                # sidebar_filters() — Company + Country + Date Range
│
├── .streamlit/
│   └── config.toml               # Theme
│
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml                # Ruff config
├── requirements.txt
└── .github/workflows/ci.yml      # Lint + Docker build on push
```

---

## Data Source

Kaggle dataset: `asaniczka/1-3m-linkedin-jobs-and-skills-2024`. Downloaded directly via Kaggle API — no local CSV storage.

| File | Used for |
|---|---|
| `linkedin_job_postings.csv` | Job titles, companies, locations, timestamps — **full 1.3M rows** |
| `job_skills.csv` | Skills per job — chunked-read filtered against full job_links |
| `job_summary.csv` | Free-text job summaries — chunked-read filtered against full job_links |

---

## Single Dataset Model

**Source of truth:** `jobs.parquet` (1.3M rows) — every page draws from this.

**Speed caches** (regenerable from `jobs.parquet`):
- `topic_rankings.parquet` — per-`search_position` aggregates with score + OLS trend.
- `skill_theme_map.parquet` — top 5k skills with ML-assigned theme + confidence.

3 parquets total. No `merged.parquet`, `label_rollup.parquet`, or `skill_bundles.parquet` — those rolled into the source or recomputed by DuckDB on demand.

---

## Data Flow

### Offline pipeline (`python -m data.pipeline`)

```
Kaggle CSVs (raw, ~several GB)
        │
        ▼
data/pipeline.py::download_kaggle_data()
        │
        ▼
data/processor.py::get_merged()
  ├── load_postings()       full 1.3M rows, parse timestamps, fill nulls
  ├── aggregate_skills()    chunked read, filter to job_links, groupby → comma-sep
  └── load_summary()        chunked read, filter to job_links
        │
        ▼
train_skill_theme_model()   TF-IDF char-ngram + SGD on top 8k skills
        │
        ▼
build_features()
  ├── derived: job_title_len, n_skills, combined_text, skills_norm
  ├── category: weakly supervised seed-keyword + SGD on search_position
  └── (9 themes + "Domain / Other")
        │
        ▼
score_topics()
  ├── volume, log_volume, salary_proxy, breadth_score
  ├── course_opportunity_score (see formula below)
  ├── opportunity_label: High Opportunity (≥60) / Emerging (≥40) / Saturated (<40)
  └── add_trend_forecast() × 2     OLS on weekly counts → forecast_4w + forecast_12w
        │
        ▼
build_skill_theme_map(top_n=5000)  ML predictions for top skills
        │
        ▼ MD5-dedup uploads (skip if R2 metadata md5 matches)
  ├── jobs.parquet               (1.3M rows, source of truth)
  ├── topic_rankings.parquet     (~few hundred topics, cached for page 3)
  └── skill_theme_map.parquet    (5k skills, cached for page 2)
        │
        ▼
Cloudflare R2: arusto-skills/  (no local disk write)
```

### App (production)

```
Cloudflare R2 (S3 API)
        │
        └─▶ data/db.py::get_db_connection()  — DuckDB mounts R2 via httpfs
                │
                ├─▶ pages/01_overview.py    aggregations on jobs.parquet
                ├─▶ pages/02_skills.py      aggregations on jobs.parquet + skill_theme_map.parquet
                └─▶ pages/03_opportunities.py  reads topic_rankings.parquet
```

DuckDB uses HTTP Range Requests — only the columns each query needs are fetched.

---

## Course Opportunity Score (locked)

```
volume_score = minmax(log1p(volume))                                  # log-scaled
salary_proxy = 0.40·remote_rate + 0.25·hybrid_rate + 0.35·senior_rate
breadth      = 0.50·minmax(city_count) + 0.50·minmax(company_count)

score = 0.40·volume_score + 0.35·minmax(salary_proxy) + 0.25·breadth
opportunity_label = High (≥60) / Emerging (≥40) / Saturated (<40)
```

Why log-scale volume: a 50k-posting topic shouldn't beat a 5k-posting topic by 10×. log1p tames skew, lets quality signals (salary, breadth) actually matter.

Trend lives **alongside** the score, not inside it. Honest separation: a topic can be Saturated and still Growing.

---

## OLS Trend Forecast

Replaces the old two-window `trend_30d` diff. For each qualifying topic:

1. Bucket `first_seen` into ISO weeks → `weekly_count`.
2. Fit OLS: `weekly_count ~ week_number` (closed-form).
3. Extract `slope` (postings/week) and `r2`.
4. Project forward 4 weeks AND 12 weeks (clipped at zero).
5. Label from 12w slope:
   - `Growing` if `slope > 1.0` and `r² ≥ 0.30`
   - `Declining` if `slope < -1.0` and `r² ≥ 0.30`
   - `Stable` otherwise
   - `Insufficient data` if fewer than 4 non-zero weeks

Output columns added to `topic_rankings.parquet`: `trend_slope, trend_r2, forecast_4w, forecast_12w, trend_label`.

> **TODO:** Workshop `R2_THRESHOLD` and `SLOPE_THRESHOLD` against actual data percentiles.

---

## MD5 Upload Dedup

R2 egress is free but uploads still cost time. Pipeline computes `md5(parquet_bytes)` before each upload, compares against `Metadata['md5']` on the existing R2 object via `head_object`:

- 404 → upload, set `Metadata={'md5': hex}`.
- 200 + md5 match → skip.
- 200 + md5 mismatch → upload, overwrite metadata.

Cost: 1 HEAD per file per run (R2 class B ops are free under 10M/month). Custom metadata (not S3 ETag) is used so multipart uploads work transparently. md5 is not a secret; no creds are ever logged.

---

## Job Category Derivation (unchanged)

| Theme | Seed Keywords |
|---|---|
| Communication & Collaboration | communication, written, verbal, presentation, stakeholder, teamwork, collaboration, interpersonal, customer service, client, documentation |
| Problem Solving & Critical Thinking | problem, problem solving, problemsolving, troubleshooting, debug, analysis, critical thinking, root cause, decision |
| Adaptability & Learning Agility | adaptability, flexible, fast learner, learning, change |
| Project & Program Management | project, program, agile, scrum, planning, roadmap, coordination, schedule |
| Data & Analytics | data, analytics, excel, sql, statistics, dashboard, power bi, tableau, reporting |
| Software & Cloud | python, java, javascript, react, aws, azure, gcp, cloud, devops, docker, kubernetes |
| Sales, Marketing & Customer | sales, marketing, business development, crm, lead generation, account, customer success |
| Operations & Quality | operations, logistics, supply chain, inventory, warehouse, quality, safety, compliance |
| Leadership & People Management | leadership, management, coaching, mentoring, hiring, recruiting, performance |
| Domain / Other | *(ML classifier handles — no seed keywords)* |

The skill-theme classifier (TF-IDF char-ngram + SGD on top 8k skills) is trained offline. Predictions are baked into `skill_theme_map.parquet`. The app only reads predictions — no training at runtime.

---

## Caching Strategy

| Layer | Mechanism | Scope |
|---|---|---|
| DuckDB connection | `@st.cache_resource` | Per Streamlit process |
| Parameterized query functions | `@st.cache_data` | Per unique args, persists across page nav |

`@st.cache_data` is process-level and persists across page navigation in the same session — first page load pays the R2 round-trip, subsequent navigation hits the cache. DuckDB streams via HTTP Range Requests; raw parquets are never cached in Streamlit.

---

## Page Specs

### Page 1 — Overview
- Metrics row: total postings, unique companies, unique locations.
- Charts: top job titles, top companies, top locations, job-level distribution, day-of-week, hour-of-day, search-position distribution.
- **Sidebar filters: Company + Country + Date Range.** Only on this page.

### Page 2 — Skills
- Top skills by frequency (N slider).
- Category dropdown (9 themes + "Domain / Other") → ranked skills, descending count.
- **Two stacked heatmaps:** skill × skill co-occurrence (top 15) + skill × category (top 15 skills × 10 themes).

### Page 3 — Course Opportunities
- Header: "Course opportunity ranking — top 15."
- Metrics row: total topics, # High Opportunity, # Emerging, # Saturated.
- **Top 15 ranked table:** rank, course_topic, score, opportunity_label, volume, forecast_4w, forecast_12w, trend_label.
- Scatter: volume vs score, color = trend_label.
- Search box for topic name.

---

## Tech Stack

- **Python 3.14** (pyenv-managed, pinned via `.python-version`; project venv at `.venv`)
- **Streamlit** — UI, multipage nav, session-level caching
- **DuckDB + httpfs** — query R2 parquets directly via SQL, sub-second column reads
- **Cloudflare R2** — free egress, S3-compatible, 3 parquets total
- **pandas + scikit-learn** — pipeline only (TF-IDF + SGD for weak supervision)
- **pyarrow** — parquet IO
- **boto3** — R2 upload + MD5 dedup
- **kaggle** — dataset download

---

## Credentials

- `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` — Cloudflare R2 access
- `KAGGLE_USERNAME`, `KAGGLE_KEY` — Kaggle API access
- Set in `.env` for local development, Streamlit secrets for production.

---

## Deployment

**Streamlit Cloud:** connects to `main` branch, entry point `streamlit_app.py`. Set secrets in the Streamlit Cloud dashboard:

```toml
R2_ACCESS_KEY_ID     = "..."
R2_SECRET_ACCESS_KEY = "..."
KAGGLE_USERNAME      = "..."
KAGGLE_KEY           = "..."
```

**Docker (local):**
```bash
docker compose up
```

---

## Pipeline Refresh

Manual only (`python -m data.pipeline`). Kaggle dataset is a static snapshot. CI runs lint + Docker build only — does NOT auto-run the pipeline. Re-run is needed only when seed keywords, score weights, or trend thresholds change. MD5 dedup ensures unchanged parquets aren't re-uploaded.

---

## Out of Scope

- Live "type a job description, get a category" predictor (cut — `baseline.pkl`, `models/train.py`, `models/predict.py` removed).
- CI auto-run of pipeline.
- Live data refresh from RapidAPI (post-prod consideration).
- Trend threshold tuning against data percentiles (TODO).

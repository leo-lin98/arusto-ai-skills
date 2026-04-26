# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`arusto-skills-taxonomy` — Streamlit dashboard surfacing labor market intelligence from the full 1.3M LinkedIn job postings. Audience is dean / curriculum committee. Anchors: "What courses should institutions build next?" Python 3.14.

## Environment

pyenv-managed Python 3.14. Project root contains `.python-version` pinning `3.14`.

```bash
pyenv install 3.14         # one-time
pyenv local 3.14           # auto-loaded by .python-version
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Always activate `.venv` before running any command locally. All tools (ruff, streamlit, python) must be run from this env.

## Commands

```bash
# Run locally
streamlit run streamlit_app.py

# Lint
ruff check .
ruff format .

# Run pipeline (offline, manual only)
python -m data.pipeline

# Build container
docker build -t arusto-skills-taxonomy .

# Run container (OrbStack)
docker compose up
```

## Structure

```
streamlit_app.py              # Entry point — multipage nav
pages/
  01_overview.py              # Job market overview: titles, companies, locations, time
  02_skills.py                # Skills frequency, category breakdown, two stacked heatmaps
  03_opportunities.py         # Course opportunity ranking (top 15)
data/
  pipeline.py                 # CLI: Kaggle → jobs.parquet → R2 (MD5 dedup)
  processor.py                # get_merged → build_features → score_topics → trend_forecast
  loader.py                   # R2 creds, MD5-dedup uploads, Kaggle download
  db.py                       # cached DuckDB conn + filter helpers
components/
  charts.py                   # Shared chart helpers
  filters.py                  # sidebar_filters() — Company + Country + Date Range, page 1 only
.github/workflows/ci.yml      # Lint + Docker build on push
```

## Data Source

Kaggle dataset: `asaniczka/1-3m-linkedin-jobs-and-skills-2024`. Downloaded directly via Kaggle API — no local CSV storage.

| File                        | Used for                                                         |
| --------------------------- | ---------------------------------------------------------------- |
| `linkedin_job_postings.csv` | Job titles, companies, locations, timestamps — full 1.3M rows    |
| `job_skills.csv`            | Skills per job — chunked-read filtered to all job_links          |
| `job_summary.csv`           | Free-text job summaries — chunked-read filtered to all job_links |

## Single Dataset Model

**Source of truth:** `jobs.parquet` (1.3M rows). Every page draws from this.

**Speed caches** (regenerable from `jobs.parquet`):

- `topic_rankings.parquet` — per-`search_position` aggregates with score + OLS trend.
- `skill_theme_map.parquet` — top 5k skills with ML-assigned theme + confidence.

3 parquets total, all in Cloudflare R2 bucket `arusto-skills/`.

## Offline Pipeline (`python -m data.pipeline`)

1. Download CSVs from Kaggle API directly into memory.
2. `get_merged()` — load full 1.3M postings, merge skills + summary — no local disk write.
3. `train_skill_theme_model()` — TF-IDF char-ngram + SGD on top 8k skills.
4. `build_features()` — derive columns + assign category via seed keyword + SGD on `search_position`.
5. `score_topics()` — per-topic aggregates + course opportunity score + OLS trend forecast (4w + 12w).
6. `build_skill_theme_map(top_n=5000)` — ML-predicted theme for top skills.
7. Stream three parquets to Cloudflare R2 via MD5-dedup upload.

### build_features() outputs

- `job_title_len`, `n_skills`, `combined_text`, `skills_norm` (comma-separated string)
- `category` — weakly supervised: seed keywords → TF-IDF char-ngram + SGD classifier (9 themes + "Domain / Other")

### score_topics() outputs (per search_position)

- `volume`, `log_volume`, `salary_proxy`, `breadth_score`
- `course_opportunity_score` — `0.40·minmax(log1p(volume)) + 0.35·minmax(salary_proxy) + 0.25·breadth`
  - `salary_proxy` = `0.40·remote_rate + 0.25·hybrid_rate + 0.35·senior_rate`
  - `breadth` = `0.50·minmax(city_count) + 0.50·minmax(company_count)`
- `opportunity_label` — High Opportunity (≥60) / Emerging (≥40) / Saturated (<40)
- `trend_slope`, `trend_r2`, `forecast_4w`, `forecast_12w`, `trend_label` — OLS on weekly counts
  - `trend_label`: `Growing` / `Declining` / `Stable` / `Insufficient data`
  - Defaults: `R2_THRESHOLD=0.30`, `SLOPE_THRESHOLD=1.0` (TODO: workshop against data percentiles)

### Parquets written to R2 (`arusto-skills/`)

| File                      | Contents                                               |
| ------------------------- | ------------------------------------------------------ |
| `jobs.parquet`            | Full merged + featured 1.3M postings (source of truth) |
| `topic_rankings.parquet`  | Ranked course topics with scores + trend forecast      |
| `skill_theme_map.parquet` | Top 5k skills → theme (ML predictions)                 |

## MD5 Upload Dedup

To avoid wasting R2 ops, `data/loader.py::upload_parquet_with_md5_dedup()` does:

1. Compute md5 of parquet bytes in memory.
2. `head_object` on R2; if 200, read `Metadata['md5']`.
3. Match → skip; mismatch or 404 → upload with `Metadata={'md5': hex}`.

Custom metadata is used (not S3 ETag) so multipart uploads work transparently. md5 is not a secret; no creds logged.

## App (production)

- DuckDB mounts R2 via `httpfs` — no full parquet download.
- HTTP Range Requests fetch only required columns.
- Cache DuckDB connection with `@st.cache_resource`.
- Cache parameterized query functions with `@st.cache_data` (e.g. `get_top_skills(category="Software")`). Persists across page navigation in the same session.
- Do **not** cache raw parquet files in Streamlit.

## Page Specs

### Page 1 — Overview

- Metrics: total postings, unique companies, unique locations.
- Charts: top titles, top companies, top locations, job level, day-of-week, hour-of-day, search-position.
- **Sidebar filters: Company + Country + Date Range.** Only on this page.

### Page 2 — Skills

- Top skills by frequency (N slider).
- Category dropdown → ranked skills (descending count).
- **Two stacked heatmaps:** skill × skill co-occurrence (top 15) + skill × category (top 15 × 10 themes).

### Page 3 — Course Opportunities

- Top-15 ranked table: rank, topic, score, label, volume, forecast_4w, forecast_12w, trend_label.
- Scatter: volume vs score, color = trend_label.
- Topic search box.

## Seed Keywords (SEED_KEYWORDS)

| Theme                               | Seed Keywords                                                                                                                              |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Communication & Collaboration       | communication, written, verbal, presentation, stakeholder, teamwork, collaboration, interpersonal, customer service, client, documentation |
| Problem Solving & Critical Thinking | problem, problem solving, problemsolving, troubleshooting, debug, analysis, critical thinking, root cause, decision                        |
| Adaptability & Learning Agility     | adaptability, flexible, fast learner, learning, change                                                                                     |
| Project & Program Management        | project, program, agile, scrum, planning, roadmap, coordination, schedule                                                                  |
| Data & Analytics                    | data, analytics, excel, sql, statistics, dashboard, power bi, tableau, reporting                                                           |
| Software & Cloud                    | python, java, javascript, react, aws, azure, gcp, cloud, devops, docker, kubernetes                                                        |
| Sales, Marketing & Customer         | sales, marketing, business development, crm, lead generation, account, customer success                                                    |
| Operations & Quality                | operations, logistics, supply chain, inventory, warehouse, quality, safety, compliance                                                     |
| Leadership & People Management      | leadership, management, coaching, mentoring, hiring, recruiting, performance                                                               |
| Domain / Other                      | _(ML classifier handles — no seed keywords)_                                                                                               |

## Credentials

- `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` — Cloudflare R2 access
- `KAGGLE_USERNAME`, `KAGGLE_KEY` — Kaggle API access
- Set in `.env` for local development, Streamlit secrets for production.

## SECURITY

- NEVER read or process .env files

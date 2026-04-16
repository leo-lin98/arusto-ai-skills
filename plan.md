# Plan: Skills-in-Demand Dashboard

## Overview

Streamlit dashboard analyzing 1.3M LinkedIn job postings to surface which skills
are in demand. Hosted on Streamlit Cloud (unlisted — link-only access). Zero cloud
cost: data pulled from Kaggle API at runtime and cached in session.

## Directory Structure

```
skills-dashboard/
├── app.py                        # Entry point, page navigation
├── pages/
│   ├── 01_overview.py            # Top companies chart (skeleton)
│   ├── 02_skills.py              # Skills demand analysis
│   └── 03_model.py               # ML model predictions placeholder
├── data/
│   ├── loader.py                 # Kaggle API download + st.cache_data
│   └── processor.py              # Merge 3 CSVs, clean, feature engineer
├── models/
│   ├── train.py                  # Training script (run locally, output .pkl)
│   └── predict.py                # Load .pkl + run inference
├── components/
│   ├── charts.py                 # Reusable chart functions (native Streamlit)
│   └── filters.py                # Sidebar filter widgets
├── assets/
│   └── models/                   # Committed .pkl files
├── .streamlit/
│   ├── config.toml               # Theme, layout
│   └── secrets.toml.example      # KAGGLE_USERNAME, KAGGLE_KEY template
├── requirements.txt
└── README.md
```

## Data Sources

Kaggle dataset: `asaniczka/1-3m-linkedin-jobs-and-skills-2024`

Three CSVs:
- `linkedin_job_postings.csv` — job metadata (title, company, location, job level, job type)
- `job_skills.csv` — job_link → skill mappings (long format: one row per skill)
- `job_summary.csv` — job_link → job description text

**Loading strategy** (zero cost):
1. On app start, check if `/tmp/data/merged.parquet` exists
2. If yes, load directly — skip all CSV processing
3. If no, download CSVs via `kaggle.api.dataset_download_files()`, merge + clean, write `merged.parquet`
4. Cache loaded DataFrame with `@st.cache_data` (TTL: 1 session)
5. Kaggle credentials stored in Streamlit Cloud secrets, never committed

**Why parquet**: columnar format reads 5-10x faster than CSV at this scale; preserves
dtypes; `pd.read_parquet()` supports column projection so pages only load what they need.

**Scale**: 1.3M rows is heavy for Streamlit. Default to 200k row sample on load;
provide a toggle for full dataset. Apply sampling before writing parquet to keep file small.

## Pages

### 01_overview.py

- Sidebar filters: company, location, date range
- Top 10 job titles (`st.bar_chart`)
- Job processing time by day of week (`st.bar_chart`)
- Top 10 companies by job count (`st.bar_chart`)
- Job level distribution (`st.bar_chart`)
- Top 10 job locations (`st.bar_chart`)
- Job postings by hour of day (`st.bar_chart`)
- Job openings by day of month (`st.bar_chart`)
- Search position distribution (`st.bar_chart`)

### 02_skills.py

- Top N skills by frequency across all postings
- Skills breakdown by industry or company (filterable)
- Trend over time using `first_seen` / `last_processed_time` if available

### 03_model.py

- Input: free-text job title or description
- Output: predicted `category` label from loaded `.pkl`
- Displays confusion matrix and classification report on held-out test split
- Placeholder UI until model is finalized

## ML Model

Baseline pipeline (TF-IDF + Logistic Regression):

```python
Pipeline(steps=[
    ("tfidf", TfidfVectorizer(
        lowercase=True, stop_words="english",
        ngram_range=(1, 2), min_df=2, max_df=0.9, max_features=50000,
    )),
    ("clf", LogisticRegression(
        max_iter=2000, class_weight="balanced", solver="saga",
    )),
])
```

Input (`combined_text`): `job_title + job_summary + skills`
Target (`category`): derived from `job_title` via keyword map (8 buckets)
Artifact: `assets/models/baseline.pkl` (committed to repo)

### Category Derivation

8 categories derived by keyword matching on `job_title`. Rows that match no bucket are dropped (Option A — keep signal clean).

| Category | Example keywords |
|---|---|
| TECHNOLOGY | software engineer, devops, cloud engineer, cybersecurity |
| DATA-ANALYTICS | data analyst, data scientist, business intelligence, ml engineer |
| MARKETING | marketing manager, seo, content strategist, digital marketing |
| SALES | account executive, sales representative, bdr, sdr |
| FINANCE | financial analyst, accountant, fp&a, auditor |
| HR-OPERATIONS | human resources, recruiter, talent acquisition, supply chain |
| PRODUCT-DESIGN | product manager, ux designer, product owner |
| CUSTOMER-SUCCESS | customer success, customer support, technical support |

## Shared Contracts

```python
# data/loader.py
def download_kaggle_data(dest_dir: str) -> None: ...

# data/processor.py
def load_postings(path: str, sample_n: int) -> pd.DataFrame: ...
def aggregate_skills(path: str) -> pd.DataFrame: ...
# groups job_skills.csv (long format) by job_link → list of skills per job

def load_summary(path: str) -> pd.DataFrame: ...
def merge_datasets(
    postings: pd.DataFrame,
    skills_agg: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame: ...
def derive_category(title: str) -> str | None: ...
def build_features(df: pd.DataFrame) -> pd.DataFrame: ...
# adds: combined_text, skills_list, skills_norm, n_skills, *_word_count, category

def get_merged(parquet_path: str, sample_n: int) -> pd.DataFrame: ...
# checks for parquet → loads it; otherwise runs full pipeline and writes parquet

# components/charts.py
def top_companies_chart(df: pd.DataFrame, n: int) -> None: ...  # st.bar_chart
def skills_frequency_chart(df: pd.DataFrame, n: int) -> None: ...  # st.bar_chart

# models/predict.py
def load_model(path: str) -> Pipeline: ...
def predict_category(model: Pipeline, text: str) -> str: ...
```

## Key Technical Decisions

- **Streamlit unlisted**: not indexed publicly; accessible by link. Set in Streamlit Cloud dashboard under app settings.
- **Kaggle secrets**: `KAGGLE_USERNAME` and `KAGGLE_KEY` stored in Streamlit Cloud secrets. Access via `st.secrets["KAGGLE_USERNAME"]`.
- **No GCP**: all data lives in `/tmp/` during session. No persistence across cold starts — acceptable for demo.
- **`.pkl` in repo**: model expected to be small (<100MB). Committed directly. If it exceeds GitHub's limit, document a manual upload step.
- **Sampling**: default 200k rows for performance; toggle in sidebar for full load.
- **Native Streamlit charts**: use `st.bar_chart`, `st.line_chart`, `st.dataframe` etc. No matplotlib/seaborn unless a chart type is unsupported natively.

## Dependencies

```
streamlit
kaggle
pandas
pyarrow
scikit-learn
joblib
matplotlib
seaborn
```

## Deployment

1. Push repo to GitHub
2. Connect at share.streamlit.io → New app
3. Add secrets: `KAGGLE_USERNAME`, `KAGGLE_KEY`
4. Set visibility to **Unlisted**
5. Share URL with team

## Verification Checklist

- [ ] `streamlit run app.py` starts locally without errors
- [ ] Kaggle download completes, CSVs appear in `/tmp/data/`
- [ ] Merged DataFrame has non-zero rows
- [ ] Top companies chart renders with real data
- [ ] `.pkl` loads without error, `predict_category()` returns a string
- [ ] Streamlit Cloud deploy succeeds, unlisted URL is accessible

## Out of Scope (for now)

- Auth / login
- Real-time data refresh
- Database or cloud storage
- Additional chart pages beyond current set
- Model retraining in-app

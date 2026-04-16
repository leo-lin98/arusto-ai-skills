# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`arusto-skills-taxonomy` — Streamlit dashboard for skills taxonomy analysis. Single-page, single-container. Python 3.12.

## Commands

```bash
# Run locally
streamlit run streamlit_app.py

# Lint
ruff check .
ruff format .

# Build container
docker build -t arusto-skills-taxonomy .

# Run container (OrbStack)
docker compose up
```

## Structure

```
streamlit_app.py   # Entry point — calls app/main.main()
app/
  main.py        # All UI lives here, wrapped in main()
  skills.py      # Skills class — loads .pkl models from /models/
data/
  skills_data.csv  # Static CSV, bundled in repo
models/
  *.pkl            # Drop trained model files here manually
.github/workflows/ci.yml  # Lint + Docker build on push
```

## Key conventions

- `Skills` class in `app/skills.py` auto-loads every `.pkl` in `/models/` on init. To add a new model, drop the `.pkl` — no code change needed.
- Models are pre-trained offline. No training happens inside the app.
- `@st.cache_data` for CSV, `@st.cache_resource` for `Skills` instance — do not remove these caches.
- Ruff is the only linter/formatter. Config in `pyproject.toml`.
- CI runs lint then Docker build. Both must pass before merge.
- `docker-compose.yml` mounts `./data` and `./models` as volumes so files can be swapped without rebuilding the image.

from __future__ import annotations

import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

import pandas as pd
from dotenv import load_dotenv

from data.loader import (
    _get_s3_client,
    download_kaggle_data,
    upload_parquet_with_md5_dedup,
)
from data.processor import (
    build_features,
    build_skill_theme_map,
    get_merged,
    score_topics,
    train_skill_theme_model,
)


@contextmanager
def timed(label: str):
    t0 = time.perf_counter()
    yield
    print(f"  ✓ {label} ({time.perf_counter() - t0:.1f}s)")


def _upload(args: tuple[pd.DataFrame, str]) -> str:
    df, key = args
    s3 = _get_s3_client()
    upload_parquet_with_md5_dedup(df, key, s3)
    return key


def main() -> None:
    load_dotenv()
    t_start = time.perf_counter()

    with tempfile.TemporaryDirectory() as tmpdir:

        with timed("Download Kaggle dataset"):
            download_kaggle_data(tmpdir)

        with timed("Load and merge data"):
            merged, skills_raw = get_merged(tmpdir)
        print(f"     {len(merged):,} rows")

        with timed("Train skill theme model"):
            vec, clf = train_skill_theme_model(skills_raw)

        with timed("Build features + derived tables (parallel)"):
            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_themes = pool.submit(build_skill_theme_map, skills_raw, vec, clf)

                featured = build_features(merged, vec, clf)
                topic_rankings = score_topics(featured)

                skill_theme_map = fut_themes.result()

        print(f"     {len(topic_rankings):,} qualifying topics")

        uploads = [
            (featured,        "jobs.parquet"),
            (topic_rankings,  "topic_rankings.parquet"),
            (skill_theme_map, "skill_theme_map.parquet"),
        ]

        with timed("Upload to R2 (parallel, md5 dedup)"):
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = {pool.submit(_upload, args): args[1] for args in uploads}
                for fut in as_completed(futures):
                    exc = fut.exception()
                    if exc:
                        raise RuntimeError(f"Upload failed for {futures[fut]}") from exc
                    print(f"     {fut.result()}")

    print(f"\nDone. Total wall time: {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()

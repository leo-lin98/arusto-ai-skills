from __future__ import annotations

import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

from dotenv import load_dotenv

load_dotenv()

from data.loader import _get_s3_client, download_kaggle_data, upload_parquet_if_missing
from data.processor import (
    build_features,
    build_label_rollup,
    build_skill_bundle_pairs,
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


def _upload(args: tuple) -> str:
    df, name = args
    s3 = _get_s3_client()
    upload_parquet_if_missing(df, name, s3)
    return name


def main() -> None:
    t_start = time.perf_counter()

    with tempfile.TemporaryDirectory() as tmpdir:

        with timed("Download Kaggle dataset"):
            download_kaggle_data(tmpdir)

        with timed("Load and merge data"):
            merged, skills_raw = get_merged(tmpdir)
        print(f"     {len(merged):,} rows")

        with timed("Train skill theme model"):
            vec, clf = train_skill_theme_model(skills_raw)

        with timed("Build features"):
            featured = build_features(merged, vec, clf)

        with timed("Score topics"):
            topic_rankings = score_topics(featured)
        print(f"     {len(topic_rankings):,} qualifying topics")

        # three derived tables are independent — threads avoid IPC cost of processes
        with timed("Build derived tables (parallel)"):
            with ThreadPoolExecutor(max_workers=3) as pool:
                fut_rollup  = pool.submit(build_label_rollup,    topic_rankings)
                fut_themes  = pool.submit(build_skill_theme_map, skills_raw, vec, clf)
                fut_bundles = pool.submit(build_skill_bundle_pairs, skills_raw)

                label_rollup    = fut_rollup.result()
                skill_theme_map = fut_themes.result()
                skill_bundles   = fut_bundles.result()

        # uploads are pure I/O — one s3 client per thread (boto3 not thread-safe)
        uploads = [
            (featured,        "merged.parquet"),
            (topic_rankings,  "topic_rankings.parquet"),
            (label_rollup,    "label_rollup.parquet"),
            (skill_theme_map, "skill_theme_map.parquet"),
            (skill_bundles,   "skill_bundles.parquet"),
        ]

        with timed("Upload to R2 (parallel)"):
            with ThreadPoolExecutor(max_workers=5) as pool:
                futures = {pool.submit(_upload, args): args[1] for args in uploads}
                for fut in as_completed(futures):
                    exc = fut.exception()
                    if exc:
                        raise RuntimeError(f"Upload failed for {futures[fut]}") from exc
                    print(f"     uploaded {fut.result()}")

    print(f"\nDone. Total wall time: {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()

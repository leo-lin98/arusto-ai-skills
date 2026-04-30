from __future__ import annotations

import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

import pandas as pd
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from data.loader import (
    _get_s3_client,
    download_kaggle_data,
    upload_parquet_with_md5_dedup,
)
from data.processor import (
    build_features,
    build_skill_theme_map,
    compute_location_toplists,
    get_merged,
    pipeline_config_hash,
    score_topics,
    skill_bundle_pairs,
    topic_breakdowns,
    topic_theme_mix,
    train_skill_theme_model,
)

R2_BUCKET: str = os.environ.get("R2_BUCKET", "arusto-skills")


@contextmanager
def timed(label: str):
    t0 = time.perf_counter()
    yield
    print(f"  ✓ {label} ({time.perf_counter() - t0:.1f}s)")


def _upload(args: tuple[pd.DataFrame, str, dict[str, str]]) -> str:
    df, key, extra_meta = args
    s3 = _get_s3_client()
    upload_parquet_with_md5_dedup(df, key, s3, extra_meta)
    return key


def _config_current(s3: object, config_hash: str) -> bool:
    try:
        head = s3.head_object(Bucket=R2_BUCKET, Key="jobs.parquet")
        return head.get("Metadata", {}).get("config_hash") == config_hash
    except ClientError:
        return False


def main() -> None:
    load_dotenv()
    t_start = time.perf_counter()

    config_hash = pipeline_config_hash()
    s3_check = _get_s3_client()
    if _config_current(s3_check, config_hash):
        print("Pipeline config unchanged — parquets are current, skipping.")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        with timed("Download Kaggle dataset"):
            download_kaggle_data(tmpdir)

        with timed("Load and merge data"):
            merged, skills_raw = get_merged(tmpdir)
        print(f"     {len(merged):,} rows")

        with timed("Train skill theme model"):
            vec, clf = train_skill_theme_model(skills_raw)

        with timed("Build features + derived tables"):
            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_themes = pool.submit(build_skill_theme_map, skills_raw, vec, clf)
                featured = build_features(merged, vec, clf)
                topic_rankings = score_topics(featured)
                skill_theme_map = fut_themes.result()

        print(f"     {len(topic_rankings):,} qualifying topics")
        top_roles = topic_rankings["job_role"].astype(str).tolist()

        with timed("Build breakdown tables"):
            by_country, by_type, by_level = topic_breakdowns(featured, top_roles)
            location_toplists = compute_location_toplists(featured)

        with timed("Build topic theme mix"):
            skill_to_theme = dict(
                zip(
                    skill_theme_map["skill"].astype(str),
                    skill_theme_map["ml_theme"].astype(str),
                )
            )
            theme_mix = topic_theme_mix(featured, skills_raw, top_roles, skill_to_theme)

        with timed("Build skill bundle pairs"):
            bundles = skill_bundle_pairs(skills_raw)

        uploads = [
            (featured, "jobs.parquet", {"config_hash": config_hash}),
            (topic_rankings, "topic_rankings.parquet", {}),
            (skill_theme_map, "skill_theme_map.parquet", {}),
            (by_country, "topic_country_volume.parquet", {}),
            (by_type, "topic_type_volume.parquet", {}),
            (by_level, "topic_level_volume.parquet", {}),
            (theme_mix, "topic_theme_mix.parquet", {}),
            (bundles, "skill_bundle_pairs.parquet", {}),
            (location_toplists, "location_toplists.parquet", {}),
        ]

        with timed("Upload to R2 (parallel, md5 dedup)"):
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = {pool.submit(_upload, args): args[1] for args in uploads}
                for fut in as_completed(futures):
                    exc = fut.exception()
                    if exc:
                        raise RuntimeError(f"Upload failed for {futures[fut]}") from exc
                    print(f"     {fut.result()}")

    print(f"\nDone. Total wall time: {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()

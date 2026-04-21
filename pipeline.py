from __future__ import annotations

import tempfile

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from data.loader import R2_BUCKET, _get_s3_client, download_kaggle_data, upload_parquet_to_r2
from data.processor import (
    build_features,
    build_label_rollup,
    build_skill_bundle_pairs,
    build_skill_theme_map,
    get_merged,
    score_topics,
    train_skill_theme_model,
)


def _exists_in_r2(key: str, s3: boto3.client) -> bool:
    try:
        s3.head_object(Bucket=R2_BUCKET, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise


def _upload_if_missing(df, key: str, s3: boto3.client) -> None:
    if _exists_in_r2(key, s3):
        print(f"Skipping {key} (already exists in R2)")
        return
    upload_parquet_to_r2(df, key, s3)


def main() -> None:
    load_dotenv()
    with tempfile.TemporaryDirectory() as tmpdir:
        print("Downloading Kaggle dataset...")
        download_kaggle_data(tmpdir)

        print("Loading and merging data...")
        merged, skills_raw = get_merged(tmpdir)
        print(f"Merged: {len(merged):,} rows")

        print("Training skill theme model...")
        vec, clf = train_skill_theme_model(skills_raw)

        print("Building features...")
        featured = build_features(merged, vec, clf)

        print("Scoring course topics...")
        topic_rankings = score_topics(featured)
        print(f"Qualifying topics: {len(topic_rankings):,}")

        print("Building derived tables...")
        label_rollup = build_label_rollup(topic_rankings)
        skill_theme_map = build_skill_theme_map(skills_raw, vec, clf)
        skill_bundles = build_skill_bundle_pairs(skills_raw)

        print("Uploading to R2...")
        s3 = _get_s3_client()
        _upload_if_missing(featured, "merged.parquet", s3)
        _upload_if_missing(topic_rankings, "topic_rankings.parquet", s3)
        _upload_if_missing(label_rollup, "label_rollup.parquet", s3)
        _upload_if_missing(skill_theme_map, "skill_theme_map.parquet", s3)
        _upload_if_missing(skill_bundles, "skill_bundles.parquet", s3)
        print("Done.")


if __name__ == "__main__":
    main()

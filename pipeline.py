from __future__ import annotations

import tempfile

from data.loader import _get_s3_client, download_kaggle_data, upload_parquet_to_r2
from data.processor import (
    build_features,
    build_label_rollup,
    build_skill_bundle_pairs,
    build_skill_theme_map,
    get_merged,
    score_topics,
    train_skill_theme_model,
)


def main() -> None:
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
        upload_parquet_to_r2(featured, "merged.parquet", s3)
        upload_parquet_to_r2(topic_rankings, "topic_rankings.parquet", s3)
        upload_parquet_to_r2(label_rollup, "label_rollup.parquet", s3)
        upload_parquet_to_r2(skill_theme_map, "skill_theme_map.parquet", s3)
        upload_parquet_to_r2(skill_bundles, "skill_bundles.parquet", s3)
        print("Done.")


if __name__ == "__main__":
    main()

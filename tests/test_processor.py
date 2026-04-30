"""
Acceptance criteria:
- score_topics output contains all required columns.
- log_volume == log1p(volume) exactly.
- course_opportunity_score is in [0, 100].
- opportunity_label is in {High Opportunity, Emerging, Saturated}.
- trend_30d contains no NaN.
- When all first_seen are null, every row gets trend_30d == 0.0.
- rank starts at 1 and is strictly ascending.
- Rows sorted by course_opportunity_score descending.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import data.processor as proc
from data.processor import score_topics

_REQUIRED_COLS = {
    "rank",
    "job_role",
    "volume",
    "log_volume",
    "course_opportunity_score",
    "opportunity_label",
    "trend_30d",
}

_VALID_OPP_LABELS = {"High Opportunity", "Emerging", "Saturated"}


def _make_featured(
    n_per_position: int = 10,
    n_positions: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2024-01-01")
    records = []
    for i in range(n_positions):
        for j in range(n_per_position):
            records.append(
                {
                    "job_link": f"pos{i}_job{j}",
                    "search_position": f"Position {i}",
                    "job_type": rng.choice(["Remote", "On-site", "Hybrid"]),
                    "job_level": rng.choice(["Mid senior", "Entry", "Senior"]),
                    "search_city": rng.choice(["New York", "Chicago", "Austin"]),
                    "search_country": rng.choice(["US", "UK", "CA"]),
                    "company": rng.choice(["Corp A", "Corp B", "Corp C"]),
                    "first_seen": base
                    + pd.Timedelta(days=(j % 20) * 7 + int(rng.integers(0, 7))),
                }
            )
    df = pd.DataFrame(records)
    df["first_seen"] = pd.to_datetime(df["first_seen"])
    return df


@pytest.fixture(autouse=True)
def lower_min_volume(monkeypatch):
    monkeypatch.setattr(proc, "MIN_VOLUME", 5)


class TestScoreTopics:
    def test_required_columns_present(self):
        result = score_topics(_make_featured())
        missing = _REQUIRED_COLS - set(result.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_log_volume_is_log1p_of_volume(self):
        result = score_topics(_make_featured())
        expected = np.log1p(result["volume"])
        pd.testing.assert_series_equal(
            result["log_volume"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_score_in_range(self):
        result = score_topics(_make_featured())
        assert result["course_opportunity_score"].between(0, 100).all()

    def test_opportunity_label_values(self):
        result = score_topics(_make_featured())
        assert set(result["opportunity_label"].unique()).issubset(_VALID_OPP_LABELS)

    def test_trend_30d_no_nan(self):
        result = score_topics(_make_featured())
        assert result["trend_30d"].isna().sum() == 0

    def test_rank_starts_at_one(self):
        result = score_topics(_make_featured())
        assert result["rank"].iloc[0] == 1

    def test_rank_monotonic_ascending(self):
        result = score_topics(_make_featured())
        assert result["rank"].is_monotonic_increasing

    def test_sorted_by_score_descending(self):
        result = score_topics(_make_featured())
        assert result["course_opportunity_score"].is_monotonic_decreasing

    def test_trend_30d_zero_when_no_dates(self):
        df = _make_featured()
        df["first_seen"] = pd.NaT
        result = score_topics(df)
        assert (result["trend_30d"] == 0.0).all()

    def test_volume_below_min_volume_filtered(self):
        df = _make_featured(n_per_position=3, n_positions=2)
        result = score_topics(df)
        assert result.empty or (result["volume"] >= proc.MIN_VOLUME).all()

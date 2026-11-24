"""Unit tests for the drift-triggered retraining logic (utils/drift.py)."""

import pandas as pd
import pytest
from utils.drift import (
    COOLDOWN_SECONDS,
    GRID_ZSCORE_THRESHOLD,
    MIN_NEW_ROWS,
    RATE_DELTA_THRESHOLD,
    count_new_labelled_rows,
    decide_retrain,
    load_last_trigger,
    parse_metrics,
    save_last_trigger,
)
from utils.training import FEATURES, label_rows

NOW = 1_000_000.0
QUIET = {"grid_zscore": 0.2, "rate_delta": 0.05}
GRID_DRIFT = {"grid_zscore": 1.4, "rate_delta": 0.05}
RATE_DRIFT = {"grid_zscore": 0.2, "rate_delta": 0.3}


def _decide(metrics, new_rows=150, last=None):
    return decide_retrain(metrics, new_rows, last, NOW)


# ---------------------------------------------------------------- decision


def test_no_drift_never_retrains_even_with_lots_of_data():
    d = _decide(QUIET, new_rows=5000)
    assert not d.retrain and not d.drift
    assert "no drift" in d.reason


def test_grid_drift_with_enough_data_retrains():
    d = _decide(GRID_DRIFT)
    assert d.retrain and d.drift and d.enough_data and not d.cooling_down
    assert "Grid z-score" in d.reason


def test_rate_drift_with_enough_data_retrains():
    d = _decide(RATE_DRIFT)
    assert d.retrain
    assert "top-10 rate" in d.reason


def test_both_drifts_are_reported():
    d = _decide({"grid_zscore": 2.0, "rate_delta": 0.5})
    assert d.retrain
    assert "Grid z-score" in d.reason and "top-10 rate" in d.reason


def test_drift_without_enough_new_rows_does_not_retrain():
    d = _decide(GRID_DRIFT, new_rows=MIN_NEW_ROWS - 1)
    assert d.drift and not d.enough_data and not d.retrain
    assert f"only {MIN_NEW_ROWS - 1}" in d.reason


def test_exactly_the_minimum_rows_is_enough():
    # same rule as training_dag, which only skips below the minimum
    assert _decide(GRID_DRIFT, new_rows=MIN_NEW_ROWS).retrain


def test_thresholds_are_strict_greater_than_like_grafana():
    at = {"grid_zscore": GRID_ZSCORE_THRESHOLD, "rate_delta": RATE_DELTA_THRESHOLD}
    assert not _decide(at).drift
    above = {"grid_zscore": GRID_ZSCORE_THRESHOLD + 0.01, "rate_delta": 0.0}
    assert _decide(above).drift


def test_cooldown_blocks_a_second_trigger():
    ten_min_ago = NOW - 600
    d = _decide(GRID_DRIFT, last=ten_min_ago)
    assert d.drift and d.enough_data and d.cooling_down and not d.retrain
    assert "10 min ago" in d.reason


def test_cooldown_expires():
    just_over = NOW - COOLDOWN_SECONDS - 1
    assert _decide(GRID_DRIFT, last=just_over).retrain


def test_never_triggered_before_means_no_cooldown():
    assert not _decide(GRID_DRIFT, last=None).cooling_down


def test_missing_metrics_count_as_no_drift():
    assert not _decide({}).drift
    assert not _decide({"grid_zscore": None, "rate_delta": None}).drift


# ------------------------------------------------------------ SQL row parsing


def test_parse_metrics_from_a_row():
    assert parse_metrics((1.5, 0.25, 40)) == {
        "grid_zscore": 1.5,
        "rate_delta": 0.25,
        "n_predictions": 40,
    }


@pytest.mark.parametrize("row", [None, (), (None, None, None)])
def test_parse_metrics_without_data_means_no_drift(row):
    assert parse_metrics(row) == {
        "grid_zscore": 0.0,
        "rate_delta": 0.0,
        "n_predictions": 0,
    }


# ------------------------------------------------------------- cooldown state


def test_state_roundtrip(tmp_path):
    path = str(tmp_path / "state.json")
    save_last_trigger(path, 123.5)
    assert load_last_trigger(path) == 123.5


def test_state_missing_or_corrupt_means_never_triggered(tmp_path):
    assert load_last_trigger(str(tmp_path / "nope.json")) is None
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    assert load_last_trigger(str(bad)) is None


# --------------------------------------------------------- new labelled rows


def _rows(n, start=0):
    return pd.DataFrame(
        {
            "Season": [2023] * n,
            "Round": [1 + (start + i) % 20 for i in range(n)],
            "Grid": [1 + i % 20 for i in range(n)],
            "CircuitID": ["monaco"] * n,
            "DriverID": [f"driver_{start + i}" for i in range(n)],
            "ConstructorName": ["Ferrari"] * n,
        }
    )[FEATURES]


def _labels(rows):
    out = rows[["Season", "Round", "DriverID"]].copy()
    out["top10"] = 1
    return out


def test_count_matches_the_training_dags_labelled_rows(tmp_path):
    good = tmp_path / "good"
    good.mkdir()
    rows = _rows(60)
    rows.iloc[:30].to_csv(good / "a.csv", index=False)
    rows.iloc[30:].to_csv(good / "b.csv", index=False)
    labels = tmp_path / "labels.csv"
    _labels(rows).to_csv(labels, index=False)

    expected = len(label_rows(rows, _labels(rows)))
    assert 0 < expected < 60  # some rows are held out for the test set
    assert count_new_labelled_rows(str(good), str(labels)) == expected


def test_unlabelled_rows_are_not_counted(tmp_path):
    good = tmp_path / "good"
    good.mkdir()
    _rows(40).to_csv(good / "a.csv", index=False)
    labels = tmp_path / "labels.csv"
    _labels(_rows(10, start=1000)).to_csv(labels, index=False)  # other drivers
    assert count_new_labelled_rows(str(good), str(labels)) == 0


def test_empty_good_data_or_missing_labels_counts_zero(tmp_path):
    good = tmp_path / "good"
    good.mkdir()
    assert count_new_labelled_rows(str(good), str(tmp_path / "labels.csv")) == 0
    _rows(10).to_csv(good / "a.csv", index=False)
    assert count_new_labelled_rows(str(good), str(tmp_path / "labels.csv")) == 0

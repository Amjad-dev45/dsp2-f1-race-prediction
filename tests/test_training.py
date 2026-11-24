"""Unit tests for the training DAG logic."""

import pandas as pd
import pytest
from utils.training import (
    FEATURES,
    TARGET,
    build_pipeline,
    holdout_set,
    is_test_row,
    label_rows,
    promotion_decision,
    training_set,
    training_stats,
)


def _pool(n=60):
    return pd.DataFrame(
        {
            "Season": [2022 + i % 3 for i in range(n)],
            "Round": [1 + i % 20 for i in range(n)],
            "Grid": [1 + i % 20 for i in range(n)],
            "CircuitID": [f"c{i % 4}" for i in range(n)],
            "DriverID": [f"d{i}" for i in range(n)],
            "ConstructorName": [f"t{i % 5}" for i in range(n)],
        }
    )


def _labels(pool):
    out = pool[["Season", "Round", "DriverID"]].copy()
    out[TARGET] = (pool["Grid"] <= 10).astype(int)
    return out


def test_test_set_and_training_rows_never_overlap():
    pool = _pool()
    labels = _labels(pool)
    test = holdout_set(pool, labels)
    train = label_rows(pool, labels)
    keys = ["Season", "Round", "DriverID"]
    assert len(test) + len(train) == len(pool)
    assert test.merge(train, on=keys).empty


def test_test_membership_is_stable_for_float_formatted_rows():
    pool = _pool()
    floaty = pool.astype({"Season": float, "Round": float, "Grid": float})
    labels = _labels(pool)
    assert len(label_rows(floaty, labels)) == len(label_rows(pool, labels))


def test_is_test_row_is_deterministic():
    pool = _pool()
    assert is_test_row(pool).equals(is_test_row(pool.copy()))


def test_unlabelled_rows_are_dropped():
    pool = _pool()
    labels = _labels(pool).iloc[:10]
    assert len(label_rows(pool, labels)) <= 10


def test_training_stats_cover_numeric_categorical_and_label():
    df = _pool()
    df[TARGET] = 1
    stats = training_stats(df)
    by_key = {(s["feature"], s["stat"], s["category"]): s["value"] for s in stats}
    assert by_key[("_all", "row_count", None)] == 60
    assert by_key[("Grid", "mean", None)] == pytest.approx(df["Grid"].mean())
    shares = [s["value"] for s in stats if s["feature"] == "ConstructorName"]
    assert sum(shares) == pytest.approx(1.0)
    assert by_key[(TARGET, "positive_rate", None)] == 1.0


@pytest.mark.parametrize(
    "cand, champ, ms, ok",
    [
        (0.80, None, 10, True),  # first model
        (0.80, 0.79, 10, True),  # better
        (0.80, 0.80, 10, True),  # equal is enough
        (0.78, 0.80, 10, False),  # worse
        (0.90, 0.80, 2000, False),  # too slow
    ],
)
def test_promotion_decision(cand, champ, ms, ok):
    decided, reason = promotion_decision(cand, champ, ms, 1000)
    assert decided is ok and reason


def test_pipeline_step_names_match_the_api():
    pool = _pool()
    model = build_pipeline().fit(pool[FEATURES], (pool["Grid"] <= 10).astype(int))
    encoder = model.named_steps["preprocessor"].named_transformers_["cat"]
    assert list(encoder.feature_names_in_) == [
        "CircuitID",
        "DriverID",
        "ConstructorName",
    ]


def test_training_set_keeps_int_target_with_no_new_rows():
    pool = _pool()
    base = pool.assign(**{TARGET: [i % 2 for i in range(len(pool))]})
    empty = label_rows(pool.iloc[0:0], _labels(pool))
    train = training_set(base, empty)
    assert len(train) == len(base)
    assert train[TARGET].dtype.kind == "i"
    build_pipeline().fit(train[FEATURES], train[TARGET])

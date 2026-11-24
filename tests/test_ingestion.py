"""Unit tests for the ingestion DAG logic (criticality, splitting, stats)."""

import pandas as pd
import pytest
from utils.ge_helpers import compute_criticality, validate_dataframe
from utils.ingestion import STATS_COLUMNS, build_stats_row, split_good_bad


@pytest.mark.parametrize(
    "schema_error, percent_invalid, expected",
    [
        (False, 0, "none"),
        (False, 5, "low"),
        (False, 9.99, "low"),
        (False, 10, "medium"),
        (False, 50, "medium"),
        (False, 50.01, "high"),
        (False, 100, "high"),
        (True, 0, "high"),  # missing column is always high
    ],
)
def test_compute_criticality(schema_error, percent_invalid, expected):
    assert compute_criticality(schema_error, percent_invalid) == expected


def _frame(n=10):
    return pd.DataFrame({"Grid": range(1, n + 1), "DriverID": ["x"] * n})


def test_split_all_good():
    good, bad = split_good_bad(_frame(), [])
    assert len(good) == 10 and bad.empty


def test_split_all_bad():
    good, bad = split_good_bad(_frame(), list(range(10)))
    assert good.empty and len(bad) == 10


def test_split_mixed_keeps_every_row_exactly_once():
    df = _frame()
    good, bad = split_good_bad(df, [0, 3, 7])
    assert list(bad.index) == [0, 3, 7]
    assert len(good) + len(bad) == len(df)
    assert set(good.index).isdisjoint(bad.index)


def _validation_result(**overrides):
    v = {
        "file_name": "batch_001.csv",
        "total_rows": 10,
        "invalid_rows": 3,
        "criticality": "medium",
        "schema_error": False,
        "error_counts": {
            "null": 1,
            "type_error": 2,
            "range": 0,
            "consistency": 4,
            "format": 0,
            "duplicate": 2,
            "schema": 0,
        },
    }
    v.update(overrides)
    return v


def test_build_stats_row_maps_columns():
    row = dict(zip(STATS_COLUMNS, build_stats_row(_validation_result())))
    assert row["valid_rows"] == 7
    assert row["invalid_rows"] == 3
    assert row["null_count"] == 1
    assert row["type_error_count"] == 2
    assert row["consistency_count"] == 4
    assert row["duplicate_count"] == 2
    assert row["schema_error"] is False


def test_build_stats_row_length_matches_columns():
    assert len(build_stats_row(_validation_result())) == len(STATS_COLUMNS)


def test_missing_column_is_schema_error_without_running_gx():
    # schema pre-check returns before Great Expectations is touched
    df = pd.DataFrame({"Season": [2024], "Round": [1]})
    v = validate_dataframe(df)
    assert v["schema_error"] is True
    assert v["criticality"] == "high"
    assert v["invalid_rows"] == v["total_rows"] == 1
    assert "Grid" in v["missing_columns"]


def test_teams_alert_is_an_adaptive_card():
    from utils.ingestion import build_teams_alert

    v = _validation_result(
        percent_invalid=30.0,
        missing_columns=[],
        report_filename="validations/s/__none__/r.html",
    )
    msg = build_teams_alert(v)
    attachment = msg["attachments"][0]
    assert msg["type"] == "message"
    assert attachment["contentType"] == "application/vnd.microsoft.card.adaptive"
    card = attachment["content"]
    assert card["type"] == "AdaptiveCard"
    assert "MEDIUM" in card["body"][0]["text"]
    facts = {f["title"]: f["value"] for f in card["body"][1]["facts"]}
    assert facts["File"] == "batch_001.csv"
    assert facts["Invalid rows"] == "3/10 (30.0%)"
    assert "consistency: 4" in facts["Errors"] and "range" not in facts["Errors"]
    assert facts["Report"] == "validations/s/\\_\\_none\\_\\_/r.html"


def test_teams_alert_links_report_when_data_docs_are_served():
    from utils.ingestion import build_teams_alert

    v = _validation_result(
        percent_invalid=30.0, missing_columns=[], report_filename="validations/r.html"
    )
    card = build_teams_alert(v, "http://localhost:8080/")["attachments"][0]["content"]
    facts = {f["title"]: f["value"] for f in card["body"][1]["facts"]}
    assert (
        facts["Report"]
        == "[validations/r.html](http://localhost:8080/validations/r.html)"
    )

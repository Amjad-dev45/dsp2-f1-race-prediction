"""Pydantic request/response validation of the model service."""

import pytest

VALID = {
    "Season": 2025,
    "Round": 1,
    "CircuitID": "bahrain",
    "DriverID": "verstappen",
    "ConstructorName": "Red Bull",
    "Grid": 1,
}


def test_prediction_response_has_features_and_model_version(client):
    response = client.post("/predict", json={"features": [VALID]})
    assert response.status_code == 200
    row = response.json()[0]
    assert row["driver_id"] == "verstappen"
    assert row["grid"] == 1
    assert row["source"] == "webapp"
    assert row["model_version"]
    assert 0.0 <= row["probability"] <= 1.0


def test_empty_features_is_400(client):
    response = client.post("/predict", json={"features": []})
    assert response.status_code == 400


@pytest.mark.parametrize(
    "field, value",
    [
        ("Grid", "abc"),  # wrong type
        ("Grid", -5),  # out of range
        ("Season", None),  # missing value
        ("DriverID", ""),  # empty string
    ],
)
def test_invalid_feature_is_422(client, field, value):
    response = client.post("/predict", json={"features": [{**VALID, field: value}]})
    assert response.status_code == 422


def test_missing_feature_column_is_422(client):
    row = {k: v for k, v in VALID.items() if k != "Grid"}
    response = client.post("/predict", json={"features": [row]})
    assert response.status_code == 422


def test_unknown_source_is_422(client):
    response = client.post("/predict", json={"features": [VALID], "source": "cli"})
    assert response.status_code == 422


def test_past_predictions_filters_by_source_and_limit(client):
    client.post("/predict", json={"features": [VALID, VALID], "source": "scheduled"})
    client.post("/predict", json={"features": [VALID]})

    scheduled = client.get("/past-predictions", params={"source": "scheduled"})
    assert scheduled.status_code == 200
    assert {r["source"] for r in scheduled.json()} == {"scheduled"}

    limited = client.get("/past-predictions", params={"limit": 1})
    assert len(limited.json()) == 1


def test_past_predictions_rejects_inverted_dates(client):
    response = client.get(
        "/past-predictions", params={"start": "2026-02-01", "end": "2026-01-01"}
    )
    assert response.status_code == 400

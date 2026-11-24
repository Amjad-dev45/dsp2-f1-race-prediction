from unittest.mock import MagicMock, patch

from model_service import app as app_module
from model_service.app import ml


def test_reload_model_swaps_in_champion(client):
    new_model = MagicMock()
    with patch.object(app_module, "load_champion", return_value=(new_model, "7")):
        old_version = ml["version"]
        resp = client.post("/reload-model")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "reloaded",
        "previous_version": old_version,
        "version": "7",
    }
    assert ml["model"] is new_model
    assert ml["version"] == "7"


def test_reload_model_failure_keeps_current_model(client):
    old_model, old_version = ml["model"], ml["version"]
    with patch.object(
        app_module, "load_champion", side_effect=RuntimeError("no champion")
    ):
        resp = client.post("/reload-model")
    assert resp.status_code == 503
    assert ml["model"] is old_model
    assert ml["version"] == old_version

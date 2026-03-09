"""Tests for imagined speech model and API route (#166)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI


COMMANDS = ["yes", "no", "stop", "go", "left", "right", "up", "down"]


def make_signal(n=512, freq=10.0, fs=256.0):
    t = np.linspace(0, n / fs, n)
    return (np.sin(2 * np.pi * freq * t)).tolist()


@pytest.fixture
def client():
    from api.routes.imagined_speech import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_model_import():
    from models.imagined_speech import ImaginedSpeechModel, get_model
    assert ImaginedSpeechModel is not None
    assert get_model() is not None


def test_predict_1d():
    from models.imagined_speech import get_model
    result = get_model().predict(np.random.randn(512))
    assert "predicted_command" in result


def test_predict_multichannel():
    from models.imagined_speech import get_model
    result = get_model().predict(np.random.randn(4, 512))
    assert "lateral_asymmetry" in result


def test_command_valid():
    from models.imagined_speech import get_model
    result = get_model().predict(np.random.randn(512))
    assert result["predicted_command"] in COMMANDS


def test_confidence_range():
    from models.imagined_speech import get_model
    result = get_model().predict(np.random.randn(512))
    assert 0.0 <= result["confidence"] <= 1.0


def test_probabilities_sum():
    from models.imagined_speech import get_model
    result = get_model().predict(np.random.randn(512))
    total = sum(result["probabilities"].values())
    assert abs(total - 1.0) < 0.01


def test_probabilities_all_commands():
    from models.imagined_speech import get_model
    result = get_model().predict(np.random.randn(512))
    for cmd in COMMANDS:
        assert cmd in result["probabilities"]


def test_lateral_asymmetry_range():
    from models.imagined_speech import get_model
    result = get_model().predict(np.random.randn(4, 512))
    assert -1.0 <= result["lateral_asymmetry"] <= 1.0


def test_is_reliable_bool():
    from models.imagined_speech import get_model
    result = get_model().predict(np.random.randn(512))
    assert isinstance(result["is_reliable"], bool)


def test_model_used_field():
    from models.imagined_speech import get_model
    result = get_model().predict(np.random.randn(512))
    assert "imagined_speech" in result["model_used"]


def test_short_signal():
    from models.imagined_speech import get_model
    result = get_model().predict(np.random.randn(64))
    assert "predicted_command" in result


def test_api_analyze(client):
    payload = {"signals": [make_signal()], "fs": 256.0, "user_id": "u1"}
    resp = client.post("/imagined-speech/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["predicted_command"] in COMMANDS


def test_api_multichannel(client):
    payload = {"signals": [make_signal(), make_signal(freq=8.0), make_signal(freq=20.0), make_signal(freq=6.0)]}
    resp = client.post("/imagined-speech/analyze", json=payload)
    assert resp.status_code == 200


def test_api_history_empty(client):
    resp = client.get("/imagined-speech/history/newuser")
    assert resp.json()["count"] == 0


def test_api_history_populated(client):
    client.post("/imagined-speech/analyze", json={"signals": [make_signal()], "user_id": "h_u"})
    resp = client.get("/imagined-speech/history/h_u")
    assert resp.json()["count"] >= 1


def test_api_reset(client):
    client.post("/imagined-speech/analyze", json={"signals": [make_signal()], "user_id": "r_u"})
    resp = client.post("/imagined-speech/reset/r_u")
    assert resp.json()["status"] == "reset"


def test_api_history_after_reset(client):
    client.post("/imagined-speech/analyze", json={"signals": [make_signal()], "user_id": "r2"})
    client.post("/imagined-speech/reset/r2")
    assert client.get("/imagined-speech/history/r2").json()["count"] == 0


def test_api_processed_at(client):
    resp = client.post("/imagined-speech/analyze", json={"signals": [make_signal()]})
    assert resp.json()["processed_at"] > 0


def test_api_history_limit(client):
    for _ in range(5):
        client.post("/imagined-speech/analyze", json={"signals": [make_signal()], "user_id": "lim"})
    resp = client.get("/imagined-speech/history/lim?limit=3")
    assert resp.json()["count"] <= 3

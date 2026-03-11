"""Tests for altered consciousness model and API route (#161)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI


def make_signal(n=512, freq=10.0, fs=256.0, amp=1.0):
    t = np.linspace(0, n / fs, n)
    return (amp * np.sin(2 * np.pi * freq * t)).tolist()


@pytest.fixture
def client():
    from api.routes.altered_consciousness import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_model_import():
    from models.altered_consciousness import AlteredConsciousnessModel, get_model
    assert AlteredConsciousnessModel is not None
    assert get_model() is not None


def test_predict_1d():
    from models.altered_consciousness import get_model
    sig = np.random.randn(512)
    result = get_model().predict(sig)
    assert "state" in result
    assert "altered_consciousness_index" in result


def test_predict_2d():
    from models.altered_consciousness import get_model
    sig = np.random.randn(4, 512)
    result = get_model().predict(sig)
    assert "state" in result


def test_state_values():
    from models.altered_consciousness import get_model
    states = {"normal", "light_altered", "moderate_altered", "deep_altered", "transcendent"}
    sig = np.random.randn(512)
    result = get_model().predict(sig)
    assert result["state"] in states


def test_index_range():
    from models.altered_consciousness import get_model
    sig = np.random.randn(512)
    result = get_model().predict(sig)
    assert 0.0 <= result["altered_consciousness_index"] <= 1.0


def test_theta_fraction_range():
    from models.altered_consciousness import get_model
    result = get_model().predict(np.random.randn(512))
    assert 0.0 <= result["theta_fraction"] <= 1.0


def test_alpha_fraction_range():
    from models.altered_consciousness import get_model
    result = get_model().predict(np.random.randn(512))
    assert 0.0 <= result["alpha_fraction"] <= 1.0


def test_beta_suppression_range():
    from models.altered_consciousness import get_model
    result = get_model().predict(np.random.randn(512))
    assert 0.0 <= result["beta_suppression"] <= 1.0


def test_spectral_entropy_range():
    from models.altered_consciousness import get_model
    result = get_model().predict(np.random.randn(512))
    assert 0.0 <= result["spectral_entropy"] <= 1.0


def test_model_used_field():
    from models.altered_consciousness import get_model
    result = get_model().predict(np.random.randn(512))
    assert result["model_used"] == "feature_based_spectral"


def test_short_signal():
    from models.altered_consciousness import get_model
    sig = np.random.randn(64)
    result = get_model().predict(sig)
    assert "state" in result


def test_api_analyze(client):
    payload = {"signals": [make_signal()], "fs": 256.0, "user_id": "u1"}
    resp = client.post("/altered-consciousness/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == "u1"
    assert "state" in data


def test_api_analyze_2ch(client):
    payload = {"signals": [make_signal(), make_signal(freq=6.0)], "fs": 256.0, "user_id": "test-user"}
    resp = client.post("/altered-consciousness/analyze", json=payload)
    assert resp.status_code == 200


def test_api_history_empty(client):
    resp = client.get("/altered-consciousness/history/newuser")
    assert resp.status_code == 200
    assert resp.json()["count"] == 0


def test_api_history_populated(client):
    payload = {"signals": [make_signal()], "user_id": "hist_user"}
    client.post("/altered-consciousness/analyze", json=payload)
    resp = client.get("/altered-consciousness/history/hist_user")
    assert resp.json()["count"] >= 1


def test_api_reset(client):
    payload = {"signals": [make_signal()], "user_id": "reset_user"}
    client.post("/altered-consciousness/analyze", json=payload)
    resp = client.post("/altered-consciousness/reset/reset_user")
    assert resp.status_code == 200
    assert resp.json()["status"] == "reset"


def test_api_history_after_reset(client):
    payload = {"signals": [make_signal()], "user_id": "reset2"}
    client.post("/altered-consciousness/analyze", json=payload)
    client.post("/altered-consciousness/reset/reset2")
    resp = client.get("/altered-consciousness/history/reset2")
    assert resp.json()["count"] == 0


def test_api_processed_at(client):
    payload = {"signals": [make_signal()], "user_id": "test-user"}
    resp = client.post("/altered-consciousness/analyze", json=payload)
    assert resp.json()["processed_at"] > 0


def test_api_history_limit(client):
    for _ in range(5):
        client.post("/altered-consciousness/analyze", json={"signals": [make_signal()], "user_id": "limit_user"})
    resp = client.get("/altered-consciousness/history/limit_user?limit=3")
    assert resp.json()["count"] <= 3

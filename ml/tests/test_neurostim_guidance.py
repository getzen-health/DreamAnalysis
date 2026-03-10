"""Tests for neurostimulation guidance model and API route (#163)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI


def make_signal(n=512, freq=10.0, fs=256.0):
    t = np.linspace(0, n / fs, n)
    return (np.sin(2 * np.pi * freq * t)).tolist()


@pytest.fixture
def client():
    from api.routes.neurostim_guidance import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_model_import():
    from models.neurostim_guidance import NeurostimGuidanceModel, get_model
    assert NeurostimGuidanceModel is not None
    assert get_model() is not None


def test_predict_default_protocol():
    from models.neurostim_guidance import get_model
    result = get_model().predict(np.random.randn(1, 512))
    assert result["protocol"] == "alpha_entrainment"


def test_predict_theta_suppression():
    from models.neurostim_guidance import get_model
    result = get_model().predict(np.random.randn(1, 512), target_protocol="theta_suppression")
    assert result["protocol"] == "theta_suppression"
    assert result["stim_frequency_hz"] == 40.0


def test_predict_beta_upregulation():
    from models.neurostim_guidance import get_model
    result = get_model().predict(np.random.randn(1, 512), target_protocol="beta_upregulation")
    assert result["protocol"] == "beta_upregulation"
    assert result["stim_frequency_hz"] == 20.0


def test_predict_delta_suppression():
    from models.neurostim_guidance import get_model
    result = get_model().predict(np.random.randn(1, 512), target_protocol="delta_suppression")
    assert result["protocol"] == "delta_suppression"
    assert result["stim_frequency_hz"] == 1.0


def test_intensity_range():
    from models.neurostim_guidance import get_model
    result = get_model().predict(np.random.randn(512))
    assert 0.0 <= result["suggested_intensity_normalized"] <= 1.0


def test_readiness_range():
    from models.neurostim_guidance import get_model
    result = get_model().predict(np.random.randn(512))
    assert 0.0 <= result["readiness_score"] <= 1.0


def test_iaf_in_alpha_range():
    from models.neurostim_guidance import get_model
    result = get_model().predict(np.random.randn(512))
    assert 7.0 <= result["iaf_hz"] <= 13.0


def test_should_stimulate_bool():
    from models.neurostim_guidance import get_model
    result = get_model().predict(np.random.randn(512))
    assert isinstance(result["should_stimulate"], bool)


def test_model_used_field():
    from models.neurostim_guidance import get_model
    result = get_model().predict(np.random.randn(512))
    assert "closed_loop" in result["model_used"]


def test_short_signal():
    from models.neurostim_guidance import get_model
    result = get_model().predict(np.random.randn(64))
    assert "protocol" in result


def test_api_analyze(client):
    payload = {"signals": [make_signal()], "fs": 256.0, "user_id": "u1", "target_protocol": "alpha_entrainment"}
    resp = client.post("/neurostim/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["protocol"] == "alpha_entrainment"


def test_api_theta_protocol(client):
    payload = {"signals": [make_signal()], "target_protocol": "theta_suppression", "user_id": "test-user"}
    resp = client.post("/neurostim/analyze", json=payload)
    assert resp.json()["stim_frequency_hz"] == 40.0


def test_api_history_empty(client):
    resp = client.get("/neurostim/history/newuser")
    assert resp.json()["count"] == 0


def test_api_history_populated(client):
    client.post("/neurostim/analyze", json={"signals": [make_signal()], "user_id": "h_u"})
    resp = client.get("/neurostim/history/h_u")
    assert resp.json()["count"] >= 1


def test_api_reset(client):
    client.post("/neurostim/analyze", json={"signals": [make_signal()], "user_id": "r_u"})
    resp = client.post("/neurostim/reset/r_u")
    assert resp.status_code == 200
    assert resp.json()["status"] == "reset"


def test_api_history_after_reset(client):
    client.post("/neurostim/analyze", json={"signals": [make_signal()], "user_id": "r2"})
    client.post("/neurostim/reset/r2")
    assert client.get("/neurostim/history/r2").json()["count"] == 0


def test_api_processed_at(client):
    resp = client.post("/neurostim/analyze", json={"signals": [make_signal()], "user_id": "test-user"})
    assert resp.json()["processed_at"] > 0


def test_api_history_limit(client):
    for _ in range(5):
        client.post("/neurostim/analyze", json={"signals": [make_signal()], "user_id": "lim"})
    resp = client.get("/neurostim/history/lim?limit=2")
    assert resp.json()["count"] <= 2

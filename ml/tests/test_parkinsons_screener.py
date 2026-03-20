"""Tests for Parkinson's screener model and API route (#168)."""
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
    from api.routes.parkinsons_screener import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_model_import():
    from models.parkinsons_screener import ParkinsonsScreener, get_model
    assert ParkinsonsScreener is not None
    assert get_model() is not None


def test_predict_1d():
    from models.parkinsons_screener import get_model
    result = get_model().predict(np.random.randn(512))
    assert "risk_category" in result


def test_predict_2d():
    from models.parkinsons_screener import get_model
    result = get_model().predict(np.random.randn(4, 512))
    assert "pd_risk_score" in result


def test_risk_categories():
    from models.parkinsons_screener import get_model
    cats = {"low_risk", "mild_concern", "moderate_concern", "high_risk"}
    result = get_model().predict(np.random.randn(512))
    assert result["risk_category"] in cats


def test_risk_score_range():
    from models.parkinsons_screener import get_model
    result = get_model().predict(np.random.randn(512))
    assert 0.0 <= result["pd_risk_score"] <= 1.0


def test_beta_burden_range():
    from models.parkinsons_screener import get_model
    result = get_model().predict(np.random.randn(512))
    assert 0.0 <= result["beta_burden"] <= 1.0


def test_tremor_index_range():
    from models.parkinsons_screener import get_model
    result = get_model().predict(np.random.randn(512))
    assert 0.0 <= result["tremor_oscillation_index"] <= 1.0


def test_paf_range():
    from models.parkinsons_screener import get_model
    result = get_model().predict(np.random.randn(512))
    assert 7.0 <= result["peak_alpha_freq_hz"] <= 13.0


def test_note_field():
    from models.parkinsons_screener import get_model
    result = get_model().predict(np.random.randn(512))
    assert "Wellness indicator only" in result["note"]


def test_model_used_field():
    from models.parkinsons_screener import get_model
    result = get_model().predict(np.random.randn(512))
    assert "beta_tremor" in result["model_used"]


def test_short_signal():
    from models.parkinsons_screener import get_model
    result = get_model().predict(np.random.randn(64))
    assert "risk_category" in result


def test_api_analyze(client):
    payload = {"signals": [make_signal()], "fs": 256.0, "user_id": "u1"}
    resp = client.post("/parkinsons-screen/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == "u1"
    assert "risk_category" in data


def test_api_note_field(client):
    resp = client.post("/parkinsons-screen/analyze", json={"signals": [make_signal()], "user_id": "test-user"})
    assert "Wellness indicator only" in resp.json()["note"]


def test_api_history_empty(client):
    resp = client.get("/parkinsons-screen/history/newuser")
    assert resp.json()["count"] == 0


def test_api_history_populated(client):
    client.post("/parkinsons-screen/analyze", json={"signals": [make_signal()], "user_id": "h_u"})
    resp = client.get("/parkinsons-screen/history/h_u")
    assert resp.json()["count"] >= 1


def test_api_reset(client):
    client.post("/parkinsons-screen/analyze", json={"signals": [make_signal()], "user_id": "r_u"})
    resp = client.post("/parkinsons-screen/reset/r_u")
    assert resp.json()["status"] == "reset"


def test_api_history_after_reset(client):
    client.post("/parkinsons-screen/analyze", json={"signals": [make_signal()], "user_id": "r2"})
    client.post("/parkinsons-screen/reset/r2")
    assert client.get("/parkinsons-screen/history/r2").json()["count"] == 0


def test_api_processed_at(client):
    resp = client.post("/parkinsons-screen/analyze", json={"signals": [make_signal()], "user_id": "test-user"})
    assert resp.json()["processed_at"] > 0


def test_api_history_limit(client):
    for _ in range(5):
        client.post("/parkinsons-screen/analyze", json={"signals": [make_signal()], "user_id": "lim"})
    resp = client.get("/parkinsons-screen/history/lim?limit=3")
    assert resp.json()["count"] <= 3


def test_api_multichannel(client):
    payload = {"signals": [make_signal(), make_signal(freq=5.0)], "user_id": "test-user"}
    resp = client.post("/parkinsons-screen/analyze", json=payload)
    assert resp.status_code == 200

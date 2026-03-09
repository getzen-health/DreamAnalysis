"""Tests for MCI screener model and API route (#162)."""
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
    from api.routes.mci_screener import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_model_import():
    from models.mci_screener import MCIScreener, get_model
    assert MCIScreener is not None
    assert get_model() is not None


def test_predict_1d():
    from models.mci_screener import get_model
    result = get_model().predict(np.random.randn(512))
    assert "risk_category" in result


def test_predict_2d():
    from models.mci_screener import get_model
    result = get_model().predict(np.random.randn(4, 512))
    assert "mci_risk_score" in result


def test_risk_categories():
    from models.mci_screener import get_model
    cats = {"low_risk", "mild_concern", "moderate_concern", "high_risk"}
    result = get_model().predict(np.random.randn(512))
    assert result["risk_category"] in cats


def test_risk_score_range():
    from models.mci_screener import get_model
    result = get_model().predict(np.random.randn(512))
    assert 0.0 <= result["mci_risk_score"] <= 1.0


def test_slowing_ratio_positive():
    from models.mci_screener import get_model
    result = get_model().predict(np.random.randn(512))
    assert result["slowing_ratio"] >= 0.0


def test_paf_in_range():
    from models.mci_screener import get_model
    result = get_model().predict(np.random.randn(512))
    assert 7.0 <= result["peak_alpha_freq_hz"] <= 13.0


def test_delta_burden_range():
    from models.mci_screener import get_model
    result = get_model().predict(np.random.randn(512))
    assert 0.0 <= result["delta_burden"] <= 1.0


def test_note_field():
    from models.mci_screener import get_model
    result = get_model().predict(np.random.randn(512))
    assert "Screening only" in result["note"]


def test_model_used_field():
    from models.mci_screener import get_model
    result = get_model().predict(np.random.randn(512))
    assert "eeg_slowing" in result["model_used"]


def test_short_signal():
    from models.mci_screener import get_model
    result = get_model().predict(np.random.randn(64))
    assert "risk_category" in result


def test_api_analyze(client):
    payload = {"signals": [make_signal()], "fs": 256.0, "user_id": "u1"}
    resp = client.post("/mci-screen/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == "u1"
    assert "risk_category" in data


def test_api_note_field(client):
    resp = client.post("/mci-screen/analyze", json={"signals": [make_signal()]})
    assert "Screening only" in resp.json()["note"]


def test_api_history_empty(client):
    resp = client.get("/mci-screen/history/newuser")
    assert resp.json()["count"] == 0


def test_api_history_populated(client):
    client.post("/mci-screen/analyze", json={"signals": [make_signal()], "user_id": "hist_u"})
    resp = client.get("/mci-screen/history/hist_u")
    assert resp.json()["count"] >= 1


def test_api_reset(client):
    client.post("/mci-screen/analyze", json={"signals": [make_signal()], "user_id": "reset_u"})
    resp = client.post("/mci-screen/reset/reset_u")
    assert resp.status_code == 200
    assert resp.json()["status"] == "reset"


def test_api_history_after_reset(client):
    client.post("/mci-screen/analyze", json={"signals": [make_signal()], "user_id": "reset2"})
    client.post("/mci-screen/reset/reset2")
    assert client.get("/mci-screen/history/reset2").json()["count"] == 0


def test_api_processed_at(client):
    resp = client.post("/mci-screen/analyze", json={"signals": [make_signal()]})
    assert resp.json()["processed_at"] > 0


def test_api_history_limit(client):
    for _ in range(5):
        client.post("/mci-screen/analyze", json={"signals": [make_signal()], "user_id": "lim"})
    resp = client.get("/mci-screen/history/lim?limit=2")
    assert resp.json()["count"] <= 2


def test_api_multi_channel(client):
    payload = {"signals": [make_signal(), make_signal(freq=6.0)]}
    resp = client.post("/mci-screen/analyze", json=payload)
    assert resp.status_code == 200

"""Tests for decision confidence model and API route (#165)."""
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
    from api.routes.decision_confidence import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_model_import():
    from models.decision_confidence import DecisionConfidenceModel, get_model
    assert DecisionConfidenceModel is not None
    assert get_model() is not None


def test_predict_1d():
    from models.decision_confidence import get_model
    result = get_model().predict(np.random.randn(512))
    assert "confidence_label" in result


def test_predict_2d():
    from models.decision_confidence import get_model
    result = get_model().predict(np.random.randn(4, 512))
    assert "confidence_score" in result


def test_confidence_labels():
    from models.decision_confidence import get_model
    labels = {"uncertain", "moderate", "confident", "highly_confident"}
    result = get_model().predict(np.random.randn(512))
    assert result["confidence_label"] in labels


def test_confidence_score_range():
    from models.decision_confidence import get_model
    result = get_model().predict(np.random.randn(512))
    assert 0.0 <= result["confidence_score"] <= 1.0


def test_conflict_index_range():
    from models.decision_confidence import get_model
    result = get_model().predict(np.random.randn(512))
    assert 0.0 <= result["conflict_index"] <= 1.0


def test_risk_propensity_range():
    from models.decision_confidence import get_model
    result = get_model().predict(np.random.randn(512))
    assert 0.0 <= result["risk_propensity"] <= 1.0


def test_decision_readiness_range():
    from models.decision_confidence import get_model
    result = get_model().predict(np.random.randn(512))
    assert 0.0 <= result["decision_readiness"] <= 1.0


def test_model_used_field():
    from models.decision_confidence import get_model
    result = get_model().predict(np.random.randn(512))
    assert "theta_beta" in result["model_used"]


def test_short_signal():
    from models.decision_confidence import get_model
    result = get_model().predict(np.random.randn(64))
    assert "confidence_label" in result


def test_api_analyze(client):
    payload = {"signals": [make_signal()], "fs": 256.0, "user_id": "u1"}
    resp = client.post("/decision-confidence/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == "u1"
    assert "confidence_label" in data


def test_api_confidence_label_valid(client):
    labels = {"uncertain", "moderate", "confident", "highly_confident"}
    resp = client.post("/decision-confidence/analyze", json={"signals": [make_signal()]})
    assert resp.json()["confidence_label"] in labels


def test_api_history_empty(client):
    resp = client.get("/decision-confidence/history/newuser")
    assert resp.json()["count"] == 0


def test_api_history_populated(client):
    client.post("/decision-confidence/analyze", json={"signals": [make_signal()], "user_id": "h_u"})
    resp = client.get("/decision-confidence/history/h_u")
    assert resp.json()["count"] >= 1


def test_api_reset(client):
    client.post("/decision-confidence/analyze", json={"signals": [make_signal()], "user_id": "r_u"})
    resp = client.post("/decision-confidence/reset/r_u")
    assert resp.json()["status"] == "reset"


def test_api_history_after_reset(client):
    client.post("/decision-confidence/analyze", json={"signals": [make_signal()], "user_id": "r2"})
    client.post("/decision-confidence/reset/r2")
    assert client.get("/decision-confidence/history/r2").json()["count"] == 0


def test_api_processed_at(client):
    resp = client.post("/decision-confidence/analyze", json={"signals": [make_signal()]})
    assert resp.json()["processed_at"] > 0


def test_api_history_limit(client):
    for _ in range(5):
        client.post("/decision-confidence/analyze", json={"signals": [make_signal()], "user_id": "lim"})
    resp = client.get("/decision-confidence/history/lim?limit=2")
    assert resp.json()["count"] <= 2


def test_api_multichannel(client):
    payload = {"signals": [make_signal(), make_signal(freq=6.0)]}
    resp = client.post("/decision-confidence/analyze", json=payload)
    assert resp.status_code == 200

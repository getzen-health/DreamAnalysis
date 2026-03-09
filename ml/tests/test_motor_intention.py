"""Tests for motor intention model and API route (#167)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI


INTENTIONS = ["rest", "left_hand", "right_hand", "both_hands", "feet"]


def make_signal(n=512, freq=10.0, fs=256.0):
    t = np.linspace(0, n / fs, n)
    return (np.sin(2 * np.pi * freq * t)).tolist()


@pytest.fixture
def client():
    from api.routes.motor_intention import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_model_import():
    from models.motor_intention import MotorIntentionModel, get_model
    assert MotorIntentionModel is not None
    assert get_model() is not None


def test_predict_1d():
    from models.motor_intention import get_model
    result = get_model().predict(np.random.randn(512))
    assert "intention" in result


def test_predict_multichannel():
    from models.motor_intention import get_model
    result = get_model().predict(np.random.randn(4, 512))
    assert "lateral_beta_erd" in result


def test_intention_valid():
    from models.motor_intention import get_model
    result = get_model().predict(np.random.randn(512))
    assert result["intention"] in INTENTIONS


def test_control_signal_range():
    from models.motor_intention import get_model
    result = get_model().predict(np.random.randn(512))
    assert 0.0 <= result["control_signal"] <= 1.0


def test_erd_magnitude_range():
    from models.motor_intention import get_model
    result = get_model().predict(np.random.randn(4, 512))
    assert 0.0 <= result["erd_magnitude"] <= 1.0


def test_probabilities_sum():
    from models.motor_intention import get_model
    result = get_model().predict(np.random.randn(512))
    total = sum(result["probabilities"].values())
    assert abs(total - 1.0) < 0.01


def test_probabilities_all_intentions():
    from models.motor_intention import get_model
    result = get_model().predict(np.random.randn(4, 512))
    for intention in INTENTIONS:
        assert intention in result["probabilities"]


def test_lateral_mu_range():
    from models.motor_intention import get_model
    result = get_model().predict(np.random.randn(4, 512))
    assert -1.0 <= result["lateral_mu_asymmetry"] <= 1.0


def test_model_used_field():
    from models.motor_intention import get_model
    result = get_model().predict(np.random.randn(512))
    assert "erd" in result["model_used"]


def test_short_signal():
    from models.motor_intention import get_model
    result = get_model().predict(np.random.randn(64))
    assert "intention" in result


def test_api_analyze(client):
    payload = {"signals": [make_signal()], "fs": 256.0, "user_id": "u1"}
    resp = client.post("/motor-intention/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["intention"] in INTENTIONS


def test_api_multichannel(client):
    payload = {"signals": [make_signal(), make_signal(freq=10.0), make_signal(freq=12.0), make_signal(freq=8.0)]}
    resp = client.post("/motor-intention/analyze", json=payload)
    assert resp.status_code == 200


def test_api_history_empty(client):
    resp = client.get("/motor-intention/history/newuser")
    assert resp.json()["count"] == 0


def test_api_history_populated(client):
    client.post("/motor-intention/analyze", json={"signals": [make_signal()], "user_id": "h_u"})
    resp = client.get("/motor-intention/history/h_u")
    assert resp.json()["count"] >= 1


def test_api_reset(client):
    client.post("/motor-intention/analyze", json={"signals": [make_signal()], "user_id": "r_u"})
    resp = client.post("/motor-intention/reset/r_u")
    assert resp.json()["status"] == "reset"


def test_api_history_after_reset(client):
    client.post("/motor-intention/analyze", json={"signals": [make_signal()], "user_id": "r2"})
    client.post("/motor-intention/reset/r2")
    assert client.get("/motor-intention/history/r2").json()["count"] == 0


def test_api_processed_at(client):
    resp = client.post("/motor-intention/analyze", json={"signals": [make_signal()]})
    assert resp.json()["processed_at"] > 0


def test_api_history_limit(client):
    for _ in range(5):
        client.post("/motor-intention/analyze", json={"signals": [make_signal()], "user_id": "lim"})
    resp = client.get("/motor-intention/history/lim?limit=2")
    assert resp.json()["count"] <= 2

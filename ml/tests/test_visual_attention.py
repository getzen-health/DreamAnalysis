"""Tests for EEG visual attention and gaze zone estimation."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

RNG = np.random.default_rng(7)
ZONES = [
    "top-left","top-center","top-right",
    "mid-left","center","mid-right",
    "bot-left","bot-center","bot-right",
]


def make_sigs(n_ch=4, n=512):
    return (RNG.standard_normal((n_ch, n)) * 5.0).tolist()


# ── Model tests ───────────────────────────────────────────────────────────────

def test_model_keys():
    from models.visual_attention import get_model
    r = get_model().predict(np.zeros((4, 512)), 256.0)
    for k in ("attention_zone","horizontal_bias","vertical_bias",
              "alpha_suppression","visual_engagement","sustained_attention_index"):
        assert k in r


def test_attention_zone_valid():
    from models.visual_attention import get_model
    r = get_model().predict(np.random.randn(4, 512) * 3, 256.0)
    assert r["attention_zone"] in ZONES


def test_horizontal_bias_range():
    from models.visual_attention import get_model
    r = get_model().predict(np.random.randn(4, 512) * 3, 256.0)
    assert -1.0 <= r["horizontal_bias"] <= 1.0


def test_vertical_bias_range():
    from models.visual_attention import get_model
    r = get_model().predict(np.random.randn(4, 512) * 3, 256.0)
    assert -1.0 <= r["vertical_bias"] <= 1.0


def test_alpha_suppression_range():
    from models.visual_attention import get_model
    r = get_model().predict(np.random.randn(4, 512) * 3, 256.0)
    assert 0.0 <= r["alpha_suppression"] <= 1.0


def test_visual_engagement_range():
    from models.visual_attention import get_model
    r = get_model().predict(np.random.randn(4, 512) * 3, 256.0)
    assert 0.0 <= r["visual_engagement"] <= 1.0


def test_sustained_attention_range():
    from models.visual_attention import get_model
    r = get_model().predict(np.random.randn(4, 512) * 3, 256.0)
    assert 0.0 <= r["sustained_attention_index"] <= 1.0


def test_grid_col_range():
    from models.visual_attention import get_model
    r = get_model().predict(np.random.randn(4, 512) * 3, 256.0)
    assert r["grid_col"] in (0, 1, 2)


def test_grid_row_range():
    from models.visual_attention import get_model
    r = get_model().predict(np.random.randn(4, 512) * 3, 256.0)
    assert r["grid_row"] in (0, 1, 2)


def test_1d_input():
    from models.visual_attention import get_model
    r = get_model().predict(np.random.randn(256) * 3, 256.0)
    assert r["attention_zone"] in ZONES


def test_zero_signal_stable():
    from models.visual_attention import get_model
    r = get_model().predict(np.zeros((4, 512)), 256.0)
    assert r["attention_zone"] in ZONES


def test_singleton():
    from models.visual_attention import get_model
    assert get_model() is get_model()


# ── API tests ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from fastapi import FastAPI
    from api.routes.visual_attention import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_analyze_200(client):
    r = client.post("/visual-attention/analyze", json={"signals": make_sigs(), "user_id": "u1"})
    assert r.status_code == 200


def test_analyze_fields(client):
    r = client.post("/visual-attention/analyze", json={"signals": make_sigs(), "user_id": "u1"})
    d = r.json()
    for k in ("attention_zone","horizontal_bias","alpha_suppression","visual_engagement"):
        assert k in d


def test_zone_in_valid_set(client):
    r = client.post("/visual-attention/analyze", json={"signals": make_sigs(), "user_id": "u1"})
    assert r.json()["attention_zone"] in ZONES


def test_history_populated(client):
    r = client.get("/visual-attention/history/u1")
    assert r.json()["count"] >= 1


def test_history_empty_user(client):
    r = client.get("/visual-attention/history/nobody_xyz")
    assert r.json()["count"] == 0


def test_reset(client):
    client.post("/visual-attention/analyze", json={"signals": make_sigs(), "user_id": "reset_u"})
    client.post("/visual-attention/reset/reset_u")
    assert client.get("/visual-attention/history/reset_u").json()["count"] == 0


def test_reset_status(client):
    r = client.post("/visual-attention/reset/x")
    assert r.json()["status"] == "reset"


def test_single_channel(client):
    r = client.post("/visual-attention/analyze", json={
        "signals": [list(np.random.randn(512) * 3)], "user_id": "sc"
    })
    assert r.status_code == 200

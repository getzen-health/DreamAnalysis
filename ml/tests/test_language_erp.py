"""Tests for N400/P600 language ERP model and API routes."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

RNG = np.random.default_rng(42)


def make_signals(n_ch=4, n_samp=512, fs=256.0):
    return (RNG.standard_normal((n_ch, n_samp)) * 5.0).tolist()


# ── Model unit tests ─────────────────────────────────────────────────────────

def test_model_returns_required_keys():
    from models.language_erp import get_model
    sig = np.zeros((4, 512))
    result = get_model().predict(sig, 256.0)
    for key in ("n400_amplitude_uv", "p600_amplitude_uv",
                "semantic_surprise_index", "syntactic_load_index",
                "comprehension_score", "model_used"):
        assert key in result


def test_model_1d_input():
    from models.language_erp import get_model
    sig = np.random.randn(512) * 3.0
    result = get_model().predict(sig, 256.0)
    assert "n400_amplitude_uv" in result


def test_model_4ch_input():
    from models.language_erp import get_model
    sig = np.random.randn(4, 512) * 3.0
    result = get_model().predict(sig, 256.0)
    assert isinstance(result["n400_amplitude_uv"], float)


def test_semantic_surprise_in_range():
    from models.language_erp import get_model
    sig = np.random.randn(4, 512) * 3.0
    result = get_model().predict(sig, 256.0)
    assert 0.0 <= result["semantic_surprise_index"] <= 1.0


def test_syntactic_load_in_range():
    from models.language_erp import get_model
    sig = np.random.randn(4, 512) * 3.0
    result = get_model().predict(sig, 256.0)
    assert 0.0 <= result["syntactic_load_index"] <= 1.0


def test_comprehension_score_in_range():
    from models.language_erp import get_model
    sig = np.random.randn(4, 512) * 3.0
    result = get_model().predict(sig, 256.0)
    assert 0.0 <= result["comprehension_score"] <= 1.0


def test_model_used_field():
    from models.language_erp import get_model
    sig = np.zeros((2, 256))
    result = get_model().predict(sig, 256.0)
    assert result["model_used"] == "feature_based_erp"


def test_zero_signal_stable():
    from models.language_erp import get_model
    sig = np.zeros((4, 512))
    result = get_model().predict(sig, 256.0)
    assert result["n400_amplitude_uv"] == 0.0


def test_word_onset_offset():
    from models.language_erp import get_model
    sig = np.random.randn(4, 1024) * 5.0
    r1 = get_model().predict(sig, 256.0, word_onset_ms=0.0)
    r2 = get_model().predict(sig, 256.0, word_onset_ms=200.0)
    # Different onsets should (usually) give different amplitudes
    assert isinstance(r1["n400_amplitude_uv"], float)
    assert isinstance(r2["n400_amplitude_uv"], float)


def test_singleton_same_object():
    from models.language_erp import get_model
    assert get_model() is get_model()


# ── API route tests ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from fastapi import FastAPI
    from api.routes.language_erp import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_analyze_200(client):
    r = client.post("/language-erp/analyze", json={
        "signals": make_signals(), "fs": 256.0, "user_id": "u1"
    })
    assert r.status_code == 200


def test_analyze_response_fields(client):
    r = client.post("/language-erp/analyze", json={
        "signals": make_signals(), "fs": 256.0, "user_id": "u1"
    })
    d = r.json()
    for key in ("n400_amplitude_uv", "p600_amplitude_uv",
                "semantic_surprise_index", "comprehension_score",
                "is_semantically_surprising", "is_syntactically_violated"):
        assert key in d


def test_analyze_is_deceptive_bool(client):
    r = client.post("/language-erp/analyze", json={
        "signals": make_signals(), "fs": 256.0, "user_id": "u2"
    })
    assert isinstance(r.json()["is_semantically_surprising"], bool)


def test_analyze_with_word_onset(client):
    r = client.post("/language-erp/analyze", json={
        "signals": make_signals(n_samp=1024), "fs": 256.0,
        "word_onset_ms": 200.0, "user_id": "u3"
    })
    assert r.status_code == 200


def test_history_starts_populated(client):
    r = client.get("/language-erp/history/u1")
    assert r.status_code == 200
    assert r.json()["count"] >= 1


def test_history_empty_user(client):
    r = client.get("/language-erp/history/nonexistent_xyz")
    assert r.json()["count"] == 0


def test_reset_clears_history(client):
    client.post("/language-erp/analyze", json={
        "signals": make_signals(), "fs": 256.0, "user_id": "resetme"
    })
    client.post("/language-erp/reset/resetme")
    r = client.get("/language-erp/history/resetme")
    assert r.json()["count"] == 0


def test_reset_returns_status(client):
    r = client.post("/language-erp/reset/someuser")
    assert r.json()["status"] == "reset"


def test_history_limit(client):
    for _ in range(5):
        client.post("/language-erp/analyze", json={
            "signals": make_signals(), "fs": 256.0, "user_id": "limituser"
        })
    r = client.get("/language-erp/history/limituser?limit=3")
    assert r.json()["count"] <= 3


def test_single_channel_input(client):
    r = client.post("/language-erp/analyze", json={
        "signals": [list(np.random.randn(512) * 3.0)],
        "fs": 256.0, "user_id": "sc"
    })
    assert r.status_code == 200

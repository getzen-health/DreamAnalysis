"""Tests for the Emotion OS open API platform (issue #442).

Covers: create_emotion_vector, fuse_emotion_sources, register_app,
register_webhook, check_webhook_triggers, register_plugin,
compute_platform_stats, platform_to_dict, and API route integration.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from models.emotion_os import (
    BASIC_EMOTIONS,
    DEFAULT_RATE_LIMIT,
    DEFAULT_SOURCE_WEIGHTS,
    MAX_WEBHOOKS_PER_APP,
    VALID_SOURCES,
    EmotionVector,
    check_webhook_triggers,
    compute_platform_stats,
    create_emotion_vector,
    fuse_emotion_sources,
    platform_to_dict,
    register_app,
    register_plugin,
    register_webhook,
    _reset_platform,
)


@pytest.fixture(autouse=True)
def clean_platform():
    """Reset platform state before every test."""
    _reset_platform()
    yield
    _reset_platform()


# ---------------------------------------------------------------------------
# 1. create_emotion_vector
# ---------------------------------------------------------------------------


def test_create_emotion_vector_defaults():
    """Default vector has neutral probabilities and valid ranges."""
    vec = create_emotion_vector()
    assert vec.valence == 0.0
    assert vec.arousal == 0.5
    assert vec.dominance == 0.5
    assert vec.confidence == 0.5
    assert vec.source == "unknown"
    assert len(vec.probabilities) == len(BASIC_EMOTIONS)
    total = sum(vec.probabilities.values())
    assert abs(total - 1.0) < 1e-6


def test_create_emotion_vector_clamps_values():
    """Values outside valid ranges are clamped."""
    vec = create_emotion_vector(valence=2.0, arousal=-0.5, dominance=1.5, confidence=3.0)
    assert vec.valence == 1.0
    assert vec.arousal == 0.0
    assert vec.dominance == 1.0
    assert vec.confidence == 1.0


def test_create_emotion_vector_normalizes_probabilities():
    """Probabilities are normalized to sum to 1.0."""
    probs = {"happy": 3.0, "sad": 1.0, "angry": 0.0, "fear": 0.0, "surprise": 0.0, "neutral": 0.0}
    vec = create_emotion_vector(probabilities=probs)
    total = sum(vec.probabilities.values())
    assert abs(total - 1.0) < 1e-6
    assert vec.probabilities["happy"] == pytest.approx(0.75, abs=1e-6)
    assert vec.probabilities["sad"] == pytest.approx(0.25, abs=1e-6)


def test_create_emotion_vector_missing_emotions_filled():
    """Probabilities dict with missing emotion keys gets them filled with 0."""
    probs = {"happy": 1.0}
    vec = create_emotion_vector(probabilities=probs)
    assert len(vec.probabilities) == len(BASIC_EMOTIONS)
    # happy=1.0, rest=0.0 -> normalized happy=1.0
    assert vec.probabilities["happy"] == pytest.approx(1.0, abs=1e-6)


def test_create_emotion_vector_all_zero_probs_fallback():
    """All-zero probabilities fall back to uniform distribution."""
    probs = {e: 0.0 for e in BASIC_EMOTIONS}
    vec = create_emotion_vector(probabilities=probs)
    expected = 1.0 / len(BASIC_EMOTIONS)
    for e in BASIC_EMOTIONS:
        assert vec.probabilities[e] == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# 2. EmotionVector methods
# ---------------------------------------------------------------------------


def test_dominant_emotion():
    """dominant_emotion returns the highest-probability label."""
    vec = create_emotion_vector(probabilities={"happy": 0.8, "sad": 0.2})
    assert vec.dominant_emotion() == "happy"


def test_to_array_shape():
    """to_array returns a 9-element numpy array."""
    vec = create_emotion_vector()
    arr = vec.to_array()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (9,)
    assert arr[0] == vec.valence
    assert arr[1] == vec.arousal
    assert arr[2] == vec.dominance


# ---------------------------------------------------------------------------
# 3. fuse_emotion_sources
# ---------------------------------------------------------------------------


def test_fuse_empty_sources():
    """Fusing no sources returns a zero-confidence vector."""
    fused = fuse_emotion_sources([])
    assert fused.confidence == 0.0
    assert fused.source == "fused"


def test_fuse_single_source():
    """Fusing a single source preserves its values."""
    vec = create_emotion_vector(valence=0.8, arousal=0.9, source="eeg", confidence=0.95)
    fused = fuse_emotion_sources([vec])
    assert fused.valence == pytest.approx(0.8, abs=1e-4)
    assert fused.arousal == pytest.approx(0.9, abs=1e-4)
    assert fused.source == "fused"


def test_fuse_two_sources_weighted():
    """Fusing two sources blends values by weight and confidence."""
    eeg = create_emotion_vector(valence=0.8, arousal=0.7, source="eeg", confidence=0.9)
    voice = create_emotion_vector(valence=-0.4, arousal=0.3, source="voice", confidence=0.6)
    fused = fuse_emotion_sources([eeg, voice])
    # EEG has higher weight (0.30) and confidence (0.9) than voice (0.25, 0.6).
    # So fused valence should be closer to 0.8 than -0.4.
    assert fused.valence > 0.0
    assert fused.source == "fused"


def test_fuse_increments_platform_counter():
    """Each fusion increments the platform's total_fusions counter."""
    vec1 = create_emotion_vector(source="eeg")
    vec2 = create_emotion_vector(source="voice")
    fuse_emotion_sources([vec1, vec2])
    fuse_emotion_sources([vec1, vec2])
    stats = compute_platform_stats()
    assert stats["total_fusions"] == 2


def test_fuse_custom_weights():
    """Custom weights override defaults."""
    eeg = create_emotion_vector(valence=1.0, source="eeg", confidence=1.0)
    voice = create_emotion_vector(valence=-1.0, source="voice", confidence=1.0)
    # Give voice all the weight.
    fused = fuse_emotion_sources([eeg, voice], weights={"eeg": 0.0, "voice": 1.0})
    assert fused.valence == pytest.approx(-1.0, abs=1e-2)


# ---------------------------------------------------------------------------
# 4. register_app
# ---------------------------------------------------------------------------


def test_register_app_creates_unique_ids():
    """Each registration produces a unique app_id and api_key."""
    app1 = register_app("TestApp1")
    app2 = register_app("TestApp2")
    assert app1.app_id != app2.app_id
    assert app1.api_key != app2.api_key
    assert app1.active is True


def test_register_app_custom_rate_limit():
    """Custom rate limit is stored correctly."""
    app = register_app("RateLimited", rate_limit=100)
    assert app.rate_limit == 100


def test_register_app_appears_in_stats():
    """Registered apps appear in platform stats."""
    register_app("StatsApp")
    stats = compute_platform_stats()
    assert stats["total_apps"] == 1
    assert stats["active_apps"] == 1


# ---------------------------------------------------------------------------
# 5. register_webhook
# ---------------------------------------------------------------------------


def test_register_webhook_success():
    """Webhook registration returns webhook_id and metadata."""
    app = register_app("HookApp")
    result = register_webhook(app.app_id, "https://example.com/hook", "happy", 0.8)
    assert "error" not in result
    assert "webhook_id" in result
    assert result["emotion"] == "happy"
    assert result["threshold"] == 0.8


def test_register_webhook_unknown_app():
    """Webhook for non-existent app returns error."""
    result = register_webhook("nonexistent", "https://x.com", "happy", 0.5)
    assert "error" in result
    assert "Unknown app_id" in result["error"]


def test_register_webhook_invalid_emotion():
    """Webhook with invalid emotion key returns error."""
    app = register_app("HookApp2")
    result = register_webhook(app.app_id, "https://x.com", "jealousy", 0.5)
    assert "error" in result
    assert "Invalid emotion key" in result["error"]


def test_register_webhook_invalid_direction():
    """Webhook with invalid direction returns error."""
    app = register_app("HookApp3")
    result = register_webhook(app.app_id, "https://x.com", "happy", 0.5, direction="sideways")
    assert "error" in result
    assert "Invalid direction" in result["error"]


def test_register_webhook_limit_enforced():
    """Per-app webhook limit is enforced."""
    app = register_app("LimitApp")
    for i in range(MAX_WEBHOOKS_PER_APP):
        result = register_webhook(app.app_id, f"https://example.com/{i}", "happy", 0.5)
        assert "error" not in result
    # One more should fail.
    result = register_webhook(app.app_id, "https://example.com/overflow", "happy", 0.5)
    assert "error" in result
    assert "Maximum webhooks" in result["error"]


# ---------------------------------------------------------------------------
# 6. check_webhook_triggers
# ---------------------------------------------------------------------------


def test_check_webhook_triggers_above():
    """Webhook fires when value is at or above threshold (direction=above)."""
    app = register_app("TriggerApp")
    register_webhook(app.app_id, "https://example.com/trigger", "happy", 0.7, "above")

    vec = create_emotion_vector(probabilities={"happy": 0.9, "sad": 0.1})
    triggered = check_webhook_triggers(vec)
    assert len(triggered) == 1
    assert triggered[0]["emotion"] == "happy"
    assert triggered[0]["value"] >= 0.7


def test_check_webhook_triggers_below():
    """Webhook fires when value is at or below threshold (direction=below)."""
    app = register_app("TriggerBelowApp")
    register_webhook(app.app_id, "https://example.com/low", "arousal", 0.3, "below")

    vec = create_emotion_vector(arousal=0.1)
    triggered = check_webhook_triggers(vec)
    assert len(triggered) == 1
    assert triggered[0]["emotion"] == "arousal"


def test_check_webhook_triggers_not_fired():
    """Webhook does not fire when threshold is not crossed."""
    app = register_app("NoFireApp")
    register_webhook(app.app_id, "https://example.com/nope", "happy", 0.9, "above")

    vec = create_emotion_vector(probabilities={"happy": 0.3, "neutral": 0.7})
    triggered = check_webhook_triggers(vec)
    assert len(triggered) == 0


def test_check_webhook_valence_trigger():
    """Webhook can trigger on dimensional values like valence."""
    app = register_app("ValenceApp")
    register_webhook(app.app_id, "https://example.com/val", "valence", -0.5, "below")

    vec = create_emotion_vector(valence=-0.8)
    triggered = check_webhook_triggers(vec)
    assert len(triggered) == 1
    assert triggered[0]["emotion"] == "valence"


# ---------------------------------------------------------------------------
# 7. register_plugin
# ---------------------------------------------------------------------------


def test_register_plugin_success():
    """Plugin registration returns plugin_id and metadata."""
    result = register_plugin("music-emotion", "Maps music features to emotion vectors")
    assert "error" not in result
    assert "plugin_id" in result
    assert result["name"] == "music-emotion"


def test_register_plugin_appears_in_stats():
    """Registered plugins appear in platform stats."""
    register_plugin("food-emotion")
    stats = compute_platform_stats()
    assert stats["total_plugins"] == 1


# ---------------------------------------------------------------------------
# 8. compute_platform_stats
# ---------------------------------------------------------------------------


def test_platform_stats_empty():
    """Stats on a fresh platform are all zeros."""
    stats = compute_platform_stats()
    assert stats["total_apps"] == 0
    assert stats["total_webhooks"] == 0
    assert stats["total_plugins"] == 0
    assert stats["total_fusions"] == 0
    assert stats["uptime_seconds"] >= 0


def test_platform_stats_after_operations():
    """Stats reflect operations performed."""
    register_app("App1")
    register_app("App2")
    register_plugin("P1")
    v1 = create_emotion_vector(source="eeg")
    v2 = create_emotion_vector(source="voice")
    fuse_emotion_sources([v1, v2])

    stats = compute_platform_stats()
    assert stats["total_apps"] == 2
    assert stats["active_apps"] == 2
    assert stats["total_plugins"] == 1
    assert stats["total_fusions"] == 1


# ---------------------------------------------------------------------------
# 9. platform_to_dict
# ---------------------------------------------------------------------------


def test_platform_to_dict_structure():
    """platform_to_dict returns expected top-level keys."""
    app = register_app("DictApp")
    register_webhook(app.app_id, "https://x.com", "happy", 0.5)
    register_plugin("TestPlugin")

    d = platform_to_dict()
    assert "apps" in d
    assert "webhooks" in d
    assert "plugins" in d
    assert "stats" in d
    assert len(d["apps"]) == 1
    assert len(d["webhooks"]) == 1
    assert len(d["plugins"]) == 1


def test_platform_to_dict_api_key_masked():
    """API keys are masked in the dict output."""
    app = register_app("MaskedApp")
    d = platform_to_dict()
    app_data = list(d["apps"].values())[0]
    assert app_data["api_key"].endswith("...")


# ---------------------------------------------------------------------------
# 10. API route integration (via TestClient)
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """FastAPI test client with emotion_os routes mounted."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.routes.emotion_os import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_api_fuse(client):
    """POST /emotion-os/fuse returns fused emotion vector."""
    resp = client.post("/emotion-os/fuse", json={
        "sources": [
            {"source": "eeg", "valence": 0.7, "arousal": 0.8, "confidence": 0.9},
            {"source": "voice", "valence": -0.2, "arousal": 0.4, "confidence": 0.6},
        ],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "fused" in data
    assert "dominant_emotion" in data["fused"]
    assert data["source_count"] == 2


def test_api_register_app(client):
    """POST /emotion-os/register-app creates a new app."""
    resp = client.post("/emotion-os/register-app", json={"name": "TestApp"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert len(data["app_id"]) > 0
    assert len(data["api_key"]) > 0


def test_api_webhook_requires_valid_app(client):
    """POST /emotion-os/webhook with invalid app returns 400."""
    resp = client.post("/emotion-os/webhook", json={
        "app_id": "nonexistent",
        "url": "https://example.com",
        "emotion": "happy",
        "threshold": 0.5,
    })
    assert resp.status_code == 400


def test_api_stats(client):
    """GET /emotion-os/stats returns platform statistics."""
    resp = client.get("/emotion-os/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "stats" in data


def test_api_status(client):
    """GET /emotion-os/status returns availability."""
    resp = client.get("/emotion-os/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "available"

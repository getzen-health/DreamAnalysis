"""Tests for elderly cognitive-emotional monitoring (issue #448).

Covers: processing speed computation, emotional flattening detection,
cognitive-emotional coupling, emotional memory advantage, social
engagement scoring, composite MCI risk, full monitoring profile,
serialization, and API route integration.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from models.elderly_monitoring import (
    CognitiveReading,
    EmotionRangeReading,
    EmotionalMemoryReading,
    MonitoringProfile,
    ProcessingSpeedReading,
    SocialEngagementReading,
    _CLINICAL_DISCLAIMER,
    _compute_social_engagement,
    assess_emotional_memory,
    compute_cognitive_emotional_coupling,
    compute_mci_risk_score,
    compute_monitoring_profile,
    compute_processing_speed,
    detect_emotional_flattening,
    profile_to_dict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _healthy_ps_readings(n=5):
    """Processing speed readings within healthy norms."""
    return [
        ProcessingSpeedReading(
            reaction_time_ms=800.0 + i * 10,
            identification_time_ms=1100.0 + i * 15,
            correct=True,
        )
        for i in range(n)
    ]


def _slow_ps_readings(n=5):
    """Processing speed readings indicating decline."""
    return [
        ProcessingSpeedReading(
            reaction_time_ms=1400.0 + i * 20,
            identification_time_ms=2000.0 + i * 30,
            correct=(i % 3 != 0),  # ~67% accuracy
        )
        for i in range(n)
    ]


def _healthy_range_readings(n=4):
    return [
        EmotionRangeReading(
            distinct_emotions=6,
            valence_range=1.2,
            arousal_range=0.7,
        )
        for _ in range(n)
    ]


def _flat_range_readings(n=4):
    return [
        EmotionRangeReading(
            distinct_emotions=2,
            valence_range=0.3,
            arousal_range=0.2,
        )
        for _ in range(n)
    ]


def _declining_range_readings():
    """Range narrows over time."""
    return [
        EmotionRangeReading(distinct_emotions=6, valence_range=1.2, arousal_range=0.7),
        EmotionRangeReading(distinct_emotions=6, valence_range=1.1, arousal_range=0.6),
        EmotionRangeReading(distinct_emotions=4, valence_range=0.6, arousal_range=0.4),
        EmotionRangeReading(distinct_emotions=2, valence_range=0.3, arousal_range=0.2),
    ]


def _healthy_cog_readings(n=4):
    return [
        CognitiveReading(
            attention_score=0.7, memory_score=0.7,
            executive_score=0.7, processing_speed_score=0.7,
        )
        for _ in range(n)
    ]


def _declining_cog_readings():
    return [
        CognitiveReading(attention_score=0.8, memory_score=0.8, executive_score=0.8, processing_speed_score=0.8),
        CognitiveReading(attention_score=0.7, memory_score=0.7, executive_score=0.7, processing_speed_score=0.7),
        CognitiveReading(attention_score=0.5, memory_score=0.5, executive_score=0.5, processing_speed_score=0.5),
        CognitiveReading(attention_score=0.3, memory_score=0.3, executive_score=0.3, processing_speed_score=0.3),
    ]


def _healthy_memory_readings(n=3):
    return [
        EmotionalMemoryReading(
            emotional_items_recalled=8,
            neutral_items_recalled=5,
            emotional_items_total=10,
            neutral_items_total=10,
        )
        for _ in range(n)
    ]


def _impaired_memory_readings(n=3):
    return [
        EmotionalMemoryReading(
            emotional_items_recalled=4,
            neutral_items_recalled=4,
            emotional_items_total=10,
            neutral_items_total=10,
        )
        for _ in range(n)
    ]


def _healthy_social_readings(n=4):
    return [
        SocialEngagementReading(
            interactions_count=12,
            unique_contacts=6,
            initiated_count=5,
        )
        for _ in range(n)
    ]


def _isolated_social_readings(n=4):
    return [
        SocialEngagementReading(
            interactions_count=2,
            unique_contacts=1,
            initiated_count=0,
        )
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# Processing speed tests
# ---------------------------------------------------------------------------


def test_processing_speed_empty():
    result = compute_processing_speed([])
    assert result["score"] == 0.0
    assert result["n_readings"] == 0


def test_processing_speed_healthy():
    result = compute_processing_speed(_healthy_ps_readings())
    assert 0.0 <= result["score"] <= 0.3
    assert result["accuracy"] == 1.0
    assert result["flags"] == []


def test_processing_speed_slow():
    result = compute_processing_speed(_slow_ps_readings())
    assert result["score"] > 0.3
    assert any("slow" in f or "low" in f for f in result["flags"])


def test_processing_speed_score_range():
    result = compute_processing_speed(_slow_ps_readings(10))
    assert 0.0 <= result["score"] <= 1.0


# ---------------------------------------------------------------------------
# Emotional flattening tests
# ---------------------------------------------------------------------------


def test_flattening_empty():
    result = detect_emotional_flattening([])
    assert result["score"] == 0.0
    assert result["trend"] == "insufficient_data"


def test_flattening_healthy():
    result = detect_emotional_flattening(_healthy_range_readings())
    assert result["score"] < 0.3
    assert "critically_narrow_range" not in result["flags"]


def test_flattening_flat():
    result = detect_emotional_flattening(_flat_range_readings())
    assert result["score"] > 0.3
    assert "critically_narrow_range" in result["flags"]


def test_flattening_declining_trend():
    result = detect_emotional_flattening(_declining_range_readings())
    assert result["trend"] == "declining"
    assert "declining_range_trend" in result["flags"]


# ---------------------------------------------------------------------------
# Cognitive-emotional coupling tests
# ---------------------------------------------------------------------------


def test_coupling_empty():
    result = compute_cognitive_emotional_coupling([], [])
    assert result["score"] == 0.0
    assert result["both_declining"] is False


def test_coupling_both_declining():
    result = compute_cognitive_emotional_coupling(
        _declining_cog_readings(), _declining_range_readings(),
    )
    assert result["both_declining"] is True
    assert result["score"] > 0.3
    assert "cognitive_emotional_co_decline" in result["flags"]


def test_coupling_stable():
    result = compute_cognitive_emotional_coupling(
        _healthy_cog_readings(), _healthy_range_readings(),
    )
    assert result["both_declining"] is False


# ---------------------------------------------------------------------------
# Emotional memory tests
# ---------------------------------------------------------------------------


def test_emotional_memory_empty():
    result = assess_emotional_memory([])
    assert result["score"] == 0.0
    assert result["advantage_preserved"] is True


def test_emotional_memory_healthy():
    result = assess_emotional_memory(_healthy_memory_readings())
    assert result["ema_ratio"] > 1.1
    assert result["advantage_preserved"] is True
    assert result["score"] < 0.3


def test_emotional_memory_impaired():
    result = assess_emotional_memory(_impaired_memory_readings())
    assert result["ema_ratio"] <= 1.1
    assert result["score"] > 0.0
    assert any("memory" in f for f in result["flags"])


# ---------------------------------------------------------------------------
# Social engagement tests
# ---------------------------------------------------------------------------


def test_social_engagement_empty():
    result = _compute_social_engagement([])
    assert result["score"] == 0.0


def test_social_engagement_healthy():
    result = _compute_social_engagement(_healthy_social_readings())
    assert result["score"] < 0.3
    assert "critically_low_social_engagement" not in result["flags"]


def test_social_engagement_isolated():
    result = _compute_social_engagement(_isolated_social_readings())
    assert result["score"] > 0.3
    assert "critically_low_social_engagement" in result["flags"]


# ---------------------------------------------------------------------------
# Composite MCI risk tests
# ---------------------------------------------------------------------------


def test_mci_risk_all_low():
    zero = {"score": 0.0, "flags": []}
    result = compute_mci_risk_score(zero, zero, zero, zero, zero)
    assert result["mci_risk_score"] == 0.0
    assert result["risk_category"] == "low_risk"
    assert _CLINICAL_DISCLAIMER in result["disclaimer"]


def test_mci_risk_all_high():
    high = {"score": 0.9, "flags": ["test_flag"]}
    result = compute_mci_risk_score(high, high, high, high, high)
    assert result["mci_risk_score"] > 0.7
    assert result["risk_category"] == "high_risk"
    assert len(result["flags"]) == 5


def test_mci_risk_mixed():
    low = {"score": 0.1, "flags": []}
    high = {"score": 0.8, "flags": ["concern"]}
    result = compute_mci_risk_score(high, low, low, high, low)
    assert 0.0 < result["mci_risk_score"] < 1.0
    assert result["risk_category"] in {"low_risk", "mild_concern", "moderate_concern", "high_risk"}


def test_mci_risk_score_range():
    med = {"score": 0.5, "flags": []}
    result = compute_mci_risk_score(med, med, med, med, med)
    assert 0.0 <= result["mci_risk_score"] <= 1.0


# ---------------------------------------------------------------------------
# Full monitoring profile tests
# ---------------------------------------------------------------------------


def test_monitoring_profile_empty():
    profile = compute_monitoring_profile()
    assert isinstance(profile, MonitoringProfile)
    assert profile.mci_risk_score == 0.0
    assert profile.risk_category == "low_risk"


def test_monitoring_profile_with_data():
    profile = compute_monitoring_profile(
        processing_speed_readings=_slow_ps_readings(),
        emotion_range_readings=_flat_range_readings(),
        social_engagement_readings=_isolated_social_readings(),
    )
    assert profile.mci_risk_score > 0.0
    assert len(profile.flags) > 0


def test_profile_to_dict():
    profile = compute_monitoring_profile()
    d = profile_to_dict(profile)
    assert isinstance(d, dict)
    assert "mci_risk_score" in d
    assert "risk_category" in d
    assert "disclaimer" in d
    assert "processing_speed_detail" in d
    assert "flattening_detail" in d
    assert "coupling_detail" in d
    assert "memory_detail" in d
    assert "social_detail" in d


def test_profile_to_dict_fields_match_profile():
    profile = compute_monitoring_profile(
        processing_speed_readings=_healthy_ps_readings(),
    )
    d = profile_to_dict(profile)
    assert d["processing_speed_score"] == profile.processing_speed_score
    assert d["mci_risk_score"] == profile.mci_risk_score


# ---------------------------------------------------------------------------
# API route tests
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    from api.routes.elderly_monitoring import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_api_status(client):
    resp = client.get("/elderly/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is True
    assert "domains" in data
    assert len(data["domains"]) == 5


def test_api_assess_empty(client):
    resp = client.post("/elderly/assess", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["mci_risk_score"] == 0.0
    assert data["risk_category"] == "low_risk"


def test_api_assess_with_data(client):
    payload = {
        "processing_speed_readings": [
            {"reaction_time_ms": 1400, "identification_time_ms": 2000, "correct": False}
            for _ in range(3)
        ],
        "emotion_range_readings": [
            {"distinct_emotions": 2, "valence_range": 0.3, "arousal_range": 0.2}
        ],
        "social_engagement_readings": [
            {"interactions_count": 2, "unique_contacts": 1, "initiated_count": 0}
        ],
    }
    resp = client.post("/elderly/assess", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["mci_risk_score"] > 0.0
    assert "sub_scores" in data
    assert "disclaimer" in data


def test_api_profile_empty(client):
    resp = client.post("/elderly/profile", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["mci_risk_score"] == 0.0
    assert "processing_speed_detail" in data


def test_api_profile_with_data(client):
    payload = {
        "processing_speed_readings": [
            {"reaction_time_ms": 900, "identification_time_ms": 1300}
        ],
        "emotional_memory_readings": [
            {
                "emotional_items_recalled": 8,
                "neutral_items_recalled": 5,
                "emotional_items_total": 10,
                "neutral_items_total": 10,
            }
        ],
    }
    resp = client.post("/elderly/profile", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "memory_detail" in data
    assert "social_detail" in data


def test_api_disclaimer_present(client):
    resp = client.post("/elderly/assess", json={})
    assert _CLINICAL_DISCLAIMER in resp.json()["disclaimer"]

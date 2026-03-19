"""Tests for collective emotional intelligence — issue #461.

Covers: anonymous aggregation, collective mood computation, event detection,
temporal patterns, geographic patterns, privacy enforcement (min group size),
profile serialization, edge cases, and API routes.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from models.collective_emotion import (
    AnonymousEmotionSample,
    CollectiveEvent,
    CollectiveMood,
    CollectiveProfile,
    aggregate_anonymous_emotions,
    compute_collective_mood,
    compute_collective_profile,
    detect_collective_events,
    profile_to_dict,
    _classify_mood,
    _classify_event,
    _MIN_GROUP_SIZE,
)
from api.routes.collective_emotion import router

# ---------------------------------------------------------------------------
# Test client setup
# ---------------------------------------------------------------------------

app = FastAPI()
app.include_router(router)
client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_samples(
    n: int,
    valence: float = 0.0,
    arousal: float = 0.5,
    stress: float = 0.3,
    energy: float = 0.5,
    region: str = "test_region",
    base_ts: float = 1_000_000.0,
) -> list:
    """Generate n AnonymousEmotionSample with given defaults."""
    return [
        AnonymousEmotionSample(
            valence=valence,
            arousal=arousal,
            stress=stress,
            energy=energy,
            timestamp=base_ts + i * 60,
            region=region,
        )
        for i in range(n)
    ]


# ===========================================================================
# Test aggregate_anonymous_emotions
# ===========================================================================

class TestAggregateAnonymousEmotions:

    def test_sufficient_data(self):
        """With enough samples, should return valid mood statistics."""
        samples = _make_samples(10, valence=0.5, arousal=0.6)
        mood = aggregate_anonymous_emotions(samples)
        assert mood.sample_count == 10
        assert mood.mean_valence == pytest.approx(0.5, abs=0.01)
        assert mood.mean_arousal == pytest.approx(0.6, abs=0.01)
        assert mood.mood_label != "insufficient_data"

    def test_insufficient_data_privacy(self):
        """Below minimum group size, should return insufficient_data."""
        samples = _make_samples(_MIN_GROUP_SIZE - 1)
        mood = aggregate_anonymous_emotions(samples)
        assert mood.mood_label == "insufficient_data"
        assert mood.sample_count == _MIN_GROUP_SIZE - 1

    def test_coherence_high_when_uniform(self):
        """All same values should produce high coherence."""
        samples = _make_samples(10, valence=0.5, arousal=0.5)
        mood = aggregate_anonymous_emotions(samples)
        assert mood.coherence > 0.8

    def test_coherence_lower_with_variance(self):
        """Diverse valence values should produce lower coherence."""
        samples = []
        for i in range(10):
            v = -1.0 + (2.0 * i / 9)  # spread from -1 to +1
            samples.append(AnonymousEmotionSample(valence=v, arousal=0.5, timestamp=float(i)))
        mood = aggregate_anonymous_emotions(samples)
        assert mood.coherence < 0.5

    def test_empty_samples(self):
        """Empty list should return insufficient_data."""
        mood = aggregate_anonymous_emotions([])
        assert mood.mood_label == "insufficient_data"
        assert mood.sample_count == 0


# ===========================================================================
# Test compute_collective_mood
# ===========================================================================

class TestComputeCollectiveMood:

    def test_returns_dict(self):
        """compute_collective_mood should return a plain dict."""
        samples = _make_samples(10, valence=0.3, arousal=0.7)
        result = compute_collective_mood(samples)
        assert isinstance(result, dict)
        assert "mean_valence" in result
        assert "mood_label" in result

    def test_mood_label_classification(self):
        """Positive valence + high arousal should be excited_positive."""
        samples = _make_samples(10, valence=0.5, arousal=0.7)
        result = compute_collective_mood(samples)
        assert result["mood_label"] == "excited_positive"


# ===========================================================================
# Test detect_collective_events
# ===========================================================================

class TestDetectCollectiveEvents:

    def test_stress_spike_detected(self):
        """High stress in current vs low baseline should detect a stress_spike."""
        # Baseline needs slight variance so std > 0 (otherwise detection skips)
        baseline = [
            AnonymousEmotionSample(
                valence=0.0, arousal=0.5,
                stress=0.2 + (i % 3) * 0.02,  # small variance around 0.2
                energy=0.5, timestamp=float(i),
            )
            for i in range(20)
        ]
        current = _make_samples(10, stress=0.9, valence=0.0, arousal=0.5)
        events = detect_collective_events(current, baseline, z_threshold=1.5)
        stress_events = [e for e in events if e.metric == "stress"]
        assert len(stress_events) >= 1
        assert stress_events[0].event_type == "stress_spike"

    def test_no_event_when_similar(self):
        """Same distribution should produce no events."""
        baseline = _make_samples(20, valence=0.3, arousal=0.5, stress=0.3)
        current = _make_samples(10, valence=0.3, arousal=0.5, stress=0.3)
        events = detect_collective_events(current, baseline)
        assert len(events) == 0

    def test_no_events_without_baseline(self):
        """No baseline should return empty events."""
        current = _make_samples(10)
        events = detect_collective_events(current, baseline_samples=None)
        assert events == []

    def test_insufficient_current_samples(self):
        """Below min group size should return no events."""
        baseline = _make_samples(20, stress=0.2)
        current = _make_samples(2, stress=0.9)
        events = detect_collective_events(current, baseline)
        assert events == []

    def test_joy_wave_detected(self):
        """Valence spike should be detected as joy_wave."""
        # Baseline needs slight variance so std > 0
        baseline = [
            AnonymousEmotionSample(
                valence=-0.2 + (i % 4) * 0.02,
                arousal=0.5, stress=0.3, energy=0.5, timestamp=float(i),
            )
            for i in range(20)
        ]
        current = _make_samples(10, valence=0.8, arousal=0.5)
        events = detect_collective_events(current, baseline, z_threshold=1.5)
        valence_events = [e for e in events if e.metric == "valence"]
        assert len(valence_events) >= 1
        assert valence_events[0].event_type == "joy_wave"
        assert valence_events[0].z_score > 0


# ===========================================================================
# Test compute_collective_profile
# ===========================================================================

class TestComputeCollectiveProfile:

    def test_full_profile(self):
        """Full profile should have mood, events, temporal, geographic data."""
        samples = _make_samples(10, valence=0.4, arousal=0.6, region="north")
        profile = compute_collective_profile(samples)
        assert profile.sufficient_data is True
        assert profile.sample_count == 10
        assert profile.mood.mean_valence == pytest.approx(0.4, abs=0.01)

    def test_insufficient_data_flag(self):
        """Below min group size should flag insufficient_data."""
        samples = _make_samples(2)
        profile = compute_collective_profile(samples)
        assert profile.sufficient_data is False

    def test_geographic_patterns_with_regions(self):
        """Multiple regions with sufficient data should appear in patterns."""
        samples_north = _make_samples(6, valence=0.5, region="north")
        samples_south = _make_samples(6, valence=-0.3, region="south")
        profile = compute_collective_profile(samples_north + samples_south)
        regions = profile.geographic_patterns.get("regions", {})
        assert "north" in regions
        assert "south" in regions

    def test_geographic_privacy_small_region(self):
        """Region with fewer than min samples should not appear."""
        samples_big = _make_samples(10, region="big_region")
        samples_small = _make_samples(2, region="tiny_region")
        profile = compute_collective_profile(samples_big + samples_small)
        regions = profile.geographic_patterns.get("regions", {})
        assert "big_region" in regions
        assert "tiny_region" not in regions


# ===========================================================================
# Test helper classifications
# ===========================================================================

class TestClassifications:

    def test_classify_mood_positive_excited(self):
        assert _classify_mood(0.5, 0.7) == "excited_positive"

    def test_classify_mood_calm_positive(self):
        assert _classify_mood(0.5, 0.3) == "calm_positive"

    def test_classify_mood_stressed_negative(self):
        assert _classify_mood(-0.5, 0.7) == "stressed_negative"

    def test_classify_mood_neutral(self):
        assert _classify_mood(0.0, 0.5) == "neutral"

    def test_classify_event_joy_wave(self):
        assert _classify_event("valence", 2.5) == "joy_wave"

    def test_classify_event_distress_wave(self):
        assert _classify_event("valence", -2.5) == "distress_wave"

    def test_classify_event_stress_spike(self):
        assert _classify_event("stress", 3.0) == "stress_spike"


# ===========================================================================
# Test serialization
# ===========================================================================

class TestSerialization:

    def test_profile_to_dict(self):
        """profile_to_dict should produce a dict with all top-level keys."""
        samples = _make_samples(10, valence=0.3)
        profile = compute_collective_profile(samples)
        d = profile_to_dict(profile)
        assert "mood" in d
        assert "events" in d
        assert "temporal_patterns" in d
        assert "geographic_patterns" in d
        assert "sample_count" in d
        assert "sufficient_data" in d

    def test_mood_dict_fields(self):
        """Mood dict should contain all mood fields."""
        samples = _make_samples(10, valence=0.5, arousal=0.6)
        profile = compute_collective_profile(samples)
        d = profile_to_dict(profile)
        mood = d["mood"]
        assert "mean_valence" in mood
        assert "mean_arousal" in mood
        assert "coherence" in mood
        assert "mood_label" in mood


# ===========================================================================
# Test API routes
# ===========================================================================

class TestAPIRoutes:

    def test_status_endpoint(self):
        """GET /collective/status should return ready."""
        resp = client.get("/collective/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"

    def test_aggregate_endpoint(self):
        """POST /collective/aggregate should return collective profile."""
        samples = [
            {"valence": 0.3, "arousal": 0.5, "stress": 0.2, "energy": 0.5, "region": "test"}
            for _ in range(10)
        ]
        resp = client.post("/collective/aggregate", json={"samples": samples})
        assert resp.status_code == 200
        data = resp.json()
        assert "mood" in data
        assert data["mood"]["sample_count"] == 10
        assert data["sufficient_data"] is True

    def test_mood_endpoint_after_aggregate(self):
        """GET /collective/mood should return cached mood after aggregation."""
        # First submit data
        samples = [
            {"valence": 0.5, "arousal": 0.6, "stress": 0.2, "energy": 0.5}
            for _ in range(10)
        ]
        client.post("/collective/aggregate", json={"samples": samples})
        # Then query mood
        resp = client.get("/collective/mood")
        assert resp.status_code == 200
        data = resp.json()
        assert "mood" in data

    def test_mood_endpoint_no_data(self):
        """GET /collective/mood with no prior data should return no_data."""
        # Reset module state
        import api.routes.collective_emotion as mod
        mod._latest_mood = {}
        mod._latest_samples = []

        resp = client.get("/collective/mood")
        assert resp.status_code == 200
        data = resp.json()
        assert data["mood"]["mood_label"] == "no_data"

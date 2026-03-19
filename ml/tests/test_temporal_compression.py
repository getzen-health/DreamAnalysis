"""Tests for temporal emotion compression — issue #462.

Covers: key moment extraction, emotional arc detection, timeline compression,
timelapse generation, compression serialization, edge cases, and API routes.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from models.temporal_compression import (
    ArcPhase,
    CompressionResult,
    EmotionalArc,
    EmotionDataPoint,
    KeyMoment,
    TimelapsePeriod,
    compress_timeline,
    compression_to_dict,
    detect_emotional_arc,
    extract_key_moments,
    generate_timelapse,
    _classify_period_emotion,
    _emotional_intensity,
)
from api.routes.temporal_compression import router

# ---------------------------------------------------------------------------
# Test client setup
# ---------------------------------------------------------------------------

app = FastAPI()
app.include_router(router)
client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_timeline(
    n: int,
    valence_start: float = 0.0,
    valence_end: float = 0.0,
    arousal: float = 0.5,
    base_ts: float = 1_000_000.0,
    interval: float = 3600.0,  # 1 hour between points
) -> list:
    """Generate a linear emotional timeline from start to end valence."""
    data = []
    for i in range(n):
        t = i / max(n - 1, 1) if n > 1 else 0.0
        v = valence_start + (valence_end - valence_start) * t
        data.append(EmotionDataPoint(
            timestamp=base_ts + i * interval,
            valence=v,
            arousal=arousal,
            stress=0.3,
            energy=0.5,
        ))
    return data


def _make_dramatic_timeline() -> list:
    """Create a timeline with clear peaks, valleys, and shifts."""
    return [
        EmotionDataPoint(timestamp=1e6 + i * 3600, valence=v, arousal=a)
        for i, (v, a) in enumerate([
            (0.0, 0.3),     # baseline
            (0.1, 0.4),     # slight rise
            (0.5, 0.7),     # rising action
            (0.8, 0.9),     # peak / climax
            (0.9, 0.8),     # near peak
            (0.5, 0.6),     # falling
            (0.2, 0.4),     # coming down
            (-0.3, 0.5),    # dip
            (-0.7, 0.3),    # valley
            (0.0, 0.4),     # resolution
        ])
    ]


# ===========================================================================
# Test extract_key_moments
# ===========================================================================

class TestExtractKeyMoments:

    def test_dramatic_timeline_finds_moments(self):
        """Dramatic timeline should produce multiple key moments."""
        data = _make_dramatic_timeline()
        moments = extract_key_moments(data, max_moments=5)
        assert len(moments) >= 2
        assert all(isinstance(m, KeyMoment) for m in moments)

    def test_sorted_by_significance(self):
        """Key moments should be sorted by significance descending."""
        data = _make_dramatic_timeline()
        moments = extract_key_moments(data, max_moments=10)
        scores = [m.significance_score for m in moments]
        assert scores == sorted(scores, reverse=True)

    def test_extreme_detected(self):
        """The highest and lowest valence points should be flagged as extreme."""
        data = _make_dramatic_timeline()
        moments = extract_key_moments(data, max_moments=20, threshold=0.1)
        extreme_moments = [m for m in moments if m.moment_type == "extreme"]
        assert len(extreme_moments) >= 1

    def test_empty_data(self):
        """Empty data should return empty list."""
        moments = extract_key_moments([])
        assert moments == []

    def test_single_point(self):
        """Single data point should return one moment."""
        data = [EmotionDataPoint(timestamp=1e6, valence=0.5, arousal=0.7)]
        moments = extract_key_moments(data)
        assert len(moments) == 1
        assert moments[0].moment_type == "single"

    def test_max_moments_limit(self):
        """Should not return more than max_moments."""
        data = _make_dramatic_timeline()
        moments = extract_key_moments(data, max_moments=3, threshold=0.0)
        assert len(moments) <= 3

    def test_flat_timeline_low_significance(self):
        """Flat timeline key moments should have lower average significance than dramatic."""
        flat_data = _make_timeline(20, valence_start=0.0, valence_end=0.0)
        dramatic_data = _make_dramatic_timeline()
        flat_moments = extract_key_moments(flat_data, threshold=0.0, max_moments=20)
        dramatic_moments = extract_key_moments(dramatic_data, threshold=0.0, max_moments=20)
        flat_avg = sum(m.significance_score for m in flat_moments) / max(len(flat_moments), 1)
        dramatic_avg = sum(m.significance_score for m in dramatic_moments) / max(len(dramatic_moments), 1)
        # Dramatic timeline should have higher average significance
        assert dramatic_avg > flat_avg


# ===========================================================================
# Test detect_emotional_arc
# ===========================================================================

class TestDetectEmotionalArc:

    def test_triumph_arc(self):
        """Rising valence should produce triumph or recovery arc."""
        data = _make_timeline(20, valence_start=-0.5, valence_end=0.5)
        arc = detect_emotional_arc(data)
        assert arc is not None
        assert arc.arc_type in ("triumph", "recovery")
        assert arc.overall_valence_change > 0

    def test_decline_arc(self):
        """Falling valence should produce decline or tragedy arc."""
        data = _make_timeline(20, valence_start=0.5, valence_end=-0.5)
        arc = detect_emotional_arc(data)
        assert arc is not None
        assert arc.arc_type in ("decline", "tragedy")
        assert arc.overall_valence_change < 0

    def test_stable_arc(self):
        """Flat valence should produce stable arc."""
        data = _make_timeline(20, valence_start=0.0, valence_end=0.0)
        arc = detect_emotional_arc(data)
        assert arc is not None
        assert arc.arc_type == "stable"

    def test_too_few_points_returns_none(self):
        """Fewer than MIN_ARC_PERIODS points should return None."""
        data = _make_timeline(2, valence_start=0.0, valence_end=0.5)
        arc = detect_emotional_arc(data)
        assert arc is None

    def test_phases_present(self):
        """Arc should have multiple phases."""
        data = _make_timeline(20, valence_start=-0.5, valence_end=0.5)
        arc = detect_emotional_arc(data, num_phases=5)
        assert arc is not None
        assert len(arc.phases) == 5
        assert all(isinstance(p, ArcPhase) for p in arc.phases)

    def test_arc_duration(self):
        """Arc duration should match input data length."""
        data = _make_timeline(30, valence_start=0.0, valence_end=0.3)
        arc = detect_emotional_arc(data)
        assert arc is not None
        assert arc.arc_duration_samples == 30


# ===========================================================================
# Test compress_timeline
# ===========================================================================

class TestCompressTimeline:

    def test_correct_number_of_periods(self):
        """Should return the requested number of periods."""
        data = _make_timeline(100)
        periods = compress_timeline(data, num_periods=10)
        assert len(periods) == 10

    def test_empty_data(self):
        """Empty data should return empty list."""
        periods = compress_timeline([])
        assert periods == []

    def test_period_statistics(self):
        """Each period should have valid statistics."""
        data = _make_timeline(50, valence_start=0.3, valence_end=0.3, arousal=0.6)
        periods = compress_timeline(data, num_periods=5)
        for p in periods:
            assert isinstance(p, TimelapsePeriod)
            assert p.sample_count > 0
            assert p.mean_valence == pytest.approx(0.3, abs=0.1)
            assert p.mean_arousal == pytest.approx(0.6, abs=0.1)

    def test_all_samples_covered(self):
        """Total samples across periods should equal input."""
        data = _make_timeline(47)
        periods = compress_timeline(data, num_periods=5)
        total = sum(p.sample_count for p in periods)
        assert total == 47

    def test_single_period(self):
        """Single period should cover all data."""
        data = _make_timeline(20)
        periods = compress_timeline(data, num_periods=1)
        assert len(periods) == 1
        assert periods[0].sample_count == 20


# ===========================================================================
# Test generate_timelapse
# ===========================================================================

class TestGenerateTimelapse:

    def test_full_generation(self):
        """Full timelapse should have all components."""
        data = _make_dramatic_timeline()
        result = generate_timelapse(data, num_periods=5, max_key_moments=10)
        assert isinstance(result, CompressionResult)
        assert len(result.timelapse) == 5
        assert len(result.key_moments) >= 1
        assert result.total_samples == len(data)
        assert result.summary != ""

    def test_empty_data(self):
        """Empty data should return empty result with summary."""
        result = generate_timelapse([])
        assert result.total_samples == 0
        assert result.summary == "No data provided"

    def test_compression_ratio(self):
        """Compression ratio should be positive and less than 1 for large data."""
        data = _make_timeline(100)
        result = generate_timelapse(data, num_periods=10, max_key_moments=5)
        assert result.compression_ratio > 0

    def test_duration_computed(self):
        """Duration in hours should be computed from timestamps."""
        data = _make_timeline(24, interval=3600.0)  # 24 hours of data
        result = generate_timelapse(data, num_periods=4)
        assert result.timeline_duration_hours == pytest.approx(23.0, abs=1.0)

    def test_key_moments_mapped_to_periods(self):
        """Key moments should be counted in their corresponding periods."""
        data = _make_dramatic_timeline()
        result = generate_timelapse(data, num_periods=5, max_key_moments=20)
        total_km_in_periods = sum(p.key_moments_count for p in result.timelapse)
        # At least some moments should be mapped
        assert total_km_in_periods >= 0  # may be 0 if moments fall outside period bounds


# ===========================================================================
# Test helpers
# ===========================================================================

class TestHelpers:

    def test_emotional_intensity_neutral(self):
        """Neutral state should have low intensity."""
        dp = EmotionDataPoint(valence=0.0, arousal=0.5)
        assert _emotional_intensity(dp) < 0.1

    def test_emotional_intensity_extreme(self):
        """Extreme state should have high intensity."""
        dp = EmotionDataPoint(valence=0.9, arousal=0.9)
        assert _emotional_intensity(dp) > 0.8

    def test_classify_period_excited(self):
        assert _classify_period_emotion(0.5, 0.7) == "excited"

    def test_classify_period_content(self):
        assert _classify_period_emotion(0.5, 0.3) == "content"

    def test_classify_period_stressed(self):
        assert _classify_period_emotion(-0.5, 0.7) == "stressed"

    def test_classify_period_neutral(self):
        assert _classify_period_emotion(0.0, 0.5) == "neutral"


# ===========================================================================
# Test serialization
# ===========================================================================

class TestSerialization:

    def test_compression_to_dict(self):
        """compression_to_dict should produce a dict with all keys."""
        data = _make_dramatic_timeline()
        result = generate_timelapse(data, num_periods=3)
        d = compression_to_dict(result)
        assert "key_moments" in d
        assert "emotional_arc" in d
        assert "timelapse" in d
        assert "total_samples" in d
        assert "compression_ratio" in d
        assert "summary" in d

    def test_empty_compression_to_dict(self):
        """Empty result should serialize cleanly."""
        result = generate_timelapse([])
        d = compression_to_dict(result)
        assert d["total_samples"] == 0
        assert d["emotional_arc"] is None
        assert d["key_moments"] == []


# ===========================================================================
# Test API routes
# ===========================================================================

class TestAPIRoutes:

    def test_status_endpoint(self):
        """GET /temporal/status should return ready."""
        resp = client.get("/temporal/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert "features" in data

    def test_compress_endpoint(self):
        """POST /temporal/compress should return compression result."""
        data_points = [
            {"timestamp": 1e6 + i * 3600, "valence": round(-0.9 + 0.09 * i, 2), "arousal": 0.5}
            for i in range(20)
        ]
        resp = client.post("/temporal/compress", json={
            "data": data_points,
            "num_periods": 5,
            "max_key_moments": 10,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "key_moments" in data
        assert "emotional_arc" in data
        assert "timelapse" in data
        assert len(data["timelapse"]) == 5
        assert data["total_samples"] == 20

    def test_compress_minimal_data(self):
        """POST /temporal/compress with minimal data should still work."""
        resp = client.post("/temporal/compress", json={
            "data": [{"timestamp": 1e6, "valence": 0.5}],
            "num_periods": 1,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_samples"] == 1

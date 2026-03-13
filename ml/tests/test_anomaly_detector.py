"""Tests for ml/notifications/anomaly_detector.py (issue #374)."""

from __future__ import annotations

import sys
import os

# Make sure ml/ is on the path when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from notifications.anomaly_detector import Anomaly, AnomalyDetector


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def detector() -> AnomalyDetector:
    return AnomalyDetector(z_threshold=1.5)


def _baseline(**overrides) -> dict:
    """Return a minimal 30-day baseline dict."""
    base = {
        "sleep_quality": 0.75,
        "sleep_quality_std": 0.08,
        "voice_valence": 0.1,
        "voice_valence_std": 0.15,
        "hrv_avg": 55.0,
        "hrv_avg_std": 10.0,
        "dream_recall_rate": 0.5,
        "dream_recall_rate_std": 0.2,
        "stress_index": 0.35,
        "stress_index_std": 0.1,
    }
    base.update(overrides)
    return base


def _user_data(**overrides) -> dict:
    """Return a minimal user snapshot close to baseline (no anomalies)."""
    data = {
        "sleep_quality": 0.75,
        "voice_valence": 0.1,
        "hrv_avg": 55.0,
        "dream_recall_rate": 0.5,
        "stress_index": 0.35,
    }
    data.update(overrides)
    return data


# ── detect_anomalies — normal (no anomalies) ─────────────────────────────

class TestNoAnomalies:
    def test_returns_empty_when_values_match_baseline(self, detector):
        anomalies = detector.detect_anomalies(_user_data(), _baseline())
        assert anomalies == []

    def test_returns_empty_when_user_data_empty(self, detector):
        anomalies = detector.detect_anomalies({}, _baseline())
        assert anomalies == []

    def test_returns_empty_when_baseline_empty(self, detector):
        anomalies = detector.detect_anomalies(_user_data(), {})
        assert anomalies == []

    def test_ignores_missing_std(self, detector):
        """If std key absent, that metric should be silently skipped."""
        baseline = _baseline()
        del baseline["sleep_quality_std"]
        anomalies = detector.detect_anomalies(_user_data(), baseline)
        metric_names = [a.metric for a in anomalies]
        assert "sleep_quality" not in metric_names

    def test_ignores_zero_std(self, detector):
        """Zero std would cause division by zero — metric must be skipped."""
        baseline = _baseline(sleep_quality_std=0.0)
        anomalies = detector.detect_anomalies(_user_data(), baseline)
        metric_names = [a.metric for a in anomalies]
        assert "sleep_quality" not in metric_names


# ── detect_anomalies — anomalies detected ────────────────────────────────

class TestAnomalyDetected:
    def test_detects_low_sleep_quality(self, detector):
        # 0.75 - 2 * 0.08 = 0.59 → z ≈ -2.0 → exceeds 1.5
        data = _user_data(sleep_quality=0.59)
        anomalies = detector.detect_anomalies(data, _baseline())
        metrics = [a.metric for a in anomalies]
        assert "sleep_quality" in metrics

    def test_detects_high_stress(self, detector):
        # baseline mean=0.35, std=0.1 → 0.35 + 2*0.1 = 0.55 → z=2.0
        data = _user_data(stress_index=0.55)
        anomalies = detector.detect_anomalies(data, _baseline())
        metrics = [a.metric for a in anomalies]
        assert "stress_index" in metrics

    def test_detects_low_hrv(self, detector):
        # baseline mean=55, std=10 → 55 - 2*10 = 35 → z=-2.0
        data = _user_data(hrv_avg=35.0)
        anomalies = detector.detect_anomalies(data, _baseline())
        metrics = [a.metric for a in anomalies]
        assert "hrv" in metrics

    def test_detects_negative_valence(self, detector):
        # baseline mean=0.1, std=0.15 → 0.1 - 2*0.15 = -0.2 → z≈-2.0
        data = _user_data(voice_valence=-0.2)
        anomalies = detector.detect_anomalies(data, _baseline())
        metrics = [a.metric for a in anomalies]
        assert "voice_valence" in metrics

    def test_sorted_by_abs_z_score_descending(self, detector):
        # Low sleep (z≈-2.0) and high stress (z=2.0) both exceed threshold
        data = _user_data(sleep_quality=0.59, stress_index=0.57)
        anomalies = detector.detect_anomalies(data, _baseline())
        assert len(anomalies) >= 2
        z_scores = [abs(a.z_score) for a in anomalies]
        assert z_scores == sorted(z_scores, reverse=True)


# ── Anomaly dataclass ─────────────────────────────────────────────────────

class TestAnomalyFields:
    def test_anomaly_has_required_fields(self, detector):
        data = _user_data(sleep_quality=0.50)  # z ≈ -3.1
        anomalies = detector.detect_anomalies(data, _baseline())
        sleep_anomaly = next(a for a in anomalies if a.metric == "sleep_quality")

        assert isinstance(sleep_anomaly.metric, str)
        assert isinstance(sleep_anomaly.value, float)
        assert isinstance(sleep_anomaly.baseline_mean, float)
        assert isinstance(sleep_anomaly.baseline_std, float)
        assert isinstance(sleep_anomaly.z_score, float)
        assert sleep_anomaly.direction in ("above", "below")
        assert isinstance(sleep_anomaly.description, str)
        assert len(sleep_anomaly.description) > 10

    def test_direction_below_for_low_value(self, detector):
        data = _user_data(sleep_quality=0.50)
        anomalies = detector.detect_anomalies(data, _baseline())
        sleep_anomaly = next(a for a in anomalies if a.metric == "sleep_quality")
        assert sleep_anomaly.direction == "below"

    def test_direction_above_for_high_stress(self, detector):
        data = _user_data(stress_index=0.60)
        anomalies = detector.detect_anomalies(data, _baseline())
        stress_anomaly = next(a for a in anomalies if a.metric == "stress_index")
        assert stress_anomaly.direction == "above"

    def test_z_score_matches_formula(self, detector):
        # sleep_quality baseline: mean=0.75, std=0.08
        # value=0.59 → z = (0.59 - 0.75) / 0.08 = -2.0
        data = _user_data(sleep_quality=0.59)
        anomalies = detector.detect_anomalies(data, _baseline())
        sleep_anomaly = next(a for a in anomalies if a.metric == "sleep_quality")
        assert abs(sleep_anomaly.z_score - (-2.0)) < 0.05

    def test_description_contains_metric_name(self, detector):
        data = _user_data(hrv_avg=35.0)
        anomalies = detector.detect_anomalies(data, _baseline())
        hrv_anomaly = next(a for a in anomalies if a.metric == "hrv")
        assert "HRV" in hrv_anomaly.description or "heart-rate" in hrv_anomaly.description


# ── Custom threshold ──────────────────────────────────────────────────────

class TestCustomThreshold:
    def test_higher_threshold_produces_fewer_anomalies(self):
        strict = AnomalyDetector(z_threshold=3.0)
        lenient = AnomalyDetector(z_threshold=1.0)
        data = _user_data(sleep_quality=0.59)  # z ≈ -2.0
        strict_anomalies = strict.detect_anomalies(data, _baseline())
        lenient_anomalies = lenient.detect_anomalies(data, _baseline())
        assert len(strict_anomalies) <= len(lenient_anomalies)

    def test_threshold_not_exceeded_returns_empty(self):
        strict = AnomalyDetector(z_threshold=5.0)
        data = _user_data(sleep_quality=0.59)
        anomalies = strict.detect_anomalies(data, _baseline())
        assert anomalies == []


# ── MorningReportGenerator integration ───────────────────────────────────

class TestMorningReportIntegration:
    def test_generate_without_baseline_returns_no_anomalies(self):
        from notifications.smart_notifications import MorningReportGenerator
        gen = MorningReportGenerator()
        voice = {"avg_valence": 0.1, "avg_stress": 0.3, "dominant_emotion": "neutral"}
        health = {"sleep_efficiency": 75.0, "hrv_sdnn": 55.0}
        result = gen.generate("user1", voice_data=voice, health_data=health)
        assert "anomalies" in result
        assert result["anomalies"] == []

    def test_generate_with_baseline_includes_anomaly_in_body(self):
        from notifications.smart_notifications import MorningReportGenerator
        gen = MorningReportGenerator()
        voice = {"avg_valence": -0.2, "avg_stress": 0.6, "dominant_emotion": "sad"}
        health = {"sleep_efficiency": 50.0, "hrv_sdnn": 35.0}
        baseline = {
            "sleep_quality": 0.75,
            "sleep_quality_std": 0.08,
            "voice_valence": 0.1,
            "voice_valence_std": 0.15,
            "hrv_avg": 55.0,
            "hrv_avg_std": 10.0,
            "stress_index": 0.35,
            "stress_index_std": 0.1,
        }
        result = gen.generate(
            "user1", voice_data=voice, health_data=health, baseline=baseline
        )
        assert result["has_data"] is True
        # At least one anomaly should be detected
        assert len(result["anomalies"]) >= 1
        # Most extreme anomaly description should appear in notification body
        if result["anomalies"]:
            top_desc = result["anomalies"][0]["description"]
            assert top_desc[:20] in result["body"]

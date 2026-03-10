"""Tests for voice-based cognitive screening and elderly emotional monitoring."""

import numpy as np
import pytest

from models.cognitive_screening import (
    ElderlyEmotionMonitor,
    VoiceCognitiveScreener,
)

FS = 16000


def _make_speech_audio(duration: float = 5.0, with_pauses: bool = False) -> np.ndarray:
    """Generate synthetic speech-like audio for testing."""
    n = int(FS * duration)
    t = np.linspace(0, duration, n)
    # Simulated speech: mix of tones + noise
    signal = (
        0.3 * np.sin(2 * np.pi * 150 * t)
        + 0.15 * np.sin(2 * np.pi * 300 * t)
        + 0.1 * np.random.randn(n)
    )
    if with_pauses:
        # Insert silence gaps at regular intervals
        pause_len = int(FS * 0.8)  # 800ms pauses
        for i in range(3):
            start = int(n * (i + 1) / 4)
            end = min(start + pause_len, n)
            signal[start:end] = 0.0
    return signal.astype(np.float64)


def _make_monotone_audio(duration: float = 5.0) -> np.ndarray:
    """Generate monotone audio (low prosodic variation)."""
    n = int(FS * duration)
    t = np.linspace(0, duration, n)
    # Single pure tone — no variation
    return (0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float64)


# ── VoiceCognitiveScreener tests ──────────────────────────────────────────────


class TestFeatureExtraction:
    def test_extracts_features_from_speech(self):
        audio = _make_speech_audio(5.0)
        screener = VoiceCognitiveScreener()
        features = screener.extract_cognitive_features(audio, FS)
        assert "duration_seconds" in features
        assert features["duration_seconds"] >= 4.5
        assert "pause_count" in features
        assert "speech_rate_syl_s" in features
        assert "f0_mean" in features
        assert "energy_cv" in features
        assert "jitter" in features
        assert "shimmer" in features
        assert "hnr" in features

    def test_empty_features_for_short_audio(self):
        audio = np.zeros(100)  # Way too short
        screener = VoiceCognitiveScreener()
        features = screener.extract_cognitive_features(audio, FS)
        assert features["duration_seconds"] == 0.0
        assert features["pause_count"] == 0

    def test_detects_pauses(self):
        audio = _make_speech_audio(5.0, with_pauses=True)
        screener = VoiceCognitiveScreener()
        features = screener.extract_cognitive_features(audio, FS)
        assert features["pause_count"] >= 1
        assert features["pause_duration_mean"] > 0

    def test_speech_rate_positive(self):
        audio = _make_speech_audio(5.0)
        screener = VoiceCognitiveScreener()
        features = screener.extract_cognitive_features(audio, FS)
        assert features["speech_rate_syl_s"] >= 0


class TestScreening:
    def test_screening_returns_required_fields(self):
        audio = _make_speech_audio(5.0)
        screener = VoiceCognitiveScreener()
        result = screener.screen(audio, FS)
        assert "cognitive_risk_score" in result
        assert "risk_level" in result
        assert "feature_flags" in result
        assert "disclaimer" in result
        assert "confidence" in result
        assert "component_scores" in result

    def test_risk_score_in_range(self):
        audio = _make_speech_audio(10.0)
        screener = VoiceCognitiveScreener()
        result = screener.screen(audio, FS)
        assert 0.0 <= result["cognitive_risk_score"] <= 1.0

    def test_risk_level_valid(self):
        audio = _make_speech_audio(5.0)
        screener = VoiceCognitiveScreener()
        result = screener.screen(audio, FS)
        assert result["risk_level"] in ("normal", "monitor", "evaluate", "insufficient_data")

    def test_insufficient_data_for_short_audio(self):
        audio = np.zeros(int(FS * 0.5))  # 0.5 seconds
        screener = VoiceCognitiveScreener()
        result = screener.screen(audio, FS)
        assert result["risk_level"] == "insufficient_data"
        assert result["confidence"] == 0.0

    def test_disclaimer_always_present(self):
        audio = _make_speech_audio(5.0)
        screener = VoiceCognitiveScreener()
        result = screener.screen(audio, FS)
        assert "screening tool only" in result["disclaimer"].lower()
        assert "NOT" in result["disclaimer"]

    def test_component_scores_present(self):
        audio = _make_speech_audio(5.0)
        screener = VoiceCognitiveScreener()
        result = screener.screen(audio, FS)
        cs = result["component_scores"]
        assert "pause_abnormality" in cs
        assert "speech_fluency" in cs
        assert "prosodic_variation" in cs
        assert "voice_quality" in cs
        for v in cs.values():
            assert 0.0 <= v <= 1.0

    def test_confidence_scales_with_duration(self):
        short = _make_speech_audio(3.0)
        long = _make_speech_audio(30.0)
        screener = VoiceCognitiveScreener()
        r_short = screener.screen(short, FS)
        r_long = screener.screen(long, FS)
        assert r_long["confidence"] > r_short["confidence"]


class TestLongitudinal:
    def test_add_and_get_trajectory(self):
        screener = VoiceCognitiveScreener()
        screener.add_longitudinal_point("test_user", {
            "cognitive_risk_score": 0.3,
            "risk_level": "normal",
            "feature_flags": [],
        })
        screener.add_longitudinal_point("test_user", {
            "cognitive_risk_score": 0.5,
            "risk_level": "monitor",
            "feature_flags": ["elevated_pause_frequency"],
        })
        traj = screener.get_trajectory("test_user")
        assert traj["n_assessments"] == 2
        assert len(traj["trajectory"]) == 2
        assert traj["trajectory"][1]["risk_score"] == 0.5

    def test_trend_detection(self):
        screener = VoiceCognitiveScreener()
        # Add worsening trajectory
        for score in [0.2, 0.25, 0.3, 0.4, 0.5, 0.55]:
            screener.add_longitudinal_point("worsening_user", {
                "cognitive_risk_score": score,
                "risk_level": "monitor",
                "feature_flags": [],
            })
        traj = screener.get_trajectory("worsening_user")
        assert traj["trend"] == "worsening"

    def test_insufficient_data_trend(self):
        screener = VoiceCognitiveScreener()
        screener.add_longitudinal_point("single_user", {
            "cognitive_risk_score": 0.3,
            "risk_level": "normal",
            "feature_flags": [],
        })
        traj = screener.get_trajectory("single_user")
        assert traj["trend"] == "insufficient_data"

    def test_empty_trajectory(self):
        screener = VoiceCognitiveScreener()
        traj = screener.get_trajectory("nonexistent_user")
        assert traj["n_assessments"] == 0
        assert traj["trend"] == "insufficient_data"

    def test_last_n_filter(self):
        screener = VoiceCognitiveScreener()
        for i in range(10):
            screener.add_longitudinal_point("many_user", {
                "cognitive_risk_score": i * 0.1,
                "risk_level": "normal",
                "feature_flags": [],
            })
        traj = screener.get_trajectory("many_user", last_n=3)
        assert len(traj["trajectory"]) == 3


# ── ElderlyEmotionMonitor tests ───────────────────────────────────────────────


class TestElderlyEmotionMonitor:
    def test_assess_returns_required_fields(self):
        monitor = ElderlyEmotionMonitor()
        features = {"f0_std": 30.0, "energy_cv": 0.5, "pause_rate_per_min": 8.0,
                     "speech_rate_syl_s": 3.5}
        result = monitor.assess(features, {"valence": 0.2}, age=70)
        assert "adjusted_valence" in result
        assert "wellbeing_concern" in result
        assert "loneliness_risk" in result
        assert "positivity_deviation" in result
        assert result["is_elderly_adjusted"] is True

    def test_elderly_negative_affect_amplified(self):
        monitor = ElderlyEmotionMonitor()
        features = {"f0_std": 30.0, "energy_cv": 0.5, "pause_rate_per_min": 8.0,
                     "speech_rate_syl_s": 3.5}
        result_elderly = monitor.assess(features, {"valence": -0.3}, age=70)
        result_young = monitor.assess(features, {"valence": -0.3}, age=30)
        # Elderly negative affect should be amplified
        assert result_elderly["adjusted_valence"] < result_young["adjusted_valence"]

    def test_non_elderly_no_adjustment(self):
        monitor = ElderlyEmotionMonitor()
        features = {"f0_std": 30.0, "energy_cv": 0.5, "pause_rate_per_min": 8.0,
                     "speech_rate_syl_s": 3.5}
        result = monitor.assess(features, {"valence": 0.5}, age=30)
        assert result["is_elderly_adjusted"] is False
        assert result["adjusted_valence"] == 0.5

    def test_wellbeing_concern_levels(self):
        monitor = ElderlyEmotionMonitor()
        # Happy elderly — should be "none"
        features_good = {"f0_std": 35.0, "energy_cv": 0.55, "pause_rate_per_min": 7.0,
                         "speech_rate_syl_s": 3.8}
        result = monitor.assess(features_good, {"valence": 0.4}, age=75)
        assert result["wellbeing_concern"] == "none"

    def test_loneliness_risk_from_monotone(self):
        monitor = ElderlyEmotionMonitor()
        # Low prosodic variation = loneliness risk
        features = {"f0_std": 5.0, "energy_cv": 0.1, "pause_rate_per_min": 15.0,
                     "speech_rate_syl_s": 1.5}
        result = monitor.assess_loneliness_risk(features)
        assert result["loneliness_risk_score"] > 0.0
        assert len(result["markers"]) > 0
        assert result["risk_level"] in ("low", "moderate", "high")

    def test_loneliness_risk_normal_speech(self):
        monitor = ElderlyEmotionMonitor()
        features = {"f0_std": 30.0, "energy_cv": 0.5, "pause_rate_per_min": 8.0,
                     "speech_rate_syl_s": 3.5}
        result = monitor.assess_loneliness_risk(features)
        assert result["loneliness_risk_score"] == 0.0
        assert result["risk_level"] == "low"
        assert len(result["markers"]) == 0

"""Tests for Ambient Sound Environment Profiler.

Covers:
  - Sound classification for each of the 7 categories
  - Sound feature computation and normalisation
  - Sound-emotion correlation with paired data
  - Insight generation (best/worst environments, transitions)
  - Edge cases: silence, very loud, mixed, empty inputs, missing optionals
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.ambient_sound_model import (
    SOUND_CATEGORIES,
    classify_sound_environment,
    compute_sound_features,
    correlate_sound_emotion,
    generate_sound_insights,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _features_for_category(category: str) -> dict:
    """Return synthetic normalised features that should classify as *category*.

    These are hand-crafted to match the category profile templates closely.
    """
    profiles = {
        "silence": dict(
            spectral_centroid=100.0,
            spectral_energy=0.01,
            zero_crossing_rate=0.01,
            spectral_bandwidth=100.0,
            spectral_rolloff=200.0,
        ),
        "nature": dict(
            spectral_centroid=3800.0,
            spectral_energy=0.30,
            zero_crossing_rate=0.10,
            spectral_bandwidth=4400.0,
            spectral_rolloff=5000.0,
        ),
        "urban": dict(
            spectral_centroid=6000.0,
            spectral_energy=0.75,
            zero_crossing_rate=0.35,
            spectral_bandwidth=7100.0,
            spectral_rolloff=6600.0,
        ),
        "social": dict(
            spectral_centroid=5000.0,
            spectral_energy=0.50,
            zero_crossing_rate=0.225,
            spectral_bandwidth=3850.0,
            spectral_rolloff=5500.0,
        ),
        "music": dict(
            spectral_centroid=5500.0,
            spectral_energy=0.55,
            zero_crossing_rate=0.125,
            spectral_bandwidth=5500.0,
            spectral_rolloff=7700.0,
        ),
        "indoor": dict(
            spectral_centroid=2200.0,
            spectral_energy=0.25,
            zero_crossing_rate=0.075,
            spectral_bandwidth=2200.0,
            spectral_rolloff=2750.0,
        ),
        "white_noise": dict(
            spectral_centroid=5500.0,
            spectral_energy=0.60,
            zero_crossing_rate=0.325,
            spectral_bandwidth=9900.0,
            spectral_rolloff=9350.0,
        ),
    }
    return profiles[category]


def _make_sound_records(categories: list, start_time: float = 1000.0, interval: float = 10.0):
    """Create a list of sound records with sequential timestamps."""
    return [
        {"category": cat, "timestamp": start_time + i * interval}
        for i, cat in enumerate(categories)
    ]


def _make_emotion_records(
    emotions: list,
    valences: list,
    arousals: list,
    start_time: float = 1000.0,
    interval: float = 10.0,
):
    """Create a list of emotion records with sequential timestamps."""
    return [
        {
            "emotion": e,
            "valence": v,
            "arousal": a,
            "timestamp": start_time + i * interval,
        }
        for i, (e, v, a) in enumerate(zip(emotions, valences, arousals))
    ]


# ---------------------------------------------------------------------------
# TestSoundCategories
# ---------------------------------------------------------------------------


class TestSoundCategories:
    """All 7 sound categories are defined and classifiable."""

    def test_all_categories_defined(self):
        assert len(SOUND_CATEGORIES) == 7
        for cat in ["silence", "nature", "urban", "social", "music", "indoor", "white_noise"]:
            assert cat in SOUND_CATEGORIES

    def test_classify_silence(self):
        raw = _features_for_category("silence")
        features = compute_sound_features(**raw)
        result = classify_sound_environment(features)
        assert result["category"] == "silence"
        assert result["confidence"] >= 0.5

    def test_classify_nature(self):
        raw = _features_for_category("nature")
        features = compute_sound_features(**raw)
        result = classify_sound_environment(features)
        assert result["category"] == "nature"

    def test_classify_urban(self):
        raw = _features_for_category("urban")
        features = compute_sound_features(**raw)
        result = classify_sound_environment(features)
        assert result["category"] == "urban"

    def test_classify_indoor(self):
        raw = _features_for_category("indoor")
        features = compute_sound_features(**raw)
        result = classify_sound_environment(features)
        assert result["category"] == "indoor"

    def test_classify_music(self):
        raw = _features_for_category("music")
        features = compute_sound_features(**raw)
        result = classify_sound_environment(features)
        assert result["category"] == "music"

    def test_classify_social(self):
        raw = _features_for_category("social")
        features = compute_sound_features(**raw)
        result = classify_sound_environment(features)
        assert result["category"] == "social"

    def test_classify_white_noise(self):
        raw = _features_for_category("white_noise")
        features = compute_sound_features(**raw)
        result = classify_sound_environment(features)
        assert result["category"] == "white_noise"


# ---------------------------------------------------------------------------
# TestClassificationOutput
# ---------------------------------------------------------------------------


class TestClassificationOutput:
    """classify_sound_environment returns correct schema and value ranges."""

    def test_output_keys(self):
        raw = _features_for_category("nature")
        features = compute_sound_features(**raw)
        result = classify_sound_environment(features)
        assert "category" in result
        assert "confidence" in result
        assert "scores" in result
        assert "feature_summary" in result

    def test_confidence_range(self):
        raw = _features_for_category("urban")
        features = compute_sound_features(**raw)
        result = classify_sound_environment(features)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_scores_sum_to_one(self):
        raw = _features_for_category("music")
        features = compute_sound_features(**raw)
        result = classify_sound_environment(features)
        total = sum(result["scores"].values())
        assert abs(total - 1.0) < 1e-6

    def test_all_categories_in_scores(self):
        raw = _features_for_category("indoor")
        features = compute_sound_features(**raw)
        result = classify_sound_environment(features)
        for cat in SOUND_CATEGORIES:
            assert cat in result["scores"]

    def test_category_is_valid(self):
        raw = _features_for_category("social")
        features = compute_sound_features(**raw)
        result = classify_sound_environment(features)
        assert result["category"] in SOUND_CATEGORIES


# ---------------------------------------------------------------------------
# TestComputeSoundFeatures
# ---------------------------------------------------------------------------


class TestComputeSoundFeatures:
    """compute_sound_features normalises inputs correctly."""

    def test_output_keys(self):
        features = compute_sound_features(
            spectral_centroid=3000.0,
            spectral_energy=0.5,
            zero_crossing_rate=0.1,
        )
        assert "centroid" in features
        assert "energy" in features
        assert "zcr" in features
        assert "bandwidth" in features
        assert "rolloff" in features
        assert "mfcc" in features
        assert "raw" in features

    def test_normalisation_range(self):
        features = compute_sound_features(
            spectral_centroid=5000.0,
            spectral_energy=0.8,
            zero_crossing_rate=0.2,
            spectral_bandwidth=3000.0,
            spectral_rolloff=8000.0,
        )
        assert 0.0 <= features["centroid"] <= 1.0
        assert 0.0 <= features["energy"] <= 1.0
        assert 0.0 <= features["zcr"] <= 1.0
        assert 0.0 <= features["bandwidth"] <= 1.0
        assert 0.0 <= features["rolloff"] <= 1.0

    def test_mfcc_default_length(self):
        features = compute_sound_features(
            spectral_centroid=1000.0,
            spectral_energy=0.3,
            zero_crossing_rate=0.05,
        )
        assert len(features["mfcc"]) == 13

    def test_mfcc_custom(self):
        mfcc_in = [float(i) for i in range(13)]
        features = compute_sound_features(
            spectral_centroid=2000.0,
            spectral_energy=0.4,
            zero_crossing_rate=0.1,
            mfcc=mfcc_in,
        )
        assert features["mfcc"] == mfcc_in

    def test_raw_values_preserved(self):
        features = compute_sound_features(
            spectral_centroid=4321.0,
            spectral_energy=0.67,
            zero_crossing_rate=0.15,
        )
        assert features["raw"]["spectral_centroid"] == 4321.0
        assert features["raw"]["spectral_energy"] == 0.67
        assert features["raw"]["zero_crossing_rate"] == 0.15


# ---------------------------------------------------------------------------
# TestSoundEmotionCorrelation
# ---------------------------------------------------------------------------


class TestSoundEmotionCorrelation:
    """correlate_sound_emotion pairs records and computes stats."""

    def test_empty_inputs(self):
        result = correlate_sound_emotion([], [])
        assert result["paired_count"] == 0

    def test_no_matching_timestamps(self):
        sounds = _make_sound_records(["nature"], start_time=0.0)
        emotions = _make_emotion_records(["happy"], [0.8], [0.6], start_time=1000.0)
        result = correlate_sound_emotion(sounds, emotions)
        assert result["paired_count"] == 0

    def test_paired_count(self):
        sounds = _make_sound_records(["nature", "urban", "silence"])
        emotions = _make_emotion_records(
            ["happy", "angry", "neutral"],
            [0.8, -0.5, 0.0],
            [0.6, 0.9, 0.2],
        )
        result = correlate_sound_emotion(sounds, emotions)
        assert result["paired_count"] == 3

    def test_per_category_stats(self):
        sounds = _make_sound_records(["nature", "nature", "urban"])
        emotions = _make_emotion_records(
            ["happy", "happy", "angry"],
            [0.7, 0.9, -0.6],
            [0.5, 0.4, 0.8],
        )
        result = correlate_sound_emotion(sounds, emotions)
        assert "nature" in result["per_category"]
        assert result["per_category"]["nature"]["n_samples"] == 2
        assert result["per_category"]["nature"]["dominant_emotion"] == "happy"

    def test_valence_by_category(self):
        sounds = _make_sound_records(["music", "music"])
        emotions = _make_emotion_records(
            ["happy", "happy"],
            [0.6, 0.8],
            [0.5, 0.5],
        )
        result = correlate_sound_emotion(sounds, emotions)
        assert "music" in result["valence_by_category"]
        assert abs(result["valence_by_category"]["music"]["mean"] - 0.7) < 1e-6


# ---------------------------------------------------------------------------
# TestInsightGeneration
# ---------------------------------------------------------------------------


class TestInsightGeneration:
    """generate_sound_insights returns actionable insights."""

    def test_empty_correlation(self):
        result = generate_sound_insights({"per_category": {}})
        assert result["best_environments"] == []
        assert result["worst_environments"] == []
        assert len(result["recommendations"]) >= 1

    def test_best_and_worst(self):
        correlation = {
            "per_category": {
                "nature": {"mean_valence": 0.7, "mean_arousal": 0.4, "n_samples": 5},
                "urban": {"mean_valence": -0.3, "mean_arousal": 0.8, "n_samples": 5},
            },
        }
        result = generate_sound_insights(correlation)
        best_cats = [e["category"] for e in result["best_environments"]]
        worst_cats = [e["category"] for e in result["worst_environments"]]
        assert "nature" in best_cats
        assert "urban" in worst_cats

    def test_transition_effects(self):
        correlation = {
            "per_category": {
                "nature": {"mean_valence": 0.5, "mean_arousal": 0.3, "n_samples": 3},
            },
        }
        history = [
            {"category": "silence"},
            {"category": "nature"},
            {"category": "nature"},
            {"category": "urban"},
            {"category": "silence"},
        ]
        result = generate_sound_insights(correlation, history=history)
        assert len(result["transition_effects"]) > 0
        transitions_str = [t["transition"] for t in result["transition_effects"]]
        assert "silence -> nature" in transitions_str

    def test_recommendations_not_empty(self):
        correlation = {
            "per_category": {
                "music": {"mean_valence": 0.6, "mean_arousal": 0.5, "n_samples": 10},
            },
        }
        result = generate_sound_insights(correlation)
        assert len(result["recommendations"]) >= 1


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: silence, very loud, mixed features, boundary values."""

    def test_zero_energy_is_silence(self):
        features = compute_sound_features(
            spectral_centroid=0.0,
            spectral_energy=0.0,
            zero_crossing_rate=0.0,
        )
        result = classify_sound_environment(features)
        assert result["category"] == "silence"
        assert result["confidence"] == 1.0

    def test_very_loud_signal(self):
        """Energy above normalisation cap should still classify without error."""
        features = compute_sound_features(
            spectral_centroid=8000.0,
            spectral_energy=5.0,
            zero_crossing_rate=0.4,
            spectral_bandwidth=9000.0,
            spectral_rolloff=10000.0,
        )
        result = classify_sound_environment(features)
        assert result["category"] in SOUND_CATEGORIES
        assert 0.0 <= result["confidence"] <= 1.0

    def test_mixed_features(self):
        """Features that don't cleanly match any category still produce a result."""
        features = compute_sound_features(
            spectral_centroid=4000.0,
            spectral_energy=0.45,
            zero_crossing_rate=0.25,
            spectral_bandwidth=5000.0,
            spectral_rolloff=5000.0,
        )
        result = classify_sound_environment(features)
        assert result["category"] in SOUND_CATEGORIES

    def test_single_sound_single_emotion_correlation(self):
        sounds = [{"category": "nature", "timestamp": 100.0}]
        emotions = [{"emotion": "happy", "valence": 0.9, "arousal": 0.3, "timestamp": 100.0}]
        result = correlate_sound_emotion(sounds, emotions)
        assert result["paired_count"] == 1

    def test_insights_with_no_history(self):
        correlation = {
            "per_category": {
                "indoor": {"mean_valence": 0.1, "mean_arousal": 0.2, "n_samples": 2},
            },
        }
        result = generate_sound_insights(correlation, history=None)
        assert result["transition_effects"] == []

    def test_all_negative_valence_no_best(self):
        correlation = {
            "per_category": {
                "urban": {"mean_valence": -0.5, "mean_arousal": 0.9, "n_samples": 5},
                "indoor": {"mean_valence": -0.1, "mean_arousal": 0.3, "n_samples": 3},
            },
        }
        result = generate_sound_insights(correlation)
        assert result["best_environments"] == []
        assert len(result["worst_environments"]) == 2

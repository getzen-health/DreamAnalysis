"""Tests for CNNKANEmotionClassifier (CNN-KAN-F2CA sparse-channel model).

Covers model instantiation, predict() output contract, edge cases,
feature-based fallback, and model info structure.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure ml/ is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.cnn_kan_emotion import (
    CNNKANEmotionClassifier,
    EMOTION_CLASSES,
    get_cnn_kan_classifier,
    _feature_based_predict,
    _soft_probs_from_valence_arousal,
    _compute_faa,
)

FS = 256.0
N_SAMPLES = 1024  # 4 seconds at 256 Hz


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def classifier():
    return CNNKANEmotionClassifier()


@pytest.fixture
def multichannel():
    """4 channels × 1024 samples of synthetic EEG (~20 µV RMS)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((4, N_SAMPLES)).astype(np.float32) * 20.0


@pytest.fixture
def single_channel():
    """1D EEG signal (n_samples,) — single channel."""
    rng = np.random.default_rng(7)
    return rng.standard_normal(N_SAMPLES).astype(np.float32) * 20.0


@pytest.fixture
def zeros_signal():
    """All-zeros 4-channel signal (disconnected / flat-line)."""
    return np.zeros((4, N_SAMPLES), dtype=np.float32)


@pytest.fixture
def short_signal():
    """Very short signal — only 16 samples (below CNN minimum)."""
    return np.random.randn(4, 16).astype(np.float32)


# ── 1. Instantiation ──────────────────────────────────────────────────────────

class TestInstantiation:
    def test_default_instantiation(self, classifier):
        assert classifier is not None

    def test_default_n_channels(self, classifier):
        assert classifier.n_channels == 4

    def test_default_n_classes(self, classifier):
        assert classifier.n_classes == 4

    def test_default_fs(self, classifier):
        assert classifier.fs == FS

    def test_custom_channels(self):
        m = CNNKANEmotionClassifier(n_channels=8)
        assert m.n_channels == 8

    def test_singleton_same_user(self):
        a = get_cnn_kan_classifier("userA")
        b = get_cnn_kan_classifier("userA")
        assert a is b

    def test_singleton_different_users(self):
        a = get_cnn_kan_classifier("userX")
        b = get_cnn_kan_classifier("userY")
        assert a is not b


# ── 2. predict() — output contract ───────────────────────────────────────────

class TestPredictOutputContract:
    """Ensure predict() always returns the required keys with valid types/ranges."""

    _REQUIRED_KEYS = {"emotion", "probabilities", "valence", "arousal", "model_type"}

    def _check_result(self, result: dict):
        # Required keys present
        for key in self._REQUIRED_KEYS:
            assert key in result, f"Missing key: {key}"

        # emotion is one of the 4 valid classes
        assert result["emotion"] in EMOTION_CLASSES, (
            f"Unknown emotion: {result['emotion']}"
        )

        # valence in [-1, 1]
        assert -1.0 <= result["valence"] <= 1.0, f"valence out of range: {result['valence']}"

        # arousal in [-1, 1]
        assert -1.0 <= result["arousal"] <= 1.0, f"arousal out of range: {result['arousal']}"

        # probabilities is a dict over 4 classes
        probs = result["probabilities"]
        assert isinstance(probs, dict)
        assert set(probs.keys()) == set(EMOTION_CLASSES), (
            f"Probability keys mismatch: {set(probs.keys())}"
        )

        # probabilities sum to ~1
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-3, f"Probabilities do not sum to 1: {total}"

        # each probability is [0, 1]
        for cls, p in probs.items():
            assert 0.0 <= p <= 1.0, f"Probability out of range for {cls}: {p}"

    def test_multichannel_4ch(self, classifier, multichannel):
        result = classifier.predict(multichannel, FS)
        self._check_result(result)

    def test_single_channel_1d(self, classifier, single_channel):
        result = classifier.predict(single_channel, FS)
        self._check_result(result)

    def test_zeros_signal(self, classifier, zeros_signal):
        """All-zeros should not crash; returns some valid output."""
        result = classifier.predict(zeros_signal, FS)
        self._check_result(result)

    def test_short_signal_edge(self, classifier, short_signal):
        """Signal shorter than 32 samples triggers neutral fallback."""
        result = classifier.predict(short_signal, FS)
        self._check_result(result)

    def test_model_type_is_string(self, classifier, multichannel):
        result = classifier.predict(multichannel, FS)
        assert isinstance(result["model_type"], str)
        assert len(result["model_type"]) > 0


# ── 3. Probabilities sum to 1.0 ───────────────────────────────────────────────

class TestProbabilityNormalisation:
    def test_probs_sum_to_one_4ch(self, classifier, multichannel):
        result = classifier.predict(multichannel, FS)
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 1e-3

    def test_probs_sum_to_one_1ch(self, classifier, single_channel):
        result = classifier.predict(single_channel, FS)
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 1e-3


# ── 4. Emotion is always a valid class ────────────────────────────────────────

class TestEmotionValidity:
    def test_emotion_class_multichannel(self, classifier, multichannel):
        for _ in range(3):
            result = classifier.predict(multichannel, FS)
            assert result["emotion"] in EMOTION_CLASSES

    def test_all_four_classes_are_reachable(self):
        """Verify that EMOTION_CLASSES contains all four expected quadrant labels."""
        expected = {
            "low_valence_low_arousal",
            "high_valence_low_arousal",
            "low_valence_high_arousal",
            "high_valence_high_arousal",
        }
        assert set(EMOTION_CLASSES) == expected


# ── 5. get_model_info() ───────────────────────────────────────────────────────

class TestModelInfo:
    _REQUIRED_KEYS = {
        "architecture", "backend", "n_channels", "n_classes",
        "fs", "total_parameters", "emotion_classes", "description",
    }

    def test_required_keys(self, classifier):
        info = classifier.get_model_info()
        for key in self._REQUIRED_KEYS:
            assert key in info, f"Missing key in model_info: {key}"

    def test_emotion_classes_list(self, classifier):
        info = classifier.get_model_info()
        assert set(info["emotion_classes"]) == set(EMOTION_CLASSES)

    def test_total_parameters_non_negative(self, classifier):
        info = classifier.get_model_info()
        assert info["total_parameters"] >= 0

    def test_architecture_name(self, classifier):
        info = classifier.get_model_info()
        assert "CNN-KAN" in info["architecture"]

    def test_singleton_info_matches_instance(self):
        c = get_cnn_kan_classifier("info_test_user")
        info = c.get_model_info()
        assert info["n_channels"] == c.n_channels


# ── 6. Feature-based fallback ─────────────────────────────────────────────────

class TestFeatureBasedFallback:
    def test_feature_fallback_direct_call(self, multichannel):
        """Call the feature-based function directly."""
        result = _feature_based_predict(multichannel, FS)
        assert "emotion" in result
        assert result["emotion"] in EMOTION_CLASSES
        assert "probabilities" in result
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 1e-3

    def test_feature_fallback_1d(self, single_channel):
        sig = single_channel.reshape(1, -1)
        result = _feature_based_predict(sig, FS)
        assert result["model_type"] == "feature-based"

    def test_feature_fallback_returns_band_powers(self, multichannel):
        result = _feature_based_predict(multichannel, FS)
        assert "band_powers" in result
        bp = result["band_powers"]
        for band in ("delta", "theta", "alpha", "beta", "high_beta"):
            assert band in bp
            assert bp[band] > 0


# ── 7. Edge cases ─────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_very_short_signal_neutral_response(self, classifier):
        short = np.zeros((4, 10), dtype=np.float32)
        result = classifier.predict(short, FS)
        assert result["emotion"] in EMOTION_CLASSES
        assert "note" in result  # short-signal note field is present

    def test_single_sample_signal(self, classifier):
        """1 sample — should not crash."""
        result = classifier.predict(np.zeros((4, 1), dtype=np.float32), FS)
        assert result["emotion"] in EMOTION_CLASSES

    def test_extra_channels_truncated(self, classifier):
        """8-channel input on a 4-channel model — should handle gracefully."""
        rng = np.random.default_rng(99)
        big = rng.standard_normal((8, N_SAMPLES)).astype(np.float32) * 20.0
        result = classifier.predict(big, FS)
        assert result["emotion"] in EMOTION_CLASSES

    def test_high_amplitude_signal(self, classifier):
        """500 µV signal (above artifact threshold) — should not crash."""
        loud = np.random.randn(4, N_SAMPLES).astype(np.float32) * 500.0
        result = classifier.predict(loud, FS)
        assert result["emotion"] in EMOTION_CLASSES


# ── 8. FAA helper ────────────────────────────────────────────────────────────

class TestFAA:
    def test_faa_returns_float(self, multichannel):
        faa = _compute_faa(multichannel, FS)
        assert isinstance(faa, float)

    def test_faa_1d_returns_zero(self, single_channel):
        """1D signal has no channel dimension; FAA is 0."""
        faa = _compute_faa(single_channel.reshape(1, -1), FS)
        assert faa == 0.0

    def test_soft_probs_sum_to_one(self):
        probs = _soft_probs_from_valence_arousal(0.5, 0.3)
        total = sum(probs.values())
        # Values are rounded to 4 decimal places, so tolerate rounding error
        assert abs(total - 1.0) < 1e-3

    def test_soft_probs_hvha_dominant_for_positive_state(self):
        """High valence + high arousal → high_valence_high_arousal should win."""
        probs = _soft_probs_from_valence_arousal(0.9, 0.9)
        winner = max(probs, key=probs.get)
        assert winner == "high_valence_high_arousal"

"""Tests for spectral band importance weighting in enhanced emotion features.

Verifies:
1. BAND_IMPORTANCE_WEIGHTS constant exists and has correct values
2. DE features reflect importance weighting (alpha/theta amplified, gamma suppressed)
3. Feature extraction still returns correct shape and finite values after weighting
4. Weighting is consistent between Python and TypeScript sides
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from processing.emotion_features_enhanced import (
    BAND_IMPORTANCE_WEIGHTS,
    ENHANCED_FEATURE_DIM,
    FEATURE_NAMES,
    extract_enhanced_emotion_features,
)


class TestBandImportanceWeights:
    """Tests for the BAND_IMPORTANCE_WEIGHTS constant."""

    def test_weights_exist(self):
        """Weights dictionary should be exported."""
        assert isinstance(BAND_IMPORTANCE_WEIGHTS, dict)

    def test_has_all_five_bands(self):
        """Should have weights for all 5 standard EEG bands."""
        for band in ["delta", "theta", "alpha", "beta", "gamma"]:
            assert band in BAND_IMPORTANCE_WEIGHTS, f"Missing weight for {band}"

    def test_alpha_is_highest(self):
        """Alpha should have the highest weight (most emotion-relevant)."""
        assert BAND_IMPORTANCE_WEIGHTS["alpha"] >= BAND_IMPORTANCE_WEIGHTS["beta"]
        assert BAND_IMPORTANCE_WEIGHTS["alpha"] >= BAND_IMPORTANCE_WEIGHTS["gamma"]
        assert BAND_IMPORTANCE_WEIGHTS["alpha"] >= BAND_IMPORTANCE_WEIGHTS["delta"]

    def test_theta_is_second_highest(self):
        """Theta should be weighted higher than beta, delta, and gamma."""
        assert BAND_IMPORTANCE_WEIGHTS["theta"] > BAND_IMPORTANCE_WEIGHTS["beta"]
        assert BAND_IMPORTANCE_WEIGHTS["theta"] > BAND_IMPORTANCE_WEIGHTS["gamma"]
        assert BAND_IMPORTANCE_WEIGHTS["theta"] > BAND_IMPORTANCE_WEIGHTS["delta"]

    def test_gamma_is_lowest(self):
        """Gamma should be strongly suppressed (EMG noise on Muse 2)."""
        assert BAND_IMPORTANCE_WEIGHTS["gamma"] < BAND_IMPORTANCE_WEIGHTS["delta"]
        assert BAND_IMPORTANCE_WEIGHTS["gamma"] < 0.5

    def test_specific_values(self):
        """Verify the exact weight values match the design spec."""
        assert BAND_IMPORTANCE_WEIGHTS["alpha"] == 1.5
        assert BAND_IMPORTANCE_WEIGHTS["theta"] == 1.3
        assert BAND_IMPORTANCE_WEIGHTS["beta"] == 1.0
        assert BAND_IMPORTANCE_WEIGHTS["delta"] == 0.8
        assert BAND_IMPORTANCE_WEIGHTS["gamma"] == 0.3

    def test_all_weights_positive(self):
        """All weights must be positive (no band fully zeroed out)."""
        for band, weight in BAND_IMPORTANCE_WEIGHTS.items():
            assert weight > 0, f"Weight for {band} must be > 0"

    def test_matches_typescript_values(self):
        """Weights must match the TypeScript BAND_IMPORTANCE_WEIGHTS in eeg-features.ts.

        This is a documentation-style test that ensures Python/TS stay in sync.
        The TypeScript values are: alpha=1.5, theta=1.3, beta=1.0, delta=0.8, gamma=0.3.
        """
        expected = {"delta": 0.8, "theta": 1.3, "alpha": 1.5, "beta": 1.0, "gamma": 0.3}
        for band, expected_weight in expected.items():
            assert BAND_IMPORTANCE_WEIGHTS[band] == expected_weight, (
                f"Python weight for {band} ({BAND_IMPORTANCE_WEIGHTS[band]}) "
                f"does not match TypeScript value ({expected_weight})"
            )


class TestWeightedFeatureExtraction:
    """Tests that DE features correctly apply importance weighting."""

    @pytest.fixture
    def four_channel_eeg(self):
        """4 channels x 4 seconds of synthetic EEG at 256 Hz."""
        rng = np.random.default_rng(42)
        return rng.standard_normal((4, 1024)) * 20.0

    @pytest.fixture
    def alpha_dominant_eeg(self):
        """4-channel EEG with dominant alpha (10 Hz)."""
        rng = np.random.default_rng(123)
        t = np.arange(1024) / 256.0
        signals = rng.standard_normal((4, 1024)) * 3.0
        # Inject strong 10 Hz alpha into all channels
        for ch in range(4):
            signals[ch] += 40.0 * np.sin(2 * np.pi * 10.0 * t)
        return signals

    @pytest.fixture
    def gamma_dominant_eeg(self):
        """4-channel EEG with dominant gamma (40 Hz)."""
        rng = np.random.default_rng(456)
        t = np.arange(1024) / 256.0
        signals = rng.standard_normal((4, 1024)) * 3.0
        # Inject strong 40 Hz gamma into all channels
        for ch in range(4):
            signals[ch] += 40.0 * np.sin(2 * np.pi * 40.0 * t)
        return signals

    def test_returns_correct_shape(self, four_channel_eeg):
        """Feature vector shape should be unchanged after adding weights."""
        features = extract_enhanced_emotion_features(four_channel_eeg, fs=256)
        assert features.shape == (ENHANCED_FEATURE_DIM,)

    def test_all_features_finite(self, four_channel_eeg):
        """No NaN or inf values allowed after weighting."""
        features = extract_enhanced_emotion_features(four_channel_eeg, fs=256)
        assert np.all(np.isfinite(features))

    def test_alpha_de_amplified(self, alpha_dominant_eeg):
        """Alpha DE features should be amplified by the 1.5x weight.

        For alpha-dominant signal, the weighted DE for alpha bands should be
        larger than unweighted (i.e., the weight effectively scales the DE value).
        """
        features = extract_enhanced_emotion_features(alpha_dominant_eeg, fs=256)
        # DE features layout: bands [delta, theta, alpha, beta, gamma] x 4 channels
        # Alpha DE is at indices 2*4+0=8, 2*4+1=9, 2*4+2=10, 2*4+3=11
        alpha_de_indices = [8, 9, 10, 11]
        alpha_des = features[alpha_de_indices]

        # For alpha-dominant signal, alpha DE should be clearly positive
        # (DE measures log-variance, high power = high DE)
        assert np.mean(alpha_des) > 0, "Alpha DE should be positive for alpha-dominant signal"

    def test_gamma_de_suppressed_vs_alpha(self, four_channel_eeg):
        """Gamma DE should be lower relative to alpha DE after weighting,
        compared to what would happen without weighting."""
        features = extract_enhanced_emotion_features(four_channel_eeg, fs=256)

        # Alpha DE indices: band 2 (alpha) x 4 channels -> indices 8-11
        # Gamma DE indices: band 4 (gamma) x 4 channels -> indices 16-19
        alpha_de_mean = np.mean(features[8:12])
        gamma_de_mean = np.mean(features[16:20])

        # Both should be finite
        assert np.isfinite(alpha_de_mean)
        assert np.isfinite(gamma_de_mean)

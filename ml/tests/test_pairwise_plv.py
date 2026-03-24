"""Tests for pairwise PLV functional connectivity features.

Tests compute_pairwise_plv() which extracts per-pair, per-band PLV values
from 4-channel Muse 2 EEG for use in emotion classification.

Reference: Wang et al. (2024), "Fusion of Multi-domain EEG Signatures
Improves Emotion Recognition" -- PLV + microstates + PSD fusion achieved
64.69% cross-subject accuracy (4-class), 7%+ improvement over single-domain.
"""

import numpy as np
import pytest

import sys
sys.path.insert(0, "/Users/sravyalu/NeuralDreamWorkshop/ml")

from processing.eeg_processor import compute_pairwise_plv


FS = 256
DURATION_SEC = 4
N_SAMPLES = FS * DURATION_SEC
N_CHANNELS = 4

# Muse 2 channel order: TP9=0, AF7=1, AF8=2, TP10=3
# 4C2 = 6 pairs: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
# Frontal pair: AF7-AF8 = (1,2)
# Fronto-temporal: AF7-TP9 = (0,1), AF8-TP10 = (2,3)


class TestPairwisePLVReturnStructure:
    """Verify the return dict has the expected keys and shapes."""

    def test_returns_dict(self):
        signals = np.random.randn(N_CHANNELS, N_SAMPLES)
        result = compute_pairwise_plv(signals, FS)
        assert isinstance(result, dict)

    def test_has_band_keys(self):
        signals = np.random.randn(N_CHANNELS, N_SAMPLES)
        result = compute_pairwise_plv(signals, FS)
        for band in ["theta", "alpha", "beta"]:
            assert f"plv_{band}" in result, f"Missing plv_{band}"

    def test_each_band_has_6_pairs(self):
        signals = np.random.randn(N_CHANNELS, N_SAMPLES)
        result = compute_pairwise_plv(signals, FS)
        for band in ["theta", "alpha", "beta"]:
            assert len(result[f"plv_{band}"]) == 6

    def test_has_summary_keys(self):
        signals = np.random.randn(N_CHANNELS, N_SAMPLES)
        result = compute_pairwise_plv(signals, FS)
        assert "plv_frontal_alpha" in result
        assert "plv_frontal_theta" in result
        assert "plv_frontal_beta" in result
        assert "plv_fronto_temporal_alpha" in result
        assert "plv_mean_alpha" in result
        assert "plv_mean_theta" in result
        assert "plv_mean_beta" in result

    def test_has_feature_vector(self):
        signals = np.random.randn(N_CHANNELS, N_SAMPLES)
        result = compute_pairwise_plv(signals, FS)
        assert "feature_vector" in result
        assert "n_features" in result
        assert isinstance(result["feature_vector"], list)
        assert result["n_features"] == len(result["feature_vector"])


class TestPairwisePLVValueRanges:
    """PLV values must be in [0, 1]."""

    def test_all_plv_values_in_range(self):
        signals = np.random.randn(N_CHANNELS, N_SAMPLES)
        result = compute_pairwise_plv(signals, FS)
        for band in ["theta", "alpha", "beta"]:
            for v in result[f"plv_{band}"]:
                assert 0.0 <= v <= 1.0, f"PLV out of range: {v}"

    def test_summary_values_in_range(self):
        signals = np.random.randn(N_CHANNELS, N_SAMPLES)
        result = compute_pairwise_plv(signals, FS)
        for key in ["plv_frontal_alpha", "plv_frontal_theta", "plv_frontal_beta",
                     "plv_fronto_temporal_alpha", "plv_mean_alpha",
                     "plv_mean_theta", "plv_mean_beta"]:
            assert 0.0 <= result[key] <= 1.0, f"{key} out of range: {result[key]}"


class TestPairwisePLVSynchronousSignals:
    """Identical/synchronous signals should produce high PLV."""

    def test_identical_channels_high_plv(self):
        """4 channels of the same 10 Hz sine -> PLV near 1."""
        t = np.arange(N_SAMPLES) / FS
        sine_10hz = np.sin(2 * np.pi * 10 * t)
        signals = np.tile(sine_10hz, (N_CHANNELS, 1))
        # Add tiny noise to avoid degenerate phase
        signals += np.random.randn(N_CHANNELS, N_SAMPLES) * 0.001
        result = compute_pairwise_plv(signals, FS)
        assert result["plv_mean_alpha"] > 0.90

    def test_independent_random_low_plv(self):
        """Independent random noise -> PLV should be low."""
        signals = np.random.randn(N_CHANNELS, N_SAMPLES)
        result = compute_pairwise_plv(signals, FS)
        # Random signals: PLV should be below 0.5 on average
        assert result["plv_mean_alpha"] < 0.5


class TestPairwisePLVFrontalPair:
    """The frontal pair (AF7-AF8) is pair index 3 in our ordering:
    pairs: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    AF7=1, AF8=2 -> index 3 in the list.
    """

    def test_frontal_alpha_equals_pair3(self):
        signals = np.random.randn(N_CHANNELS, N_SAMPLES)
        result = compute_pairwise_plv(signals, FS)
        # plv_frontal_alpha should be the PLV of pair (1,2) in alpha band
        assert result["plv_frontal_alpha"] == result["plv_alpha"][3]


class TestPairwisePLVEdgeCases:
    """Edge cases: short signals, too few channels."""

    def test_single_channel_returns_defaults(self):
        signal = np.random.randn(1, N_SAMPLES)
        result = compute_pairwise_plv(signal, FS)
        assert result["plv_mean_alpha"] == 0.0
        assert result["n_features"] > 0
        assert all(v == 0.0 for v in result["feature_vector"])

    def test_two_channels_returns_1_pair(self):
        signals = np.random.randn(2, N_SAMPLES)
        result = compute_pairwise_plv(signals, FS)
        # Only 1 pair for 2 channels
        for band in ["theta", "alpha", "beta"]:
            assert len(result[f"plv_{band}"]) == 1

    def test_short_signal_no_crash(self):
        """Very short signal should not crash."""
        signals = np.random.randn(N_CHANNELS, 32)  # only 32 samples
        result = compute_pairwise_plv(signals, FS)
        assert isinstance(result, dict)
        assert "feature_vector" in result

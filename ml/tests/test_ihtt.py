"""Tests for Inter-Hemispheric Transfer Time (IHTT) computation.

IHTT measures the time delay for neural information to travel between
homologous electrode pairs (AF7<>AF8, TP9<>TP10) via the corpus callosum.
Typical values: 5-25 ms. Faster IHTT correlates with better cognitive
integration, emotional regulation, and focus.

Method: cross-correlation peak lag between bandpass-filtered (1-40 Hz)
homologous channels. The lag at maximum correlation = IHTT estimate.
"""

import numpy as np
import pytest

import sys
sys.path.insert(0, "/Users/sravyalu/NeuralDreamWorkshop/ml")

from processing.eeg_processor import compute_ihtt

FS = 256
DURATION_SEC = 4
N_SAMPLES = FS * DURATION_SEC


class TestComputeIHTT:
    """Unit tests for compute_ihtt()."""

    def test_returns_dict_with_expected_keys(self):
        """Should return frontal_lag_ms, temporal_lag_ms, and mean_ihtt_ms."""
        signals = np.random.randn(4, N_SAMPLES) * 10
        result = compute_ihtt(signals, FS)
        assert "frontal_lag_ms" in result
        assert "temporal_lag_ms" in result
        assert "mean_ihtt_ms" in result

    def test_identical_channels_give_zero_lag(self):
        """If left and right channels are identical, lag should be ~0 ms."""
        t = np.arange(N_SAMPLES) / FS
        base_signal = np.sin(2 * np.pi * 10 * t) * 10
        signals = np.tile(base_signal, (4, 1))
        # Add tiny noise to avoid degenerate zero-variance
        signals += np.random.randn(4, N_SAMPLES) * 0.01
        result = compute_ihtt(signals, FS)
        # Lag should be effectively zero (within 1 sample = ~3.9 ms)
        assert abs(result["frontal_lag_ms"]) < 5.0
        assert abs(result["temporal_lag_ms"]) < 5.0

    def test_known_lag_detected(self):
        """A known 10-sample shift (~39 ms at 256 Hz) should be detected."""
        t = np.arange(N_SAMPLES) / FS
        base = np.sin(2 * np.pi * 10 * t) * 10
        lag_samples = 10  # ~39.06 ms
        signals = np.zeros((4, N_SAMPLES))
        # AF7 = ch1, AF8 = ch2 (frontal pair)
        signals[1] = base
        signals[2] = np.roll(base, lag_samples)
        # TP9 = ch0, TP10 = ch3 (temporal pair)
        signals[0] = base
        signals[3] = np.roll(base, lag_samples)
        # Add slight noise
        signals += np.random.randn(4, N_SAMPLES) * 0.5

        result = compute_ihtt(signals, FS)
        expected_ms = lag_samples / FS * 1000  # ~39 ms
        # Allow +/- 1 sample tolerance = ~3.9 ms
        assert abs(result["frontal_lag_ms"] - expected_ms) < 5.0
        assert abs(result["temporal_lag_ms"] - expected_ms) < 5.0

    def test_lag_values_are_non_negative(self):
        """IHTT lag should always be non-negative (absolute lag)."""
        signals = np.random.randn(4, N_SAMPLES) * 10
        result = compute_ihtt(signals, FS)
        assert result["frontal_lag_ms"] >= 0.0
        assert result["temporal_lag_ms"] >= 0.0
        assert result["mean_ihtt_ms"] >= 0.0

    def test_mean_is_average_of_frontal_and_temporal(self):
        """mean_ihtt_ms should be the average of frontal and temporal lags."""
        signals = np.random.randn(4, N_SAMPLES) * 10
        result = compute_ihtt(signals, FS)
        expected_mean = (result["frontal_lag_ms"] + result["temporal_lag_ms"]) / 2.0
        assert abs(result["mean_ihtt_ms"] - expected_mean) < 0.01

    def test_max_lag_capped(self):
        """IHTT should be capped at physiologically plausible range (max ~50 ms)."""
        signals = np.random.randn(4, N_SAMPLES) * 10
        result = compute_ihtt(signals, FS)
        # Physiological IHTT is 5-25 ms; cap at 50 ms
        assert result["frontal_lag_ms"] <= 50.0
        assert result["temporal_lag_ms"] <= 50.0

    def test_too_few_channels_returns_zeros(self):
        """With <4 channels, should return zeros (no homologous pairs)."""
        signals = np.random.randn(2, N_SAMPLES) * 10
        result = compute_ihtt(signals, FS)
        assert result["frontal_lag_ms"] == 0.0
        assert result["temporal_lag_ms"] == 0.0
        assert result["mean_ihtt_ms"] == 0.0

    def test_short_signal_returns_zeros(self):
        """Very short signals should return zeros gracefully."""
        signals = np.random.randn(4, 10) * 10
        result = compute_ihtt(signals, FS)
        assert result["frontal_lag_ms"] == 0.0
        assert result["temporal_lag_ms"] == 0.0

    def test_1d_input_returns_zeros(self):
        """1D input (single channel) should return zeros."""
        signal = np.random.randn(N_SAMPLES) * 10
        result = compute_ihtt(signal, FS)
        assert result["mean_ihtt_ms"] == 0.0

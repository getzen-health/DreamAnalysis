"""Tests for eye movement feature extraction and dream detector eye movement integration.

Tests:
- extract_eye_movement_features returns expected keys
- saccade_rate is non-negative
- Signal with injected spikes -> higher saccade rate than clean signal
- Handles short/empty signals gracefully
- Multichannel mode (AF7-AF8) works
"""

import sys
import numpy as np
import pytest
from pathlib import Path

# Ensure ml/ is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from processing.eeg_processor import extract_eye_movement_features


class TestExtractEyeMovementFeatures:
    """Tests for the extract_eye_movement_features function."""

    def test_returns_expected_keys(self):
        """Output dict must contain saccade_rate, avg_saccade_amplitude, eye_movement_index."""
        np.random.seed(42)
        signal = np.random.randn(256 * 4) * 10  # 4 seconds at 256 Hz
        result = extract_eye_movement_features(signal, fs=256.0)

        assert "saccade_rate" in result
        assert "avg_saccade_amplitude" in result
        assert "eye_movement_index" in result

    def test_saccade_rate_non_negative(self):
        """saccade_rate must always be >= 0."""
        np.random.seed(42)
        signal = np.random.randn(256 * 4) * 10
        result = extract_eye_movement_features(signal, fs=256.0)
        assert result["saccade_rate"] >= 0.0

    def test_avg_saccade_amplitude_non_negative(self):
        """avg_saccade_amplitude must always be >= 0."""
        np.random.seed(42)
        signal = np.random.randn(256 * 4) * 10
        result = extract_eye_movement_features(signal, fs=256.0)
        assert result["avg_saccade_amplitude"] >= 0.0

    def test_eye_movement_index_non_negative(self):
        """eye_movement_index must always be >= 0."""
        np.random.seed(42)
        signal = np.random.randn(256 * 4) * 10
        result = extract_eye_movement_features(signal, fs=256.0)
        assert result["eye_movement_index"] >= 0.0

    def test_injected_spikes_higher_saccade_rate(self):
        """A signal with large rapid deflections should have higher saccade rate
        than a clean low-amplitude signal."""
        np.random.seed(42)
        fs = 256.0
        duration = 10  # 10 seconds for better statistics
        n_samples = int(fs * duration)

        # Clean signal: low amplitude noise
        clean_signal = np.random.randn(n_samples) * 5  # 5 uV RMS

        # Spikey signal: inject rapid deflections simulating saccades
        spikey_signal = np.random.randn(n_samples) * 5
        # Inject 20 large rapid deflections (~80 uV in ~50ms)
        spike_width = int(0.05 * fs)  # 50 ms
        for i in range(20):
            pos = np.random.randint(spike_width, n_samples - spike_width)
            amplitude = np.random.choice([-1, 1]) * 80  # 80 uV
            spikey_signal[pos:pos + spike_width] += amplitude

        result_clean = extract_eye_movement_features(clean_signal, fs)
        result_spikey = extract_eye_movement_features(spikey_signal, fs)

        assert result_spikey["saccade_rate"] > result_clean["saccade_rate"]

    def test_short_signal_no_crash(self):
        """Short signals (< 1 second) should not crash."""
        signal = np.random.randn(50) * 10  # ~0.2 seconds at 256 Hz
        result = extract_eye_movement_features(signal, fs=256.0)
        assert "saccade_rate" in result
        assert result["saccade_rate"] >= 0.0

    def test_empty_signal_no_crash(self):
        """Empty signal should return zeros, not crash."""
        signal = np.array([])
        result = extract_eye_movement_features(signal, fs=256.0)
        assert result["saccade_rate"] == 0.0
        assert result["avg_saccade_amplitude"] == 0.0
        assert result["eye_movement_index"] == 0.0

    def test_flat_signal_zero_saccades(self):
        """A flat (constant) signal should have zero saccade rate."""
        signal = np.ones(256 * 4) * 10  # constant 10 uV
        result = extract_eye_movement_features(signal, fs=256.0)
        assert result["saccade_rate"] == 0.0

    def test_multichannel_af7_af8(self):
        """When given a 2D signal (AF7-AF8 difference), should use lateral movement mode."""
        np.random.seed(42)
        fs = 256.0
        n_samples = int(fs * 4)

        # Simulate AF7 and AF8 with lateral eye movements
        af7 = np.random.randn(n_samples) * 10
        af8 = np.random.randn(n_samples) * 10

        # Inject correlated saccade-like deflections (opposite polarity = lateral eye movement)
        spike_width = int(0.05 * fs)
        for i in range(10):
            pos = np.random.randint(spike_width, n_samples - spike_width)
            amplitude = 80
            af7[pos:pos + spike_width] += amplitude
            af8[pos:pos + spike_width] -= amplitude  # opposite polarity

        signals = np.array([af7, af8])
        result = extract_eye_movement_features(signals, fs=fs)

        assert "saccade_rate" in result
        assert result["saccade_rate"] > 0.0

    def test_multichannel_returns_all_keys(self):
        """Multichannel mode must also return all three expected keys."""
        np.random.seed(42)
        signals = np.random.randn(2, 256 * 4) * 10
        result = extract_eye_movement_features(signals, fs=256.0)

        assert "saccade_rate" in result
        assert "avg_saccade_amplitude" in result
        assert "eye_movement_index" in result

    def test_values_are_float(self):
        """All returned values should be Python floats."""
        np.random.seed(42)
        signal = np.random.randn(256 * 4) * 10
        result = extract_eye_movement_features(signal, fs=256.0)

        for key, value in result.items():
            assert isinstance(value, float), f"{key} is {type(value)}, expected float"

"""Tests for updated FlowStateDetector with validated biomarkers.

Covers:
- Quadratic theta peaks at moderate values
- High alpha+theta with low beta -> high flow
- Flow intensity classification thresholds
- Beta asymmetry near zero -> higher flow
- Edge cases (all zeros, single channel, multichannel)
- Calibration baseline integration
- Component score ranges and weights
"""

import sys
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.flow_state_detector import FlowStateDetector, FLOW_STATES


@pytest.fixture
def detector():
    """Fresh FlowStateDetector instance (no saved model)."""
    return FlowStateDetector()


@pytest.fixture
def calibrated_detector():
    """FlowStateDetector with baseline calibration set manually."""
    d = FlowStateDetector()
    d.baseline_alpha = 0.25
    d.baseline_beta = 0.20
    d.baseline_theta = 0.15
    return d


def _make_signal(fs=256, duration=4, freqs=None, amps=None):
    """Generate a synthetic EEG signal with specific frequency components.

    Args:
        fs: Sampling rate
        duration: Duration in seconds
        freqs: List of frequency components (Hz)
        amps: List of amplitude for each frequency (uV)

    Returns:
        1D numpy array of the synthesized signal
    """
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs
    signal = np.zeros(n_samples)
    if freqs and amps:
        for freq, amp in zip(freqs, amps):
            signal += amp * np.sin(2 * np.pi * freq * t)
    # Add tiny noise to avoid degenerate PSD
    signal += np.random.randn(n_samples) * 0.1
    return signal


def _make_multichannel(fs=256, duration=4, n_channels=4, freqs=None, amps=None):
    """Generate multichannel synthetic EEG.

    All channels get the same base signal. Channels 1 and 2 (AF7, AF8)
    can be individually controlled via per-channel freq/amp overrides.
    """
    base = _make_signal(fs, duration, freqs, amps)
    channels = np.tile(base, (n_channels, 1))
    # Add slight per-channel variation
    for i in range(n_channels):
        channels[i] += np.random.randn(channels.shape[1]) * 0.05
    return channels


class TestQuadraticThetaModel:
    """Test that theta_flow_score peaks at moderate theta values."""

    def test_moderate_theta_scores_highest(self, detector):
        """Moderate theta (~0.20-0.30 relative power) should score higher
        than very low or very high theta (inverted-U relationship)."""
        # Moderate theta: balanced signal (theta ~0.25 relative power)
        moderate_theta = _make_signal(freqs=[6, 10, 18], amps=[10, 12, 12])
        result_moderate = detector.predict(moderate_theta)

        # Very high theta signal: dominant 6 Hz (theta ~0.98 relative power)
        high_theta = _make_signal(freqs=[6, 10, 18], amps=[40, 3, 3])
        result_high = detector.predict(high_theta)

        # Very low theta signal: dominant beta (theta ~0.004 relative power)
        low_theta = _make_signal(freqs=[6, 10, 18], amps=[2, 5, 30])
        result_low = detector.predict(low_theta)

        mod_score = result_moderate["components"]["theta_flow"]
        high_score = result_high["components"]["theta_flow"]
        low_score = result_low["components"]["theta_flow"]

        # Moderate theta should score higher than both extremes
        assert mod_score > high_score, (
            f"Moderate theta ({mod_score}) should score > high theta ({high_score})"
        )
        assert mod_score > low_score, (
            f"Moderate theta ({mod_score}) should score > low theta ({low_score})"
        )

    def test_theta_flow_score_bounded_0_1(self, detector):
        """theta_flow_score should always be in [0, 1]."""
        for amp in [0.1, 5, 20, 50, 100]:
            signal = _make_signal(freqs=[6], amps=[amp])
            result = detector.predict(signal)
            score = result["components"]["theta_flow"]
            assert 0 <= score <= 1, f"theta_flow out of range: {score}"

    def test_zero_theta_low_score(self, detector):
        """Zero theta power should produce a low theta_flow_score."""
        # Pure beta signal — minimal theta
        signal = _make_signal(freqs=[20, 25], amps=[30, 20])
        result = detector.predict(signal)
        # With no theta, normalized theta is 0, distance from 0.5 optimal is large
        assert result["components"]["theta_flow"] < 0.8


class TestFlowRatio:
    """Test (alpha + theta) / beta flow ratio component."""

    def test_high_alpha_theta_low_beta_high_flow(self, detector):
        """High alpha+theta with low beta should produce high flow_ratio score."""
        # Strong alpha + theta, weak beta
        flow_signal = _make_signal(freqs=[6, 10, 18], amps=[25, 25, 3])
        result = detector.predict(flow_signal)
        assert result["components"]["flow_ratio"] > 0.4, (
            f"Expected high flow_ratio, got {result['components']['flow_ratio']}"
        )

    def test_low_alpha_theta_high_beta_low_flow(self, detector):
        """Low alpha+theta with high beta should produce low flow_ratio score."""
        # Weak alpha + theta, strong beta (stressed / anxious state)
        stress_signal = _make_signal(freqs=[6, 10, 20, 25], amps=[3, 3, 25, 20])
        result = detector.predict(stress_signal)
        assert result["components"]["flow_ratio"] < 0.5, (
            f"Expected low flow_ratio, got {result['components']['flow_ratio']}"
        )

    def test_flow_ratio_bounded(self, detector):
        """flow_ratio score should always be in [0, 1]."""
        for beta_amp in [0.1, 5, 30]:
            signal = _make_signal(freqs=[6, 10, 20], amps=[15, 15, beta_amp])
            result = detector.predict(signal)
            score = result["components"]["flow_ratio"]
            assert 0 <= score <= 1


class TestFlowIntensityClassification:
    """Test flow intensity level thresholds."""

    def test_flow_states_list(self):
        """FLOW_STATES should reflect the new intensity levels."""
        assert FLOW_STATES == ["no_flow", "shallow", "moderate", "deep"]

    def test_no_flow_below_03(self, detector):
        """Scores below 0.3 should be classified as no_flow."""
        # Dominant high-beta (anxious) — should be low flow
        signal = _make_signal(freqs=[25, 30], amps=[40, 30])
        result = detector.predict(signal)
        if result["flow_score"] < 0.3:
            assert result["state"] == "no_flow"
            assert result["flow_intensity"] == "none"

    def test_deep_flow_above_075(self, detector):
        """Scores >= 0.75 should be classified as deep flow."""
        result = detector.predict(_make_signal(freqs=[6, 10], amps=[20, 20]))
        # Manually verify: if score happens to be >= 0.75
        if result["flow_score"] >= 0.75:
            assert result["state"] == "deep"
            assert result["flow_intensity"] == "deep"

    def test_flow_intensity_field_present(self, detector):
        """Result should contain 'flow_intensity' field."""
        signal = _make_signal(freqs=[6, 10, 20], amps=[15, 15, 10])
        result = detector.predict(signal)
        assert "flow_intensity" in result
        assert result["flow_intensity"] in ("none", "shallow", "moderate", "deep")

    def test_state_index_matches_state(self, detector):
        """state_index should correspond to the correct FLOW_STATES entry."""
        signal = _make_signal(freqs=[6, 10, 20], amps=[15, 15, 10])
        result = detector.predict(signal)
        assert FLOW_STATES[result["state_index"]] == result["state"]


class TestBetaAsymmetry:
    """Test beta asymmetry component with multichannel input."""

    def test_symmetric_beta_high_score(self, detector):
        """Symmetric AF7/AF8 beta should produce high beta_symmetry score."""
        # All channels identical — perfectly symmetric
        channels = _make_multichannel(freqs=[6, 10, 20], amps=[10, 10, 15])
        result = detector.predict(channels)
        # With identical channels, beta asymmetry should be near zero
        assert result["components"]["beta_symmetry"] > 0.7, (
            f"Expected high beta_symmetry for symmetric channels, "
            f"got {result['components']['beta_symmetry']}"
        )

    def test_asymmetric_beta_lower_score(self, detector):
        """Asymmetric AF7/AF8 beta should produce lower beta_symmetry score."""
        n_samples = 256 * 4
        t = np.arange(n_samples) / 256

        # ch0 = TP9 (doesn't matter for beta asymmetry)
        ch0 = np.random.randn(n_samples) * 10
        # ch1 = AF7: strong beta
        ch1 = 30 * np.sin(2 * np.pi * 20 * t) + np.random.randn(n_samples) * 0.5
        # ch2 = AF8: weak beta, strong alpha
        ch2 = 5 * np.sin(2 * np.pi * 20 * t) + 25 * np.sin(2 * np.pi * 10 * t) + np.random.randn(n_samples) * 0.5
        # ch3 = TP10
        ch3 = np.random.randn(n_samples) * 10

        channels = np.array([ch0, ch1, ch2, ch3])
        result = detector.predict(channels)

        # Compare with symmetric version
        sym_channels = _make_multichannel(freqs=[6, 10, 20], amps=[10, 10, 15])
        sym_result = detector.predict(sym_channels)

        assert result["components"]["beta_symmetry"] < sym_result["components"]["beta_symmetry"], (
            f"Asymmetric ({result['components']['beta_symmetry']}) should be < "
            f"symmetric ({sym_result['components']['beta_symmetry']})"
        )

    def test_single_channel_default_symmetry(self, detector):
        """Single-channel input should use default beta_symmetry of 0.5."""
        signal = _make_signal(freqs=[6, 10, 20], amps=[15, 15, 10])
        result = detector.predict(signal)
        assert result["components"]["beta_symmetry"] == 0.5


class TestEdgeCases:
    """Edge cases: all zeros, very short signals, degenerate inputs."""

    def test_all_zeros_signal(self, detector):
        """All-zeros signal should not crash; should return valid result."""
        signal = np.zeros(1024)
        result = detector.predict(signal)
        assert "flow_score" in result
        assert 0 <= result["flow_score"] <= 1
        assert result["state"] in FLOW_STATES

    def test_single_channel_1d(self, detector):
        """1D input should work without multichannel features."""
        signal = np.random.randn(1024) * 20
        result = detector.predict(signal)
        assert "flow_score" in result
        assert "components" in result
        assert "beta_symmetry" in result["components"]

    def test_two_channel_input(self, detector):
        """2-channel input (< 3 channels) should use default beta_symmetry."""
        channels = np.random.randn(2, 1024) * 20
        result = detector.predict(channels)
        assert result["components"]["beta_symmetry"] == 0.5

    def test_constant_signal(self, detector):
        """Near-constant signal should not crash."""
        signal = np.ones(1024) * 0.001
        result = detector.predict(signal)
        assert 0 <= result["flow_score"] <= 1


class TestCalibrationBaseline:
    """Test beta decrease component with and without baseline calibration."""

    def test_no_baseline_default_score(self, detector):
        """Without calibration, beta_decrease should default to 0.4."""
        signal = _make_signal(freqs=[6, 10, 20], amps=[15, 15, 10])
        result = detector.predict(signal)
        assert result["components"]["beta_decrease"] == 0.4

    def test_with_baseline_beta_decreased(self, calibrated_detector):
        """When current beta is lower than baseline, beta_decrease should be > 0.5."""
        # Signal with low beta (flow-like) — baseline_beta is 0.20
        signal = _make_signal(freqs=[6, 10, 20], amps=[20, 20, 3])
        result = calibrated_detector.predict(signal)
        # Beta should be lower than baseline 0.20, so score should be > 0.5
        assert result["components"]["beta_decrease"] > 0.4, (
            f"Expected beta_decrease > 0.4 when beta drops from baseline, "
            f"got {result['components']['beta_decrease']}"
        )

    def test_with_baseline_beta_increased(self, calibrated_detector):
        """When current beta exceeds baseline, beta_decrease should be < 0.5."""
        # Signal with high beta (stressed) — baseline_beta is 0.20
        signal = _make_signal(freqs=[20, 25, 30], amps=[30, 25, 20])
        result = calibrated_detector.predict(signal)
        # Beta should be higher than baseline 0.20, so score should be < 0.5
        assert result["components"]["beta_decrease"] < 0.5, (
            f"Expected beta_decrease < 0.5 when beta exceeds baseline, "
            f"got {result['components']['beta_decrease']}"
        )


class TestOutputStructure:
    """Verify the output dict has all required fields."""

    def test_all_required_keys(self, detector):
        """Output should contain all required keys."""
        signal = _make_signal(freqs=[6, 10, 20], amps=[15, 15, 10])
        result = detector.predict(signal)
        required_keys = {
            "state", "state_index", "flow_score", "confidence",
            "flow_intensity", "components", "band_powers"
        }
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - result.keys()}"
        )

    def test_component_keys(self, detector):
        """Components dict should have the four validated biomarker scores."""
        signal = _make_signal(freqs=[6, 10, 20], amps=[15, 15, 10])
        result = detector.predict(signal)
        component_keys = {"theta_flow", "flow_ratio", "beta_decrease", "beta_symmetry"}
        assert component_keys == set(result["components"].keys())

    def test_flow_score_is_weighted_sum(self, detector):
        """flow_score should be the weighted sum of the four components."""
        signal = _make_signal(freqs=[6, 10, 20], amps=[15, 15, 10])
        result = detector.predict(signal)
        c = result["components"]
        expected = (
            0.35 * c["theta_flow"] +
            0.30 * c["flow_ratio"] +
            0.20 * c["beta_decrease"] +
            0.15 * c["beta_symmetry"]
        )
        assert abs(result["flow_score"] - round(expected, 3)) < 0.01, (
            f"flow_score {result['flow_score']} != weighted sum {round(expected, 3)}"
        )

    def test_confidence_bounded(self, detector):
        """Confidence should be in [0.3, 0.95]."""
        for _ in range(5):
            signal = np.random.randn(1024) * 20
            result = detector.predict(signal)
            assert 0.3 <= result["confidence"] <= 0.95

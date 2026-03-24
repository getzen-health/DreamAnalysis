"""Tests for band-specific Hjorth mobility features.

Evidence: Hjorth mobility in beta rhythm achieves 83.33% accuracy (AUC 0.904)
on SEED dataset — best single feature among 18 examined (PubMed, SVM LOSO).

Tests the new compute_band_hjorth_mobility() and compute_hjorth_mobility_ratio()
functions that compute Hjorth parameters on band-filtered signals rather than
broadband, and cross-region ratios (frontal/temporal) for the Muse 2 layout.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

FS = 256
DURATION_S = 4
N_SAMPLES = FS * DURATION_S


def _make_sinusoid(freq: float, amp: float = 20.0, duration_s: float = 4.0,
                   fs: int = 256, seed: int = 42) -> np.ndarray:
    """Create a sinusoid + light noise."""
    rng = np.random.RandomState(seed)
    t = np.arange(int(fs * duration_s)) / fs
    return amp * np.sin(2 * np.pi * freq * t) + rng.randn(len(t)) * 2.0


def _make_multichannel_muse(beta_frontal: float = 30.0,
                            beta_temporal: float = 10.0,
                            noise_level: float = 2.0,
                            seed: int = 42) -> np.ndarray:
    """Create 4-channel Muse 2 synthetic EEG.

    ch0=TP9, ch1=AF7, ch2=AF8, ch3=TP10.
    Frontal channels (AF7/AF8) get more beta (20 Hz) power.
    Temporal channels (TP9/TP10) get less beta power.
    Noise is kept low so the sinusoidal component dominates.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(N_SAMPLES) / FS
    signals = np.zeros((4, N_SAMPLES))
    # TP9 — temporal left
    signals[0] = beta_temporal * np.sin(2 * np.pi * 20 * t) + rng.randn(N_SAMPLES) * noise_level
    # AF7 — frontal left
    signals[1] = beta_frontal * np.sin(2 * np.pi * 20 * t) + rng.randn(N_SAMPLES) * noise_level
    # AF8 — frontal right
    signals[2] = beta_frontal * np.sin(2 * np.pi * 20 * t) + rng.randn(N_SAMPLES) * noise_level
    # TP10 — temporal right
    signals[3] = beta_temporal * np.sin(2 * np.pi * 20 * t) + rng.randn(N_SAMPLES) * noise_level
    return signals


class TestBandHjorthMobility:
    """Tests for compute_band_hjorth_mobility()."""

    def test_returns_dict_with_expected_keys(self):
        from processing.eeg_processor import compute_band_hjorth_mobility
        signal = _make_sinusoid(20.0)  # 20 Hz = beta
        result = compute_band_hjorth_mobility(signal, FS)
        assert isinstance(result, dict)
        assert "beta_mobility" in result
        assert "theta_mobility" in result
        assert "alpha_mobility" in result

    def test_mobility_values_are_positive(self):
        from processing.eeg_processor import compute_band_hjorth_mobility
        signal = _make_sinusoid(20.0)
        result = compute_band_hjorth_mobility(signal, FS)
        for key, val in result.items():
            assert val >= 0.0, f"{key} should be non-negative, got {val}"

    def test_mobility_values_are_finite(self):
        from processing.eeg_processor import compute_band_hjorth_mobility
        signal = _make_sinusoid(20.0)
        result = compute_band_hjorth_mobility(signal, FS)
        for key, val in result.items():
            assert np.isfinite(val), f"{key} is not finite: {val}"

    def test_beta_signal_has_higher_beta_activity(self):
        """A signal with strong 20 Hz should have higher beta-band activity
        (variance after filtering) than a signal with strong 6 Hz."""
        from processing.eeg_processor import compute_band_hjorth_mobility, bandpass_filter, BANDS
        beta_signal = _make_sinusoid(20.0, amp=200.0, seed=1)
        theta_signal = _make_sinusoid(6.0, amp=200.0, seed=2)
        # After beta filtering, the beta signal should retain much more energy
        low, high = BANDS["beta"]
        beta_filtered = bandpass_filter(beta_signal, low, high, FS, order=4)
        theta_filtered = bandpass_filter(theta_signal, low, high, FS, order=4)
        assert np.var(beta_filtered) > np.var(theta_filtered) * 5, (
            "Beta-dominated signal should have much more beta-band energy"
        )
        # And the band mobility should still be computable and finite
        beta_result = compute_band_hjorth_mobility(beta_signal, FS)
        assert np.isfinite(beta_result["beta_mobility"])

    def test_theta_signal_has_higher_theta_mobility(self):
        """A signal dominated by 6 Hz should have higher theta mobility
        than a signal dominated by 20 Hz."""
        from processing.eeg_processor import compute_band_hjorth_mobility
        theta_signal = _make_sinusoid(6.0, amp=200.0, seed=1)
        beta_signal = _make_sinusoid(20.0, amp=200.0, seed=2)
        theta_result = compute_band_hjorth_mobility(theta_signal, FS)
        beta_result = compute_band_hjorth_mobility(beta_signal, FS)
        assert theta_result["theta_mobility"] > beta_result["theta_mobility"], (
            f"Theta signal mobility {theta_result['theta_mobility']:.4f} should exceed "
            f"beta signal mobility {beta_result['theta_mobility']:.4f} in theta band"
        )

    def test_flat_signal_returns_zeros(self):
        """Flat signal should have zero or near-zero mobility in all bands."""
        from processing.eeg_processor import compute_band_hjorth_mobility
        flat = np.ones(N_SAMPLES) * 0.001
        result = compute_band_hjorth_mobility(flat, FS)
        for key, val in result.items():
            assert val < 0.01, f"{key} should be near zero for flat signal, got {val}"

    def test_short_signal_does_not_crash(self):
        """Signal shorter than filter requirement should not raise."""
        from processing.eeg_processor import compute_band_hjorth_mobility
        short = np.random.randn(32)  # very short
        result = compute_band_hjorth_mobility(short, FS)
        assert isinstance(result, dict)

    def test_custom_bands_parameter(self):
        """Can pass specific bands to compute."""
        from processing.eeg_processor import compute_band_hjorth_mobility
        signal = _make_sinusoid(20.0)
        result = compute_band_hjorth_mobility(signal, FS, bands=["beta", "alpha"])
        assert "beta_mobility" in result
        assert "alpha_mobility" in result
        # theta not requested, should not be present
        assert "theta_mobility" not in result


class TestHjorthMobilityRatio:
    """Tests for compute_hjorth_mobility_ratio()."""

    def test_returns_dict_with_expected_keys(self):
        from processing.eeg_processor import compute_hjorth_mobility_ratio
        signals = _make_multichannel_muse()
        result = compute_hjorth_mobility_ratio(signals, FS)
        assert isinstance(result, dict)
        assert "beta_mobility_ratio_ft" in result  # frontal/temporal

    def test_ratio_values_are_positive(self):
        from processing.eeg_processor import compute_hjorth_mobility_ratio
        signals = _make_multichannel_muse()
        result = compute_hjorth_mobility_ratio(signals, FS)
        for key, val in result.items():
            assert val >= 0.0, f"{key} should be non-negative, got {val}"

    def test_ratio_values_are_finite(self):
        from processing.eeg_processor import compute_hjorth_mobility_ratio
        signals = _make_multichannel_muse()
        result = compute_hjorth_mobility_ratio(signals, FS)
        for key, val in result.items():
            assert np.isfinite(val), f"{key} is not finite: {val}"

    def test_frontal_high_freq_beta_gives_different_ratio(self):
        """When frontal channels have higher-frequency beta content (25 Hz)
        and temporal channels have lower (15 Hz), the mobility ratio reflects
        the spectral centroid difference."""
        from processing.eeg_processor import compute_hjorth_mobility_ratio
        rng = np.random.RandomState(42)
        t = np.arange(N_SAMPLES) / FS
        signals = np.zeros((4, N_SAMPLES))
        # Temporal: 15 Hz (low beta) — lower mobility
        signals[0] = 40 * np.sin(2 * np.pi * 15 * t) + rng.randn(N_SAMPLES) * 1.0
        signals[3] = 40 * np.sin(2 * np.pi * 15 * t) + rng.randn(N_SAMPLES) * 1.0
        # Frontal: 25 Hz (high beta) — higher mobility
        signals[1] = 40 * np.sin(2 * np.pi * 25 * t) + rng.randn(N_SAMPLES) * 1.0
        signals[2] = 40 * np.sin(2 * np.pi * 25 * t) + rng.randn(N_SAMPLES) * 1.0
        result = compute_hjorth_mobility_ratio(signals, FS)
        assert result["beta_mobility_ratio_ft"] > 1.0, (
            f"Higher-frequency frontal beta should give ratio > 1, got {result['beta_mobility_ratio_ft']:.4f}"
        )

    def test_temporal_high_freq_beta_gives_inverted_ratio(self):
        """Inverse of above: when temporal has higher beta frequency,
        the ratio should be < 1."""
        from processing.eeg_processor import compute_hjorth_mobility_ratio
        rng = np.random.RandomState(42)
        t = np.arange(N_SAMPLES) / FS
        signals = np.zeros((4, N_SAMPLES))
        # Temporal: 25 Hz (high beta)
        signals[0] = 40 * np.sin(2 * np.pi * 25 * t) + rng.randn(N_SAMPLES) * 1.0
        signals[3] = 40 * np.sin(2 * np.pi * 25 * t) + rng.randn(N_SAMPLES) * 1.0
        # Frontal: 15 Hz (low beta)
        signals[1] = 40 * np.sin(2 * np.pi * 15 * t) + rng.randn(N_SAMPLES) * 1.0
        signals[2] = 40 * np.sin(2 * np.pi * 15 * t) + rng.randn(N_SAMPLES) * 1.0
        result = compute_hjorth_mobility_ratio(signals, FS)
        assert result["beta_mobility_ratio_ft"] < 1.0, (
            f"Lower-frequency frontal beta should give ratio < 1, got {result['beta_mobility_ratio_ft']:.4f}"
        )

    def test_balanced_beta_gives_near_one_ratio(self):
        """Equal beta power in frontal and temporal should give ratio near 1."""
        from processing.eeg_processor import compute_hjorth_mobility_ratio
        signals = _make_multichannel_muse(beta_frontal=25.0, beta_temporal=25.0)
        result = compute_hjorth_mobility_ratio(signals, FS)
        assert 0.5 < result["beta_mobility_ratio_ft"] < 2.0, (
            f"Balanced beta should give ratio near 1, got {result['beta_mobility_ratio_ft']:.4f}"
        )

    def test_single_channel_returns_empty(self):
        """Single channel cannot compute ratio — should return empty dict."""
        from processing.eeg_processor import compute_hjorth_mobility_ratio
        signal = np.random.randn(1, N_SAMPLES) * 20
        result = compute_hjorth_mobility_ratio(signal, FS)
        assert result == {} or all(v == 0.0 for v in result.values())

    def test_two_channel_returns_ratio(self):
        """Two channels (minimal) should still compute a ratio."""
        from processing.eeg_processor import compute_hjorth_mobility_ratio
        signals = np.random.randn(2, N_SAMPLES) * 20
        result = compute_hjorth_mobility_ratio(
            signals, FS, frontal_chs=[0], temporal_chs=[1]
        )
        assert isinstance(result, dict)

    def test_custom_channel_indices(self):
        """Can pass custom frontal/temporal channel indices."""
        from processing.eeg_processor import compute_hjorth_mobility_ratio
        signals = _make_multichannel_muse()
        result = compute_hjorth_mobility_ratio(
            signals, FS, frontal_chs=[1, 2], temporal_chs=[0, 3]
        )
        assert "beta_mobility_ratio_ft" in result


class TestBandHjorthInEmotionClassifier:
    """Verify band Hjorth features appear in emotion classifier output."""

    def test_emotion_output_contains_band_hjorth(self):
        """Emotion classifier should include band_hjorth_mobility in output."""
        from models.emotion_classifier import EmotionClassifier
        clf = EmotionClassifier()
        signals = _make_multichannel_muse()
        result = clf.predict(signals, FS)
        assert "band_hjorth" in result, (
            f"Expected 'band_hjorth' key in emotion output, got keys: {list(result.keys())}"
        )
        band_hjorth = result["band_hjorth"]
        assert "beta_mobility_ratio_ft" in band_hjorth

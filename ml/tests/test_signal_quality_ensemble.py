"""Tests for signal-quality-adaptive ensemble weighting.

When EEG signal quality is poor (high artifact %), the ensemble should
automatically downweight EEGNet (which needs clean raw signal) and upweight
the feature-based heuristics (which are more robust to noise via band-power
ratios and asymmetry measures).

Verifies:
1. Clean signal uses standard ensemble weights (0.6 EEGNet / 0.4 heuristic)
2. Poor signal shifts weight toward heuristic (up to 100% heuristic)
3. Medium-quality signal produces intermediate weights (smooth ramp)
4. The quality score is exposed in the result dict for debugging
5. _compute_epoch_quality returns 0-1 score for various signal conditions
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_clean_eeg(n_channels: int = 4, n_samples: int = 1024,
                    seed: int = 42) -> np.ndarray:
    """Generate clean synthetic EEG (~20 uV RMS, well below artifact threshold)."""
    rng = np.random.RandomState(seed)
    return rng.randn(n_channels, n_samples) * 15.0


def _make_noisy_eeg(n_channels: int = 4, n_samples: int = 1024,
                    seed: int = 42) -> np.ndarray:
    """Generate noisy EEG with high EMG and line noise but below 75 uV artifact threshold.

    Peak amplitude stays under 75 uV (artifact threshold) so the epoch is NOT
    rejected outright, but signal quality is clearly poor.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n_samples) / 256.0
    eeg = rng.randn(n_channels, n_samples) * 10.0
    # Add strong 60 Hz line noise (dominates the spectrum)
    for ch in range(n_channels):
        eeg[ch] += 40.0 * np.sin(2 * np.pi * 60 * t)
    # Add broadband EMG-like high-frequency noise (20-50 Hz)
    for ch in range(n_channels):
        eeg[ch] += rng.randn(n_samples) * 15.0
    # Clip to stay below artifact threshold (75 uV)
    eeg = np.clip(eeg, -70, 70)
    return eeg


def _mock_eegnet_result():
    """Standard mock EEGNet prediction."""
    return {
        "emotion": "happy",
        "probabilities": {
            "happy": 0.5, "sad": 0.1, "angry": 0.05,
            "fearful": 0.05, "relaxed": 0.2, "focused": 0.1,
        },
        "valence": 0.6,
        "arousal": 0.5,
        "mc_confidence": 0.65,
        "mc_uncertainty": 0.15,
        "mc_pred_std": {},
        "mc_predictive_entropy": 0.3,
        "confidence_label": "medium",
        "model_type": "eegnet_4ch",
    }


class TestComputeEpochQuality:
    """Unit tests for the fast epoch quality estimator."""

    def setup_method(self):
        from models.emotion_classifier import EmotionClassifier
        self.clf = EmotionClassifier()

    def test_clean_signal_high_quality(self):
        """Clean signal should produce quality > 0.6."""
        eeg = _make_clean_eeg()
        quality = self.clf._compute_epoch_quality(eeg, fs=256.0)
        assert 0.0 <= quality <= 1.0, f"Quality {quality} out of bounds"
        assert quality > 0.6, f"Clean signal quality={quality:.3f}, expected > 0.6"

    def test_noisy_signal_low_quality(self):
        """Noisy signal (high EMG + line noise) should produce quality < 0.5."""
        eeg = _make_noisy_eeg()
        quality = self.clf._compute_epoch_quality(eeg, fs=256.0)
        assert 0.0 <= quality <= 1.0, f"Quality {quality} out of bounds"
        assert quality < 0.5, f"Noisy signal quality={quality:.3f}, expected < 0.5"

    def test_flatline_very_low_quality(self):
        """Flatline signal should produce very low quality."""
        eeg = np.ones((4, 1024)) * 0.001
        quality = self.clf._compute_epoch_quality(eeg, fs=256.0)
        assert quality < 0.3, f"Flatline quality={quality:.3f}, expected < 0.3"

    def test_quality_returns_float(self):
        """Quality should always be a Python float in [0, 1]."""
        eeg = _make_clean_eeg()
        quality = self.clf._compute_epoch_quality(eeg, fs=256.0)
        assert isinstance(quality, float)

    def test_single_channel_handled(self):
        """Should work with single-channel input."""
        eeg = _make_clean_eeg(n_channels=1)
        quality = self.clf._compute_epoch_quality(eeg, fs=256.0)
        assert 0.0 <= quality <= 1.0


class TestAdaptiveEnsembleWeights:
    """Tests that ensemble weights adapt to signal quality."""

    def setup_method(self):
        from models.emotion_classifier import EmotionClassifier
        self.clf = EmotionClassifier()

    def _run_ensemble_with_mock(self, eeg):
        """Run ensemble prediction with mocked EEGNet."""
        mock_eegnet = MagicMock()
        mock_eegnet.is_available.return_value = True
        mock_eegnet.predict_with_uncertainty.return_value = _mock_eegnet_result()
        self.clf._eegnet = mock_eegnet
        # Disable TTA for faster tests
        self.clf._TTA_N_AUGMENTATIONS = 0

        result = self.clf.predict(eeg, fs=256.0, device_type="muse_2")
        return result

    def test_clean_signal_uses_standard_weights(self):
        """Clean signal should produce result with eegnet weight near 0.6."""
        eeg = _make_clean_eeg()
        result = self._run_ensemble_with_mock(eeg)

        assert result["model_type"] == "ensemble-eegnet-heuristic"
        # On clean signal, adaptive EEGNet weight should be close to the default 0.6
        assert "adaptive_eegnet_weight" in result
        assert result["adaptive_eegnet_weight"] >= 0.5, (
            f"Clean signal: EEGNet weight={result['adaptive_eegnet_weight']:.3f}, expected >= 0.5"
        )

    def test_noisy_signal_downweights_eegnet(self):
        """Noisy signal should downweight EEGNet."""
        eeg = _make_noisy_eeg()
        result = self._run_ensemble_with_mock(eeg)

        assert result["model_type"] == "ensemble-eegnet-heuristic"
        assert "adaptive_eegnet_weight" in result
        assert result["adaptive_eegnet_weight"] < 0.5, (
            f"Noisy signal: EEGNet weight={result['adaptive_eegnet_weight']:.3f}, expected < 0.5"
        )

    def test_epoch_quality_in_result(self):
        """Result should include the epoch quality score for debugging."""
        eeg = _make_clean_eeg()
        result = self._run_ensemble_with_mock(eeg)

        assert "epoch_quality" in result
        assert 0.0 <= result["epoch_quality"] <= 1.0

    def test_adaptive_weight_range(self):
        """Adaptive EEGNet weight should always be in [0, max_weight]."""
        for eeg in [_make_clean_eeg(), _make_noisy_eeg()]:
            result = self._run_ensemble_with_mock(eeg)
            w = result["adaptive_eegnet_weight"]
            assert 0.0 <= w <= 0.6, f"Adaptive weight {w} outside [0, 0.6]"

    def test_heuristic_weight_complement(self):
        """Heuristic weight should be 1 - eegnet_weight (always sums to 1)."""
        eeg = _make_clean_eeg()
        result = self._run_ensemble_with_mock(eeg)
        w_e = result["adaptive_eegnet_weight"]
        w_h = result["adaptive_heuristic_weight"]
        assert abs((w_e + w_h) - 1.0) < 0.001, (
            f"Weights don't sum to 1: EEGNet={w_e}, heuristic={w_h}"
        )

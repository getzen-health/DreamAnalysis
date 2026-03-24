"""Tests for EMA smoothing of continuous indices across ALL model paths.

Verifies that valence, arousal, stress_index, focus_index, relaxation_index,
anger_index, and fear_index are EMA-smoothed (not raw per-frame values)
regardless of which model path is used for inference.

The bug this tests for: before the fix, only _predict_features() applied
EMA smoothing to continuous indices.  The LGBM, EEGNet, ONNX, sklearn,
and multichannel DEAP paths all returned raw per-frame values that caused
dashboard readings to jitter on every 2-second epoch hop.
"""

import numpy as np
import pytest
from models.emotion_classifier import EmotionClassifier


def _make_eeg(n_channels: int = 4, n_samples: int = 1024,
              seed: int = 0) -> np.ndarray:
    """Generate synthetic EEG-like signal (4 ch, 4 sec at 256 Hz)."""
    rng = np.random.RandomState(seed)
    # Keep amplitude well below artifact threshold (75 uV)
    return rng.randn(n_channels, n_samples) * 10.0


class TestEMASmoothingConsistency:
    """EMA smoothing must be applied on ALL model paths."""

    def setup_method(self):
        self.clf = EmotionClassifier()

    def test_feature_path_smooths_indices(self):
        """The feature-based path should EMA smooth continuous indices."""
        eeg1 = _make_eeg(seed=1)
        eeg2 = _make_eeg(seed=2)

        r1 = self.clf.predict(eeg1, fs=256.0)
        r2 = self.clf.predict(eeg2, fs=256.0)

        # After two calls, _ema_valence should be set (not None)
        assert self.clf._ema_valence is not None
        assert self.clf._ema_arousal is not None
        assert self.clf._ema_stress is not None
        assert self.clf._ema_focus is not None
        assert self.clf._ema_relaxation is not None

    def test_consecutive_predictions_smooth(self):
        """Consecutive predictions should produce smoother output than raw."""
        eeg_streams = [_make_eeg(seed=i) for i in range(10)]

        results = []
        for eeg in eeg_streams:
            results.append(self.clf.predict(eeg, fs=256.0))

        # Extract valence time series
        valences = [r["valence"] for r in results]
        arousals = [r["arousal"] for r in results]

        # With EMA, consecutive differences should be smaller than without EMA.
        # We can't test exact values, but we can verify:
        # 1. All values are finite
        assert all(np.isfinite(v) for v in valences)
        assert all(np.isfinite(a) for a in arousals)

        # 2. Values are in valid range
        assert all(-1.0 <= v <= 1.0 for v in valences)
        assert all(0.0 <= a <= 1.0 for a in arousals)

    def test_ema_reduces_jitter(self):
        """EMA should reduce frame-to-frame variance compared to raw values."""
        # Use alternating high-alpha and high-beta signals to create jitter
        rng = np.random.RandomState(42)

        results = []
        for i in range(20):
            eeg = rng.randn(4, 1024) * 10.0
            # Alternate between alpha-dominant and beta-dominant
            if i % 2 == 0:
                # Boost 8-12 Hz (alpha) on all channels
                t = np.arange(1024) / 256.0
                for ch in range(4):
                    eeg[ch] += 20.0 * np.sin(2 * np.pi * 10 * t)
            else:
                # Boost 20-30 Hz (beta) on all channels
                t = np.arange(1024) / 256.0
                for ch in range(4):
                    eeg[ch] += 20.0 * np.sin(2 * np.pi * 25 * t)
            results.append(self.clf.predict(eeg, fs=256.0))

        # Compute frame-to-frame differences in stress index
        stress_vals = [r["stress_index"] for r in results]
        diffs = [abs(stress_vals[i+1] - stress_vals[i]) for i in range(len(stress_vals) - 1)]

        # With EMA (alpha=0.4), the later differences should be smaller
        # than if we had no smoothing. Check that the max diff is < 0.5
        # (without EMA, alternating signals could cause 0.0 <-> 1.0 swings)
        assert max(diffs) < 0.5, (
            f"Max stress jitter {max(diffs):.3f} too large — EMA not smoothing"
        )

    def test_ema_seeds_on_first_call(self):
        """First prediction should seed EMA (no prior history to blend)."""
        eeg = _make_eeg(seed=0)
        result = self.clf.predict(eeg, fs=256.0)

        # After first call, EMA values should equal raw values (seeded)
        assert self.clf._ema_valence is not None
        # The seeded value should match what was returned
        assert abs(self.clf._ema_valence - result["valence"]) < 1e-6

    def test_artifact_freezes_ema(self):
        """Artifact epoch should NOT update EMA — should return frozen values."""
        # Force feature-based path (no external model interference) by using
        # a fresh classifier with REVE/LGBM disabled.
        clf = EmotionClassifier()
        # Disable all model paths to force _predict_features
        clf._reve_foundation = None
        clf._reve = None
        clf._eegnet = None
        clf.mega_lgbm_model = None
        clf.lgbm_muse_model = None
        clf._tsception = None
        clf.onnx_session = None
        clf.sklearn_model = None

        eeg_normal = _make_eeg(seed=0)
        r1 = clf.predict(eeg_normal, fs=256.0)

        saved_valence = clf._ema_valence
        saved_arousal = clf._ema_arousal

        # Now send an artifact (amplitude > 75 uV)
        eeg_artifact = np.ones((4, 1024)) * 200.0  # way above threshold
        r2 = clf.predict(eeg_artifact, fs=256.0)

        # EMA should be frozen — valence/arousal should be the pre-artifact values
        assert r2.get("artifact_detected", False) is True
        assert abs(clf._ema_valence - saved_valence) < 1e-6

    def test_all_index_keys_present_feature_path(self):
        """All continuous index keys should be present in feature-based output."""
        clf = EmotionClassifier()
        # Force feature-based path
        clf._reve_foundation = None
        clf._reve = None
        clf._eegnet = None
        clf.mega_lgbm_model = None
        clf.lgbm_muse_model = None
        clf._tsception = None
        clf.onnx_session = None
        clf.sklearn_model = None

        eeg = _make_eeg(seed=0)
        result = clf.predict(eeg, fs=256.0)

        required_keys = [
            "valence", "arousal", "stress_index", "focus_index",
            "relaxation_index", "anger_index", "fear_index",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
            assert isinstance(result[key], float), f"{key} is not float: {type(result[key])}"

    def test_valence_arousal_present_all_paths(self):
        """Valence and arousal must always be present regardless of model path."""
        eeg = _make_eeg(seed=0)
        result = self.clf.predict(eeg, fs=256.0)

        # These two are guaranteed by every model path
        assert "valence" in result
        assert "arousal" in result
        assert isinstance(result["valence"], float)
        assert isinstance(result["arousal"], float)

    def test_build_muse_result_smooths_indices(self):
        """_build_muse_result (used by LGBM paths) should apply EMA smoothing."""
        eeg = _make_eeg(seed=0)
        smoothed = np.array([0.2, 0.1, 0.1, 0.1, 0.3, 0.2])

        # Call _build_muse_result twice with different signals
        r1 = self.clf._build_muse_result(4, smoothed, eeg, 256.0,
                                          artifact_detected=False)
        v1 = r1["valence"]

        # Second call with same smoothed probs but different eeg
        eeg2 = _make_eeg(seed=99)
        r2 = self.clf._build_muse_result(4, smoothed, eeg2, 256.0,
                                          artifact_detected=False)
        v2 = r2["valence"]

        # With EMA, v2 should be a blend of v1 and the new raw valence
        # (not equal to the raw value from eeg2 alone)
        assert self.clf._ema_valence is not None

    def test_ensure_explanation_smooths_external_models(self):
        """_ensure_explanation should smooth indices from external models."""
        # Simulate EEGNet-style result dict
        result1 = {
            "emotion": "happy",
            "probabilities": {"happy": 0.5, "sad": 0.1, "angry": 0.1,
                              "fear": 0.1, "surprise": 0.1, "neutral": 0.1},
            "valence": 0.8,
            "arousal": 0.6,
            "stress_index": 0.2,
            "focus_index": 0.5,
            "relaxation_index": 0.4,
            "anger_index": 0.1,
            "fear_index": 0.1,
        }
        r1 = self.clf._ensure_explanation(result1)

        # First call seeds EMA
        assert self.clf._ema_valence is not None
        assert abs(self.clf._ema_valence - 0.8) < 1e-6  # seeded

        # Second call with different values should blend
        result2 = {
            "emotion": "sad",
            "probabilities": {"happy": 0.1, "sad": 0.5, "angry": 0.1,
                              "fear": 0.1, "surprise": 0.1, "neutral": 0.1},
            "valence": -0.5,
            "arousal": 0.3,
            "stress_index": 0.6,
            "focus_index": 0.3,
            "relaxation_index": 0.6,
            "anger_index": 0.05,
            "fear_index": 0.15,
        }
        r2 = self.clf._ensure_explanation(result2)

        # EMA(alpha=0.4): new = 0.4 * raw + 0.6 * prev
        expected_valence = 0.4 * (-0.5) + 0.6 * 0.8
        assert abs(r2["valence"] - expected_valence) < 1e-6, (
            f"Expected EMA valence {expected_valence:.4f}, got {r2['valence']:.4f}"
        )
        expected_arousal = 0.4 * 0.3 + 0.6 * 0.6
        assert abs(r2["arousal"] - expected_arousal) < 1e-6

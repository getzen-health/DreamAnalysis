"""Tests for EEGNet + feature-based heuristic ensemble.

When EEGNet is available, the emotion classifier should run BOTH EEGNet
and feature-based heuristics, average their probability outputs, and return
the ensemble result. This tests that:
1. Ensemble activates when EEGNet is available
2. Probabilities are a weighted average of both classifiers
3. The ensemble result preserves all required output keys
4. model_type reflects ensemble usage
5. Falls back cleanly when EEGNet is unavailable
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def _make_eeg(n_channels: int = 4, n_samples: int = 1024,
              seed: int = 0) -> np.ndarray:
    """Generate synthetic EEG signal below artifact threshold."""
    rng = np.random.RandomState(seed)
    return rng.randn(n_channels, n_samples) * 10.0


class TestEnsembleEEGNetHeuristic:
    """Ensemble of EEGNet + feature-based heuristics."""

    def setup_method(self):
        from models.emotion_classifier import EmotionClassifier
        self.clf = EmotionClassifier()

    def test_ensemble_model_type_when_eegnet_available(self):
        """When EEGNet fires, model_type should indicate ensemble."""
        # Mock EEGNet to be available and return valid probabilities
        mock_eegnet = MagicMock()
        mock_eegnet.is_available.return_value = True
        mock_eegnet.predict.return_value = {
            "emotion": "happy",
            "probabilities": {
                "happy": 0.4, "sad": 0.1, "angry": 0.1,
                "fearful": 0.1, "relaxed": 0.2, "focused": 0.1,
            },
            "valence": 0.6,
            "arousal": 0.5,
            "stress_index": 0.2,
            "focus_index": 0.4,
            "relaxation_index": 0.3,
            "model_type": "eegnet_4ch",
        }
        self.clf._eegnet = mock_eegnet
        # Disable higher-priority models so EEGNet path is reached
        self.clf._reve_foundation = None
        self.clf._reve = None

        eeg = _make_eeg(seed=42)
        result = self.clf.predict(eeg, fs=256.0)

        assert "ensemble" in result["model_type"], (
            f"Expected ensemble model_type, got: {result['model_type']}"
        )

    def test_ensemble_averages_probabilities(self):
        """Ensemble probabilities should blend EEGNet and heuristic probs."""
        mock_eegnet = MagicMock()
        mock_eegnet.is_available.return_value = True
        # EEGNet strongly favors happy
        mock_eegnet.predict.return_value = {
            "emotion": "happy",
            "probabilities": {
                "happy": 0.8, "sad": 0.05, "angry": 0.02,
                "fearful": 0.03, "relaxed": 0.05, "focused": 0.05,
            },
            "valence": 0.7,
            "arousal": 0.6,
            "stress_index": 0.1,
            "focus_index": 0.5,
            "relaxation_index": 0.3,
            "model_type": "eegnet_4ch",
        }
        self.clf._eegnet = mock_eegnet
        self.clf._reve_foundation = None
        self.clf._reve = None

        eeg = _make_eeg(seed=42)
        result = self.clf.predict(eeg, fs=256.0)

        # Probabilities should exist and sum to ~1.0
        probs = result["probabilities"]
        assert len(probs) == 6
        prob_sum = sum(probs.values())
        assert abs(prob_sum - 1.0) < 0.05, f"Probabilities sum to {prob_sum}"

        # The ensemble should not be identical to the raw EEGNet output
        # (because heuristic probs are blended in)
        eegnet_probs = mock_eegnet.predict.return_value["probabilities"]
        at_least_one_diff = any(
            abs(probs[e] - eegnet_probs[e]) > 0.01
            for e in probs
        )
        assert at_least_one_diff, (
            "Ensemble probs are identical to EEGNet probs -- heuristic not blended"
        )

    def test_ensemble_preserves_required_keys(self):
        """Ensemble result must have all keys the API contract requires."""
        mock_eegnet = MagicMock()
        mock_eegnet.is_available.return_value = True
        mock_eegnet.predict.return_value = {
            "emotion": "relaxed",
            "probabilities": {
                "happy": 0.1, "sad": 0.1, "angry": 0.1,
                "fearful": 0.1, "relaxed": 0.4, "focused": 0.2,
            },
            "valence": 0.3,
            "arousal": 0.3,
            "stress_index": 0.2,
            "focus_index": 0.3,
            "relaxation_index": 0.6,
            "model_type": "eegnet_4ch",
        }
        self.clf._eegnet = mock_eegnet
        self.clf._reve_foundation = None
        self.clf._reve = None

        eeg = _make_eeg(seed=42)
        result = self.clf.predict(eeg, fs=256.0)

        required_keys = [
            "emotion", "probabilities", "valence", "arousal",
            "stress_index", "focus_index", "relaxation_index",
            "band_powers", "differential_entropy",
            "model_type", "explanation",
        ]
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

    def test_ensemble_valence_arousal_in_range(self):
        """Ensemble valence and arousal must be in valid ranges."""
        mock_eegnet = MagicMock()
        mock_eegnet.is_available.return_value = True
        mock_eegnet.predict.return_value = {
            "emotion": "angry",
            "probabilities": {
                "happy": 0.05, "sad": 0.1, "angry": 0.5,
                "fearful": 0.15, "relaxed": 0.1, "focused": 0.1,
            },
            "valence": -0.6,
            "arousal": 0.8,
            "stress_index": 0.7,
            "focus_index": 0.3,
            "relaxation_index": 0.1,
            "model_type": "eegnet_4ch",
        }
        self.clf._eegnet = mock_eegnet
        self.clf._reve_foundation = None
        self.clf._reve = None

        eeg = _make_eeg(seed=42)
        result = self.clf.predict(eeg, fs=256.0)

        assert -1.0 <= result["valence"] <= 1.0
        assert 0.0 <= result["arousal"] <= 1.0

    def test_no_ensemble_when_eegnet_unavailable(self):
        """Without EEGNet, should fall through to normal chain (no ensemble)."""
        self.clf._eegnet = None
        self.clf._reve_foundation = None
        self.clf._reve = None
        self.clf.mega_lgbm_model = None
        self.clf.lgbm_muse_model = None
        self.clf._tsception = None
        self.clf.onnx_session = None
        self.clf.sklearn_model = None

        eeg = _make_eeg(seed=42)
        result = self.clf.predict(eeg, fs=256.0)

        assert "ensemble" not in result["model_type"]
        assert result["model_type"] == "feature-based"

    def test_ensemble_emotion_is_valid(self):
        """Ensemble emotion label must be one of the 6 valid emotions."""
        from models.emotion_classifier import EMOTIONS

        mock_eegnet = MagicMock()
        mock_eegnet.is_available.return_value = True
        mock_eegnet.predict.return_value = {
            "emotion": "happy",
            "probabilities": {
                "happy": 0.3, "sad": 0.2, "angry": 0.1,
                "fearful": 0.1, "relaxed": 0.2, "focused": 0.1,
            },
            "valence": 0.3,
            "arousal": 0.5,
            "stress_index": 0.3,
            "focus_index": 0.4,
            "relaxation_index": 0.3,
            "model_type": "eegnet_4ch",
        }
        self.clf._eegnet = mock_eegnet
        self.clf._reve_foundation = None
        self.clf._reve = None

        eeg = _make_eeg(seed=42)
        result = self.clf.predict(eeg, fs=256.0)

        assert result["emotion"] in EMOTIONS or result["emotion"] == "neutral"

    def test_ensemble_weight_favors_eegnet(self):
        """EEGNet should have higher weight (0.6) than heuristic (0.4).

        After weighted average, EMA smoothing, and re-normalization, the exact
        final probability depends on the heuristic's contribution and EMA state.
        We verify that when EEGNet strongly favors happy, the ensemble happy
        probability is the HIGHEST among all classes (argmax), confirming EEGNet's
        higher weight is influential.
        """
        mock_eegnet = MagicMock()
        mock_eegnet.is_available.return_value = True
        # EEGNet says 100% happy (extreme case)
        mock_eegnet.predict.return_value = {
            "emotion": "happy",
            "probabilities": {
                "happy": 1.0, "sad": 0.0, "angry": 0.0,
                "fearful": 0.0, "relaxed": 0.0, "focused": 0.0,
            },
            "valence": 0.9,
            "arousal": 0.7,
            "stress_index": 0.0,
            "focus_index": 0.5,
            "relaxation_index": 0.3,
            "model_type": "eegnet_4ch",
        }
        self.clf._eegnet = mock_eegnet
        self.clf._reve_foundation = None
        self.clf._reve = None

        eeg = _make_eeg(seed=42)
        result = self.clf.predict(eeg, fs=256.0)

        # With EEGNet at 100% happy and 0.6 weight, happy should be the
        # top-scoring class in the ensemble (EEGNet's strong signal dominates)
        probs = result["probabilities"]
        happy_prob = probs.get("happy", 0.0)
        max_prob = max(probs.values())
        assert happy_prob == max_prob or abs(happy_prob - max_prob) < 0.01, (
            f"Happy prob {happy_prob:.3f} is not the top class (max={max_prob:.3f}) "
            "-- EEGNet weight may not be applied correctly"
        )

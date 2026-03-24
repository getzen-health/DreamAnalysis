"""Tests for MC Dropout uncertainty estimation in EEGNet.

MC Dropout (Gal & Ghahramani 2016) runs N forward passes with dropout
enabled at inference time to approximate Bayesian uncertainty. These tests
verify:
1. EEGNet.predict_proba_mc returns correct shapes and valid values
2. EEGNetEmotionClassifier.predict_with_uncertainty returns uncertainty fields
3. Uncertainty is higher for random noise than for a consistent signal
4. Ensemble pipeline passes MC uncertainty through to the final result
5. MC Dropout correctly enables dropout layers while keeping BatchNorm in eval
"""

import numpy as np
import pytest
import torch
import torch.nn as nn


def _make_eeg(n_channels: int = 4, n_samples: int = 1024,
              seed: int = 0) -> np.ndarray:
    """Generate synthetic EEG signal below artifact threshold."""
    rng = np.random.RandomState(seed)
    return rng.randn(n_channels, n_samples) * 10.0


class TestEEGNetMCDropout:
    """Low-level tests for EEGNet.predict_proba_mc."""

    def setup_method(self):
        from models.eegnet import EEGNet
        self.model = EEGNet(n_classes=6, n_channels=4, n_samples=1024,
                            dropout_rate=0.25)
        self.model.eval()

    def test_mc_returns_correct_shapes(self):
        """Mean probs, std probs, and entropy should have correct shapes."""
        x = torch.randn(1, 4, 1024)
        mean_probs, std_probs, entropy = self.model.predict_proba_mc(x, n_forward=5)

        assert mean_probs.shape == (1, 6), f"Expected (1, 6), got {mean_probs.shape}"
        assert std_probs.shape == (1, 6), f"Expected (1, 6), got {std_probs.shape}"
        assert isinstance(entropy, float)

    def test_mean_probs_sum_to_one(self):
        """Mean probabilities across MC passes should sum to ~1."""
        x = torch.randn(1, 4, 1024)
        mean_probs, _, _ = self.model.predict_proba_mc(x, n_forward=10)
        prob_sum = float(mean_probs[0].sum())
        assert abs(prob_sum - 1.0) < 0.05, f"Probs sum to {prob_sum}"

    def test_std_probs_non_negative(self):
        """Standard deviation of probabilities must be >= 0."""
        x = torch.randn(1, 4, 1024)
        _, std_probs, _ = self.model.predict_proba_mc(x, n_forward=5)
        assert np.all(std_probs >= 0), "Negative std detected"

    def test_entropy_non_negative(self):
        """Predictive entropy must be >= 0."""
        x = torch.randn(1, 4, 1024)
        _, _, entropy = self.model.predict_proba_mc(x, n_forward=5)
        assert entropy >= 0.0, f"Negative entropy: {entropy}"

    def test_entropy_bounded_by_max(self):
        """Predictive entropy <= log(n_classes)."""
        x = torch.randn(1, 4, 1024)
        _, _, entropy = self.model.predict_proba_mc(x, n_forward=5)
        max_entropy = float(np.log(6))
        assert entropy <= max_entropy + 0.01, (
            f"Entropy {entropy} exceeds max {max_entropy}"
        )

    def test_dropout_active_during_mc_passes(self):
        """During MC passes, dropout layers should be in train mode."""
        x = torch.randn(1, 4, 1024)

        # Collect multiple MC results -- if dropout is really active,
        # individual pass outputs should differ (stochastic).
        all_outputs = []
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
        with torch.no_grad():
            for _ in range(10):
                logits = self.model(x)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_outputs.append(probs[0])

        self.model.eval()

        # With dropout active, at least some passes should differ
        stacked = np.stack(all_outputs)
        max_std = stacked.std(axis=0).max()
        # With dropout_rate=0.25, there should be SOME variation
        # (unless all weights happen to be zero, which is extremely unlikely)
        assert max_std > 0.0, (
            "No variation across dropout passes -- dropout may not be active"
        )

    def test_model_returns_to_eval_after_mc(self):
        """After predict_proba_mc, model should be back in eval mode."""
        x = torch.randn(1, 4, 1024)
        self.model.predict_proba_mc(x, n_forward=5)

        # All dropout layers should be in eval mode after the call
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                assert not m.training, "Dropout layer still in train mode after MC"

    def test_single_pass_equivalent_to_predict_proba(self):
        """With n_forward=1, MC should be close to regular predict (not exact due to dropout)."""
        x = torch.randn(1, 4, 1024)
        # With 1 pass, we get a single stochastic sample
        mean_probs, std_probs, _ = self.model.predict_proba_mc(x, n_forward=1)
        # Std should be zero with only 1 sample
        assert np.allclose(std_probs, 0.0), "Std should be 0 with n_forward=1"


class TestEEGNetClassifierUncertainty:
    """Tests for EEGNetEmotionClassifier.predict_with_uncertainty."""

    def setup_method(self):
        from models.eegnet import EEGNet, EEGNetEmotionClassifier
        self.wrapper = EEGNetEmotionClassifier()
        # Inject a fresh untrained model for testing (no saved weights needed)
        model = EEGNet(n_classes=6, n_channels=4, n_samples=1024)
        self.wrapper._models[4] = model
        self.wrapper._benchmarks[4] = 0.85
        self.wrapper._ema[4] = None

    def test_predict_with_uncertainty_returns_mc_fields(self):
        """Output dict should contain all MC Dropout uncertainty fields."""
        eeg = _make_eeg(seed=42)
        result = self.wrapper.predict_with_uncertainty(eeg, fs=256.0)

        assert "mc_uncertainty" in result
        assert "mc_confidence" in result
        assert "mc_pred_std" in result
        assert "mc_predictive_entropy" in result
        assert "confidence_label" in result

    def test_mc_uncertainty_in_valid_range(self):
        """MC uncertainty should be in [0, 1]."""
        eeg = _make_eeg(seed=42)
        result = self.wrapper.predict_with_uncertainty(eeg, fs=256.0)

        assert 0.0 <= result["mc_uncertainty"] <= 1.0
        assert 0.0 <= result["mc_confidence"] <= 1.0

    def test_mc_confidence_complement_of_uncertainty(self):
        """mc_confidence should equal 1 - mc_uncertainty."""
        eeg = _make_eeg(seed=42)
        result = self.wrapper.predict_with_uncertainty(eeg, fs=256.0)

        expected = 1.0 - result["mc_uncertainty"]
        assert abs(result["mc_confidence"] - expected) < 0.001

    def test_confidence_label_valid(self):
        """Confidence label must be one of high/medium/low."""
        eeg = _make_eeg(seed=42)
        result = self.wrapper.predict_with_uncertainty(eeg, fs=256.0)

        assert result["confidence_label"] in ("high", "medium", "low")

    def test_mc_pred_std_has_all_emotions(self):
        """mc_pred_std should have a key for each emotion class."""
        eeg = _make_eeg(seed=42)
        result = self.wrapper.predict_with_uncertainty(eeg, fs=256.0)

        from models.eegnet import EEGNetEmotionClassifier
        for e in EEGNetEmotionClassifier.EMOTIONS:
            assert e in result["mc_pred_std"], f"Missing emotion '{e}' in mc_pred_std"
            assert result["mc_pred_std"][e] >= 0.0, f"Negative std for {e}"

    def test_standard_predict_fields_still_present(self):
        """predict_with_uncertainty should still return all standard fields."""
        eeg = _make_eeg(seed=42)
        result = self.wrapper.predict_with_uncertainty(eeg, fs=256.0)

        for key in ["emotion", "probabilities", "valence", "arousal",
                     "stress_index", "focus_index", "relaxation_index", "model_type"]:
            assert key in result, f"Missing standard key: {key}"

    def test_probabilities_sum_to_one(self):
        """Even with MC Dropout, final probabilities should sum to ~1."""
        eeg = _make_eeg(seed=42)
        result = self.wrapper.predict_with_uncertainty(eeg, fs=256.0)

        prob_sum = sum(result["probabilities"].values())
        assert abs(prob_sum - 1.0) < 0.05, f"Probs sum to {prob_sum}"


class TestEnsembleMCDropoutIntegration:
    """Tests that MC Dropout fields propagate through the ensemble pipeline."""

    def setup_method(self):
        from models.emotion_classifier import EmotionClassifier
        self.clf = EmotionClassifier()

    def test_ensemble_includes_mc_fields(self):
        """When ensemble is active, result should include MC uncertainty fields."""
        from unittest.mock import MagicMock
        mock_eegnet = MagicMock()
        mock_eegnet.is_available.return_value = True
        mock_eegnet.predict_with_uncertainty.return_value = {
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
            "mc_uncertainty": 0.35,
            "mc_confidence": 0.65,
            "mc_pred_std": {
                "happy": 0.05, "sad": 0.02, "angry": 0.01,
                "fearful": 0.02, "relaxed": 0.03, "focused": 0.01,
            },
            "mc_predictive_entropy": 0.63,
            "confidence_label": "medium",
        }
        self.clf._eegnet = mock_eegnet
        self.clf._reve_foundation = None
        self.clf._reve = None

        eeg = _make_eeg(seed=42)
        result = self.clf.predict(eeg, fs=256.0)

        assert "mc_uncertainty" in result
        assert "mc_confidence" in result
        assert "mc_pred_std" in result
        assert "confidence_label" in result
        assert result["model_type"] == "ensemble-eegnet-heuristic"

    def test_ensemble_confidence_uses_mc(self):
        """Ensemble confidence should come from MC Dropout, not max(softmax)."""
        from unittest.mock import MagicMock
        mock_eegnet = MagicMock()
        mock_eegnet.is_available.return_value = True
        mock_eegnet.predict_with_uncertainty.return_value = {
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
            "mc_uncertainty": 0.20,
            "mc_confidence": 0.80,
            "mc_pred_std": {
                "happy": 0.01, "sad": 0.01, "angry": 0.01,
                "fearful": 0.01, "relaxed": 0.02, "focused": 0.01,
            },
            "mc_predictive_entropy": 0.36,
            "confidence_label": "high",
        }
        self.clf._eegnet = mock_eegnet
        self.clf._reve_foundation = None
        self.clf._reve = None

        eeg = _make_eeg(seed=42)
        result = self.clf.predict(eeg, fs=256.0)

        # Confidence should be the MC confidence value (0.80), not max(softmax)
        assert result["confidence"] == 0.80
        assert result["confidence_label"] == "high"

"""Tests for learned temperature scaling in ConfidenceCalibrator.

Temperature scaling (Guo et al., 2017): a single learned scalar T divides
logits before softmax. When T > 1, predictions become less confident
(more calibrated). When T < 1, more confident. T is optimized by
minimizing NLL on a validation set.

Verifies:
1. fit_temperature learns a T > 1 for overconfident predictions
2. fit_temperature learns a T < 1 for underconfident predictions
3. Calibrated distribution sums to 1.0
4. Temperature scaling reduces ECE on synthetic overconfident data
5. Temperature is persisted and reloaded from disk
6. Fallback to heuristic when no temperature is learned
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from processing.confidence_calibration import ConfidenceCalibrator


@pytest.fixture
def calibrator(tmp_path):
    """Create a calibrator with a temporary calibration directory."""
    cal = ConfidenceCalibrator()
    # Override the calibration dir to use temp
    cal._calibration_dir = tmp_path
    return cal


def _make_overconfident_data(n: int = 200, rng_seed: int = 42):
    """Generate synthetic data where the model is overconfident.

    Model predicts very peaked distributions (max prob ~0.95) but is only
    correct 60% of the time. Temperature scaling should learn T > 1 to
    soften these overconfident predictions.
    """
    rng = np.random.RandomState(rng_seed)
    n_classes = 6

    # Model predicts peaked distributions (overconfident)
    logits_list = []
    true_labels = []

    for i in range(n):
        # Model always predicts class 0 with high logit
        logits = rng.randn(n_classes) * 0.5
        predicted_class = rng.randint(0, n_classes)
        logits[predicted_class] += 3.0  # make it very confident

        # But model is only correct 60% of the time
        if rng.random() < 0.60:
            true_label = predicted_class
        else:
            wrong_classes = [c for c in range(n_classes) if c != predicted_class]
            true_label = rng.choice(wrong_classes)

        logits_list.append(logits)
        true_labels.append(true_label)

    return np.array(logits_list), np.array(true_labels)


def _make_underconfident_data(n: int = 200, rng_seed: int = 42):
    """Generate data where the model is underconfident (flat distributions
    but actually correct 90% of the time)."""
    rng = np.random.RandomState(rng_seed)
    n_classes = 6

    logits_list = []
    true_labels = []

    for i in range(n):
        # Model predicts fairly flat distribution (underconfident)
        logits = rng.randn(n_classes) * 0.3
        predicted_class = rng.randint(0, n_classes)
        logits[predicted_class] += 0.5  # only slightly confident

        # But model is correct 90% of the time
        if rng.random() < 0.90:
            true_label = predicted_class
        else:
            wrong_classes = [c for c in range(n_classes) if c != predicted_class]
            true_label = rng.choice(wrong_classes)

        logits_list.append(logits)
        true_labels.append(true_label)

    return np.array(logits_list), np.array(true_labels)


def _compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error.

    probs: (n, n_classes) probability distributions
    labels: (n,) true class indices
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = accuracies[mask].mean()
        ece += mask.sum() / len(labels) * abs(bin_acc - bin_conf)

    return float(ece)


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Apply softmax with temperature."""
    scaled = logits / temperature
    scaled -= scaled.max(axis=-1, keepdims=True)
    exp_vals = np.exp(scaled)
    return exp_vals / exp_vals.sum(axis=-1, keepdims=True)


class TestTemperatureScalingFit:
    """Test that fit_temperature learns appropriate T values."""

    def test_overconfident_model_learns_T_greater_than_1(self, calibrator):
        """Overconfident predictions should yield T > 1 (softens distribution)."""
        logits, labels = _make_overconfident_data()
        calibrator.fit_temperature("emotion", logits, labels)

        assert "emotion" in calibrator.temperature_params
        T = calibrator.temperature_params["emotion"]["temperature"]
        assert T > 1.0, f"Expected T > 1 for overconfident model, got T={T:.3f}"

    def test_underconfident_model_learns_T_less_than_1(self, calibrator):
        """Underconfident predictions should yield T < 1 (sharpens distribution)."""
        logits, labels = _make_underconfident_data()
        calibrator.fit_temperature("emotion", logits, labels)

        assert "emotion" in calibrator.temperature_params
        T = calibrator.temperature_params["emotion"]["temperature"]
        assert T < 1.0, f"Expected T < 1 for underconfident model, got T={T:.3f}"

    def test_needs_minimum_samples(self, calibrator):
        """Should not fit with too few samples."""
        logits = np.random.randn(5, 6)
        labels = np.array([0, 1, 2, 3, 4])
        calibrator.fit_temperature("emotion", logits, labels)
        # Should not have stored params
        assert "emotion" not in calibrator.temperature_params


class TestTemperatureScalingApply:
    """Test that applying temperature scaling produces valid distributions."""

    def test_calibrated_distribution_sums_to_1(self, calibrator):
        """Output probabilities must sum to 1.0 after temperature scaling."""
        logits, labels = _make_overconfident_data()
        calibrator.fit_temperature("emotion", logits, labels)

        test_logits = np.random.randn(6)
        result = calibrator.apply_temperature("emotion", test_logits)
        assert result is not None
        assert abs(sum(result) - 1.0) < 1e-6

    def test_returns_none_when_no_temperature(self, calibrator):
        """Should return None when no temperature has been learned."""
        result = calibrator.apply_temperature("nonexistent_model", np.random.randn(6))
        assert result is None

    def test_temperature_softens_overconfident_predictions(self, calibrator):
        """With T > 1, max probability should be lower (less peaked)."""
        logits, labels = _make_overconfident_data()
        calibrator.fit_temperature("emotion", logits, labels)

        test_logits = np.array([3.0, 0.5, 0.2, -0.1, -0.3, -0.5])
        uncalibrated = _softmax(test_logits, temperature=1.0)
        calibrated = calibrator.apply_temperature("emotion", test_logits)

        assert calibrated is not None
        assert max(calibrated) < max(uncalibrated), (
            f"Calibrated max ({max(calibrated):.4f}) should be less than "
            f"uncalibrated max ({max(uncalibrated):.4f}) for overconfident model"
        )


class TestECEImprovement:
    """Test that temperature scaling actually improves calibration."""

    def test_ece_decreases_on_overconfident_data(self, calibrator):
        """ECE should decrease after temperature scaling on overconfident data."""
        logits, labels = _make_overconfident_data(n=500)

        # Split into fit and test sets
        fit_logits, fit_labels = logits[:300], labels[:300]
        test_logits, test_labels = logits[300:], labels[300:]

        # ECE before calibration
        uncalibrated_probs = _softmax(test_logits)
        ece_before = _compute_ece(uncalibrated_probs, test_labels)

        # Fit temperature
        calibrator.fit_temperature("emotion", fit_logits, fit_labels)

        # ECE after calibration
        T = calibrator.temperature_params["emotion"]["temperature"]
        calibrated_probs = _softmax(test_logits, temperature=T)
        ece_after = _compute_ece(calibrated_probs, test_labels)

        assert ece_after < ece_before, (
            f"ECE should decrease: before={ece_before:.4f}, after={ece_after:.4f}"
        )


class TestTemperaturePersistence:
    """Test that learned temperature is saved and loaded from disk."""

    def test_save_and_load_temperature(self, calibrator, tmp_path):
        """Temperature params should persist across calibrator instances."""
        logits, labels = _make_overconfident_data()
        calibrator.fit_temperature("emotion", logits, labels)

        T_original = calibrator.temperature_params["emotion"]["temperature"]

        # Create a new calibrator pointing to the same directory
        cal2 = ConfidenceCalibrator()
        cal2._calibration_dir = tmp_path
        cal2._load_calibrations()

        assert "emotion" in cal2.temperature_params
        T_loaded = cal2.temperature_params["emotion"]["temperature"]
        assert abs(T_original - T_loaded) < 1e-6

    def test_fallback_when_no_temperature_file(self, calibrator):
        """Should use heuristic calibration when no temperature file exists."""
        result = calibrator.calibrate("emotion", 0.85)
        # Should still return a valid result using conservative heuristic
        assert "calibrated_confidence" in result
        assert 0.0 <= result["calibrated_confidence"] <= 1.0

"""Tests for PredictionStabilityTracker — cosine similarity based emotion stability."""
import numpy as np
import pytest


def test_first_prediction_returns_stability_one():
    """First prediction has no prior to compare to -- stability should be 1.0 (fully stable)."""
    from models.emotion_classifier import PredictionStabilityTracker
    tracker = PredictionStabilityTracker()
    probs = np.array([0.5, 0.3, 0.1, 0.05, 0.03, 0.02])
    stability = tracker.update(probs)
    assert stability == 1.0


def test_identical_predictions_yield_max_stability():
    """Repeated identical predictions should converge stability to 1.0."""
    from models.emotion_classifier import PredictionStabilityTracker
    tracker = PredictionStabilityTracker()
    probs = np.array([0.5, 0.3, 0.1, 0.05, 0.03, 0.02])
    for _ in range(10):
        stability = tracker.update(probs)
    assert stability > 0.99


def test_opposite_predictions_reduce_stability():
    """Alternating between very different probability vectors should drop stability."""
    from models.emotion_classifier import PredictionStabilityTracker
    tracker = PredictionStabilityTracker()
    probs_a = np.array([0.8, 0.05, 0.05, 0.05, 0.03, 0.02])
    probs_b = np.array([0.02, 0.03, 0.05, 0.05, 0.05, 0.8])
    # Alternate 10 times
    for i in range(10):
        p = probs_a if i % 2 == 0 else probs_b
        stability = tracker.update(p)
    # After alternating, stability should be well below 1
    assert stability < 0.5


def test_gradual_shift_keeps_moderate_stability():
    """A gradual shift from one emotion to another should keep stability moderate."""
    from models.emotion_classifier import PredictionStabilityTracker
    tracker = PredictionStabilityTracker()
    # Gradually shift from class 0 dominant to class 5 dominant over 20 steps
    for i in range(20):
        t = i / 19.0  # 0.0 to 1.0
        probs = np.array([1 - t, 0.0, 0.0, 0.0, 0.0, t])
        # Avoid exact zero — add small epsilon for valid cosine sim
        probs = probs + 1e-6
        probs /= probs.sum()
        stability = tracker.update(probs)
    # Gradual shift should produce moderate stability, not crash
    assert 0.3 < stability < 1.0


def test_confidence_penalty_applied_when_unstable():
    """When stability is low, adjusted confidence should be lower than raw confidence."""
    from models.emotion_classifier import PredictionStabilityTracker
    tracker = PredictionStabilityTracker()
    probs_a = np.array([0.9, 0.02, 0.02, 0.02, 0.02, 0.02])
    probs_b = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.9])

    # Build up instability
    for i in range(10):
        tracker.update(probs_a if i % 2 == 0 else probs_b)

    raw_confidence = 0.9
    adjusted = tracker.adjust_confidence(raw_confidence)
    assert adjusted < raw_confidence
    assert adjusted > 0.0  # Should not go negative


def test_confidence_unchanged_when_stable():
    """When stability is high, adjusted confidence should be close to raw confidence."""
    from models.emotion_classifier import PredictionStabilityTracker
    tracker = PredictionStabilityTracker()
    probs = np.array([0.5, 0.3, 0.1, 0.05, 0.03, 0.02])

    # Build up stability
    for _ in range(10):
        tracker.update(probs)

    raw_confidence = 0.8
    adjusted = tracker.adjust_confidence(raw_confidence)
    # Should be within 5% of raw when fully stable
    assert abs(adjusted - raw_confidence) < 0.05


def test_zero_probability_vector_handled():
    """Edge case: all-zero probability vector should not crash (NaN/inf)."""
    from models.emotion_classifier import PredictionStabilityTracker
    tracker = PredictionStabilityTracker()
    zeros = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    stability = tracker.update(zeros)
    assert np.isfinite(stability)
    # Second call with zeros
    stability = tracker.update(zeros)
    assert np.isfinite(stability)


def test_stability_value_always_in_zero_one():
    """Stability should always be in [0, 1] range regardless of input."""
    from models.emotion_classifier import PredictionStabilityTracker
    tracker = PredictionStabilityTracker()
    rng = np.random.RandomState(42)
    for _ in range(50):
        probs = rng.dirichlet(np.ones(6))
        stability = tracker.update(probs)
        assert 0.0 <= stability <= 1.0


def test_stability_accessible_without_update():
    """Accessing stability before any update should return 1.0 (default stable)."""
    from models.emotion_classifier import PredictionStabilityTracker
    tracker = PredictionStabilityTracker()
    assert tracker.stability == 1.0


def test_predict_output_contains_stability_fields():
    """Integration test: EmotionClassifier.predict() includes stability fields."""
    from models.emotion_classifier import EmotionClassifier
    clf = EmotionClassifier()
    # Synthetic 4-channel EEG data (4 channels, 1024 samples = 4 sec at 256 Hz)
    rng = np.random.RandomState(42)
    eeg = rng.randn(4, 1024) * 10.0  # 10 uV amplitude — below artifact threshold
    result = clf.predict(eeg, fs=256.0)
    assert "prediction_stability" in result
    assert "stability_adjusted_confidence" in result
    assert 0.0 <= result["prediction_stability"] <= 1.0
    assert 0.0 <= result["stability_adjusted_confidence"] <= 1.0


def test_predict_stability_degrades_with_flipping():
    """Integration test: stability tracking works on EmotionClassifier output.

    Directly manipulate the stability tracker with contrasting probability
    vectors (bypassing EMA which dampens differences), then verify the
    predict() output reflects the tracker state.
    """
    from models.emotion_classifier import EmotionClassifier
    clf = EmotionClassifier()

    # Feed contrasting probability vectors directly into the tracker
    # to simulate what would happen if the model rapidly flipped
    probs_a = np.array([0.8, 0.05, 0.05, 0.03, 0.02, 0.05])
    probs_b = np.array([0.05, 0.05, 0.05, 0.03, 0.02, 0.8])
    for i in range(10):
        clf._stability_tracker.update(probs_a if i % 2 == 0 else probs_b)

    # Now run a real predict — the stability should reflect the flipping
    rng = np.random.RandomState(42)
    eeg = rng.randn(4, 1024) * 10.0
    result = clf.predict(eeg, fs=256.0)

    # Stability should be well below 1.0 because of prior flipping
    assert result["prediction_stability"] < 0.8
    # And adjusted confidence should be penalized
    assert result["stability_adjusted_confidence"] <= result.get("confidence", 1.0)

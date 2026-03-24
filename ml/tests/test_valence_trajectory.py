"""Tests for emotional trajectory tracking embedded in EmotionClassifier.predict().

Verifies that the classifier tracks valence over the last 5 predictions and
outputs trajectory direction ("improving" | "stable" | "declining"),
trajectory_magnitude (float), and trajectory_confidence (float) in every
prediction result.

This is the "Oura readiness trend" for mood — users see
"Your mood has been improving" without any extra API calls.
"""

import numpy as np
import pytest
from models.emotion_classifier import EmotionClassifier


def _make_eeg(n_channels: int = 4, n_samples: int = 1024,
              seed: int = 0) -> np.ndarray:
    """Generate synthetic EEG-like signal (4 ch, 4 sec at 256 Hz)."""
    rng = np.random.RandomState(seed)
    return rng.randn(n_channels, n_samples) * 10.0


def _force_feature_path(clf: EmotionClassifier) -> None:
    """Disable all external model paths to force feature-based heuristics."""
    clf._reve_foundation = None
    clf._reve = None
    clf._eegnet = None
    clf.mega_lgbm_model = None
    clf.lgbm_muse_model = None
    clf._tsception = None
    clf.onnx_session = None
    clf.sklearn_model = None


class TestTrajectoryKeysPresent:
    """Every prediction must include trajectory fields."""

    def test_trajectory_keys_in_first_prediction(self):
        clf = EmotionClassifier()
        _force_feature_path(clf)
        result = clf.predict(_make_eeg(seed=0), fs=256.0)
        assert "trajectory" in result
        assert "trajectory_magnitude" in result
        assert "trajectory_confidence" in result

    def test_trajectory_keys_after_multiple_predictions(self):
        clf = EmotionClassifier()
        _force_feature_path(clf)
        for i in range(7):
            result = clf.predict(_make_eeg(seed=i), fs=256.0)
        assert "trajectory" in result
        assert "trajectory_magnitude" in result
        assert "trajectory_confidence" in result


class TestTrajectoryValues:
    """Trajectory field values must be valid."""

    def test_trajectory_direction_is_valid_string(self):
        clf = EmotionClassifier()
        _force_feature_path(clf)
        for i in range(5):
            result = clf.predict(_make_eeg(seed=i), fs=256.0)
        assert result["trajectory"] in ("improving", "stable", "declining")

    def test_trajectory_magnitude_is_nonnegative_float(self):
        clf = EmotionClassifier()
        _force_feature_path(clf)
        for i in range(5):
            result = clf.predict(_make_eeg(seed=i), fs=256.0)
        assert isinstance(result["trajectory_magnitude"], float)
        assert result["trajectory_magnitude"] >= 0.0

    def test_trajectory_confidence_in_range(self):
        clf = EmotionClassifier()
        _force_feature_path(clf)
        for i in range(5):
            result = clf.predict(_make_eeg(seed=i), fs=256.0)
        assert isinstance(result["trajectory_confidence"], float)
        assert 0.0 <= result["trajectory_confidence"] <= 1.0


class TestTrajectoryWithFewSamples:
    """With < 3 samples, trajectory should be stable with low confidence."""

    def test_single_prediction_is_stable(self):
        clf = EmotionClassifier()
        _force_feature_path(clf)
        result = clf.predict(_make_eeg(seed=0), fs=256.0)
        assert result["trajectory"] == "stable"
        assert result["trajectory_confidence"] == 0.0

    def test_two_predictions_still_low_confidence(self):
        clf = EmotionClassifier()
        _force_feature_path(clf)
        clf.predict(_make_eeg(seed=0), fs=256.0)
        result = clf.predict(_make_eeg(seed=1), fs=256.0)
        # With only 2 samples, confidence should be very low
        assert result["trajectory_confidence"] < 0.5


class TestTrajectoryDirection:
    """Trajectory direction should reflect the slope of recent valence."""

    def test_rising_valence_is_improving(self):
        clf = EmotionClassifier()
        _force_feature_path(clf)
        # Manually inject a rising valence history
        clf._valence_history.clear()
        clf._valence_history.extend([-0.4, -0.2, 0.0, 0.2, 0.4])
        traj = clf._compute_trajectory()
        assert traj["trajectory"] == "improving"
        assert traj["trajectory_magnitude"] > 0.0

    def test_falling_valence_is_declining(self):
        clf = EmotionClassifier()
        _force_feature_path(clf)
        clf._valence_history.clear()
        clf._valence_history.extend([0.4, 0.2, 0.0, -0.2, -0.4])
        traj = clf._compute_trajectory()
        assert traj["trajectory"] == "declining"
        assert traj["trajectory_magnitude"] > 0.0

    def test_flat_valence_is_stable(self):
        clf = EmotionClassifier()
        _force_feature_path(clf)
        clf._valence_history.clear()
        clf._valence_history.extend([0.3, 0.3, 0.3, 0.3, 0.3])
        traj = clf._compute_trajectory()
        assert traj["trajectory"] == "stable"
        assert traj["trajectory_magnitude"] < 0.01

    def test_noisy_but_overall_improving(self):
        clf = EmotionClassifier()
        _force_feature_path(clf)
        # Overall upward trend but with noise
        clf._valence_history.clear()
        clf._valence_history.extend([-0.3, -0.1, -0.2, 0.1, 0.3])
        traj = clf._compute_trajectory()
        assert traj["trajectory"] == "improving"


class TestTrajectoryConfidence:
    """Confidence should reflect R^2 of the linear fit and sample count."""

    def test_perfect_linear_trend_high_confidence(self):
        clf = EmotionClassifier()
        clf._valence_history.clear()
        clf._valence_history.extend([-0.4, -0.2, 0.0, 0.2, 0.4])
        traj = clf._compute_trajectory()
        # Perfect linear trend: R^2 = 1.0, 5 samples -> high confidence
        assert traj["trajectory_confidence"] > 0.8

    def test_random_scatter_low_confidence(self):
        clf = EmotionClassifier()
        clf._valence_history.clear()
        clf._valence_history.extend([0.5, -0.5, 0.5, -0.5, 0.5])
        traj = clf._compute_trajectory()
        # Alternating values: poor linear fit -> low confidence
        assert traj["trajectory_confidence"] < 0.3

    def test_fewer_samples_lower_confidence(self):
        clf = EmotionClassifier()
        clf._valence_history.clear()
        clf._valence_history.extend([0.0, 0.2, 0.4])
        traj_3 = clf._compute_trajectory()

        clf._valence_history.clear()
        clf._valence_history.extend([-0.4, -0.2, 0.0, 0.2, 0.4])
        traj_5 = clf._compute_trajectory()

        # Same slope quality but fewer samples -> lower confidence
        assert traj_3["trajectory_confidence"] <= traj_5["trajectory_confidence"]


class TestTrajectoryHistoryManagement:
    """Valence history deque should maintain correct size."""

    def test_history_maxlen_is_5(self):
        clf = EmotionClassifier()
        assert clf._valence_history.maxlen == 5

    def test_history_grows_with_predictions(self):
        clf = EmotionClassifier()
        _force_feature_path(clf)
        for i in range(3):
            clf.predict(_make_eeg(seed=i), fs=256.0)
        assert len(clf._valence_history) == 3

    def test_history_caps_at_5(self):
        clf = EmotionClassifier()
        _force_feature_path(clf)
        for i in range(10):
            clf.predict(_make_eeg(seed=i), fs=256.0)
        assert len(clf._valence_history) == 5


class TestTrajectoryWithArtifacts:
    """Artifact epochs should not corrupt trajectory history."""

    def test_artifact_does_not_add_to_history(self):
        clf = EmotionClassifier()
        _force_feature_path(clf)
        # First normal prediction
        clf.predict(_make_eeg(seed=0), fs=256.0)
        history_len = len(clf._valence_history)

        # Send artifact
        eeg_artifact = np.ones((4, 1024)) * 200.0
        clf.predict(eeg_artifact, fs=256.0)

        # History should not have grown
        assert len(clf._valence_history) == history_len

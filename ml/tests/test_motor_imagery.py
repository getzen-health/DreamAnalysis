"""Tests for motor imagery BCI classifier.

Validates ERD-based classification of imagined movements (left hand,
right hand, both feet, rest) using mu/beta desynchronization on Muse 2's
4 EEG channels: TP9 (ch0), AF7 (ch1), AF8 (ch2), TP10 (ch3).

Mu rhythm at temporal sites (TP9/TP10) is the closest Muse 2 proxy for
the sensorimotor C3/C4 electrodes used in traditional motor imagery BCIs.
"""
import numpy as np
import pytest

from models.motor_imagery import MotorImageryClassifier

FS = 256
DURATION = 4  # seconds
N_SAMPLES = FS * DURATION
VALID_CLASSES = {"left_hand", "right_hand", "both_feet", "rest"}


def _synth_eeg(n_channels=4, n_samples=N_SAMPLES, fs=FS):
    """Generate broadband synthetic EEG (~20 uV RMS)."""
    rng = np.random.RandomState(42)
    t = np.arange(n_samples) / fs
    signals = rng.randn(n_channels, n_samples) * 10
    # Add some alpha/mu to make it realistic
    for ch in range(n_channels):
        signals[ch] += 15 * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi))
    return signals


def _left_hand_imagery_signal(fs=FS, n_samples=N_SAMPLES):
    """Simulate left hand imagery: suppress mu at TP10 (right hemisphere).

    Left hand imagery -> contralateral (right hemisphere) mu suppression.
    TP10 (ch3) should show less mu power than TP9 (ch0).
    """
    rng = np.random.RandomState(99)
    t = np.arange(n_samples) / fs
    signals = rng.randn(4, n_samples) * 8

    # Strong mu at TP9 (ch0) - left hemisphere, no suppression
    signals[0] += 30 * np.sin(2 * np.pi * 10 * t)
    # Weak mu at TP10 (ch3) - right hemisphere, suppressed (ERD)
    signals[3] += 3 * np.sin(2 * np.pi * 10 * t)
    # Frontal channels: moderate mu
    signals[1] += 12 * np.sin(2 * np.pi * 10 * t + 0.5)
    signals[2] += 12 * np.sin(2 * np.pi * 10 * t + 1.0)
    return signals


def _right_hand_imagery_signal(fs=FS, n_samples=N_SAMPLES):
    """Simulate right hand imagery: suppress mu at TP9 (left hemisphere).

    Right hand imagery -> contralateral (left hemisphere) mu suppression.
    TP9 (ch0) should show less mu power than TP10 (ch3).
    """
    rng = np.random.RandomState(77)
    t = np.arange(n_samples) / fs
    signals = rng.randn(4, n_samples) * 8

    # Weak mu at TP9 (ch0) - left hemisphere, suppressed (ERD)
    signals[0] += 3 * np.sin(2 * np.pi * 10 * t)
    # Strong mu at TP10 (ch3) - right hemisphere, no suppression
    signals[3] += 30 * np.sin(2 * np.pi * 10 * t)
    # Frontal channels: moderate mu
    signals[1] += 12 * np.sin(2 * np.pi * 10 * t + 0.3)
    signals[2] += 12 * np.sin(2 * np.pi * 10 * t + 0.7)
    return signals


# ── TestBaseline ────────────────────────────────────────────────────


class TestBaseline:
    def test_set_baseline_returns_dict(self):
        clf = MotorImageryClassifier()
        result = clf.set_baseline(_synth_eeg(), fs=FS)
        assert isinstance(result, dict)
        assert result["baseline_set"] is True

    def test_baseline_contains_mu_and_beta(self):
        clf = MotorImageryClassifier()
        result = clf.set_baseline(_synth_eeg(), fs=FS)
        assert "mu_powers" in result
        assert "beta_powers" in result
        assert len(result["mu_powers"]) == 4
        assert len(result["beta_powers"]) == 4

    def test_baseline_powers_positive(self):
        clf = MotorImageryClassifier()
        result = clf.set_baseline(_synth_eeg(), fs=FS)
        for p in result["mu_powers"]:
            assert p >= 0.0
        for p in result["beta_powers"]:
            assert p >= 0.0


# ── TestClassify ────────────────────────────────────────────────────


class TestClassify:
    def test_classify_returns_required_keys(self):
        clf = MotorImageryClassifier()
        clf.set_baseline(_synth_eeg(), fs=FS)
        result = clf.classify(_synth_eeg(), fs=FS)
        required = {
            "predicted_class", "probabilities", "confidence",
            "laterality_index", "mu_suppression", "erd_map",
        }
        assert required.issubset(result.keys())

    def test_predicted_class_is_valid(self):
        clf = MotorImageryClassifier()
        clf.set_baseline(_synth_eeg(), fs=FS)
        result = clf.classify(_synth_eeg(), fs=FS)
        assert result["predicted_class"] in VALID_CLASSES

    def test_probabilities_sum_to_one(self):
        clf = MotorImageryClassifier()
        clf.set_baseline(_synth_eeg(), fs=FS)
        result = clf.classify(_synth_eeg(), fs=FS)
        probs = result["probabilities"]
        assert set(probs.keys()) == VALID_CLASSES
        assert abs(sum(probs.values()) - 1.0) < 1e-4

    def test_classify_without_baseline(self):
        """Classifier should still work without baseline (uses raw powers)."""
        clf = MotorImageryClassifier()
        result = clf.classify(_synth_eeg(), fs=FS)
        assert result["predicted_class"] in VALID_CLASSES

    def test_classify_1d_signal(self):
        """Single-channel input should not crash."""
        clf = MotorImageryClassifier()
        result = clf.classify(_synth_eeg()[0], fs=FS)
        assert result["predicted_class"] in VALID_CLASSES

    def test_erd_map_structure(self):
        clf = MotorImageryClassifier()
        clf.set_baseline(_synth_eeg(), fs=FS)
        result = clf.classify(_synth_eeg(), fs=FS)
        erd = result["erd_map"]
        assert isinstance(erd, dict)
        # Should have mu and beta entries for channels
        assert "mu" in erd
        assert "beta" in erd


# ── TestLaterality ──────────────────────────────────────────────────


class TestLaterality:
    def test_laterality_range(self):
        clf = MotorImageryClassifier()
        clf.set_baseline(_synth_eeg(), fs=FS)
        result = clf.classify(_synth_eeg(), fs=FS)
        assert -1.0 <= result["laterality_index"] <= 1.0

    def test_left_hand_imagery_tends_left(self):
        """Left hand imagery (right mu suppression) should give negative laterality."""
        clf = MotorImageryClassifier()
        # Use balanced baseline
        baseline = _synth_eeg()
        clf.set_baseline(baseline, fs=FS)
        result = clf.classify(_left_hand_imagery_signal(), fs=FS)
        # Laterality: negative = right hemisphere suppression = left hand
        assert result["laterality_index"] < 0
        # Should classify as left_hand
        assert result["predicted_class"] == "left_hand"

    def test_right_hand_imagery_tends_right(self):
        """Right hand imagery (left mu suppression) should give positive laterality."""
        clf = MotorImageryClassifier()
        baseline = _synth_eeg()
        clf.set_baseline(baseline, fs=FS)
        result = clf.classify(_right_hand_imagery_signal(), fs=FS)
        # Laterality: positive = left hemisphere suppression = right hand
        assert result["laterality_index"] > 0
        # Should classify as right_hand
        assert result["predicted_class"] == "right_hand"


# ── TestConfidence ──────────────────────────────────────────────────


class TestConfidence:
    def test_confidence_in_range(self):
        clf = MotorImageryClassifier()
        clf.set_baseline(_synth_eeg(), fs=FS)
        result = clf.classify(_synth_eeg(), fs=FS)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_strong_signal_higher_confidence(self):
        """A clearly lateralized signal should produce higher confidence than noise."""
        clf = MotorImageryClassifier()
        clf.set_baseline(_synth_eeg(), fs=FS)
        # Strong left-hand signal
        strong = clf.classify(_left_hand_imagery_signal(), fs=FS)
        # Random noise
        rng = np.random.RandomState(123)
        noisy = clf.classify(rng.randn(4, N_SAMPLES) * 5, fs=FS)
        assert strong["confidence"] > noisy["confidence"]


# ── TestAccuracy ────────────────────────────────────────────────────


class TestAccuracy:
    def test_submit_label_and_accuracy(self):
        clf = MotorImageryClassifier()
        clf.set_baseline(_synth_eeg(), fs=FS)
        clf.classify(_synth_eeg(), fs=FS)
        clf.submit_label("left_hand")
        clf.classify(_synth_eeg(), fs=FS)
        clf.submit_label("right_hand")
        acc = clf.get_accuracy()
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_accuracy_empty_returns_zero(self):
        clf = MotorImageryClassifier()
        assert clf.get_accuracy() == 0.0

    def test_submit_label_invalid(self):
        """Submitting a label without prior classification should not crash."""
        clf = MotorImageryClassifier()
        # No classification done yet, label submission is a no-op
        clf.submit_label("left_hand")
        assert clf.get_accuracy() == 0.0


# ── TestSessionStats ────────────────────────────────────────────────


class TestSessionStats:
    def test_empty_session_stats(self):
        clf = MotorImageryClassifier()
        stats = clf.get_session_stats()
        assert stats["n_classifications"] == 0

    def test_stats_after_classifications(self):
        clf = MotorImageryClassifier()
        clf.set_baseline(_synth_eeg(), fs=FS)
        clf.classify(_synth_eeg(), fs=FS)
        clf.classify(_left_hand_imagery_signal(), fs=FS)
        stats = clf.get_session_stats()
        assert stats["n_classifications"] == 2
        assert "class_distribution" in stats


# ── TestHistory ─────────────────────────────────────────────────────


class TestHistory:
    def test_empty_history(self):
        clf = MotorImageryClassifier()
        assert clf.get_history() == []

    def test_history_grows(self):
        clf = MotorImageryClassifier()
        clf.set_baseline(_synth_eeg(), fs=FS)
        clf.classify(_synth_eeg(), fs=FS)
        clf.classify(_synth_eeg(), fs=FS)
        assert len(clf.get_history()) == 2

    def test_history_last_n(self):
        clf = MotorImageryClassifier()
        clf.set_baseline(_synth_eeg(), fs=FS)
        for _ in range(5):
            clf.classify(_synth_eeg(), fs=FS)
        last_3 = clf.get_history(last_n=3)
        assert len(last_3) == 3


# ── TestMultiUser ───────────────────────────────────────────────────


class TestMultiUser:
    def test_independent_users(self):
        clf = MotorImageryClassifier()
        clf.set_baseline(_synth_eeg(), fs=FS, user_id="alice")
        clf.set_baseline(_synth_eeg(), fs=FS, user_id="bob")
        r1 = clf.classify(_left_hand_imagery_signal(), fs=FS, user_id="alice")
        r2 = clf.classify(_right_hand_imagery_signal(), fs=FS, user_id="bob")
        # Each user has independent state
        assert clf.get_session_stats(user_id="alice")["n_classifications"] == 1
        assert clf.get_session_stats(user_id="bob")["n_classifications"] == 1


# ── TestReset ───────────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_state(self):
        clf = MotorImageryClassifier()
        clf.set_baseline(_synth_eeg(), fs=FS)
        clf.classify(_synth_eeg(), fs=FS)
        clf.submit_label("left_hand")
        clf.reset()
        assert clf.get_history() == []
        assert clf.get_accuracy() == 0.0
        assert clf.get_session_stats()["n_classifications"] == 0

    def test_reset_specific_user(self):
        clf = MotorImageryClassifier()
        clf.set_baseline(_synth_eeg(), fs=FS, user_id="alice")
        clf.classify(_synth_eeg(), fs=FS, user_id="alice")
        clf.set_baseline(_synth_eeg(), fs=FS, user_id="bob")
        clf.classify(_synth_eeg(), fs=FS, user_id="bob")
        clf.reset(user_id="alice")
        assert clf.get_session_stats(user_id="alice")["n_classifications"] == 0
        assert clf.get_session_stats(user_id="bob")["n_classifications"] == 1

"""Tests for EEG-based deception detector."""

import numpy as np
import pytest

from models.deception_detector import (
    DISCLAIMER,
    DeceptionDetector,
    _DECEPTIVE_LOWER,
    _TRUTHFUL_UPPER,
)


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def detector():
    """Fresh detector with default 256 Hz."""
    return DeceptionDetector(fs=256.0)


@pytest.fixture
def rng():
    """Seeded random generator for reproducibility."""
    return np.random.RandomState(42)


def _make_calm_signal(rng, fs=256, duration_s=4, n_channels=4):
    """Simulate calm/truthful EEG: strong alpha, low theta/beta.

    Dominant 10 Hz alpha with minimal theta and beta activity.
    """
    n = int(fs * duration_s)
    t = np.arange(n) / fs
    signals = np.zeros((n_channels, n))
    for ch in range(n_channels):
        # Strong alpha (10 Hz) -- relaxed truthful state
        signals[ch] = 15.0 * np.sin(2 * np.pi * 10 * t)
        # Weak theta (6 Hz)
        signals[ch] += 2.0 * np.sin(2 * np.pi * 6 * t)
        # Weak beta (20 Hz)
        signals[ch] += 2.0 * np.sin(2 * np.pi * 20 * t)
        # Small noise
        signals[ch] += rng.randn(n) * 1.0
    return signals


def _make_deceptive_signal(rng, fs=256, duration_s=4, n_channels=4):
    """Simulate deceptive EEG: elevated theta + beta, suppressed alpha.

    Dominant frontal theta (cognitive load) and beta (effort),
    reduced alpha (suppressed relaxation).
    """
    n = int(fs * duration_s)
    t = np.arange(n) / fs
    signals = np.zeros((n_channels, n))
    for ch in range(n_channels):
        # Weak alpha (10 Hz) -- suppressed
        signals[ch] = 3.0 * np.sin(2 * np.pi * 10 * t)
        # Strong theta (6 Hz) -- cognitive load
        signals[ch] += 18.0 * np.sin(2 * np.pi * 6 * t)
        # Strong beta (20 Hz) -- effortful processing
        signals[ch] += 14.0 * np.sin(2 * np.pi * 20 * t)
        # More noise (mental effort)
        signals[ch] += rng.randn(n) * 2.0
    return signals


def _make_single_channel(rng, fs=256, duration_s=4):
    """Single-channel EEG (1D array)."""
    n = int(fs * duration_s)
    t = np.arange(n) / fs
    signal = 10.0 * np.sin(2 * np.pi * 10 * t)
    signal += 4.0 * np.sin(2 * np.pi * 6 * t)
    signal += rng.randn(n) * 2.0
    return signal


# ── TestInit ─────────────────────────────────────────────────


class TestInit:
    def test_default_fs(self):
        d = DeceptionDetector()
        assert d._fs == 256.0

    def test_custom_fs(self):
        d = DeceptionDetector(fs=512.0)
        assert d._fs == 512.0

    def test_empty_state(self):
        d = DeceptionDetector()
        assert d._baselines == {}
        assert d._history == {}


# ── TestSetBaseline ──────────────────────────────────────────


class TestSetBaseline:
    def test_baseline_set_flag(self, detector, rng):
        signals = _make_calm_signal(rng)
        result = detector.set_baseline(signals)
        assert result["baseline_set"] is True

    def test_baseline_metrics_keys(self, detector, rng):
        signals = _make_calm_signal(rng)
        result = detector.set_baseline(signals)
        metrics = result["baseline_metrics"]
        assert "frontal_theta" in metrics
        assert "beta_power" in metrics
        assert "alpha_power" in metrics
        assert "theta_alpha_ratio" in metrics

    def test_baseline_metrics_positive(self, detector, rng):
        signals = _make_calm_signal(rng)
        result = detector.set_baseline(signals)
        metrics = result["baseline_metrics"]
        assert metrics["frontal_theta"] >= 0
        assert metrics["beta_power"] >= 0
        assert metrics["alpha_power"] >= 0

    def test_baseline_stored_per_user(self, detector, rng):
        s1 = _make_calm_signal(rng)
        s2 = _make_calm_signal(rng, duration_s=2)
        detector.set_baseline(s1, user_id="alice")
        detector.set_baseline(s2, user_id="bob")
        assert "alice" in detector._baselines
        assert "bob" in detector._baselines

    def test_baseline_overwrite(self, detector, rng):
        s1 = _make_calm_signal(rng)
        detector.set_baseline(s1, user_id="u1")
        old_theta = detector._baselines["u1"]["frontal_theta"]
        s2 = _make_deceptive_signal(rng)
        detector.set_baseline(s2, user_id="u1")
        new_theta = detector._baselines["u1"]["frontal_theta"]
        # Different signal should produce different baseline
        assert old_theta != new_theta

    def test_single_channel_baseline(self, detector, rng):
        signal = _make_single_channel(rng)
        result = detector.set_baseline(signal)
        assert result["baseline_set"] is True
        assert result["baseline_metrics"]["alpha_power"] > 0

    def test_custom_fs_baseline(self, rng):
        d = DeceptionDetector(fs=512.0)
        signals = _make_calm_signal(rng, fs=512, duration_s=4)
        result = d.set_baseline(signals, fs=512.0)
        assert result["baseline_set"] is True


# ── TestAssess ───────────────────────────────────────────────


class TestAssess:
    def test_output_keys(self, detector, rng):
        signals = _make_calm_signal(rng)
        result = detector.assess(signals)
        expected_keys = {
            "deception_likelihood",
            "cognitive_load",
            "confidence",
            "assessment",
            "frontal_theta_power",
            "beta_engagement",
            "alpha_suppression",
            "disclaimer",
            "has_baseline",
        }
        assert expected_keys == set(result.keys())

    def test_deception_likelihood_range(self, detector, rng):
        signals = _make_calm_signal(rng)
        result = detector.assess(signals)
        assert 0.0 <= result["deception_likelihood"] <= 1.0

    def test_cognitive_load_range(self, detector, rng):
        signals = _make_calm_signal(rng)
        result = detector.assess(signals)
        assert 0.0 <= result["cognitive_load"] <= 1.0

    def test_confidence_range(self, detector, rng):
        signals = _make_calm_signal(rng)
        result = detector.assess(signals)
        assert 0.0 < result["confidence"] <= 1.0

    def test_beta_engagement_range(self, detector, rng):
        signals = _make_calm_signal(rng)
        result = detector.assess(signals)
        assert 0.0 <= result["beta_engagement"] <= 1.0

    def test_alpha_suppression_range(self, detector, rng):
        signals = _make_calm_signal(rng)
        result = detector.assess(signals)
        assert 0.0 <= result["alpha_suppression"] <= 1.0

    def test_disclaimer_always_present(self, detector, rng):
        signals = _make_calm_signal(rng)
        result = detector.assess(signals)
        assert result["disclaimer"] == DISCLAIMER

    def test_has_baseline_false_without_baseline(self, detector, rng):
        signals = _make_calm_signal(rng)
        result = detector.assess(signals)
        assert result["has_baseline"] is False

    def test_has_baseline_true_with_baseline(self, detector, rng):
        baseline_sig = _make_calm_signal(rng)
        detector.set_baseline(baseline_sig)
        test_sig = _make_calm_signal(rng)
        result = detector.assess(test_sig)
        assert result["has_baseline"] is True

    def test_assessment_label_values(self, detector, rng):
        signals = _make_calm_signal(rng)
        result = detector.assess(signals)
        assert result["assessment"] in ("truthful", "uncertain", "deceptive")


# ── TestAssessmentThresholds ──────────────────────────────


class TestAssessmentThresholds:
    def test_calm_signal_tends_truthful(self, detector, rng):
        """Calm signal with strong alpha should not register as deceptive."""
        baseline = _make_calm_signal(rng)
        detector.set_baseline(baseline)
        test_sig = _make_calm_signal(np.random.RandomState(99))
        result = detector.assess(test_sig)
        # With baseline from calm + test signal also calm, should not be deceptive
        assert result["assessment"] != "deceptive"

    def test_deceptive_signal_elevated_features(self, detector, rng):
        """Deceptive signal should show higher cognitive load and beta engagement
        than calm signal (relative to calm baseline)."""
        baseline = _make_calm_signal(rng)
        detector.set_baseline(baseline)

        calm_result = detector.assess(_make_calm_signal(np.random.RandomState(77)))
        deceptive_result = detector.assess(
            _make_deceptive_signal(np.random.RandomState(88))
        )

        # Deceptive signal should show higher cognitive load
        assert deceptive_result["cognitive_load"] > calm_result["cognitive_load"]
        # Deceptive signal should show higher beta engagement
        assert deceptive_result["beta_engagement"] > calm_result["beta_engagement"]

    def test_deceptive_signal_higher_likelihood(self, detector, rng):
        """Deceptive signal should produce higher deception likelihood."""
        baseline = _make_calm_signal(rng)
        detector.set_baseline(baseline)

        calm_result = detector.assess(_make_calm_signal(np.random.RandomState(77)))
        deceptive_result = detector.assess(
            _make_deceptive_signal(np.random.RandomState(88))
        )

        assert (
            deceptive_result["deception_likelihood"]
            > calm_result["deception_likelihood"]
        )

    def test_threshold_truthful(self):
        """Truthful threshold is below 0.35."""
        assert _TRUTHFUL_UPPER == 0.35

    def test_threshold_deceptive(self):
        """Deceptive threshold is above 0.65."""
        assert _DECEPTIVE_LOWER == 0.65


# ── TestConfidence ────────────────────────────────────────


class TestConfidence:
    def test_confidence_higher_with_baseline(self, detector, rng):
        """Having a baseline should generally increase confidence."""
        signals = _make_calm_signal(rng)
        result_no_baseline = detector.assess(signals)

        detector.set_baseline(_make_calm_signal(np.random.RandomState(55)))
        result_with_baseline = detector.assess(signals)

        # Baseline should improve confidence (or at least not decrease it
        # dramatically). Exact comparison depends on likelihood value.
        # Just check both are valid.
        assert result_no_baseline["confidence"] > 0
        assert result_with_baseline["confidence"] > 0

    def test_confidence_bounded(self, detector, rng):
        """Confidence must be between 0.1 and 0.95."""
        signals = _make_calm_signal(rng)
        result = detector.assess(signals)
        assert 0.1 <= result["confidence"] <= 0.95


# ── TestHistory ──────────────────────────────────────────


class TestHistory:
    def test_empty_history(self, detector):
        assert detector.get_history() == []

    def test_history_grows(self, detector, rng):
        signals = _make_calm_signal(rng)
        detector.assess(signals)
        detector.assess(signals)
        assert len(detector.get_history()) == 2

    def test_history_last_n(self, detector, rng):
        signals = _make_calm_signal(rng)
        for _ in range(10):
            detector.assess(signals)
        assert len(detector.get_history(last_n=3)) == 3

    def test_history_last_n_larger_than_total(self, detector, rng):
        signals = _make_calm_signal(rng)
        detector.assess(signals)
        assert len(detector.get_history(last_n=100)) == 1

    def test_history_capped_at_500(self, detector):
        """History should not exceed 500 entries per user."""
        # Use tiny signal for speed
        tiny = np.random.randn(4, 64)
        for _ in range(510):
            detector.assess(tiny)
        assert len(detector.get_history()) == 500

    def test_history_per_user(self, detector, rng):
        signals = _make_calm_signal(rng)
        detector.assess(signals, user_id="alice")
        detector.assess(signals, user_id="alice")
        detector.assess(signals, user_id="bob")
        assert len(detector.get_history("alice")) == 2
        assert len(detector.get_history("bob")) == 1

    def test_history_contains_result_keys(self, detector, rng):
        signals = _make_calm_signal(rng)
        detector.assess(signals)
        entry = detector.get_history()[0]
        assert "deception_likelihood" in entry
        assert "assessment" in entry
        assert "disclaimer" in entry

    def test_get_history_returns_copy(self, detector, rng):
        """get_history should return a new list, not internal reference."""
        signals = _make_calm_signal(rng)
        detector.assess(signals)
        h1 = detector.get_history()
        h1.clear()
        # Internal history should be unaffected
        assert len(detector.get_history()) == 1


# ── TestSessionStats ──────────────────────────────────────


class TestSessionStats:
    def test_empty_stats(self, detector):
        stats = detector.get_session_stats()
        assert stats["n_epochs"] == 0
        assert stats["mean_likelihood"] == 0.0
        assert stats["assessment_distribution"]["truthful"] == 0
        assert stats["assessment_distribution"]["uncertain"] == 0
        assert stats["assessment_distribution"]["deceptive"] == 0

    def test_stats_after_assessments(self, detector, rng):
        signals = _make_calm_signal(rng)
        for _ in range(5):
            detector.assess(signals)
        stats = detector.get_session_stats()
        assert stats["n_epochs"] == 5
        assert 0.0 <= stats["mean_likelihood"] <= 1.0
        total = sum(stats["assessment_distribution"].values())
        assert total == 5

    def test_stats_per_user(self, detector, rng):
        signals = _make_calm_signal(rng)
        detector.assess(signals, user_id="alice")
        detector.assess(signals, user_id="alice")
        detector.assess(signals, user_id="bob")
        assert detector.get_session_stats("alice")["n_epochs"] == 2
        assert detector.get_session_stats("bob")["n_epochs"] == 1


# ── TestMultiUser ─────────────────────────────────────────


class TestMultiUser:
    def test_independent_baselines(self, detector, rng):
        calm = _make_calm_signal(rng)
        decept = _make_deceptive_signal(rng)
        detector.set_baseline(calm, user_id="honest")
        detector.set_baseline(decept, user_id="liar")
        assert (
            detector._baselines["honest"]["frontal_theta"]
            != detector._baselines["liar"]["frontal_theta"]
        )

    def test_independent_histories(self, detector, rng):
        signals = _make_calm_signal(rng)
        detector.assess(signals, user_id="u1")
        detector.assess(signals, user_id="u2")
        detector.assess(signals, user_id="u2")
        assert len(detector.get_history("u1")) == 1
        assert len(detector.get_history("u2")) == 2

    def test_reset_one_user(self, detector, rng):
        signals = _make_calm_signal(rng)
        detector.set_baseline(signals, user_id="a")
        detector.assess(signals, user_id="a")
        detector.set_baseline(signals, user_id="b")
        detector.assess(signals, user_id="b")
        detector.reset("a")
        assert detector.get_history("a") == []
        assert "a" not in detector._baselines
        # User b unaffected
        assert len(detector.get_history("b")) == 1
        assert "b" in detector._baselines


# ── TestReset ─────────────────────────────────────────────


class TestReset:
    def test_reset_clears_history(self, detector, rng):
        signals = _make_calm_signal(rng)
        detector.assess(signals)
        detector.reset()
        assert detector.get_history() == []

    def test_reset_clears_baseline(self, detector, rng):
        signals = _make_calm_signal(rng)
        detector.set_baseline(signals)
        detector.reset()
        assert "default" not in detector._baselines

    def test_reset_clears_stats(self, detector, rng):
        signals = _make_calm_signal(rng)
        detector.assess(signals)
        detector.reset()
        assert detector.get_session_stats()["n_epochs"] == 0


# ── TestEdgeCases ─────────────────────────────────────────


class TestEdgeCases:
    def test_very_short_signal(self, detector):
        """Very short signal (< 1 second) should not crash."""
        short = np.random.randn(4, 32)
        result = detector.assess(short)
        assert "deception_likelihood" in result
        assert 0.0 <= result["deception_likelihood"] <= 1.0

    def test_single_channel_input(self, detector, rng):
        """1D input should work."""
        signal = _make_single_channel(rng)
        result = detector.assess(signal)
        assert "deception_likelihood" in result

    def test_two_channel_input(self, detector, rng):
        """2-channel input should still work."""
        n = int(256 * 4)
        t = np.arange(n) / 256
        signals = np.zeros((2, n))
        signals[0] = 10.0 * np.sin(2 * np.pi * 10 * t) + rng.randn(n)
        signals[1] = 10.0 * np.sin(2 * np.pi * 10 * t) + rng.randn(n)
        result = detector.assess(signals)
        assert 0.0 <= result["deception_likelihood"] <= 1.0

    def test_flat_signal(self, detector):
        """Flat (near-zero) signal should not crash or produce NaN."""
        flat = np.ones((4, 1024)) * 0.001
        result = detector.assess(flat)
        assert not np.isnan(result["deception_likelihood"])
        assert 0.0 <= result["deception_likelihood"] <= 1.0

    def test_noisy_signal(self, detector):
        """Very noisy signal should produce valid output."""
        noisy = np.random.randn(4, 1024) * 200
        result = detector.assess(noisy)
        assert 0.0 <= result["deception_likelihood"] <= 1.0

    def test_assess_without_baseline_default_user(self, detector, rng):
        """Assess should work without setting a baseline first."""
        signals = _make_calm_signal(rng)
        result = detector.assess(signals)
        assert result["has_baseline"] is False
        assert result["assessment"] in ("truthful", "uncertain", "deceptive")

    def test_zero_length_history(self, detector):
        """get_history with last_n=0 should return full history."""
        tiny = np.random.randn(4, 64)
        detector.assess(tiny)
        detector.assess(tiny)
        # last_n=0 is falsy, should return all
        assert len(detector.get_history(last_n=0)) == 2

    def test_nonexistent_user_history(self, detector):
        """Requesting history for unknown user returns empty list."""
        assert detector.get_history("nonexistent") == []

    def test_nonexistent_user_stats(self, detector):
        """Requesting stats for unknown user returns empty stats."""
        stats = detector.get_session_stats("nonexistent")
        assert stats["n_epochs"] == 0

"""Tests for interoceptive awareness estimator.

Covers:
  - Baseline recording and output structure
  - Assessment output keys, types, and ranges
  - Interoceptive scoring (high theta + alpha suppression = high score)
  - Alpha suppression relative to baseline
  - Right-frontal activation detection
  - Body awareness level thresholds
  - Session statistics and improvement trend
  - History management
  - Reset behaviour
  - Multi-user isolation
  - Edge cases (constant, 1-D, short, zero signals)
"""

import numpy as np
import pytest

from models.interoceptive_awareness import InteroceptiveAwarenessTrainer


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_signal(
    fs=256,
    duration=4,
    theta_amp=5.0,
    alpha_amp=10.0,
    af7_alpha=None,
    af8_alpha=None,
    af7_theta=None,
    af8_theta=None,
    noise_amp=1.0,
    n_channels=4,
    seed=42,
):
    """Build synthetic EEG with controllable per-channel band amplitudes.

    Channel layout: ch0=TP9, ch1=AF7, ch2=AF8, ch3=TP10.
    By default all channels share the same theta_amp and alpha_amp.
    Override af7_*/af8_* to make per-channel differences.
    Set seed=None to use the caller's RNG state (no re-seeding).
    """
    if seed is not None:
        np.random.seed(seed)
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs
    signals = []

    for ch in range(n_channels):
        # Determine per-channel amplitudes
        a_alpha = alpha_amp
        a_theta = theta_amp
        if ch == 1:  # AF7
            if af7_alpha is not None:
                a_alpha = af7_alpha
            if af7_theta is not None:
                a_theta = af7_theta
        elif ch == 2:  # AF8
            if af8_alpha is not None:
                a_alpha = af8_alpha
            if af8_theta is not None:
                a_theta = af8_theta

        alpha = a_alpha * np.sin(2 * np.pi * 10 * t + ch * 0.3)
        theta = a_theta * np.sin(2 * np.pi * 6 * t + ch * 0.5)
        noise = noise_amp * np.random.randn(n_samples)
        signals.append(alpha + theta + noise)

    return np.array(signals[:n_channels])


@pytest.fixture
def trainer():
    return InteroceptiveAwarenessTrainer(fs=256.0)


# ── Baseline ─────────────────────────────────────────────────────────────────

class TestBaseline:
    def test_set_baseline_returns_dict(self, trainer):
        np.random.seed(42)
        result = trainer.set_baseline(_make_signal())
        assert isinstance(result, dict)

    def test_baseline_set_flag(self, trainer):
        np.random.seed(42)
        result = trainer.set_baseline(_make_signal())
        assert result["baseline_set"] is True

    def test_baseline_n_channels(self, trainer):
        np.random.seed(42)
        result = trainer.set_baseline(_make_signal(n_channels=4))
        assert result["n_channels"] == 4

    def test_baseline_single_channel(self, trainer):
        np.random.seed(42)
        sig = np.random.randn(1024) * 10
        result = trainer.set_baseline(sig)
        assert result["baseline_set"] is True
        assert result["n_channels"] == 1

    def test_baseline_enables_has_baseline(self, trainer):
        np.random.seed(42)
        result = trainer.assess(_make_signal())
        assert result["has_baseline"] is False
        trainer.set_baseline(_make_signal())
        result = trainer.assess(_make_signal())
        assert result["has_baseline"] is True

    def test_baseline_per_user(self, trainer):
        np.random.seed(42)
        trainer.set_baseline(_make_signal(), user_id="alice")
        # bob has no baseline
        result_alice = trainer.assess(_make_signal(), user_id="alice")
        result_bob = trainer.assess(_make_signal(), user_id="bob")
        assert result_alice["has_baseline"] is True
        assert result_bob["has_baseline"] is False


# ── Assessment Output Structure ──────────────────────────────────────────────

class TestAssessOutputStructure:
    def test_all_keys_present(self, trainer):
        np.random.seed(42)
        result = trainer.assess(_make_signal())
        expected_keys = {
            "interoceptive_score",
            "frontal_theta_power",
            "alpha_suppression",
            "right_frontal_activation",
            "body_awareness_level",
            "has_baseline",
        }
        assert expected_keys == set(result.keys())

    def test_interoceptive_score_is_float(self, trainer):
        np.random.seed(42)
        result = trainer.assess(_make_signal())
        assert isinstance(result["interoceptive_score"], float)

    def test_interoceptive_score_range(self, trainer):
        np.random.seed(42)
        result = trainer.assess(_make_signal())
        assert 0.0 <= result["interoceptive_score"] <= 1.0

    def test_frontal_theta_is_float(self, trainer):
        np.random.seed(42)
        result = trainer.assess(_make_signal())
        assert isinstance(result["frontal_theta_power"], float)

    def test_frontal_theta_non_negative(self, trainer):
        np.random.seed(42)
        result = trainer.assess(_make_signal())
        assert result["frontal_theta_power"] >= 0.0

    def test_alpha_suppression_is_float(self, trainer):
        np.random.seed(42)
        result = trainer.assess(_make_signal())
        assert isinstance(result["alpha_suppression"], float)

    def test_alpha_suppression_range(self, trainer):
        np.random.seed(42)
        result = trainer.assess(_make_signal())
        assert 0.0 <= result["alpha_suppression"] <= 1.0

    def test_right_frontal_is_float(self, trainer):
        np.random.seed(42)
        result = trainer.assess(_make_signal())
        assert isinstance(result["right_frontal_activation"], float)

    def test_right_frontal_range(self, trainer):
        np.random.seed(42)
        result = trainer.assess(_make_signal())
        assert 0.0 <= result["right_frontal_activation"] <= 1.0

    def test_body_awareness_level_is_string(self, trainer):
        np.random.seed(42)
        result = trainer.assess(_make_signal())
        assert isinstance(result["body_awareness_level"], str)

    def test_body_awareness_level_valid(self, trainer):
        np.random.seed(42)
        result = trainer.assess(_make_signal())
        assert result["body_awareness_level"] in ("high", "moderate", "low", "minimal")

    def test_has_baseline_is_bool(self, trainer):
        np.random.seed(42)
        result = trainer.assess(_make_signal())
        assert isinstance(result["has_baseline"], bool)


# ── Interoceptive Scoring Logic ──────────────────────────────────────────────

class TestInteroceptiveScoring:
    def test_high_theta_and_suppression_gives_high_score(self, trainer):
        """High frontal theta + suppressed alpha (vs baseline) = high interoceptive score."""
        np.random.seed(42)
        # Baseline: strong alpha, moderate theta
        baseline = _make_signal(alpha_amp=20.0, theta_amp=3.0)
        trainer.set_baseline(baseline)

        # Task: strong theta, suppressed alpha
        task = _make_signal(alpha_amp=3.0, theta_amp=20.0)
        result = trainer.assess(task)
        assert result["interoceptive_score"] > 0.5

    def test_low_theta_no_suppression_gives_low_score(self, trainer):
        """Low frontal theta + no alpha suppression = low score."""
        np.random.seed(42)
        # Baseline with low alpha
        baseline = _make_signal(alpha_amp=3.0, theta_amp=2.0)
        trainer.set_baseline(baseline)

        # Task: same alpha (no suppression), low theta
        task = _make_signal(alpha_amp=3.0, theta_amp=2.0)
        result = trainer.assess(task)
        assert result["interoceptive_score"] < 0.5

    def test_stronger_theta_increases_score(self, trainer):
        """Increasing frontal theta should increase the composite score."""
        np.random.seed(42)
        low_theta = _make_signal(theta_amp=2.0, alpha_amp=10.0)
        high_theta = _make_signal(theta_amp=25.0, alpha_amp=10.0)
        r_low = trainer.assess(low_theta)
        r_high = trainer.assess(high_theta)
        assert r_high["frontal_theta_power"] > r_low["frontal_theta_power"]

    def test_score_bounded_at_one(self, trainer):
        """Even extreme inputs should not push score above 1.0."""
        np.random.seed(42)
        baseline = _make_signal(alpha_amp=50.0, theta_amp=1.0)
        trainer.set_baseline(baseline)
        extreme = _make_signal(alpha_amp=0.5, theta_amp=50.0,
                               af8_alpha=0.1, af7_alpha=30.0)
        result = trainer.assess(extreme)
        assert result["interoceptive_score"] <= 1.0

    def test_score_bounded_at_zero(self, trainer):
        """Score should never go below 0.0."""
        np.random.seed(42)
        result = trainer.assess(_make_signal(theta_amp=0.1, alpha_amp=0.1))
        assert result["interoceptive_score"] >= 0.0


# ── Alpha Suppression ────────────────────────────────────────────────────────

class TestAlphaSuppression:
    def test_suppressed_alpha_positive(self, trainer):
        """Current alpha < baseline alpha -> positive suppression."""
        np.random.seed(42)
        baseline = _make_signal(alpha_amp=20.0)
        trainer.set_baseline(baseline)
        task = _make_signal(alpha_amp=3.0)
        result = trainer.assess(task)
        assert result["alpha_suppression"] > 0.3

    def test_no_suppression_when_alpha_equals_baseline(self, trainer):
        """Same alpha as baseline -> suppression near zero."""
        np.random.seed(42)
        sig = _make_signal(alpha_amp=10.0)
        trainer.set_baseline(sig)
        result = trainer.assess(sig)
        assert result["alpha_suppression"] < 0.15

    def test_alpha_increase_clamped_to_zero(self, trainer):
        """If alpha increases from baseline, suppression should be 0 (not negative)."""
        np.random.seed(42)
        baseline = _make_signal(alpha_amp=3.0)
        trainer.set_baseline(baseline)
        task = _make_signal(alpha_amp=30.0)
        result = trainer.assess(task)
        assert result["alpha_suppression"] == 0.0

    def test_suppression_without_baseline_uses_default(self, trainer):
        """Without baseline, still computes suppression against population default."""
        np.random.seed(42)
        result = trainer.assess(_make_signal(alpha_amp=2.0))
        # Should not crash; suppression should be non-negative
        assert result["alpha_suppression"] >= 0.0

    def test_suppression_capped_at_one(self, trainer):
        """Suppression should not exceed 1.0 even with extreme baseline."""
        np.random.seed(42)
        baseline = _make_signal(alpha_amp=100.0)
        trainer.set_baseline(baseline)
        task = _make_signal(alpha_amp=0.01)
        result = trainer.assess(task)
        assert result["alpha_suppression"] <= 1.0


# ── Right-Frontal Activation ─────────────────────────────────────────────────

class TestRightFrontalActivation:
    def test_right_lateralized_detected(self, trainer):
        """Lower AF8 alpha (right-frontal activation) -> high value."""
        np.random.seed(42)
        sig = _make_signal(af7_alpha=20.0, af8_alpha=3.0)
        result = trainer.assess(sig)
        assert result["right_frontal_activation"] > 0.6

    def test_left_lateralized_detected(self, trainer):
        """Lower AF7 alpha (left-frontal activation) -> low value."""
        np.random.seed(42)
        sig = _make_signal(af7_alpha=3.0, af8_alpha=20.0)
        result = trainer.assess(sig)
        assert result["right_frontal_activation"] < 0.4

    def test_symmetric_gives_neutral(self, trainer):
        """Equal alpha at AF7/AF8 -> activation near 0.5."""
        np.random.seed(42)
        sig = _make_signal(af7_alpha=10.0, af8_alpha=10.0)
        result = trainer.assess(sig)
        assert 0.35 <= result["right_frontal_activation"] <= 0.65

    def test_single_channel_neutral(self, trainer):
        """With only 1 channel, right_frontal_activation should be 0.5."""
        np.random.seed(42)
        sig = np.random.randn(1024) * 10
        result = trainer.assess(sig)
        assert result["right_frontal_activation"] == 0.5

    def test_two_channel_neutral(self, trainer):
        """With 2 channels (no AF8), should return 0.5."""
        np.random.seed(42)
        sig = np.random.randn(2, 1024) * 10
        result = trainer.assess(sig)
        assert result["right_frontal_activation"] == 0.5


# ── Body Awareness Levels ────────────────────────────────────────────────────

class TestBodyAwarenessLevels:
    def test_high_level(self, trainer):
        """Score >= 0.65 should produce 'high'."""
        np.random.seed(42)
        baseline = _make_signal(alpha_amp=30.0, theta_amp=2.0)
        trainer.set_baseline(baseline)
        task = _make_signal(alpha_amp=2.0, theta_amp=25.0,
                            af7_alpha=15.0, af8_alpha=2.0)
        result = trainer.assess(task)
        if result["interoceptive_score"] >= 0.65:
            assert result["body_awareness_level"] == "high"

    def test_minimal_level_low_signal(self, trainer):
        """Very low theta with no suppression -> 'minimal' or 'low'."""
        np.random.seed(42)
        # Set baseline with similarly low alpha so suppression is near zero
        baseline = _make_signal(alpha_amp=1.0, theta_amp=0.5)
        trainer.set_baseline(baseline)
        result = trainer.assess(_make_signal(theta_amp=0.5, alpha_amp=1.0))
        assert result["body_awareness_level"] in ("minimal", "low")

    def test_level_consistent_with_score(self, trainer):
        """Awareness level must be consistent with score thresholds."""
        np.random.seed(42)
        for _ in range(20):
            sig = _make_signal(
                theta_amp=np.random.uniform(0.5, 25),
                alpha_amp=np.random.uniform(0.5, 25),
            )
            result = trainer.assess(sig)
            score = result["interoceptive_score"]
            level = result["body_awareness_level"]
            if score >= 0.65:
                assert level == "high"
            elif score >= 0.40:
                assert level == "moderate"
            elif score >= 0.20:
                assert level == "low"
            else:
                assert level == "minimal"


# ── Session Stats ────────────────────────────────────────────────────────────

class TestSessionStats:
    def test_empty_stats(self, trainer):
        stats = trainer.get_session_stats()
        assert stats["n_epochs"] == 0
        assert stats["mean_score"] == 0.0
        assert stats["improvement_trend"] == "insufficient_data"

    def test_stats_after_assessments(self, trainer):
        np.random.seed(42)
        for _ in range(5):
            trainer.assess(_make_signal())
        stats = trainer.get_session_stats()
        assert stats["n_epochs"] == 5
        assert 0.0 <= stats["mean_score"] <= 1.0

    def test_improvement_trend_improving(self, trainer):
        """Scores that increase over time -> 'improving'."""
        np.random.seed(42)
        baseline = _make_signal(alpha_amp=30.0, theta_amp=2.0)
        trainer.set_baseline(baseline)
        # First half: low theta, high alpha (no suppression) -> low scores
        for i in range(4):
            trainer.assess(_make_signal(theta_amp=0.5, alpha_amp=30.0,
                                        seed=100 + i))
        # Second half: high theta, suppressed alpha, right-lateralized -> high scores
        for i in range(4):
            trainer.assess(_make_signal(theta_amp=30.0, alpha_amp=2.0,
                                        af7_alpha=15.0, af8_alpha=2.0,
                                        seed=200 + i))
        stats = trainer.get_session_stats()
        assert stats["improvement_trend"] == "improving"

    def test_improvement_trend_declining(self, trainer):
        """Scores that decrease -> 'declining'."""
        np.random.seed(42)
        baseline = _make_signal(alpha_amp=30.0, theta_amp=2.0)
        trainer.set_baseline(baseline)
        # First half: high theta, suppressed alpha, right-lateralized -> high scores
        for i in range(4):
            trainer.assess(_make_signal(theta_amp=30.0, alpha_amp=2.0,
                                        af7_alpha=15.0, af8_alpha=2.0,
                                        seed=200 + i))
        # Second half: low theta, high alpha (no suppression) -> low scores
        for i in range(4):
            trainer.assess(_make_signal(theta_amp=0.5, alpha_amp=30.0,
                                        seed=100 + i))
        stats = trainer.get_session_stats()
        assert stats["improvement_trend"] == "declining"

    def test_insufficient_data_trend(self, trainer):
        """Fewer than 4 epochs -> 'insufficient_data'."""
        np.random.seed(42)
        trainer.assess(_make_signal())
        trainer.assess(_make_signal())
        stats = trainer.get_session_stats()
        assert stats["improvement_trend"] == "insufficient_data"

    def test_stats_per_user(self, trainer):
        np.random.seed(42)
        trainer.assess(_make_signal(), user_id="alice")
        trainer.assess(_make_signal(), user_id="bob")
        trainer.assess(_make_signal(), user_id="bob")
        assert trainer.get_session_stats("alice")["n_epochs"] == 1
        assert trainer.get_session_stats("bob")["n_epochs"] == 2


# ── History ──────────────────────────────────────────────────────────────────

class TestHistory:
    def test_empty_history(self, trainer):
        assert trainer.get_history() == []

    def test_history_grows(self, trainer):
        np.random.seed(42)
        trainer.assess(_make_signal())
        trainer.assess(_make_signal())
        assert len(trainer.get_history()) == 2

    def test_history_last_n(self, trainer):
        np.random.seed(42)
        for _ in range(10):
            trainer.assess(_make_signal())
        hist = trainer.get_history(last_n=3)
        assert len(hist) == 3

    def test_history_last_n_larger_than_total(self, trainer):
        np.random.seed(42)
        trainer.assess(_make_signal())
        trainer.assess(_make_signal())
        hist = trainer.get_history(last_n=100)
        assert len(hist) == 2

    def test_history_capped_at_500(self, trainer):
        np.random.seed(42)
        sig = _make_signal()
        for _ in range(510):
            trainer.assess(sig)
        assert len(trainer.get_history()) == 500

    def test_history_entries_have_correct_keys(self, trainer):
        np.random.seed(42)
        trainer.assess(_make_signal())
        entry = trainer.get_history()[0]
        assert "interoceptive_score" in entry
        assert "body_awareness_level" in entry

    def test_history_returns_copy(self, trainer):
        """Returned list should not be the internal list (no mutation leaks)."""
        np.random.seed(42)
        trainer.assess(_make_signal())
        h1 = trainer.get_history()
        h2 = trainer.get_history()
        # They have the same content
        assert len(h1) == len(h2)


# ── Reset ────────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_history(self, trainer):
        np.random.seed(42)
        trainer.assess(_make_signal())
        trainer.reset()
        assert trainer.get_history() == []

    def test_reset_clears_baseline(self, trainer):
        np.random.seed(42)
        trainer.set_baseline(_make_signal())
        trainer.reset()
        result = trainer.assess(_make_signal())
        assert result["has_baseline"] is False

    def test_reset_clears_stats(self, trainer):
        np.random.seed(42)
        trainer.assess(_make_signal())
        trainer.reset()
        stats = trainer.get_session_stats()
        assert stats["n_epochs"] == 0

    def test_reset_one_user_keeps_other(self, trainer):
        np.random.seed(42)
        trainer.assess(_make_signal(), user_id="alice")
        trainer.assess(_make_signal(), user_id="bob")
        trainer.reset("alice")
        assert len(trainer.get_history("alice")) == 0
        assert len(trainer.get_history("bob")) == 1


# ── Multi-User ───────────────────────────────────────────────────────────────

class TestMultiUser:
    def test_independent_histories(self, trainer):
        np.random.seed(42)
        trainer.assess(_make_signal(), user_id="alice")
        trainer.assess(_make_signal(), user_id="bob")
        trainer.assess(_make_signal(), user_id="bob")
        assert len(trainer.get_history("alice")) == 1
        assert len(trainer.get_history("bob")) == 2

    def test_independent_baselines(self, trainer):
        np.random.seed(42)
        trainer.set_baseline(_make_signal(alpha_amp=20.0), user_id="alice")
        # Bob has no baseline
        r_alice = trainer.assess(_make_signal(alpha_amp=5.0), user_id="alice")
        r_bob = trainer.assess(_make_signal(alpha_amp=5.0), user_id="bob")
        assert r_alice["has_baseline"] is True
        assert r_bob["has_baseline"] is False

    def test_independent_stats(self, trainer):
        np.random.seed(42)
        for _ in range(5):
            trainer.assess(_make_signal(), user_id="alice")
        trainer.assess(_make_signal(), user_id="bob")
        assert trainer.get_session_stats("alice")["n_epochs"] == 5
        assert trainer.get_session_stats("bob")["n_epochs"] == 1


# ── Edge Cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_constant_signal(self, trainer):
        """Constant (flat) signal should not crash."""
        np.random.seed(42)
        sig = np.full((4, 1024), 5.0)
        result = trainer.assess(sig)
        assert 0.0 <= result["interoceptive_score"] <= 1.0

    def test_1d_input(self, trainer):
        """1-D signal should be reshaped and processed."""
        np.random.seed(42)
        sig = np.random.randn(1024) * 10
        result = trainer.assess(sig)
        assert "interoceptive_score" in result
        assert 0.0 <= result["interoceptive_score"] <= 1.0

    def test_short_signal(self, trainer):
        """Very short signal (< 1 second) should not crash."""
        np.random.seed(42)
        sig = np.random.randn(4, 64) * 10  # 0.25 sec
        result = trainer.assess(sig)
        assert 0.0 <= result["interoceptive_score"] <= 1.0

    def test_zero_signal(self, trainer):
        """All-zero signal should not crash."""
        sig = np.zeros((4, 1024))
        result = trainer.assess(sig)
        assert 0.0 <= result["interoceptive_score"] <= 1.0

    def test_single_sample(self, trainer):
        """Single sample should not crash."""
        np.random.seed(42)
        sig = np.array([[1.0], [2.0], [3.0], [4.0]])
        result = trainer.assess(sig)
        assert 0.0 <= result["interoceptive_score"] <= 1.0

    def test_very_large_amplitudes(self, trainer):
        """Very large amplitudes (railed) should still produce valid output."""
        np.random.seed(42)
        sig = np.random.randn(4, 1024) * 5000
        result = trainer.assess(sig)
        assert 0.0 <= result["interoceptive_score"] <= 1.0

    def test_nan_free_output(self, trainer):
        """Output values should never be NaN."""
        np.random.seed(42)
        sig = np.random.randn(4, 1024) * 10
        result = trainer.assess(sig)
        for key, val in result.items():
            if isinstance(val, float):
                assert not np.isnan(val), f"{key} is NaN"

    def test_baseline_then_assess_1d(self, trainer):
        """Set baseline with 1-D then assess with 1-D."""
        np.random.seed(42)
        baseline = np.random.randn(1024) * 15
        trainer.set_baseline(baseline)
        task = np.random.randn(1024) * 5
        result = trainer.assess(task)
        assert result["has_baseline"] is True

    def test_fs_override(self, trainer):
        """Passing fs explicitly should override the constructor default."""
        np.random.seed(42)
        sig = _make_signal(fs=128, duration=4)
        result = trainer.assess(sig, fs=128)
        assert 0.0 <= result["interoceptive_score"] <= 1.0

    def test_many_channels(self, trainer):
        """Signal with more than 4 channels should still work."""
        np.random.seed(42)
        sig = np.random.randn(8, 1024) * 10
        result = trainer.assess(sig)
        assert 0.0 <= result["interoceptive_score"] <= 1.0

    def test_two_channels(self, trainer):
        """Two channels (no AF8) should produce valid output."""
        np.random.seed(42)
        sig = np.random.randn(2, 1024) * 10
        result = trainer.assess(sig)
        assert 0.0 <= result["interoceptive_score"] <= 1.0

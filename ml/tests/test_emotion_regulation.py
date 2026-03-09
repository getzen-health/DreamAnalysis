"""Tests for closed-loop emotion regulation biofeedback."""
import numpy as np
import pytest

from models.emotion_regulation import EmotionRegulationTrainer


@pytest.fixture
def trainer():
    return EmotionRegulationTrainer(success_threshold=0.05, fs=256.0)


def _make_eeg(
    fs=256, duration=4, alpha_amp_left=10.0, alpha_amp_right=10.0, n_channels=4
):
    """Synthetic EEG with controllable alpha amplitude per hemisphere.

    Muse 2 layout: ch0=TP9, ch1=AF7 (left), ch2=AF8 (right), ch3=TP10.
    AF7 and AF8 are the frontal channels used for FAA.
    """
    np.random.seed(None)  # Allow variance between calls
    t = np.arange(int(fs * duration)) / fs
    signals = []
    for ch in range(n_channels):
        noise = 2.0 * np.random.randn(len(t))
        if ch == 1:  # AF7 (left frontal)
            alpha = alpha_amp_left * np.sin(2 * np.pi * 10 * t)
        elif ch == 2:  # AF8 (right frontal)
            alpha = alpha_amp_right * np.sin(2 * np.pi * 10 * t)
        else:  # TP9/TP10 (temporal)
            alpha = 8.0 * np.sin(2 * np.pi * 10 * t + ch * 0.5)
        signals.append(alpha + noise)
    return np.array(signals)


def _make_eeg_seeded(seed=42, **kwargs):
    """Deterministic wrapper around _make_eeg."""
    np.random.seed(seed)
    return _make_eeg(**kwargs)


# ── TestBaseline ────────────────────────────────────────────


class TestBaseline:
    def test_set_baseline_returns_faa(self, trainer):
        signals = _make_eeg_seeded(seed=42)
        result = trainer.set_baseline(signals)
        assert "baseline_faa" in result
        assert isinstance(result["baseline_faa"], float)

    def test_baseline_set_flag(self, trainer):
        signals = _make_eeg_seeded(seed=42)
        result = trainer.set_baseline(signals)
        assert result["baseline_set"] is True

    def test_baseline_alpha_powers(self, trainer):
        signals = _make_eeg_seeded(seed=42)
        result = trainer.set_baseline(signals)
        assert result["af7_alpha"] > 0
        assert result["af8_alpha"] > 0

    def test_equal_alpha_gives_near_zero_faa(self, trainer):
        # Same alpha amplitude on both sides -> FAA near 0
        signals = _make_eeg_seeded(seed=42, alpha_amp_left=10, alpha_amp_right=10)
        result = trainer.set_baseline(signals)
        assert abs(result["baseline_faa"]) < 0.5

    def test_single_channel_baseline(self, trainer):
        signal_1d = 10.0 * np.sin(2 * np.pi * 10 * np.arange(1024) / 256)
        result = trainer.set_baseline(signal_1d)
        assert result["baseline_set"] is True
        # Single channel -> FAA ~0 (same signal for both)
        assert abs(result["baseline_faa"]) < 0.01


# ── TestEvaluate ────────────────────────────────────────────


class TestEvaluate:
    def test_output_keys(self, trainer):
        trainer.set_baseline(_make_eeg_seeded(seed=10))
        result = trainer.evaluate(_make_eeg_seeded(seed=20))
        expected_keys = {
            "current_faa", "baseline_faa", "regulation_success",
            "regulation_score", "effort_index", "feedback_message",
            "target_state", "strategy_suggestion",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_score_range(self, trainer):
        trainer.set_baseline(_make_eeg_seeded(seed=10))
        result = trainer.evaluate(_make_eeg_seeded(seed=20))
        assert 0 <= result["regulation_score"] <= 100

    def test_effort_index_range(self, trainer):
        trainer.set_baseline(_make_eeg_seeded(seed=10))
        result = trainer.evaluate(_make_eeg_seeded(seed=20))
        assert 0 <= result["effort_index"] <= 1

    def test_feedback_message_present(self, trainer):
        trainer.set_baseline(_make_eeg_seeded(seed=10))
        result = trainer.evaluate(_make_eeg_seeded(seed=20))
        assert isinstance(result["feedback_message"], str)
        assert len(result["feedback_message"]) > 0

    def test_strategy_suggestion_present(self, trainer):
        trainer.set_baseline(_make_eeg_seeded(seed=10))
        result = trainer.evaluate(_make_eeg_seeded(seed=20))
        assert isinstance(result["strategy_suggestion"], str)
        assert len(result["strategy_suggestion"]) > 0

    def test_target_state_echoed(self, trainer):
        trainer.set_baseline(_make_eeg_seeded(seed=10))
        result = trainer.evaluate(
            _make_eeg_seeded(seed=20), target_state="neutral"
        )
        assert result["target_state"] == "neutral"

    def test_no_baseline_uses_zero(self, trainer):
        result = trainer.evaluate(_make_eeg_seeded(seed=20))
        assert result["baseline_faa"] == 0.0
        assert result["has_baseline"] is False


# ── TestRegulationSuccess ───────────────────────────────────


class TestRegulationSuccess:
    def test_positive_shift_is_success(self, trainer):
        # Baseline: equal alpha. Evaluate: more right alpha -> positive FAA shift
        trainer.set_baseline(
            _make_eeg_seeded(seed=42, alpha_amp_left=10, alpha_amp_right=10)
        )
        result = trainer.evaluate(
            _make_eeg_seeded(seed=43, alpha_amp_left=5, alpha_amp_right=20),
            target_state="positive",
        )
        assert result["regulation_success"] is True
        assert result["faa_shift"] > 0

    def test_negative_shift_is_failure(self, trainer):
        # Baseline: equal. Evaluate: more left alpha -> negative FAA shift
        trainer.set_baseline(
            _make_eeg_seeded(seed=42, alpha_amp_left=10, alpha_amp_right=10)
        )
        result = trainer.evaluate(
            _make_eeg_seeded(seed=43, alpha_amp_left=20, alpha_amp_right=5),
            target_state="positive",
        )
        assert result["regulation_success"] is False
        assert result["faa_shift"] < 0

    def test_neutral_target_success_near_baseline(self, trainer):
        # For neutral target, staying close to baseline = success
        signals = _make_eeg_seeded(seed=42, alpha_amp_left=10, alpha_amp_right=10)
        trainer.set_baseline(signals)
        result = trainer.evaluate(signals, target_state="neutral")
        assert result["regulation_success"] is True


# ── TestStrategies ──────────────────────────────────────────


class TestStrategies:
    def test_all_strategies_returned(self, trainer):
        strategies = trainer.get_strategies()
        assert "positive" in strategies
        assert "neutral" in strategies
        assert "negative_high_arousal" in strategies
        assert "negative_low_arousal" in strategies

    def test_single_state_strategies(self, trainer):
        strategies = trainer.get_strategies(state="positive")
        assert len(strategies) == 1
        assert "positive" in strategies
        assert len(strategies["positive"]) >= 1

    def test_strategies_are_strings(self, trainer):
        strategies = trainer.get_strategies()
        for state, items in strategies.items():
            for s in items:
                assert isinstance(s, str)
                assert len(s) > 0

    def test_strategy_varies_by_faa_state(self, trainer):
        # Positive FAA should give positive strategy
        trainer.set_baseline(
            _make_eeg_seeded(seed=42, alpha_amp_left=10, alpha_amp_right=10)
        )
        # Strong positive FAA
        result_pos = trainer.evaluate(
            _make_eeg_seeded(seed=43, alpha_amp_left=3, alpha_amp_right=20),
            target_state="positive",
        )
        # Strong negative FAA
        result_neg = trainer.evaluate(
            _make_eeg_seeded(seed=44, alpha_amp_left=20, alpha_amp_right=3),
            target_state="positive",
        )
        # Strategies should differ for very different states
        assert result_pos["strategy_suggestion"] != result_neg["strategy_suggestion"]


# ── TestEffortIndex ─────────────────────────────────────────


class TestEffortIndex:
    def test_high_variability_high_effort(self, trainer):
        # Create signal with rapidly varying alpha amplitude (high effort)
        fs = 256
        duration = 4
        t = np.arange(int(fs * duration)) / fs
        signals = np.zeros((4, len(t)))
        np.random.seed(99)
        for ch in range(4):
            # Amplitude-modulated alpha: varies from 2 to 30 uV
            am = 15 + 13 * np.sin(2 * np.pi * 0.5 * t)  # slow AM
            signals[ch] = am * np.sin(2 * np.pi * 10 * t) + np.random.randn(len(t))

        trainer.set_baseline(_make_eeg_seeded(seed=42))
        result = trainer.evaluate(signals)
        assert result["effort_index"] > 0

    def test_constant_alpha_low_effort(self, trainer):
        # Constant alpha amplitude -> low variability -> low effort
        fs = 256
        duration = 4
        t = np.arange(int(fs * duration)) / fs
        signals = np.zeros((4, len(t)))
        for ch in range(4):
            signals[ch] = 10.0 * np.sin(2 * np.pi * 10 * t)

        trainer.set_baseline(_make_eeg_seeded(seed=42))
        result = trainer.evaluate(signals)
        # Pure sinusoid should have very low effort
        assert result["effort_index"] < 0.3

    def test_effort_index_bounded(self, trainer):
        trainer.set_baseline(_make_eeg_seeded(seed=42))
        for seed in range(50, 60):
            result = trainer.evaluate(_make_eeg_seeded(seed=seed))
            assert 0 <= result["effort_index"] <= 1


# ── TestSessionStats ────────────────────────────────────────


class TestSessionStats:
    def test_empty_stats(self, trainer):
        stats = trainer.get_session_stats()
        assert stats["n_epochs"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["has_baseline"] is False

    def test_stats_after_evaluations(self, trainer):
        trainer.set_baseline(_make_eeg_seeded(seed=42))
        for seed in range(10, 16):
            trainer.evaluate(_make_eeg_seeded(seed=seed))
        stats = trainer.get_session_stats()
        assert stats["n_epochs"] == 6
        assert "success_rate" in stats
        assert "mean_score" in stats
        assert "mean_faa_shift" in stats

    def test_success_rate_range(self, trainer):
        trainer.set_baseline(
            _make_eeg_seeded(seed=42, alpha_amp_left=10, alpha_amp_right=10)
        )
        # Mix of positive and negative shifts
        for amp_l, amp_r, seed in [(5, 20, 1), (20, 5, 2), (5, 20, 3), (10, 10, 4)]:
            trainer.evaluate(
                _make_eeg_seeded(seed=seed, alpha_amp_left=amp_l, alpha_amp_right=amp_r)
            )
        stats = trainer.get_session_stats()
        assert 0.0 <= stats["success_rate"] <= 1.0

    def test_trend_after_enough_data(self, trainer):
        trainer.set_baseline(
            _make_eeg_seeded(seed=42, alpha_amp_left=10, alpha_amp_right=10)
        )
        # Improving: gradually increasing right alpha
        for i in range(20):
            trainer.evaluate(
                _make_eeg_seeded(
                    seed=100 + i,
                    alpha_amp_left=10,
                    alpha_amp_right=10 + i * 0.8,
                )
            )
        stats = trainer.get_session_stats()
        assert stats["trend"] in ("improving", "stable", "declining")


# ── TestHistory ─────────────────────────────────────────────


class TestHistory:
    def test_empty_history(self, trainer):
        history = trainer.get_history()
        assert history == []

    def test_history_grows(self, trainer):
        trainer.set_baseline(_make_eeg_seeded(seed=42))
        trainer.evaluate(_make_eeg_seeded(seed=10))
        trainer.evaluate(_make_eeg_seeded(seed=20))
        trainer.evaluate(_make_eeg_seeded(seed=30))
        history = trainer.get_history()
        assert len(history) == 3

    def test_last_n(self, trainer):
        trainer.set_baseline(_make_eeg_seeded(seed=42))
        for seed in range(10, 20):
            trainer.evaluate(_make_eeg_seeded(seed=seed))
        last_3 = trainer.get_history(last_n=3)
        assert len(last_3) == 3
        full = trainer.get_history()
        assert last_3 == full[-3:]

    def test_last_n_larger_than_history(self, trainer):
        trainer.set_baseline(_make_eeg_seeded(seed=42))
        trainer.evaluate(_make_eeg_seeded(seed=10))
        last_100 = trainer.get_history(last_n=100)
        assert len(last_100) == 1


# ── TestMultiUser ───────────────────────────────────────────


class TestMultiUser:
    def test_independent_users(self, trainer):
        trainer.set_baseline(_make_eeg_seeded(seed=42), user_id="alice")
        trainer.evaluate(_make_eeg_seeded(seed=10), user_id="alice")
        trainer.evaluate(_make_eeg_seeded(seed=11), user_id="alice")

        trainer.set_baseline(_make_eeg_seeded(seed=50), user_id="bob")
        trainer.evaluate(_make_eeg_seeded(seed=60), user_id="bob")

        alice_stats = trainer.get_session_stats("alice")
        bob_stats = trainer.get_session_stats("bob")
        assert alice_stats["n_epochs"] == 2
        assert bob_stats["n_epochs"] == 1

    def test_independent_histories(self, trainer):
        trainer.set_baseline(_make_eeg_seeded(seed=42), user_id="a")
        trainer.evaluate(_make_eeg_seeded(seed=10), user_id="a")

        trainer.set_baseline(_make_eeg_seeded(seed=50), user_id="b")

        assert len(trainer.get_history(user_id="a")) == 1
        assert len(trainer.get_history(user_id="b")) == 0


# ── TestReset ───────────────────────────────────────────────


class TestReset:
    def test_reset_clears_all_state(self, trainer):
        trainer.set_baseline(_make_eeg_seeded(seed=42))
        trainer.evaluate(_make_eeg_seeded(seed=10))
        trainer.evaluate(_make_eeg_seeded(seed=20))
        trainer.reset()
        stats = trainer.get_session_stats()
        assert stats["n_epochs"] == 0
        assert stats["has_baseline"] is False
        assert trainer.get_history() == []

    def test_reset_one_user_preserves_other(self, trainer):
        trainer.set_baseline(_make_eeg_seeded(seed=42), user_id="a")
        trainer.evaluate(_make_eeg_seeded(seed=10), user_id="a")

        trainer.set_baseline(_make_eeg_seeded(seed=50), user_id="b")
        trainer.evaluate(_make_eeg_seeded(seed=60), user_id="b")

        trainer.reset(user_id="a")
        assert trainer.get_session_stats("a")["n_epochs"] == 0
        assert trainer.get_session_stats("b")["n_epochs"] == 1

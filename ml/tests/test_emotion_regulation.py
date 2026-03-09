"""Tests for closed-loop emotion regulation biofeedback."""
import numpy as np
import pytest

from models.emotion_regulation import (
    EmotionRegulationTrainer,
    EmotionRegulationBiofeedback,
    get_emotion_regulation_biofeedback,
)


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


# ════════════════════════════════════════════════════════════════════════════
# Tests for EmotionRegulationBiofeedback (new closed-loop biofeedback class)
# ════════════════════════════════════════════════════════════════════════════

FS_BFB = 256
VALID_EMOTION_STATES = {"anxious", "calm", "focused", "stressed", "neutral"}
VALID_BIOFEEDBACK_CUES = {"increase_alpha", "decrease_beta", "balanced"}
VALID_REGULATION_TRENDS = {"improving", "declining", "stable"}
REQUIRED_PREDICT_KEYS = {
    "emotion_state",
    "regulation_score",
    "biofeedback_cue",
    "alpha_asymmetry",
    "anxiety_index",
    "regulation_trend",
    "session_duration",
}
REQUIRED_SUMMARY_KEYS = {
    "mean_regulation_score",
    "peak_score",
    "session_count",
    "dominant_state",
}


def _make_bfb_eeg(
    n_channels: int = 4,
    duration_s: float = 2.0,
    dominant_hz: float = 10.0,
    amplitude: float = 10.0,
) -> np.ndarray:
    """Synthetic sinusoidal EEG for biofeedback tests."""
    n = int(FS_BFB * duration_s)
    t = np.linspace(0, duration_s, n)
    sig = amplitude * np.sin(2 * np.pi * dominant_hz * t)
    if n_channels == 1:
        return sig.astype(np.float32).reshape(1, -1)
    return np.tile(sig, (n_channels, 1)).astype(np.float32)


def _make_stressed_bfb_eeg() -> np.ndarray:
    """EEG with high theta and high beta."""
    n = int(FS_BFB * 2.0)
    t = np.linspace(0, 2.0, n)
    theta = 20.0 * np.sin(2 * np.pi * 6 * t)
    beta = 15.0 * np.sin(2 * np.pi * 22 * t)
    alpha = 2.0 * np.sin(2 * np.pi * 10 * t)
    sig = (theta + beta + alpha).astype(np.float32)
    return np.tile(sig, (4, 1))


def _make_calm_bfb_eeg() -> np.ndarray:
    """EEG with dominant alpha and low beta."""
    n = int(FS_BFB * 2.0)
    t = np.linspace(0, 2.0, n)
    alpha = 20.0 * np.sin(2 * np.pi * 10 * t)
    theta = 2.0 * np.sin(2 * np.pi * 6 * t)
    beta = 2.0 * np.sin(2 * np.pi * 18 * t)
    sig = (alpha + theta + beta).astype(np.float32)
    return np.tile(sig, (4, 1))


@pytest.fixture
def bfb_model():
    """Fresh EmotionRegulationBiofeedback for each test."""
    return EmotionRegulationBiofeedback()


# ── Init state ────────────────────────────────────────────────────────────────

class TestBiofeedbackInit:
    def test_session_count_starts_at_zero(self, bfb_model):
        summary = bfb_model.get_session_summary()
        assert summary["session_count"] == 0

    def test_session_summary_returns_zeros_when_empty(self, bfb_model):
        summary = bfb_model.get_session_summary()
        assert summary["mean_regulation_score"] == 0.0
        assert summary["peak_score"] == 0.0

    def test_session_duration_starts_at_zero(self, bfb_model):
        eeg = _make_bfb_eeg()
        result = bfb_model.predict(eeg, fs=FS_BFB)
        assert result["session_duration"] == 0

    def test_regulation_trend_starts_stable(self, bfb_model):
        eeg = _make_bfb_eeg()
        result = bfb_model.predict(eeg, fs=FS_BFB)
        assert result["regulation_trend"] == "stable"


# ── predict() output structure ─────────────────────────────────────────────

class TestBiofeedbackPredictOutputKeys:
    def test_predict_returns_all_required_keys(self, bfb_model):
        eeg = _make_bfb_eeg()
        result = bfb_model.predict(eeg, fs=FS_BFB)
        assert REQUIRED_PREDICT_KEYS.issubset(result.keys()), (
            f"Missing keys: {REQUIRED_PREDICT_KEYS - result.keys()}"
        )

    def test_emotion_state_is_valid_string(self, bfb_model):
        eeg = _make_bfb_eeg()
        result = bfb_model.predict(eeg, fs=FS_BFB)
        assert result["emotion_state"] in VALID_EMOTION_STATES

    def test_biofeedback_cue_is_valid(self, bfb_model):
        eeg = _make_bfb_eeg()
        result = bfb_model.predict(eeg, fs=FS_BFB)
        assert result["biofeedback_cue"] in VALID_BIOFEEDBACK_CUES

    def test_regulation_trend_is_valid(self, bfb_model):
        eeg = _make_bfb_eeg()
        result = bfb_model.predict(eeg, fs=FS_BFB)
        assert result["regulation_trend"] in VALID_REGULATION_TRENDS

    def test_regulation_score_is_float(self, bfb_model):
        eeg = _make_bfb_eeg()
        result = bfb_model.predict(eeg, fs=FS_BFB)
        assert isinstance(result["regulation_score"], float)

    def test_anxiety_index_is_float(self, bfb_model):
        eeg = _make_bfb_eeg()
        result = bfb_model.predict(eeg, fs=FS_BFB)
        assert isinstance(result["anxiety_index"], float)

    def test_alpha_asymmetry_is_float(self, bfb_model):
        eeg = _make_bfb_eeg()
        result = bfb_model.predict(eeg, fs=FS_BFB)
        assert isinstance(result["alpha_asymmetry"], float)

    def test_session_duration_is_int(self, bfb_model):
        eeg = _make_bfb_eeg()
        result = bfb_model.predict(eeg, fs=FS_BFB)
        assert isinstance(result["session_duration"], int)


# ── predict() value ranges ─────────────────────────────────────────────────

class TestBiofeedbackPredictValueRanges:
    def test_regulation_score_in_range_0_1(self, bfb_model):
        eeg = _make_bfb_eeg()
        result = bfb_model.predict(eeg, fs=FS_BFB)
        assert 0.0 <= result["regulation_score"] <= 1.0

    def test_anxiety_index_in_range_0_1(self, bfb_model):
        eeg = _make_bfb_eeg()
        result = bfb_model.predict(eeg, fs=FS_BFB)
        assert 0.0 <= result["anxiety_index"] <= 1.0

    def test_regulation_score_non_negative_across_inputs(self, bfb_model):
        for hz in [4.0, 8.0, 10.0, 20.0, 25.0]:
            eeg = _make_bfb_eeg(dominant_hz=hz)
            result = bfb_model.predict(eeg, fs=FS_BFB)
            assert result["regulation_score"] >= 0.0

    def test_regulation_score_at_most_one_across_inputs(self, bfb_model):
        for hz in [4.0, 8.0, 10.0, 20.0, 25.0]:
            eeg = _make_bfb_eeg(dominant_hz=hz)
            result = bfb_model.predict(eeg, fs=FS_BFB)
            assert result["regulation_score"] <= 1.0

    def test_anxiety_index_non_negative(self, bfb_model):
        eeg = _make_bfb_eeg()
        result = bfb_model.predict(eeg, fs=FS_BFB)
        assert result["anxiety_index"] >= 0.0

    def test_anxiety_index_at_most_one(self, bfb_model):
        eeg = _make_bfb_eeg()
        result = bfb_model.predict(eeg, fs=FS_BFB)
        assert result["anxiety_index"] <= 1.0


# ── Input shape handling ───────────────────────────────────────────────────

class TestBiofeedbackInputShapes:
    def test_predict_1d_input(self, bfb_model):
        eeg_1d = _make_bfb_eeg(n_channels=1, duration_s=2.0).reshape(-1)
        result = bfb_model.predict(eeg_1d, fs=FS_BFB)
        assert REQUIRED_PREDICT_KEYS.issubset(result.keys())

    def test_predict_2ch_input(self, bfb_model):
        eeg = _make_bfb_eeg(n_channels=2)
        result = bfb_model.predict(eeg, fs=FS_BFB)
        assert REQUIRED_PREDICT_KEYS.issubset(result.keys())

    def test_predict_4ch_input(self, bfb_model):
        eeg = _make_bfb_eeg(n_channels=4)
        result = bfb_model.predict(eeg, fs=FS_BFB)
        assert REQUIRED_PREDICT_KEYS.issubset(result.keys())


# ── update_session() ───────────────────────────────────────────────────────

class TestBiofeedbackUpdateSession:
    def test_update_session_increments_session_count(self, bfb_model):
        eeg = _make_bfb_eeg()
        bfb_model.update_session(eeg, fs=FS_BFB, duration_sec=1)
        bfb_model.update_session(eeg, fs=FS_BFB, duration_sec=1)
        summary = bfb_model.get_session_summary()
        assert summary["session_count"] == 2

    def test_update_session_accumulates_duration(self, bfb_model):
        eeg = _make_bfb_eeg()
        bfb_model.update_session(eeg, fs=FS_BFB, duration_sec=3)
        bfb_model.update_session(eeg, fs=FS_BFB, duration_sec=5)
        result = bfb_model.predict(eeg, fs=FS_BFB)
        assert result["session_duration"] == 8

    def test_update_session_includes_session_count_key(self, bfb_model):
        eeg = _make_bfb_eeg()
        result = bfb_model.update_session(eeg, fs=FS_BFB, duration_sec=1)
        assert "session_count" in result

    def test_update_session_count_starts_at_1(self, bfb_model):
        eeg = _make_bfb_eeg()
        result = bfb_model.update_session(eeg, fs=FS_BFB, duration_sec=1)
        assert result["session_count"] == 1

    def test_update_session_increments_monotonically(self, bfb_model):
        eeg = _make_bfb_eeg()
        for i in range(1, 6):
            result = bfb_model.update_session(eeg, fs=FS_BFB, duration_sec=1)
            assert result["session_count"] == i


# ── get_session_summary() ──────────────────────────────────────────────────

class TestBiofeedbackGetSessionSummary:
    def test_summary_has_required_keys(self, bfb_model):
        eeg = _make_bfb_eeg()
        bfb_model.update_session(eeg, fs=FS_BFB, duration_sec=1)
        summary = bfb_model.get_session_summary()
        assert REQUIRED_SUMMARY_KEYS.issubset(summary.keys()), (
            f"Missing keys: {REQUIRED_SUMMARY_KEYS - summary.keys()}"
        )

    def test_summary_session_count_matches_updates(self, bfb_model):
        eeg = _make_bfb_eeg()
        for _ in range(7):
            bfb_model.update_session(eeg, fs=FS_BFB, duration_sec=1)
        summary = bfb_model.get_session_summary()
        assert summary["session_count"] == 7

    def test_summary_dominant_state_is_valid(self, bfb_model):
        eeg = _make_bfb_eeg()
        for _ in range(3):
            bfb_model.update_session(eeg, fs=FS_BFB, duration_sec=1)
        summary = bfb_model.get_session_summary()
        assert summary["dominant_state"] in VALID_EMOTION_STATES

    def test_summary_mean_score_in_range(self, bfb_model):
        eeg = _make_bfb_eeg()
        for _ in range(5):
            bfb_model.update_session(eeg, fs=FS_BFB, duration_sec=1)
        summary = bfb_model.get_session_summary()
        assert 0.0 <= summary["mean_regulation_score"] <= 1.0

    def test_summary_peak_score_gte_mean(self, bfb_model):
        eeg = _make_bfb_eeg()
        for _ in range(5):
            bfb_model.update_session(eeg, fs=FS_BFB, duration_sec=1)
        summary = bfb_model.get_session_summary()
        assert summary["peak_score"] >= summary["mean_regulation_score"] - 1e-9

    def test_summary_empty_session_returns_defaults(self, bfb_model):
        summary = bfb_model.get_session_summary()
        assert summary["session_count"] == 0
        assert summary["dominant_state"] == "neutral"


# ── reset() ───────────────────────────────────────────────────────────────

class TestBiofeedbackReset:
    def test_reset_clears_session_count(self, bfb_model):
        eeg = _make_bfb_eeg()
        for _ in range(4):
            bfb_model.update_session(eeg, fs=FS_BFB, duration_sec=1)
        bfb_model.reset()
        summary = bfb_model.get_session_summary()
        assert summary["session_count"] == 0

    def test_reset_clears_session_scores(self, bfb_model):
        eeg = _make_bfb_eeg()
        for _ in range(4):
            bfb_model.update_session(eeg, fs=FS_BFB, duration_sec=1)
        bfb_model.reset()
        summary = bfb_model.get_session_summary()
        assert summary["mean_regulation_score"] == 0.0
        assert summary["peak_score"] == 0.0

    def test_reset_clears_duration(self, bfb_model):
        eeg = _make_bfb_eeg()
        bfb_model.update_session(eeg, fs=FS_BFB, duration_sec=10)
        bfb_model.reset()
        result = bfb_model.predict(eeg, fs=FS_BFB)
        assert result["session_duration"] == 0

    def test_reset_allows_new_session_to_start(self, bfb_model):
        eeg = _make_bfb_eeg()
        for _ in range(3):
            bfb_model.update_session(eeg, fs=FS_BFB, duration_sec=1)
        bfb_model.reset()
        bfb_model.update_session(eeg, fs=FS_BFB, duration_sec=1)
        summary = bfb_model.get_session_summary()
        assert summary["session_count"] == 1


# ── Singleton factory ──────────────────────────────────────────────────────

class TestBiofeedbackSingleton:
    def test_singleton_same_user_returns_same_instance(self):
        # Use unique IDs to avoid cross-test interference from other tests
        a = get_emotion_regulation_biofeedback("bfb_test_user_A_v2")
        b = get_emotion_regulation_biofeedback("bfb_test_user_A_v2")
        assert a is b

    def test_singleton_different_users_return_different_instances(self):
        x = get_emotion_regulation_biofeedback("bfb_user_X_unique_v2")
        y = get_emotion_regulation_biofeedback("bfb_user_Y_unique_v2")
        assert x is not y

    def test_singleton_default_user_is_consistent(self):
        m1 = get_emotion_regulation_biofeedback()
        m2 = get_emotion_regulation_biofeedback("default")
        assert m1 is m2

    def test_singleton_is_correct_class(self):
        m = get_emotion_regulation_biofeedback("bfb_type_check_v2")
        assert isinstance(m, EmotionRegulationBiofeedback)


# ── Trend detection ────────────────────────────────────────────────────────

class TestBiofeedbackRegulationTrend:
    def test_trend_is_stable_with_insufficient_data(self, bfb_model):
        eeg = _make_bfb_eeg()
        result = bfb_model.predict(eeg, fs=FS_BFB)
        assert result["regulation_trend"] == "stable"

    def test_trend_is_valid_after_many_updates(self, bfb_model):
        eeg = _make_bfb_eeg()
        for _ in range(15):
            bfb_model.update_session(eeg, fs=FS_BFB, duration_sec=1)
        result = bfb_model.predict(eeg, fs=FS_BFB)
        assert result["regulation_trend"] in VALID_REGULATION_TRENDS


# ── EMA smoothing and different inputs ────────────────────────────────────

class TestBiofeedbackEMASmoothing:
    def test_predict_stable_score_on_same_input(self, bfb_model):
        eeg = _make_bfb_eeg()
        results = [bfb_model.predict(eeg, fs=FS_BFB) for _ in range(5)]
        scores = [r["regulation_score"] for r in results]
        # On constant input EMA converges; scores should be close
        assert max(scores) - min(scores) < 0.5

    def test_predict_calm_vs_stressed_differ_in_at_least_one_metric(self):
        calm = _make_calm_bfb_eeg()
        stressed = _make_stressed_bfb_eeg()
        m_calm = EmotionRegulationBiofeedback()
        m_stress = EmotionRegulationBiofeedback()
        r_calm = m_calm.predict(calm, fs=FS_BFB)
        r_stressed = m_stress.predict(stressed, fs=FS_BFB)
        assert (
            r_calm["regulation_score"] != r_stressed["regulation_score"]
            or r_calm["anxiety_index"] != r_stressed["anxiety_index"]
            or r_calm["alpha_asymmetry"] != r_stressed["alpha_asymmetry"]
        )

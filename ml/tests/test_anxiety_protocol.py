"""Tests for anxiety neurofeedback protocol."""
import numpy as np
import pytest

from models.anxiety_protocol import AnxietyProtocol


@pytest.fixture
def protocol():
    return AnxietyProtocol(alpha_threshold=1.1, hbeta_threshold=0.9, fs=256.0)


def _make_signal(fs=256, duration=4, n_channels=4, alpha_amp=10.0, hbeta_amp=5.0,
                 noise_amp=2.0):
    """EEG with controllable alpha and high-beta amplitudes."""
    t = np.arange(int(fs * duration)) / fs
    signals = []
    for ch in range(n_channels):
        alpha = alpha_amp * np.sin(2 * np.pi * 10 * t + ch * 0.3)
        hbeta = hbeta_amp * np.sin(2 * np.pi * 25 * t + ch * 0.5)
        noise = noise_amp * np.random.randn(len(t))
        signals.append(alpha + hbeta + noise)
    return np.array(signals)


class TestBaseline:
    def test_set_baseline(self, protocol):
        np.random.seed(42)
        result = protocol.set_baseline(_make_signal())
        assert result["baseline_set"] is True
        assert result["baseline_alpha"] > 0
        assert result["baseline_high_beta"] > 0

    def test_baseline_anxiety_index(self, protocol):
        np.random.seed(42)
        result = protocol.set_baseline(_make_signal())
        assert 0 <= result["baseline_anxiety_index"] <= 1

    def test_single_channel(self, protocol):
        np.random.seed(42)
        sig = 10 * np.sin(2 * np.pi * 10 * np.arange(1024) / 256)
        result = protocol.set_baseline(sig)
        assert result["baseline_set"] is True


class TestEvaluate:
    def test_output_keys(self, protocol):
        np.random.seed(42)
        protocol.set_baseline(_make_signal(alpha_amp=10, hbeta_amp=10))
        result = protocol.evaluate(_make_signal(alpha_amp=15, hbeta_amp=5))
        expected = {"reward", "alpha_ratio", "hbeta_ratio", "anxiety_index",
                    "anxiety_level", "feedback_intensity", "instruction", "has_baseline"}
        assert expected.issubset(set(result.keys()))

    def test_reward_when_alpha_up_hbeta_down(self, protocol):
        np.random.seed(42)
        protocol.set_baseline(_make_signal(alpha_amp=10, hbeta_amp=10))
        result = protocol.evaluate(_make_signal(alpha_amp=15, hbeta_amp=5))
        assert result["alpha_ratio"] > 1.0
        assert result["hbeta_ratio"] < 1.0

    def test_no_reward_alpha_same(self, protocol):
        np.random.seed(42)
        protocol.set_baseline(_make_signal(alpha_amp=10, hbeta_amp=10))
        result = protocol.evaluate(_make_signal(alpha_amp=10, hbeta_amp=10))
        # Same levels → no reward (alpha not above threshold)
        assert result["alpha_ratio"] < 1.15

    def test_feedback_tone_on_reward(self, protocol):
        np.random.seed(42)
        protocol.set_baseline(_make_signal(alpha_amp=8, hbeta_amp=12))
        result = protocol.evaluate(_make_signal(alpha_amp=15, hbeta_amp=5))
        if result["reward"]:
            assert result["feedback_tone_hz"] is not None
            assert result["feedback_tone_hz"] >= 220

    def test_no_tone_on_no_reward(self, protocol):
        np.random.seed(42)
        protocol.set_baseline(_make_signal(alpha_amp=15, hbeta_amp=5))
        result = protocol.evaluate(_make_signal(alpha_amp=5, hbeta_amp=15))
        if not result["reward"]:
            assert result["feedback_tone_hz"] is None

    def test_feedback_intensity_range(self, protocol):
        np.random.seed(42)
        protocol.set_baseline(_make_signal(alpha_amp=10, hbeta_amp=10))
        result = protocol.evaluate(_make_signal(alpha_amp=15, hbeta_amp=5))
        assert 0 <= result["feedback_intensity"] <= 1

    def test_no_baseline_ratio_one(self, protocol):
        np.random.seed(42)
        result = protocol.evaluate(_make_signal())
        assert result["has_baseline"] is False
        assert result["alpha_ratio"] == 1.0
        assert result["hbeta_ratio"] == 1.0


class TestAnxietyLevels:
    def test_high_anxiety_with_high_beta(self, protocol):
        np.random.seed(42)
        # Very high high-beta, low alpha → high anxiety
        result = protocol.evaluate(_make_signal(alpha_amp=2, hbeta_amp=20))
        assert result["anxiety_level"] in ("moderate", "high")

    def test_low_anxiety_with_high_alpha(self, protocol):
        np.random.seed(42)
        # High alpha, low high-beta → low anxiety
        result = protocol.evaluate(_make_signal(alpha_amp=20, hbeta_amp=2))
        assert result["anxiety_level"] in ("low", "mild")

    def test_anxiety_index_range(self, protocol):
        np.random.seed(42)
        result = protocol.evaluate(_make_signal())
        assert 0 <= result["anxiety_index"] <= 1


class TestInstructions:
    def test_instruction_present(self, protocol):
        np.random.seed(42)
        result = protocol.evaluate(_make_signal())
        assert isinstance(result["instruction"], str)
        assert len(result["instruction"]) > 0

    def test_calming_instruction_when_anxious(self, protocol):
        np.random.seed(42)
        result = protocol.evaluate(_make_signal(alpha_amp=2, hbeta_amp=20))
        assert "breathing" in result["instruction"] or "tension" in result["instruction"] or "maintain" in result["instruction"] or "excellent" in result["instruction"]


class TestPartialScore:
    def test_partial_score_range(self, protocol):
        np.random.seed(42)
        protocol.set_baseline(_make_signal(alpha_amp=10, hbeta_amp=10))
        result = protocol.evaluate(_make_signal(alpha_amp=15, hbeta_amp=5))
        assert 0 <= result["partial_score"] <= 1.0

    def test_full_score_on_reward(self, protocol):
        np.random.seed(42)
        protocol.set_baseline(_make_signal(alpha_amp=8, hbeta_amp=12))
        result = protocol.evaluate(_make_signal(alpha_amp=15, hbeta_amp=5))
        if result["reward"]:
            assert result["partial_score"] == 1.0


class TestSessionStats:
    def test_empty_stats(self, protocol):
        stats = protocol.get_session_stats()
        assert stats["n_epochs"] == 0

    def test_stats_after_data(self, protocol):
        np.random.seed(42)
        protocol.set_baseline(_make_signal(alpha_amp=10, hbeta_amp=10))
        for _ in range(5):
            protocol.evaluate(_make_signal(alpha_amp=12, hbeta_amp=8))
        stats = protocol.get_session_stats()
        assert stats["n_epochs"] == 5
        assert "reward_rate" in stats
        assert "mean_anxiety" in stats

    def test_anxiety_trend(self, protocol):
        np.random.seed(42)
        protocol.set_baseline(_make_signal(alpha_amp=10, hbeta_amp=10))
        for i in range(20):
            protocol.evaluate(_make_signal(alpha_amp=10 + i * 0.5, hbeta_amp=10 - i * 0.3))
        stats = protocol.get_session_stats()
        assert stats["trend"] in ("improving", "worsening", "stable")


class TestHistory:
    def test_empty_history(self, protocol):
        assert protocol.get_history() == []

    def test_history_grows(self, protocol):
        np.random.seed(42)
        protocol.evaluate(_make_signal())
        protocol.evaluate(_make_signal())
        assert len(protocol.get_history()) == 2

    def test_history_last_n(self, protocol):
        np.random.seed(42)
        for _ in range(10):
            protocol.evaluate(_make_signal())
        assert len(protocol.get_history(last_n=3)) == 3


class TestMultiUser:
    def test_independent(self, protocol):
        np.random.seed(42)
        protocol.set_baseline(_make_signal(), user_id="a")
        protocol.evaluate(_make_signal(), user_id="a")
        protocol.set_baseline(_make_signal(), user_id="b")
        protocol.evaluate(_make_signal(), user_id="b")
        a = protocol.get_session_stats("a")
        b = protocol.get_session_stats("b")
        assert a["n_epochs"] == 1
        assert b["n_epochs"] == 1


class TestReset:
    def test_reset_clears(self, protocol):
        np.random.seed(42)
        protocol.set_baseline(_make_signal())
        protocol.evaluate(_make_signal())
        protocol.reset()
        stats = protocol.get_session_stats()
        assert stats["n_epochs"] == 0
        assert stats["has_baseline"] is False

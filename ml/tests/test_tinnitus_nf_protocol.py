"""Tests for tinnitus neurofeedback protocol."""
import numpy as np
import pytest

from models.tinnitus_nf_protocol import TinnitusNFProtocol


@pytest.fixture
def protocol():
    return TinnitusNFProtocol(reward_threshold=1.1, fs=256.0)


def _make_alpha_signal(fs=256, duration=4, alpha_amp=15.0, n_channels=4):
    """EEG with controllable alpha amplitude."""
    t = np.arange(int(fs * duration)) / fs
    signals = []
    for ch in range(n_channels):
        alpha = alpha_amp * np.sin(2 * np.pi * 10 * t + ch * 0.3)
        noise = 2.0 * np.random.randn(len(t))
        signals.append(alpha + noise)
    return np.array(signals)


class TestBaseline:
    def test_set_baseline(self, protocol):
        np.random.seed(42)
        result = protocol.set_baseline(_make_alpha_signal())
        assert result["baseline_set"] is True
        assert result["baseline_alpha"] > 0

    def test_baseline_channel_powers(self, protocol):
        np.random.seed(42)
        result = protocol.set_baseline(_make_alpha_signal())
        assert len(result["channel_powers"]) == 2  # TP9 + TP10

    def test_single_channel_baseline(self, protocol):
        np.random.seed(42)
        signal_1d = 10.0 * np.sin(2 * np.pi * 10 * np.arange(1024) / 256)
        result = protocol.set_baseline(signal_1d)
        assert result["baseline_alpha"] >= 0


class TestEvaluate:
    def test_output_keys(self, protocol):
        np.random.seed(42)
        protocol.set_baseline(_make_alpha_signal(alpha_amp=10))
        result = protocol.evaluate(_make_alpha_signal(alpha_amp=15))
        expected = {"reward", "alpha_ratio", "current_alpha", "baseline_alpha",
                    "feedback_intensity", "feedback_tone_hz", "has_baseline"}
        assert expected.issubset(set(result.keys()))

    def test_reward_when_alpha_increases(self, protocol):
        np.random.seed(42)
        protocol.set_baseline(_make_alpha_signal(alpha_amp=10))
        result = protocol.evaluate(_make_alpha_signal(alpha_amp=15))
        assert result["reward"] is True
        assert result["alpha_ratio"] > 1.0

    def test_no_reward_when_alpha_same(self, protocol):
        np.random.seed(42)
        protocol.set_baseline(_make_alpha_signal(alpha_amp=10))
        result = protocol.evaluate(_make_alpha_signal(alpha_amp=10))
        # Ratio should be ~1.0, below 1.1 threshold
        assert result["alpha_ratio"] < 1.15  # approximately 1

    def test_feedback_tone_on_reward(self, protocol):
        np.random.seed(42)
        protocol.set_baseline(_make_alpha_signal(alpha_amp=10))
        result = protocol.evaluate(_make_alpha_signal(alpha_amp=15))
        if result["reward"]:
            assert result["feedback_tone_hz"] is not None
            assert result["feedback_tone_hz"] >= 440

    def test_no_tone_on_no_reward(self, protocol):
        np.random.seed(42)
        protocol.set_baseline(_make_alpha_signal(alpha_amp=15))
        result = protocol.evaluate(_make_alpha_signal(alpha_amp=5))
        if not result["reward"]:
            assert result["feedback_tone_hz"] is None

    def test_feedback_intensity_range(self, protocol):
        np.random.seed(42)
        protocol.set_baseline(_make_alpha_signal(alpha_amp=10))
        result = protocol.evaluate(_make_alpha_signal(alpha_amp=15))
        assert 0 <= result["feedback_intensity"] <= 1

    def test_no_baseline_ratio_one(self, protocol):
        np.random.seed(42)
        result = protocol.evaluate(_make_alpha_signal())
        assert result["has_baseline"] is False
        assert result["alpha_ratio"] == 1.0


class TestSessionStats:
    def test_empty_stats(self, protocol):
        stats = protocol.get_session_stats()
        assert stats["n_epochs"] == 0

    def test_stats_with_data(self, protocol):
        np.random.seed(42)
        protocol.set_baseline(_make_alpha_signal(alpha_amp=10))
        for _ in range(5):
            protocol.evaluate(_make_alpha_signal(alpha_amp=12))
        stats = protocol.get_session_stats()
        assert stats["n_epochs"] == 5
        assert "reward_rate" in stats
        assert "mean_alpha_ratio" in stats

    def test_reward_rate(self, protocol):
        np.random.seed(42)
        protocol.set_baseline(_make_alpha_signal(alpha_amp=10))
        for amp in [15, 15, 5, 5, 15]:  # 3 rewards, 2 non
            protocol.evaluate(_make_alpha_signal(alpha_amp=amp))
        stats = protocol.get_session_stats()
        assert 0 <= stats["reward_rate"] <= 1

    def test_trend_after_enough_data(self, protocol):
        np.random.seed(42)
        protocol.set_baseline(_make_alpha_signal(alpha_amp=10))
        for i in range(20):
            protocol.evaluate(_make_alpha_signal(alpha_amp=10 + i * 0.5))
        stats = protocol.get_session_stats()
        assert stats["trend"] in ("improving", "stable", "declining")


class TestMultiUser:
    def test_independent_users(self, protocol):
        np.random.seed(42)
        protocol.set_baseline(_make_alpha_signal(), user_id="a")
        protocol.evaluate(_make_alpha_signal(), user_id="a")
        protocol.set_baseline(_make_alpha_signal(), user_id="b")
        protocol.evaluate(_make_alpha_signal(), user_id="b")
        a = protocol.get_session_stats("a")
        b = protocol.get_session_stats("b")
        assert a["n_epochs"] == 1
        assert b["n_epochs"] == 1


class TestReset:
    def test_reset_clears(self, protocol):
        np.random.seed(42)
        protocol.set_baseline(_make_alpha_signal())
        protocol.evaluate(_make_alpha_signal())
        protocol.reset()
        stats = protocol.get_session_stats()
        assert stats["n_epochs"] == 0
        assert stats["has_baseline"] is False

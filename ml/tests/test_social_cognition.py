"""Tests for social cognition detector."""
import numpy as np
import pytest

from models.social_cognition import SocialCognitionDetector


@pytest.fixture
def detector():
    return SocialCognitionDetector(fs=256.0)


def _make_signal(fs=256, duration=4, n_channels=4, mu_amp=10.0, theta_amp=5.0,
                 beta_amp=5.0, noise_amp=2.0):
    """EEG with controllable mu, theta, and beta amplitudes."""
    t = np.arange(int(fs * duration)) / fs
    signals = []
    for ch in range(n_channels):
        # Mu at temporal (ch0, ch3), theta at frontal (ch1, ch2)
        if ch in (0, 3):
            mu = mu_amp * np.sin(2 * np.pi * 10 * t + ch * 0.3)
        else:
            mu = (mu_amp * 0.5) * np.sin(2 * np.pi * 10 * t + ch * 0.3)
        theta = theta_amp * np.sin(2 * np.pi * 6 * t + ch * 0.5)
        beta = beta_amp * np.sin(2 * np.pi * 20 * t + ch * 0.7)
        noise = noise_amp * np.random.randn(len(t))
        signals.append(mu + theta + beta + noise)
    return np.array(signals)


class TestBaseline:
    def test_set_baseline(self, detector):
        np.random.seed(42)
        result = detector.set_baseline(_make_signal())
        assert result["baseline_set"] is True
        assert result["baseline_mu"] > 0
        assert result["baseline_theta"] > 0

    def test_single_channel(self, detector):
        np.random.seed(42)
        sig = 10 * np.sin(2 * np.pi * 10 * np.arange(1024) / 256)
        result = detector.set_baseline(sig)
        assert result["baseline_set"] is True


class TestAssess:
    def test_output_keys(self, detector):
        np.random.seed(42)
        detector.set_baseline(_make_signal())
        result = detector.assess(_make_signal())
        expected = {"empathy_index", "mentalizing_index", "social_engagement",
                    "mu_suppression", "social_state", "has_baseline", "recommendations"}
        assert expected.issubset(set(result.keys()))

    def test_empathy_range(self, detector):
        np.random.seed(42)
        result = detector.assess(_make_signal())
        assert 0 <= result["empathy_index"] <= 1

    def test_mentalizing_range(self, detector):
        np.random.seed(42)
        result = detector.assess(_make_signal())
        assert 0 <= result["mentalizing_index"] <= 1

    def test_social_engagement_range(self, detector):
        np.random.seed(42)
        result = detector.assess(_make_signal())
        assert 0 <= result["social_engagement"] <= 1

    def test_social_state_valid(self, detector):
        np.random.seed(42)
        result = detector.assess(_make_signal())
        assert result["social_state"] in (
            "deeply_engaged", "socially_attentive",
            "passively_observing", "socially_disengaged"
        )


class TestMuSuppression:
    def test_mu_suppression_with_baseline(self, detector):
        np.random.seed(42)
        detector.set_baseline(_make_signal(mu_amp=15))
        result = detector.assess(_make_signal(mu_amp=5))
        # Lower mu than baseline → suppression
        assert result["mu_suppression"] > 0
        assert result["mu_ratio"] < 1.0

    def test_no_suppression_same_mu(self, detector):
        np.random.seed(42)
        detector.set_baseline(_make_signal(mu_amp=10))
        result = detector.assess(_make_signal(mu_amp=10))
        # Similar mu → low suppression
        assert result["mu_ratio"] < 1.5  # approximately 1

    def test_suppression_range(self, detector):
        np.random.seed(42)
        detector.set_baseline(_make_signal(mu_amp=10))
        result = detector.assess(_make_signal(mu_amp=5))
        assert 0 <= result["mu_suppression"] <= 1


class TestSocialStates:
    def test_engaged_with_mu_suppression_and_theta(self, detector):
        np.random.seed(42)
        # Baseline: high mu, low theta
        detector.set_baseline(_make_signal(mu_amp=15, theta_amp=3))
        # During: low mu (suppression), high theta (mentalizing)
        result = detector.assess(_make_signal(mu_amp=5, theta_amp=15))
        assert result["social_state"] in ("deeply_engaged", "socially_attentive")

    def test_context_preserved(self, detector):
        np.random.seed(42)
        result = detector.assess(_make_signal(), context="empathy_task")
        assert result["context"] == "empathy_task"


class TestRecommendations:
    def test_recommendations_list(self, detector):
        np.random.seed(42)
        result = detector.assess(_make_signal())
        assert isinstance(result["recommendations"], list)


class TestSessionStats:
    def test_empty_stats(self, detector):
        stats = detector.get_session_stats()
        assert stats["n_epochs"] == 0

    def test_stats_after_data(self, detector):
        np.random.seed(42)
        detector.set_baseline(_make_signal())
        for _ in range(5):
            detector.assess(_make_signal())
        stats = detector.get_session_stats()
        assert stats["n_epochs"] == 5
        assert "mean_empathy" in stats
        assert "dominant_state" in stats


class TestHistory:
    def test_empty_history(self, detector):
        assert detector.get_history() == []

    def test_history_grows(self, detector):
        np.random.seed(42)
        detector.assess(_make_signal())
        detector.assess(_make_signal())
        assert len(detector.get_history()) == 2

    def test_history_last_n(self, detector):
        np.random.seed(42)
        for _ in range(10):
            detector.assess(_make_signal())
        assert len(detector.get_history(last_n=3)) == 3


class TestMultiUser:
    def test_independent(self, detector):
        np.random.seed(42)
        detector.assess(_make_signal(), user_id="a")
        detector.assess(_make_signal(), user_id="b")
        assert len(detector.get_history("a")) == 1
        assert len(detector.get_history("b")) == 1


class TestReset:
    def test_reset_clears(self, detector):
        np.random.seed(42)
        detector.set_baseline(_make_signal())
        detector.assess(_make_signal())
        detector.reset()
        assert detector.get_history() == []
        stats = detector.get_session_stats()
        assert stats["n_epochs"] == 0
        assert stats["has_baseline"] is False

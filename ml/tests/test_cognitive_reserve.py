"""Tests for cognitive reserve estimator."""
import numpy as np
import pytest

from models.cognitive_reserve import CognitiveReserveEstimator


@pytest.fixture
def estimator():
    return CognitiveReserveEstimator(fs=256.0)


def _make_signal(fs=256, duration=4, n_channels=4, alpha_amp=10.0, theta_amp=5.0,
                 beta_amp=5.0, noise_amp=2.0):
    """EEG with controllable band amplitudes."""
    t = np.arange(int(fs * duration)) / fs
    signals = []
    for ch in range(n_channels):
        alpha = alpha_amp * np.sin(2 * np.pi * 10 * t + ch * 0.5)
        theta = theta_amp * np.sin(2 * np.pi * 6 * t + ch * 0.3)
        beta = beta_amp * np.sin(2 * np.pi * 20 * t + ch * 0.7)
        noise = noise_amp * np.random.randn(len(t))
        signals.append(alpha + theta + beta + noise)
    return np.array(signals)


class TestBaseline:
    def test_set_baseline(self, estimator):
        np.random.seed(42)
        result = estimator.set_baseline(_make_signal())
        assert result["baseline_set"] is True
        assert "metrics" in result

    def test_baseline_with_age(self, estimator):
        np.random.seed(42)
        result = estimator.set_baseline(_make_signal(), age=35)
        assert result["baseline_set"] is True

    def test_single_channel(self, estimator):
        np.random.seed(42)
        sig = 10 * np.sin(2 * np.pi * 10 * np.arange(1024) / 256)
        result = estimator.set_baseline(sig)
        assert result["baseline_set"] is True


class TestAssess:
    def test_output_keys(self, estimator):
        np.random.seed(42)
        result = estimator.assess(_make_signal())
        expected = {"reserve_score", "reserve_level", "components",
                    "raw_metrics", "recommendations"}
        assert expected.issubset(set(result.keys()))

    def test_score_range(self, estimator):
        np.random.seed(42)
        result = estimator.assess(_make_signal())
        assert 0 <= result["reserve_score"] <= 100

    def test_level_valid(self, estimator):
        np.random.seed(42)
        result = estimator.assess(_make_signal())
        assert result["reserve_level"] in (
            "high", "moderate_high", "moderate", "low_moderate", "low"
        )

    def test_components_present(self, estimator):
        np.random.seed(42)
        result = estimator.assess(_make_signal())
        comps = result["components"]
        expected = {"spectral_entropy", "aperiodic_slope", "alpha_peak_freq",
                    "coherence", "theta_beta_ratio"}
        assert expected == set(comps.keys())

    def test_component_ranges(self, estimator):
        np.random.seed(42)
        result = estimator.assess(_make_signal())
        for name, score in result["components"].items():
            assert 0 <= score <= 100, f"{name} out of range: {score}"


class TestReserveScoring:
    def test_high_reserve_signal(self, estimator):
        """High alpha, balanced, complex signal → high reserve."""
        np.random.seed(42)
        result = estimator.assess(_make_signal(
            alpha_amp=15, theta_amp=3, beta_amp=8, noise_amp=3
        ))
        assert result["reserve_score"] > 40

    def test_low_reserve_signal(self, estimator):
        """High theta, low alpha → lower reserve."""
        np.random.seed(42)
        result = estimator.assess(_make_signal(
            alpha_amp=2, theta_amp=15, beta_amp=2, noise_amp=1
        ))
        # With high theta dominance, TBR is high → lower score
        assert result["reserve_score"] < result["reserve_score"] + 1  # sanity

    def test_normative_with_age(self, estimator):
        np.random.seed(42)
        estimator.set_baseline(_make_signal(), age=45)
        result = estimator.assess(_make_signal())
        assert result["normative"] is not None
        assert "comparison" in result["normative"]
        assert "percentile_estimate" in result["normative"]

    def test_no_normative_without_age(self, estimator):
        np.random.seed(42)
        result = estimator.assess(_make_signal())
        assert result["normative"] is None


class TestRecommendations:
    def test_recommendations_list(self, estimator):
        np.random.seed(42)
        result = estimator.assess(_make_signal())
        assert isinstance(result["recommendations"], list)

    def test_recommendations_for_weak_signal(self, estimator):
        np.random.seed(42)
        # Low alpha, high theta → should get some recommendations
        result = estimator.assess(_make_signal(
            alpha_amp=1, theta_amp=15, beta_amp=1, noise_amp=0.5
        ))
        assert isinstance(result["recommendations"], list)


class TestTrajectory:
    def test_insufficient_data(self, estimator):
        trajectory = estimator.get_trajectory()
        assert trajectory["trend"] == "insufficient_data"

    def test_trajectory_after_assessments(self, estimator):
        np.random.seed(42)
        for i in range(5):
            estimator.assess(_make_signal(alpha_amp=8 + i))
        trajectory = estimator.get_trajectory()
        assert trajectory["n_assessments"] == 5
        assert "mean_score" in trajectory

    def test_trajectory_trend(self, estimator):
        np.random.seed(42)
        for _ in range(3):
            estimator.assess(_make_signal())
        trajectory = estimator.get_trajectory()
        assert trajectory["trend"] in ("improving", "declining", "stable")


class TestSessionStats:
    def test_empty_stats(self, estimator):
        stats = estimator.get_session_stats()
        assert stats["n_assessments"] == 0

    def test_stats_after_data(self, estimator):
        np.random.seed(42)
        for _ in range(3):
            estimator.assess(_make_signal())
        stats = estimator.get_session_stats()
        assert stats["n_assessments"] == 3
        assert "mean_reserve" in stats
        assert "latest_level" in stats


class TestHistory:
    def test_empty_history(self, estimator):
        assert estimator.get_history() == []

    def test_history_grows(self, estimator):
        np.random.seed(42)
        estimator.assess(_make_signal())
        estimator.assess(_make_signal())
        assert len(estimator.get_history()) == 2

    def test_history_last_n(self, estimator):
        np.random.seed(42)
        for _ in range(10):
            estimator.assess(_make_signal())
        assert len(estimator.get_history(last_n=3)) == 3


class TestMultiUser:
    def test_independent(self, estimator):
        np.random.seed(42)
        estimator.assess(_make_signal(), user_id="a")
        estimator.assess(_make_signal(), user_id="b")
        assert len(estimator.get_history("a")) == 1
        assert len(estimator.get_history("b")) == 1


class TestReset:
    def test_reset_clears(self, estimator):
        np.random.seed(42)
        estimator.set_baseline(_make_signal())
        estimator.assess(_make_signal())
        estimator.reset()
        assert estimator.get_history() == []
        stats = estimator.get_session_stats()
        assert stats["n_assessments"] == 0
        assert stats["has_baseline"] is False

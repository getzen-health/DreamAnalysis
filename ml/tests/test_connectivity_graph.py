"""Tests for brain connectivity graph."""
import numpy as np
import pytest

from models.connectivity_graph import ConnectivityGraph


@pytest.fixture
def graph():
    return ConnectivityGraph(fs=256.0)


def _make_signal(fs=256, duration=4, n_channels=4, noise_amp=2.0):
    """Standard 4-channel EEG."""
    t = np.arange(int(fs * duration)) / fs
    signals = []
    for ch in range(n_channels):
        alpha = 10 * np.sin(2 * np.pi * 10 * t + ch * 0.5)
        theta = 5 * np.sin(2 * np.pi * 6 * t + ch * 0.3)
        noise = noise_amp * np.random.randn(len(t))
        signals.append(alpha + theta + noise)
    return np.array(signals)


def _make_synchronous_signal(fs=256, duration=4, n_channels=4):
    """All channels identical — maximum connectivity."""
    t = np.arange(int(fs * duration)) / fs
    base = 10 * np.sin(2 * np.pi * 10 * t)
    signals = [base + 0.1 * np.random.randn(len(t)) for _ in range(n_channels)]
    return np.array(signals)


def _make_independent_signal(fs=256, duration=4, n_channels=4):
    """Each channel independent — low connectivity."""
    signals = [np.random.randn(int(fs * duration)) * 5 for _ in range(n_channels)]
    return np.array(signals)


class TestBaseline:
    def test_set_baseline(self, graph):
        np.random.seed(42)
        result = graph.set_baseline(_make_signal())
        assert result["baseline_set"] is True
        assert result["n_channels"] == 4
        assert result["n_pairs"] == 6  # C(4,2) = 6

    def test_single_channel(self, graph):
        np.random.seed(42)
        sig = 10 * np.sin(2 * np.pi * 10 * np.arange(1024) / 256)
        result = graph.set_baseline(sig)
        assert result["baseline_set"] is True


class TestAnalyze:
    def test_output_keys(self, graph):
        np.random.seed(42)
        result = graph.analyze(_make_signal())
        expected = {"pairs", "mean_coherence", "mean_plv", "graph_density",
                    "connectivity_state", "n_channels", "has_baseline"}
        assert expected.issubset(set(result.keys()))

    def test_coherence_range(self, graph):
        np.random.seed(42)
        result = graph.analyze(_make_signal())
        assert 0 <= result["mean_coherence"] <= 1

    def test_plv_range(self, graph):
        np.random.seed(42)
        result = graph.analyze(_make_signal())
        assert 0 <= result["mean_plv"] <= 1

    def test_density_range(self, graph):
        np.random.seed(42)
        result = graph.analyze(_make_signal())
        assert 0 <= result["graph_density"] <= 1

    def test_state_valid(self, graph):
        np.random.seed(42)
        result = graph.analyze(_make_signal())
        assert result["connectivity_state"] in (
            "highly_connected", "moderately_connected", "weakly_connected"
        )

    def test_pairs_count(self, graph):
        np.random.seed(42)
        result = graph.analyze(_make_signal())
        assert len(result["pairs"]) == 6  # 4 channels → 6 pairs


class TestConnectivityLevels:
    def test_high_coherence_synchronous(self, graph):
        np.random.seed(42)
        result = graph.analyze(_make_synchronous_signal())
        assert result["mean_coherence"] > 0.5

    def test_low_coherence_independent(self, graph):
        np.random.seed(42)
        result = graph.analyze(_make_independent_signal())
        assert result["mean_coherence"] < 0.5

    def test_high_plv_synchronous(self, graph):
        np.random.seed(42)
        result = graph.analyze(_make_synchronous_signal())
        assert result["mean_plv"] > 0.5


class TestSpecificConnectivity:
    def test_interhemispheric(self, graph):
        np.random.seed(42)
        result = graph.analyze(_make_signal())
        assert "interhemispheric_connectivity" in result
        assert 0 <= result["interhemispheric_connectivity"] <= 1

    def test_frontal_temporal(self, graph):
        np.random.seed(42)
        result = graph.analyze(_make_signal())
        assert "frontal_temporal_connectivity" in result
        assert 0 <= result["frontal_temporal_connectivity"] <= 1


class TestSessionStats:
    def test_empty_stats(self, graph):
        stats = graph.get_session_stats()
        assert stats["n_epochs"] == 0

    def test_stats_after_data(self, graph):
        np.random.seed(42)
        for _ in range(5):
            graph.analyze(_make_signal())
        stats = graph.get_session_stats()
        assert stats["n_epochs"] == 5
        assert "mean_coherence" in stats
        assert "dominant_state" in stats


class TestHistory:
    def test_empty_history(self, graph):
        assert graph.get_history() == []

    def test_history_grows(self, graph):
        np.random.seed(42)
        graph.analyze(_make_signal())
        graph.analyze(_make_signal())
        assert len(graph.get_history()) == 2

    def test_history_last_n(self, graph):
        np.random.seed(42)
        for _ in range(10):
            graph.analyze(_make_signal())
        assert len(graph.get_history(last_n=3)) == 3


class TestMultiUser:
    def test_independent(self, graph):
        np.random.seed(42)
        graph.analyze(_make_signal(), user_id="a")
        graph.analyze(_make_signal(), user_id="b")
        assert len(graph.get_history("a")) == 1
        assert len(graph.get_history("b")) == 1


class TestReset:
    def test_reset_clears(self, graph):
        np.random.seed(42)
        graph.set_baseline(_make_signal())
        graph.analyze(_make_signal())
        graph.reset()
        assert graph.get_history() == []
        stats = graph.get_session_stats()
        assert stats["n_epochs"] == 0
        assert stats["has_baseline"] is False

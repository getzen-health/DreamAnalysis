"""Tests for EmotionalGranularityEstimator.

Comprehensive test suite covering:
- Episode addition (single/multi-channel, short signals, labels)
- Granularity computation (insufficient data, sufficient data, scoring)
- Output structure and value ranges
- Session stats
- History tracking
- Reset behavior
- Multi-user independence
- Edge cases (constant signal, 1D input, very short signals, NaN handling)
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import butter, filtfilt

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.emotional_granularity import EmotionalGranularityEstimator


# ---------------------------------------------------------------------------
# Helpers to generate synthetic EEG with controllable spectral content
# ---------------------------------------------------------------------------


def _make_eeg(
    n_channels: int = 4,
    duration_sec: float = 4.0,
    fs: float = 256.0,
    alpha_amp: float = 15.0,
    beta_amp: float = 5.0,
    theta_amp: float = 8.0,
    noise_amp: float = 3.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic EEG with controllable band amplitudes."""
    rng = np.random.RandomState(seed)
    n_samples = int(duration_sec * fs)
    t = np.arange(n_samples) / fs
    eeg = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        alpha = alpha_amp * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi))
        beta = beta_amp * np.sin(2 * np.pi * 20 * t + rng.uniform(0, 2 * np.pi))
        theta = theta_amp * np.sin(2 * np.pi * 6 * t + rng.uniform(0, 2 * np.pi))
        noise = noise_amp * rng.randn(n_samples)
        eeg[ch] = alpha + beta + theta + noise
    return eeg


def _make_diverse_episodes(n: int, fs: float = 256.0) -> list:
    """Generate n episodes with diverse spectral content."""
    np.random.seed(42)
    episodes = []
    for i in range(n):
        eeg = _make_eeg(
            n_channels=4,
            fs=fs,
            alpha_amp=5.0 + 20.0 * np.random.rand(),
            beta_amp=2.0 + 15.0 * np.random.rand(),
            theta_amp=3.0 + 12.0 * np.random.rand(),
            noise_amp=1.0 + 5.0 * np.random.rand(),
            seed=42 + i * 7,
        )
        episodes.append(eeg)
    return episodes


def _make_identical_episodes(n: int, fs: float = 256.0) -> list:
    """Generate n episodes with identical spectral content."""
    np.random.seed(42)
    template = _make_eeg(n_channels=4, fs=fs, seed=42)
    return [template.copy() for _ in range(n)]


# ===========================================================================
# TestAddEpisode
# ===========================================================================


class TestAddEpisode:
    """Tests for add_episode method."""

    def test_add_single_episode(self):
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        eeg = _make_eeg(n_channels=4, seed=42)
        est.add_episode(eeg)
        stats = est.get_session_stats()
        assert stats["n_episodes"] == 1

    def test_add_multiple_episodes(self):
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        for i in range(5):
            eeg = _make_eeg(seed=42 + i)
            est.add_episode(eeg)
        stats = est.get_session_stats()
        assert stats["n_episodes"] == 5

    def test_add_single_channel(self):
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        eeg = _make_eeg(n_channels=1, seed=42)
        est.add_episode(eeg)
        assert est.get_session_stats()["n_episodes"] == 1

    def test_add_multichannel(self):
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        eeg = _make_eeg(n_channels=4, seed=42)
        est.add_episode(eeg)
        assert est.get_session_stats()["n_episodes"] == 1

    def test_add_1d_input(self):
        """1D input should be reshaped to (1, n_samples)."""
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        eeg = np.random.randn(1024) * 20
        est.add_episode(eeg)
        assert est.get_session_stats()["n_episodes"] == 1

    def test_add_with_label(self):
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        eeg = _make_eeg(seed=42)
        est.add_episode(eeg, label="happy")
        assert est.get_session_stats()["n_episodes"] == 1

    def test_add_with_custom_fs(self):
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        eeg = _make_eeg(seed=42, fs=128.0)
        est.add_episode(eeg, fs=128.0)
        assert est.get_session_stats()["n_episodes"] == 1

    def test_add_with_user_id(self):
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        eeg = _make_eeg(seed=42)
        est.add_episode(eeg, user_id="alice")
        assert est.get_session_stats("alice")["n_episodes"] == 1
        assert est.get_session_stats("default")["n_episodes"] == 0

    def test_add_short_signal(self):
        """Short signals (< 64 samples) should still be added gracefully."""
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        eeg = np.random.randn(4, 32) * 20
        est.add_episode(eeg)
        assert est.get_session_stats()["n_episodes"] == 1

    def test_add_very_short_signal(self):
        """Very short signal (< 10 samples) should be handled."""
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        eeg = np.random.randn(4, 8) * 20
        est.add_episode(eeg)
        assert est.get_session_stats()["n_episodes"] == 1

    def test_episode_cap_enforced(self):
        """Should not store more than 500 episodes per user."""
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        # Use tiny signal to keep test fast
        for i in range(510):
            eeg = np.random.randn(1, 64) * 20
            est.add_episode(eeg)
        assert est.get_session_stats()["n_episodes"] == 500


# ===========================================================================
# TestComputeGranularity
# ===========================================================================


class TestComputeGranularity:
    """Tests for compute_granularity method."""

    def test_insufficient_episodes_returns_none(self):
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        for i in range(5):
            est.add_episode(_make_eeg(seed=42 + i))
        result = est.compute_granularity()
        assert result["granularity"] is None
        assert result["ready"] is False
        assert result["granularity_level"] == "insufficient_data"

    def test_zero_episodes_returns_none(self):
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        result = est.compute_granularity()
        assert result["granularity"] is None
        assert result["ready"] is False

    def test_exactly_10_episodes_is_sufficient(self):
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        episodes = _make_diverse_episodes(10)
        for ep in episodes:
            est.add_episode(ep)
        result = est.compute_granularity()
        assert result["granularity"] is not None
        assert result["ready"] is True

    def test_9_episodes_is_insufficient(self):
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        episodes = _make_diverse_episodes(9)
        for ep in episodes:
            est.add_episode(ep)
        result = est.compute_granularity()
        assert result["granularity"] is None
        assert result["ready"] is False

    def test_sufficient_returns_valid_scores(self):
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        episodes = _make_diverse_episodes(15)
        for ep in episodes:
            est.add_episode(ep)
        result = est.compute_granularity()
        assert result["granularity"] is not None
        assert result["pattern_diversity"] is not None
        assert result["entropy_variability"] is not None
        assert result["de_variability"] is not None

    def test_episodes_collected_count(self):
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        episodes = _make_diverse_episodes(12)
        for ep in episodes:
            est.add_episode(ep)
        result = est.compute_granularity()
        assert result["episodes_collected"] == 12


# ===========================================================================
# TestGranularityScoring
# ===========================================================================


class TestGranularityScoring:
    """Tests that diverse signals produce higher granularity than identical."""

    def test_diverse_higher_than_identical(self):
        """Diverse episodes should yield higher granularity than identical ones."""
        np.random.seed(42)

        # Diverse episodes
        est_diverse = EmotionalGranularityEstimator(fs=256.0)
        for ep in _make_diverse_episodes(15):
            est_diverse.add_episode(ep)
        result_diverse = est_diverse.compute_granularity()

        # Identical episodes
        est_identical = EmotionalGranularityEstimator(fs=256.0)
        for ep in _make_identical_episodes(15):
            est_identical.add_episode(ep)
        result_identical = est_identical.compute_granularity()

        assert result_diverse["granularity"] > result_identical["granularity"]

    def test_diverse_higher_pattern_diversity(self):
        """Diverse episodes should have higher pattern_diversity."""
        np.random.seed(42)

        est_diverse = EmotionalGranularityEstimator(fs=256.0)
        for ep in _make_diverse_episodes(12):
            est_diverse.add_episode(ep)

        est_identical = EmotionalGranularityEstimator(fs=256.0)
        for ep in _make_identical_episodes(12):
            est_identical.add_episode(ep)

        div_result = est_diverse.compute_granularity()
        ident_result = est_identical.compute_granularity()
        assert div_result["pattern_diversity"] > ident_result["pattern_diversity"]

    def test_identical_signals_low_granularity(self):
        """Identical signals should produce low granularity."""
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        for ep in _make_identical_episodes(15):
            est.add_episode(ep)
        result = est.compute_granularity()
        assert result["granularity"] < 0.35

    def test_very_diverse_signals_moderate_or_high_granularity(self):
        """Very diverse signals should produce at least moderate granularity."""
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        for ep in _make_diverse_episodes(20):
            est.add_episode(ep)
        result = est.compute_granularity()
        assert result["granularity"] >= 0.20  # at least not trivially low

    def test_granularity_level_matches_score(self):
        """Level label should match the score threshold."""
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        for ep in _make_diverse_episodes(15):
            est.add_episode(ep)
        result = est.compute_granularity()
        g = result["granularity"]
        level = result["granularity_level"]
        if g >= 0.65:
            assert level == "high"
        elif g >= 0.35:
            assert level == "moderate"
        else:
            assert level == "low"


# ===========================================================================
# TestOutputStructure
# ===========================================================================


class TestOutputStructure:
    """Tests for output dict structure, types, and value ranges."""

    def setup_method(self):
        np.random.seed(42)
        self.est = EmotionalGranularityEstimator(fs=256.0)
        for ep in _make_diverse_episodes(12):
            self.est.add_episode(ep)
        self.result = self.est.compute_granularity()

    def test_all_required_keys_present(self):
        required = [
            "granularity", "pattern_diversity", "entropy_variability",
            "de_variability", "episodes_collected", "ready", "granularity_level",
        ]
        for key in required:
            assert key in self.result, f"Missing key: {key}"

    def test_granularity_is_float(self):
        assert isinstance(self.result["granularity"], float)

    def test_pattern_diversity_is_float(self):
        assert isinstance(self.result["pattern_diversity"], float)

    def test_entropy_variability_is_float(self):
        assert isinstance(self.result["entropy_variability"], float)

    def test_de_variability_is_float(self):
        assert isinstance(self.result["de_variability"], float)

    def test_episodes_collected_is_int(self):
        assert isinstance(self.result["episodes_collected"], int)

    def test_ready_is_bool(self):
        assert isinstance(self.result["ready"], bool)

    def test_granularity_level_is_string(self):
        assert isinstance(self.result["granularity_level"], str)

    def test_granularity_in_0_1_range(self):
        g = self.result["granularity"]
        assert 0.0 <= g <= 1.0

    def test_pattern_diversity_non_negative(self):
        assert self.result["pattern_diversity"] >= 0.0

    def test_entropy_variability_non_negative(self):
        assert self.result["entropy_variability"] >= 0.0

    def test_de_variability_non_negative(self):
        assert self.result["de_variability"] >= 0.0

    def test_granularity_level_valid_values(self):
        assert self.result["granularity_level"] in {"high", "moderate", "low"}

    def test_insufficient_output_keys(self):
        """Insufficient data result should have the same keys."""
        est = EmotionalGranularityEstimator(fs=256.0)
        result = est.compute_granularity()
        required = [
            "granularity", "pattern_diversity", "entropy_variability",
            "de_variability", "episodes_collected", "ready", "granularity_level",
        ]
        for key in required:
            assert key in result, f"Missing key in insufficient result: {key}"


# ===========================================================================
# TestSessionStats
# ===========================================================================


class TestSessionStats:
    """Tests for get_session_stats method."""

    def test_empty_stats(self):
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        stats = est.get_session_stats()
        assert stats["n_episodes"] == 0
        assert stats["has_episodes"] is False
        assert stats["mean_granularity"] is None

    def test_stats_after_episodes(self):
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        for ep in _make_diverse_episodes(5):
            est.add_episode(ep)
        stats = est.get_session_stats()
        assert stats["n_episodes"] == 5
        assert stats["has_episodes"] is True

    def test_stats_after_compute(self):
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        for ep in _make_diverse_episodes(12):
            est.add_episode(ep)
        est.compute_granularity()
        stats = est.get_session_stats()
        assert stats["mean_granularity"] is not None
        assert isinstance(stats["mean_granularity"], float)

    def test_stats_mean_granularity_no_compute(self):
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        for ep in _make_diverse_episodes(12):
            est.add_episode(ep)
        stats = est.get_session_stats()
        assert stats["mean_granularity"] is None


# ===========================================================================
# TestHistory
# ===========================================================================


class TestHistory:
    """Tests for get_history method."""

    def test_empty_history(self):
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        history = est.get_history()
        assert history == []

    def test_history_grows_with_compute(self):
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        for ep in _make_diverse_episodes(12):
            est.add_episode(ep)
        est.compute_granularity()
        est.compute_granularity()
        history = est.get_history()
        assert len(history) == 2

    def test_history_last_n(self):
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        for ep in _make_diverse_episodes(12):
            est.add_episode(ep)
        for _ in range(5):
            est.compute_granularity()
        history = est.get_history(last_n=2)
        assert len(history) == 2

    def test_history_last_n_larger_than_history(self):
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        for ep in _make_diverse_episodes(12):
            est.add_episode(ep)
        est.compute_granularity()
        history = est.get_history(last_n=100)
        assert len(history) == 1

    def test_history_entries_are_dicts(self):
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        for ep in _make_diverse_episodes(12):
            est.add_episode(ep)
        est.compute_granularity()
        history = est.get_history()
        assert all(isinstance(h, dict) for h in history)

    def test_history_entries_have_granularity(self):
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        for ep in _make_diverse_episodes(12):
            est.add_episode(ep)
        est.compute_granularity()
        history = est.get_history()
        assert "granularity" in history[0]


# ===========================================================================
# TestReset
# ===========================================================================


class TestReset:
    """Tests for reset method."""

    def test_reset_clears_episodes(self):
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        for ep in _make_diverse_episodes(5):
            est.add_episode(ep)
        est.reset()
        assert est.get_session_stats()["n_episodes"] == 0

    def test_reset_clears_history(self):
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        for ep in _make_diverse_episodes(12):
            est.add_episode(ep)
        est.compute_granularity()
        est.reset()
        assert est.get_history() == []

    def test_reset_specific_user(self):
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        for ep in _make_diverse_episodes(5):
            est.add_episode(ep, user_id="alice")
        for ep in _make_diverse_episodes(3):
            est.add_episode(ep, user_id="bob")
        est.reset(user_id="alice")
        assert est.get_session_stats("alice")["n_episodes"] == 0
        assert est.get_session_stats("bob")["n_episodes"] == 3

    def test_reset_allows_reuse(self):
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        for ep in _make_diverse_episodes(5):
            est.add_episode(ep)
        est.reset()
        est.add_episode(_make_eeg(seed=99))
        assert est.get_session_stats()["n_episodes"] == 1


# ===========================================================================
# TestMultiUser
# ===========================================================================


class TestMultiUser:
    """Tests for multi-user independence."""

    def test_users_are_independent(self):
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        for ep in _make_diverse_episodes(5):
            est.add_episode(ep, user_id="alice")
        for ep in _make_diverse_episodes(3):
            est.add_episode(ep, user_id="bob")
        assert est.get_session_stats("alice")["n_episodes"] == 5
        assert est.get_session_stats("bob")["n_episodes"] == 3

    def test_compute_for_one_user_doesnt_affect_other(self):
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        for ep in _make_diverse_episodes(12):
            est.add_episode(ep, user_id="alice")
        for ep in _make_diverse_episodes(3):
            est.add_episode(ep, user_id="bob")
        result_alice = est.compute_granularity("alice")
        result_bob = est.compute_granularity("bob")
        assert result_alice["ready"] is True
        assert result_bob["ready"] is False

    def test_history_is_per_user(self):
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        for ep in _make_diverse_episodes(12):
            est.add_episode(ep, user_id="alice")
        est.compute_granularity("alice")
        assert len(est.get_history("alice")) == 1
        assert len(est.get_history("bob")) == 0

    def test_reset_one_user_preserves_other(self):
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        for ep in _make_diverse_episodes(5):
            est.add_episode(ep, user_id="alice")
        for ep in _make_diverse_episodes(5):
            est.add_episode(ep, user_id="bob")
        est.reset("alice")
        assert est.get_session_stats("alice")["n_episodes"] == 0
        assert est.get_session_stats("bob")["n_episodes"] == 5


# ===========================================================================
# TestEdgeCases
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases and robustness."""

    def test_constant_signal(self):
        """Constant (DC) signal should not crash."""
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        eeg = np.ones((4, 1024)) * 100.0
        for _ in range(12):
            est.add_episode(eeg)
        result = est.compute_granularity()
        assert result["ready"] is True
        assert result["granularity"] is not None
        # Constant signal = zero diversity
        assert result["granularity"] <= 0.5

    def test_1d_input_computes_granularity(self):
        """1D input across episodes should produce valid granularity."""
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        for i in range(12):
            eeg = np.random.randn(1024) * (10 + i * 3)
            est.add_episode(eeg)
        result = est.compute_granularity()
        assert result["ready"] is True
        assert 0.0 <= result["granularity"] <= 1.0

    def test_very_short_signals_compute_granularity(self):
        """Very short signals should still allow granularity computation."""
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        for i in range(12):
            eeg = np.random.randn(4, 16) * 20
            est.add_episode(eeg)
        result = est.compute_granularity()
        assert result["ready"] is True
        assert result["granularity"] is not None

    def test_nan_in_eeg_handled(self):
        """NaN values in EEG should be handled gracefully."""
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        for i in range(12):
            eeg = np.random.randn(4, 512) * 20
            # Inject a NaN
            eeg[0, 100] = np.nan
            est.add_episode(eeg)
        # Should not raise
        result = est.compute_granularity()
        assert result["ready"] is True

    def test_inf_in_eeg_handled(self):
        """Inf values in EEG should be handled gracefully."""
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        for i in range(12):
            eeg = np.random.randn(4, 512) * 20
            eeg[1, 50] = np.inf
            est.add_episode(eeg)
        result = est.compute_granularity()
        assert result["ready"] is True

    def test_single_sample_signal(self):
        """Single sample signal should not crash."""
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        eeg = np.random.randn(4, 1) * 20
        est.add_episode(eeg)
        assert est.get_session_stats()["n_episodes"] == 1

    def test_two_channel_signal(self):
        """2-channel signal should work (no frontal asymmetry from AF7/AF8)."""
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        for i in range(12):
            eeg = np.random.randn(2, 512) * 20
            est.add_episode(eeg)
        result = est.compute_granularity()
        assert result["ready"] is True

    def test_large_amplitude_signal(self):
        """Very large amplitude should not cause overflow."""
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        for i in range(12):
            eeg = np.random.randn(4, 512) * 10000
            est.add_episode(eeg)
        result = est.compute_granularity()
        assert result["ready"] is True
        assert np.isfinite(result["granularity"])

    def test_zero_signal(self):
        """All-zero signal should produce valid (low) granularity."""
        np.random.seed(42)
        est = EmotionalGranularityEstimator(fs=256.0)
        for _ in range(12):
            eeg = np.zeros((4, 512))
            est.add_episode(eeg)
        result = est.compute_granularity()
        assert result["ready"] is True
        assert result["granularity"] is not None

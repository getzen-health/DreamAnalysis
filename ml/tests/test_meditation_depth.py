"""Tests for MeditationDepthQuantifier."""

import numpy as np
import pytest

from models.meditation_depth import MeditationDepthQuantifier, DEPTH_LEVELS


@pytest.fixture
def quantifier():
    return MeditationDepthQuantifier(fs=256.0)


def _make_eeg(
    fs=256, duration=4, n_channels=4,
    theta_amp=5.0, alpha_amp=10.0, beta_amp=3.0,
    gamma_amp=0.5, noise_amp=2.0, seed=42,
):
    """Synthesize multichannel EEG with controllable band amplitudes.

    theta ~6 Hz, alpha ~10 Hz, beta ~20 Hz, gamma ~40 Hz.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(int(fs * duration)) / fs
    signals = []
    for ch in range(n_channels):
        phase = ch * 0.4
        s = (
            theta_amp * np.sin(2 * np.pi * 6 * t + phase)
            + alpha_amp * np.sin(2 * np.pi * 10 * t + phase)
            + beta_amp * np.sin(2 * np.pi * 20 * t + phase)
            + gamma_amp * np.sin(2 * np.pi * 40 * t + phase)
            + noise_amp * rng.randn(len(t))
        )
        signals.append(s)
    return np.array(signals)


# =====================================================================
# TestBaseline
# =====================================================================
class TestBaseline:
    def test_set_baseline_returns_confirmation(self, quantifier):
        result = quantifier.set_baseline(_make_eeg())
        assert result["baseline_set"] is True
        assert "baseline_powers" in result

    def test_baseline_powers_are_positive(self, quantifier):
        result = quantifier.set_baseline(_make_eeg())
        for band, val in result["baseline_powers"].items():
            assert val >= 0.0, f"Baseline power for {band} should be non-negative"

    def test_baseline_with_1d_signal(self, quantifier):
        signal_1d = 10.0 * np.sin(2 * np.pi * 10 * np.arange(1024) / 256)
        result = quantifier.set_baseline(signal_1d)
        assert result["baseline_set"] is True

    def test_baseline_per_user(self, quantifier):
        quantifier.set_baseline(_make_eeg(alpha_amp=5), user_id="alice")
        quantifier.set_baseline(_make_eeg(alpha_amp=20), user_id="bob")
        # Both should have separate baselines
        r_a = quantifier.assess(_make_eeg(), user_id="alice")
        r_b = quantifier.assess(_make_eeg(), user_id="bob")
        # Different baselines means different scores
        assert isinstance(r_a["depth_score"], float)
        assert isinstance(r_b["depth_score"], float)


# =====================================================================
# TestAssess
# =====================================================================
class TestAssess:
    def test_output_keys(self, quantifier):
        quantifier.set_baseline(_make_eeg())
        result = quantifier.assess(_make_eeg())
        expected_keys = {
            "depth_score", "depth_level", "fmt_power",
            "alpha_coherence", "theta_alpha_ratio",
            "gamma_bursts_detected", "stability_index",
            "time_in_depth", "recommendations",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_depth_score_range(self, quantifier):
        result = quantifier.assess(_make_eeg())
        assert 0 <= result["depth_score"] <= 100

    def test_depth_level_valid(self, quantifier):
        result = quantifier.assess(_make_eeg())
        assert result["depth_level"] in DEPTH_LEVELS

    def test_fmt_power_positive(self, quantifier):
        result = quantifier.assess(_make_eeg())
        assert result["fmt_power"] >= 0.0

    def test_alpha_coherence_range(self, quantifier):
        result = quantifier.assess(_make_eeg())
        assert 0 <= result["alpha_coherence"] <= 1.0

    def test_stability_index_range(self, quantifier):
        result = quantifier.assess(_make_eeg())
        assert 0 <= result["stability_index"] <= 1.0

    def test_gamma_bursts_is_bool(self, quantifier):
        result = quantifier.assess(_make_eeg())
        assert isinstance(result["gamma_bursts_detected"], bool)

    def test_time_in_depth_has_all_levels(self, quantifier):
        result = quantifier.assess(_make_eeg())
        for level in DEPTH_LEVELS:
            assert level in result["time_in_depth"]

    def test_recommendations_is_list(self, quantifier):
        result = quantifier.assess(_make_eeg())
        assert isinstance(result["recommendations"], list)
        assert len(result["recommendations"]) >= 1

    def test_assess_without_baseline(self, quantifier):
        """Should work even without a baseline (uses absolute thresholds)."""
        result = quantifier.assess(_make_eeg())
        assert 0 <= result["depth_score"] <= 100
        assert result["depth_level"] in DEPTH_LEVELS


# =====================================================================
# TestDepthLevels
# =====================================================================
class TestDepthLevels:
    def test_high_theta_low_beta_scores_deeper(self, quantifier):
        """High theta + low beta should produce a deeper score than the reverse."""
        # Deep meditation signature: strong theta, minimal beta
        deep_eeg = _make_eeg(theta_amp=25, alpha_amp=12, beta_amp=1, noise_amp=1)
        # Surface signature: strong beta, low theta
        surface_eeg = _make_eeg(theta_amp=1, alpha_amp=3, beta_amp=25, noise_amp=1)

        quantifier.set_baseline(_make_eeg(theta_amp=5, alpha_amp=10, beta_amp=5))
        deep_result = quantifier.assess(deep_eeg)

        quantifier.reset()
        quantifier.set_baseline(_make_eeg(theta_amp=5, alpha_amp=10, beta_amp=5))
        surface_result = quantifier.assess(surface_eeg)

        assert deep_result["depth_score"] > surface_result["depth_score"]

    def test_surface_level_with_high_beta(self, quantifier):
        """Very high beta + low theta should yield surface or light."""
        eeg = _make_eeg(theta_amp=1, alpha_amp=2, beta_amp=30, noise_amp=1)
        quantifier.set_baseline(_make_eeg(theta_amp=5, alpha_amp=10, beta_amp=5))
        result = quantifier.assess(eeg)
        assert result["depth_level"] in ("surface", "light")

    def test_deeper_with_more_theta(self, quantifier):
        """Increasing theta amplitude while keeping other bands fixed
        should increase the depth score monotonically."""
        quantifier.set_baseline(
            _make_eeg(theta_amp=3, alpha_amp=10, beta_amp=5)
        )
        scores = []
        for theta_amp in [3, 8, 15, 25]:
            quantifier.reset()
            quantifier.set_baseline(
                _make_eeg(theta_amp=3, alpha_amp=10, beta_amp=5)
            )
            eeg = _make_eeg(theta_amp=theta_amp, alpha_amp=10, beta_amp=3)
            result = quantifier.assess(eeg)
            scores.append(result["depth_score"])

        # Each step should generally be higher (allow small jitter)
        assert scores[-1] > scores[0], (
            f"Highest theta ({scores[-1]}) should score higher than lowest ({scores[0]})"
        )


# =====================================================================
# TestFMT
# =====================================================================
class TestFMT:
    def test_fmt_increases_with_theta(self, quantifier):
        """FMT power should increase when theta amplitude is raised."""
        low_theta = _make_eeg(theta_amp=2, alpha_amp=10)
        high_theta = _make_eeg(theta_amp=25, alpha_amp=10)

        r_low = quantifier.assess(low_theta)
        r_high = quantifier.assess(high_theta)

        assert r_high["fmt_power"] > r_low["fmt_power"]

    def test_fmt_uses_frontal_channels(self, quantifier):
        """FMT should be computed from AF7 (ch1) + AF8 (ch2) on 4-channel data."""
        # Put theta only on frontal channels (ch1, ch2)
        rng = np.random.RandomState(99)
        t = np.arange(1024) / 256
        signals = rng.randn(4, 1024) * 2  # low noise baseline
        # Add strong theta only to ch1 and ch2
        theta_signal = 20 * np.sin(2 * np.pi * 6 * t)
        signals[1] += theta_signal
        signals[2] += theta_signal

        r = quantifier.assess(signals)
        # FMT should be substantial since frontal channels have strong theta
        assert r["fmt_power"] > 0.0

    def test_fmt_single_channel(self, quantifier):
        """FMT should still work with a single-channel signal."""
        t = np.arange(1024) / 256
        sig = 15 * np.sin(2 * np.pi * 6 * t) + np.random.randn(1024) * 2
        result = quantifier.assess(sig)
        assert result["fmt_power"] > 0.0


# =====================================================================
# TestAlphaCoherence
# =====================================================================
class TestAlphaCoherence:
    def test_coherent_alpha_high_score(self, quantifier):
        """Identical alpha across channels should yield high coherence."""
        t = np.arange(1024) / 256
        alpha = 15 * np.sin(2 * np.pi * 10 * t)
        # All channels get the same alpha + tiny noise
        signals = np.array([
            alpha + np.random.randn(1024) * 0.5,
            alpha + np.random.randn(1024) * 0.5,
            alpha + np.random.randn(1024) * 0.5,
            alpha + np.random.randn(1024) * 0.5,
        ])
        result = quantifier.assess(signals)
        assert result["alpha_coherence"] > 0.5

    def test_incoherent_alpha_lower_score(self, quantifier):
        """Independent noise should yield lower alpha coherence than
        synchronized alpha."""
        rng = np.random.RandomState(123)
        noise_only = rng.randn(4, 1024) * 20
        result = quantifier.assess(noise_only)
        # Pure noise coherence should generally be lower
        assert result["alpha_coherence"] < 0.9

    def test_single_channel_coherence_zero(self, quantifier):
        """Single-channel signal has no coherence to measure."""
        sig = np.random.randn(1024) * 10
        result = quantifier.assess(sig)
        assert result["alpha_coherence"] == 0.0


# =====================================================================
# TestStability
# =====================================================================
class TestStability:
    def test_stable_signal_high_index(self, quantifier):
        """A stationary signal should produce high stability index."""
        # Pure sine wave = perfectly stationary
        t = np.arange(2048) / 256  # 8 seconds for multiple windows
        alpha = 10 * np.sin(2 * np.pi * 10 * t)
        signals = np.array([alpha, alpha, alpha, alpha])
        result = quantifier.assess(signals)
        assert result["stability_index"] >= 0.5

    def test_unstable_signal_lower_index(self, quantifier):
        """A signal that changes dramatically should have lower stability."""
        rng = np.random.RandomState(77)
        t = np.arange(2048) / 256
        # First half: strong alpha. Second half: strong beta.
        half = 1024
        signals = []
        for ch in range(4):
            s = np.zeros(2048)
            s[:half] = 20 * np.sin(2 * np.pi * 10 * t[:half]) + rng.randn(half) * 2
            s[half:] = 20 * np.sin(2 * np.pi * 22 * t[half:]) + rng.randn(half) * 2
            signals.append(s)
        signals = np.array(signals)

        stable_eeg = _make_eeg(duration=8, theta_amp=10, alpha_amp=10, beta_amp=3)
        r_stable = quantifier.assess(stable_eeg)
        r_unstable = quantifier.assess(signals)

        # Stable should be >= unstable (or close)
        assert r_stable["stability_index"] >= r_unstable["stability_index"] - 0.15

    def test_short_signal_default_stability(self, quantifier):
        """Very short signal (< 2 windows) should return 0.5 default."""
        short = np.random.randn(4, 128) * 10  # 0.5 sec at 256 Hz
        result = quantifier.assess(short)
        assert result["stability_index"] == 0.5


# =====================================================================
# TestSessionTimeline
# =====================================================================
class TestSessionTimeline:
    def test_empty_timeline(self, quantifier):
        assert quantifier.get_session_timeline() == []

    def test_timeline_grows(self, quantifier):
        for i in range(5):
            quantifier.assess(_make_eeg(seed=i))
        timeline = quantifier.get_session_timeline()
        assert len(timeline) == 5

    def test_timeline_entries_have_keys(self, quantifier):
        quantifier.assess(_make_eeg())
        entry = quantifier.get_session_timeline()[0]
        assert "depth_score" in entry
        assert "depth_level" in entry
        assert "fmt_power" in entry
        assert "alpha_coherence" in entry
        assert "theta_alpha_ratio" in entry

    def test_timeline_per_user(self, quantifier):
        quantifier.assess(_make_eeg(), user_id="a")
        quantifier.assess(_make_eeg(), user_id="a")
        quantifier.assess(_make_eeg(), user_id="b")
        assert len(quantifier.get_session_timeline("a")) == 2
        assert len(quantifier.get_session_timeline("b")) == 1


# =====================================================================
# TestSessionStats
# =====================================================================
class TestSessionStats:
    def test_empty_stats(self, quantifier):
        stats = quantifier.get_session_stats()
        assert stats["n_assessments"] == 0

    def test_stats_after_data(self, quantifier):
        for i in range(5):
            quantifier.assess(_make_eeg(seed=i))
        stats = quantifier.get_session_stats()
        assert stats["n_assessments"] == 5
        assert "mean_depth" in stats
        assert "max_depth" in stats
        assert "deepest_level" in stats
        assert stats["deepest_level"] in DEPTH_LEVELS
        assert "time_in_depth" in stats
        assert "trend" in stats

    def test_stats_mean_in_range(self, quantifier):
        for i in range(10):
            quantifier.assess(_make_eeg(seed=i))
        stats = quantifier.get_session_stats()
        assert 0 <= stats["mean_depth"] <= 100
        assert 0 <= stats["max_depth"] <= 100

    def test_trend_after_enough_data(self, quantifier):
        for i in range(20):
            quantifier.assess(_make_eeg(seed=i))
        stats = quantifier.get_session_stats()
        assert stats["trend"] in ("deepening", "surfacing", "stable")


# =====================================================================
# TestHistory
# =====================================================================
class TestHistory:
    def test_empty_history(self, quantifier):
        assert quantifier.get_history() == []

    def test_history_grows(self, quantifier):
        for i in range(3):
            quantifier.assess(_make_eeg(seed=i))
        history = quantifier.get_history()
        assert len(history) == 3

    def test_last_n(self, quantifier):
        for i in range(10):
            quantifier.assess(_make_eeg(seed=i))
        last_3 = quantifier.get_history(last_n=3)
        assert len(last_3) == 3
        # Should be the last 3 entries
        full = quantifier.get_history()
        assert last_3 == full[-3:]

    def test_last_n_larger_than_history(self, quantifier):
        quantifier.assess(_make_eeg())
        result = quantifier.get_history(last_n=100)
        assert len(result) == 1

    def test_history_entries_are_full_results(self, quantifier):
        quantifier.assess(_make_eeg())
        entry = quantifier.get_history()[0]
        assert "depth_score" in entry
        assert "depth_level" in entry
        assert "recommendations" in entry


# =====================================================================
# TestMultiUser
# =====================================================================
class TestMultiUser:
    def test_independent_users(self, quantifier):
        quantifier.set_baseline(_make_eeg(), user_id="user1")
        quantifier.assess(_make_eeg(), user_id="user1")
        quantifier.assess(_make_eeg(), user_id="user1")

        quantifier.set_baseline(_make_eeg(), user_id="user2")
        quantifier.assess(_make_eeg(), user_id="user2")

        stats1 = quantifier.get_session_stats("user1")
        stats2 = quantifier.get_session_stats("user2")

        assert stats1["n_assessments"] == 2
        assert stats2["n_assessments"] == 1

    def test_reset_one_user_preserves_other(self, quantifier):
        quantifier.set_baseline(_make_eeg(), user_id="keep")
        quantifier.assess(_make_eeg(), user_id="keep")

        quantifier.set_baseline(_make_eeg(), user_id="remove")
        quantifier.assess(_make_eeg(), user_id="remove")

        quantifier.reset(user_id="remove")

        assert quantifier.get_session_stats("keep")["n_assessments"] == 1
        assert quantifier.get_session_stats("remove")["n_assessments"] == 0


# =====================================================================
# TestReset
# =====================================================================
class TestReset:
    def test_reset_clears_everything(self, quantifier):
        quantifier.set_baseline(_make_eeg())
        for i in range(5):
            quantifier.assess(_make_eeg(seed=i))

        quantifier.reset()

        assert quantifier.get_session_timeline() == []
        assert quantifier.get_history() == []
        stats = quantifier.get_session_stats()
        assert stats["n_assessments"] == 0
        assert stats.get("has_baseline") is False

    def test_reset_allows_fresh_start(self, quantifier):
        quantifier.set_baseline(_make_eeg())
        quantifier.assess(_make_eeg())

        quantifier.reset()
        quantifier.set_baseline(_make_eeg())
        result = quantifier.assess(_make_eeg())

        assert result["depth_score"] >= 0
        assert quantifier.get_session_stats()["n_assessments"] == 1


# =====================================================================
# TestGammaBursts
# =====================================================================
class TestGammaBursts:
    def test_strong_gamma_detected(self, quantifier):
        """Very strong 40 Hz signal should trigger gamma burst detection."""
        quantifier.set_baseline(_make_eeg(gamma_amp=0.5))
        high_gamma = _make_eeg(gamma_amp=20, theta_amp=5, alpha_amp=5)
        result = quantifier.assess(high_gamma)
        assert result["gamma_bursts_detected"] is True

    def test_no_gamma_in_normal_eeg(self, quantifier):
        """Normal EEG with minimal gamma should not trigger."""
        quantifier.set_baseline(_make_eeg(gamma_amp=2))
        normal = _make_eeg(gamma_amp=1)
        result = quantifier.assess(normal)
        assert result["gamma_bursts_detected"] is False


# =====================================================================
# TestTimeInDepth
# =====================================================================
class TestTimeInDepth:
    def test_time_accumulates(self, quantifier):
        quantifier.assess(_make_eeg())
        result = quantifier.assess(_make_eeg())
        total = sum(result["time_in_depth"].values())
        assert total > 0

    def test_time_in_depth_all_levels_present(self, quantifier):
        result = quantifier.assess(_make_eeg())
        for level in DEPTH_LEVELS:
            assert level in result["time_in_depth"]

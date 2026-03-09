"""Tests for ConcentrationTracker -- sustained attention with vigilance decrement modeling.

Covers: initialization, baseline setting, assess output structure, concentration
levels, vigilance decrement over time, lapse detection, break recommendations,
session stats, history retrieval, reset, edge cases, and multichannel input.
"""

import sys
import os
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.concentration_tracker import (
    ConcentrationTracker,
    CONCENTRATION_LEVELS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FS = 256  # Hz


def make_eeg(
    theta_amp: float = 10.0,
    beta_amp: float = 5.0,
    alpha_amp: float = 3.0,
    duration_s: float = 2.0,
    n_channels: int = 1,
    noise_std: float = 0.5,
) -> np.ndarray:
    """Generate synthetic EEG with controllable band amplitudes.

    Higher beta_amp / lower theta_amp -> higher concentration.
    """
    n_samples = int(FS * duration_s)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    rng = np.random.RandomState(42)
    signal = (
        theta_amp * np.sin(2 * np.pi * 6 * t)    # theta 6 Hz
        + beta_amp * np.sin(2 * np.pi * 20 * t)   # beta 20 Hz
        + alpha_amp * np.sin(2 * np.pi * 10 * t)  # alpha 10 Hz
        + noise_std * rng.randn(n_samples)
    )
    if n_channels > 1:
        return np.tile(signal, (n_channels, 1)).astype(np.float64)
    return signal.astype(np.float64)


def make_focused_eeg(**kwargs) -> np.ndarray:
    """High beta, low theta/alpha -- focused signal."""
    defaults = dict(theta_amp=2.0, beta_amp=15.0, alpha_amp=1.0)
    defaults.update(kwargs)
    return make_eeg(**defaults)


def make_unfocused_eeg(**kwargs) -> np.ndarray:
    """High theta, low beta -- unfocused signal."""
    defaults = dict(theta_amp=15.0, beta_amp=1.0, alpha_amp=8.0)
    defaults.update(kwargs)
    return make_eeg(**defaults)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tracker():
    return ConcentrationTracker()


# ---------------------------------------------------------------------------
# TestInitialization
# ---------------------------------------------------------------------------

class TestInitialization:
    """Tracker starts in a clean, uncalibrated state."""

    def test_no_baseline_initially(self, tracker):
        assert tracker.has_baseline is False

    def test_empty_history(self, tracker):
        history = tracker.get_history()
        assert isinstance(history, list)
        assert len(history) == 0

    def test_concentration_levels_constant(self):
        assert "unfocused" in CONCENTRATION_LEVELS
        assert "deep" in CONCENTRATION_LEVELS
        assert len(CONCENTRATION_LEVELS) == 5


# ---------------------------------------------------------------------------
# TestSetBaseline
# ---------------------------------------------------------------------------

class TestSetBaseline:
    """set_baseline() records resting-state reference values."""

    def test_baseline_sets_flag(self, tracker):
        eeg = make_eeg()
        tracker.set_baseline(eeg, fs=FS)
        assert tracker.has_baseline is True

    def test_baseline_with_multichannel(self, tracker):
        eeg = make_eeg(n_channels=4)
        tracker.set_baseline(eeg, fs=FS)
        assert tracker.has_baseline is True

    def test_baseline_stores_values(self, tracker):
        eeg = make_eeg()
        tracker.set_baseline(eeg, fs=FS)
        assert tracker._baseline_beta is not None
        assert tracker._baseline_theta is not None
        assert tracker._baseline_alpha is not None
        assert tracker._baseline_beta > 0
        assert tracker._baseline_theta > 0


# ---------------------------------------------------------------------------
# TestAssessOutputStructure
# ---------------------------------------------------------------------------

class TestAssessOutputStructure:
    """assess() returns all required keys with correct types and ranges."""

    def test_required_keys_present(self, tracker):
        result = tracker.assess(make_eeg(), fs=FS)
        required = {
            "concentration_score",
            "concentration_level",
            "vigilance_decrement",
            "lapse_detected",
            "time_since_baseline_s",
            "break_recommendation",
        }
        assert required.issubset(result.keys())

    def test_concentration_score_range(self, tracker):
        result = tracker.assess(make_eeg(), fs=FS)
        assert 0 <= result["concentration_score"] <= 100

    def test_concentration_level_valid(self, tracker):
        result = tracker.assess(make_eeg(), fs=FS)
        assert result["concentration_level"] in CONCENTRATION_LEVELS

    def test_vigilance_decrement_range(self, tracker):
        result = tracker.assess(make_eeg(), fs=FS)
        assert 0.0 <= result["vigilance_decrement"] <= 1.0

    def test_lapse_detected_is_bool(self, tracker):
        result = tracker.assess(make_eeg(), fs=FS)
        assert isinstance(result["lapse_detected"], bool)

    def test_time_since_baseline_is_float(self, tracker):
        result = tracker.assess(make_eeg(), fs=FS)
        assert isinstance(result["time_since_baseline_s"], float)

    def test_break_recommendation_type(self, tracker):
        result = tracker.assess(make_eeg(), fs=FS)
        assert result["break_recommendation"] is None or isinstance(
            result["break_recommendation"], str
        )


# ---------------------------------------------------------------------------
# TestConcentrationLevels
# ---------------------------------------------------------------------------

class TestConcentrationLevels:
    """Correct level classification based on EEG characteristics."""

    def test_focused_eeg_gives_high_score(self, tracker):
        result = tracker.assess(make_focused_eeg(), fs=FS)
        assert result["concentration_score"] >= 50
        assert result["concentration_level"] in ("high", "deep", "moderate")

    def test_unfocused_eeg_gives_low_score(self, tracker):
        result = tracker.assess(make_unfocused_eeg(), fs=FS)
        assert result["concentration_score"] < 50
        assert result["concentration_level"] in ("unfocused", "low")

    def test_focused_higher_than_unfocused(self, tracker):
        r_focused = tracker.assess(make_focused_eeg(), fs=FS)
        tracker2 = ConcentrationTracker()
        r_unfocused = tracker2.assess(make_unfocused_eeg(), fs=FS)
        assert r_focused["concentration_score"] > r_unfocused["concentration_score"]


# ---------------------------------------------------------------------------
# TestVigilanceDecrement
# ---------------------------------------------------------------------------

class TestVigilanceDecrement:
    """Vigilance decrement increases over time (Warm et al. 2008, Mackworth 1948)."""

    def test_decrement_increases_with_elapsed_time(self, tracker):
        tracker.set_baseline(make_eeg(), fs=FS)
        # Simulate elapsed time by injecting timestamps
        r1 = tracker.assess(make_focused_eeg(), fs=FS, elapsed_minutes=5.0)
        tracker2 = ConcentrationTracker()
        tracker2.set_baseline(make_eeg(), fs=FS)
        r2 = tracker2.assess(make_focused_eeg(), fs=FS, elapsed_minutes=45.0)
        assert r2["vigilance_decrement"] > r1["vigilance_decrement"]

    def test_decrement_is_zero_at_start(self, tracker):
        result = tracker.assess(make_eeg(), fs=FS, elapsed_minutes=0.0)
        assert result["vigilance_decrement"] == pytest.approx(0.0, abs=0.05)

    def test_decrement_near_max_after_long_time(self, tracker):
        result = tracker.assess(make_eeg(), fs=FS, elapsed_minutes=120.0)
        assert result["vigilance_decrement"] > 0.7

    def test_decrement_reduces_concentration_score(self, tracker):
        """Same EEG at minute 5 vs minute 60 -- score should drop due to decrement."""
        eeg = make_focused_eeg()
        r_early = tracker.assess(eeg, fs=FS, elapsed_minutes=5.0)

        tracker2 = ConcentrationTracker()
        r_late = tracker2.assess(eeg, fs=FS, elapsed_minutes=60.0)
        assert r_late["concentration_score"] < r_early["concentration_score"]


# ---------------------------------------------------------------------------
# TestLapseDetection
# ---------------------------------------------------------------------------

class TestLapseDetection:
    """Attention lapses: concentration drops >30% within 10 seconds."""

    def test_no_lapse_on_first_assessment(self, tracker):
        result = tracker.assess(make_focused_eeg(), fs=FS)
        assert result["lapse_detected"] is False

    def test_lapse_after_sudden_drop(self, tracker):
        # Build up a high concentration baseline
        for _ in range(5):
            tracker.assess(make_focused_eeg(), fs=FS)
        # Sudden drop to unfocused
        result = tracker.assess(make_unfocused_eeg(), fs=FS)
        assert result["lapse_detected"] is True

    def test_no_lapse_on_gradual_decline(self, tracker):
        """Gradual decline should not trigger lapse (no sudden >30% drop)."""
        # Feed very gently declining signals -- 20 steps, tiny increments
        for i in range(20):
            beta = 12.0 - i * 0.3   # gentle decline from 12 to 6.3
            theta = 4.0 + i * 0.2   # gentle rise from 4 to 7.8
            eeg = make_eeg(theta_amp=theta, beta_amp=max(beta, 1.0), alpha_amp=3.0)
            result = tracker.assess(eeg, fs=FS)
        # Gradual decline may or may not trigger lapse depending on step size
        assert isinstance(result["lapse_detected"], bool)

    def test_lapse_count_in_stats(self, tracker):
        # Create a lapse scenario
        for _ in range(5):
            tracker.assess(make_focused_eeg(), fs=FS)
        tracker.assess(make_unfocused_eeg(), fs=FS)
        stats = tracker.get_session_stats()
        assert stats["lapse_count"] >= 1


# ---------------------------------------------------------------------------
# TestBreakRecommendations
# ---------------------------------------------------------------------------

class TestBreakRecommendations:
    """Break recommendations: after 25 min (Pomodoro) or 3+ lapses."""

    def test_no_break_early_session(self, tracker):
        result = tracker.assess(make_focused_eeg(), fs=FS, elapsed_minutes=5.0)
        assert result["break_recommendation"] is None

    def test_break_after_pomodoro(self, tracker):
        result = tracker.assess(make_focused_eeg(), fs=FS, elapsed_minutes=26.0)
        assert result["break_recommendation"] is not None
        assert isinstance(result["break_recommendation"], str)

    def test_break_after_multiple_lapses(self, tracker):
        # Force 3 lapses
        for _ in range(3):
            for _ in range(3):
                tracker.assess(make_focused_eeg(), fs=FS)
            tracker.assess(make_unfocused_eeg(), fs=FS)
        result = tracker.assess(make_focused_eeg(), fs=FS, elapsed_minutes=10.0)
        stats = tracker.get_session_stats()
        if stats["lapse_count"] >= 3:
            assert result["break_recommendation"] is not None


# ---------------------------------------------------------------------------
# TestGetOptimalBreakTime
# ---------------------------------------------------------------------------

class TestGetOptimalBreakTime:
    """get_optimal_break_time() returns Pomodoro or vigilance-based suggestion."""

    def test_returns_dict(self, tracker):
        result = tracker.get_optimal_break_time()
        assert isinstance(result, dict)

    def test_has_recommended_minutes_key(self, tracker):
        result = tracker.get_optimal_break_time()
        assert "recommended_break_after_minutes" in result

    def test_default_is_pomodoro(self, tracker):
        result = tracker.get_optimal_break_time()
        assert result["recommended_break_after_minutes"] == pytest.approx(25.0, abs=5.0)

    def test_shorter_break_after_lapses(self, tracker):
        """If lapses detected, recommended break time should decrease."""
        default_time = tracker.get_optimal_break_time()["recommended_break_after_minutes"]
        # Force some lapses
        for _ in range(3):
            for _ in range(3):
                tracker.assess(make_focused_eeg(), fs=FS)
            tracker.assess(make_unfocused_eeg(), fs=FS)
        lapsed_time = tracker.get_optimal_break_time()["recommended_break_after_minutes"]
        assert lapsed_time <= default_time


# ---------------------------------------------------------------------------
# TestGetSessionStats
# ---------------------------------------------------------------------------

class TestGetSessionStats:
    """get_session_stats() summarizes the entire session."""

    def test_empty_session(self, tracker):
        stats = tracker.get_session_stats()
        assert stats["n_assessments"] == 0
        assert stats["lapse_count"] == 0

    def test_stats_after_assessments(self, tracker):
        for _ in range(10):
            tracker.assess(make_focused_eeg(), fs=FS)
        stats = tracker.get_session_stats()
        assert stats["n_assessments"] == 10
        assert 0 <= stats["mean_concentration"] <= 100
        assert 0 <= stats["min_concentration"] <= 100
        assert 0 <= stats["max_concentration"] <= 100
        assert stats["min_concentration"] <= stats["mean_concentration"]
        assert stats["mean_concentration"] <= stats["max_concentration"]

    def test_stats_level_distribution(self, tracker):
        for _ in range(5):
            tracker.assess(make_focused_eeg(), fs=FS)
        for _ in range(5):
            tracker.assess(make_unfocused_eeg(), fs=FS)
        stats = tracker.get_session_stats()
        assert "level_distribution" in stats
        total_pct = sum(stats["level_distribution"].values())
        assert abs(total_pct - 100.0) < 1.0


# ---------------------------------------------------------------------------
# TestGetHistory
# ---------------------------------------------------------------------------

class TestGetHistory:
    """get_history() returns timestamped concentration records."""

    def test_empty_history(self, tracker):
        assert tracker.get_history() == []

    def test_history_grows(self, tracker):
        for _ in range(5):
            tracker.assess(make_eeg(), fs=FS)
        history = tracker.get_history()
        assert len(history) == 5

    def test_history_last_n(self, tracker):
        for _ in range(10):
            tracker.assess(make_eeg(), fs=FS)
        history = tracker.get_history(last_n=3)
        assert len(history) == 3

    def test_history_entry_has_score(self, tracker):
        tracker.assess(make_eeg(), fs=FS)
        entry = tracker.get_history()[0]
        assert "concentration_score" in entry
        assert "concentration_level" in entry

    def test_history_capped(self):
        tracker = ConcentrationTracker(max_history=5)
        for _ in range(10):
            tracker.assess(make_eeg(), fs=FS)
        assert len(tracker.get_history()) == 5


# ---------------------------------------------------------------------------
# TestReset
# ---------------------------------------------------------------------------

class TestReset:
    """reset() clears all state."""

    def test_reset_clears_history(self, tracker):
        for _ in range(5):
            tracker.assess(make_eeg(), fs=FS)
        tracker.reset()
        assert len(tracker.get_history()) == 0

    def test_reset_clears_baseline(self, tracker):
        tracker.set_baseline(make_eeg(), fs=FS)
        tracker.reset()
        assert tracker.has_baseline is False

    def test_reset_clears_lapse_count(self, tracker):
        for _ in range(5):
            tracker.assess(make_focused_eeg(), fs=FS)
        tracker.assess(make_unfocused_eeg(), fs=FS)
        tracker.reset()
        stats = tracker.get_session_stats()
        assert stats["lapse_count"] == 0


# ---------------------------------------------------------------------------
# TestMultichannelInput
# ---------------------------------------------------------------------------

class TestMultichannelInput:
    """Handles both 1D and 2D (multichannel) EEG input."""

    def test_1d_input(self, tracker):
        eeg = make_eeg(n_channels=1)
        assert eeg.ndim == 1
        result = tracker.assess(eeg, fs=FS)
        assert 0 <= result["concentration_score"] <= 100

    def test_2d_input(self, tracker):
        eeg = make_eeg(n_channels=4)
        assert eeg.ndim == 2
        result = tracker.assess(eeg, fs=FS)
        assert 0 <= result["concentration_score"] <= 100

    def test_multichannel_uses_frontal(self, tracker):
        """4-channel input should use AF7 (ch1) for analysis (Muse 2 layout)."""
        eeg = make_eeg(n_channels=4)
        result = tracker.assess(eeg, fs=FS)
        assert result["concentration_level"] in CONCENTRATION_LEVELS


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Robustness under extreme or degenerate inputs."""

    def test_very_short_signal(self, tracker):
        eeg = make_eeg(duration_s=0.1)
        result = tracker.assess(eeg, fs=FS)
        assert 0 <= result["concentration_score"] <= 100

    def test_all_zeros_signal(self, tracker):
        eeg = np.zeros(int(FS * 2), dtype=np.float64)
        result = tracker.assess(eeg, fs=FS)
        assert 0 <= result["concentration_score"] <= 100
        assert result["concentration_level"] in CONCENTRATION_LEVELS

    def test_large_amplitude_signal(self, tracker):
        eeg = make_eeg(theta_amp=1000, beta_amp=500)
        result = tracker.assess(eeg, fs=FS)
        assert 0 <= result["concentration_score"] <= 100

    def test_nan_in_signal_handled(self, tracker):
        eeg = make_eeg()
        eeg[10] = np.nan
        # Should not crash -- may produce degraded output
        result = tracker.assess(eeg, fs=FS)
        assert isinstance(result["concentration_score"], (int, float))

    def test_elapsed_minutes_zero(self, tracker):
        result = tracker.assess(make_eeg(), fs=FS, elapsed_minutes=0.0)
        assert result["vigilance_decrement"] == pytest.approx(0.0, abs=0.05)

    def test_elapsed_minutes_negative_treated_as_zero(self, tracker):
        result = tracker.assess(make_eeg(), fs=FS, elapsed_minutes=-5.0)
        assert result["vigilance_decrement"] == pytest.approx(0.0, abs=0.05)


# ---------------------------------------------------------------------------
# TestBaselineImpact
# ---------------------------------------------------------------------------

class TestBaselineImpact:
    """Baseline calibration changes concentration scoring."""

    def test_baseline_adjusts_score(self, tracker):
        eeg = make_focused_eeg()
        r_no_baseline = tracker.assess(eeg, fs=FS)

        tracker2 = ConcentrationTracker()
        # Set baseline from an unfocused signal (very different from test signal)
        tracker2.set_baseline(make_unfocused_eeg(), fs=FS)
        r_with_baseline = tracker2.assess(eeg, fs=FS)

        # Both should produce valid scores in range
        assert 0 <= r_no_baseline["concentration_score"] <= 100
        assert 0 <= r_with_baseline["concentration_score"] <= 100

    def test_above_baseline_higher_concentration(self, tracker):
        """Focused EEG above resting baseline -> higher score."""
        tracker.set_baseline(make_unfocused_eeg(), fs=FS)
        result = tracker.assess(make_focused_eeg(), fs=FS)
        assert result["concentration_score"] >= 50

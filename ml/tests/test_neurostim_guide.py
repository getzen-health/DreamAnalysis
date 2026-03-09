"""Tests for NeurostimGuide — closed-loop neurostimulation guidance from EEG.

Covers:
- Initialization and defaults
- Baseline setting and IAF estimation
- Recommendation logic for all target states (focus, relaxation, mood, default)
- Contraindication detection (seizure-like, alpha-surplus)
- Safety disclaimer always present
- Multi-user isolation
- Session stats and history tracking
- History cap (500 entries)
- Edge cases (short signals, flat signals, 1D input, no baseline)
- Reset functionality
"""

import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.neurostim_guide import NeurostimGuide


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(fs=256, duration=4, freqs=None, amps=None):
    """Synthesize EEG with specific frequency content."""
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs
    signal = np.zeros(n_samples)
    if freqs and amps:
        for freq, amp in zip(freqs, amps):
            signal += amp * np.sin(2 * np.pi * freq * t)
    signal += np.random.randn(n_samples) * 0.1
    return signal


def _make_multichannel(fs=256, duration=4, freqs=None, amps=None):
    """4-channel synthetic Muse 2 EEG."""
    base = _make_signal(fs, duration, freqs, amps)
    channels = np.tile(base, (4, 1))
    for i in range(4):
        channels[i] += np.random.randn(channels.shape[1]) * 0.05
    return channels


def _alpha_dominant_signal(fs=256, duration=4):
    """Signal with strong alpha (10 Hz), moderate beta, low theta."""
    return _make_multichannel(fs, duration, freqs=[10, 6, 20], amps=[30, 5, 5])


def _theta_dominant_signal(fs=256, duration=4):
    """Signal with strong theta (6 Hz), moderate alpha/beta."""
    return _make_multichannel(fs, duration, freqs=[6, 10, 20], amps=[30, 5, 5])


def _beta_dominant_signal(fs=256, duration=4):
    """Signal with strong beta (20 Hz), low alpha/theta."""
    return _make_multichannel(fs, duration, freqs=[20, 10, 6], amps=[30, 5, 5])


def _seizure_like_signal(fs=256, duration=4):
    """Very high amplitude signal (>200 uV) mimicking seizure-like activity."""
    return _make_multichannel(fs, duration, freqs=[10, 6, 20], amps=[250, 100, 100])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def guide():
    return NeurostimGuide()


@pytest.fixture
def baselined_guide():
    g = NeurostimGuide()
    baseline = _alpha_dominant_signal()
    g.set_baseline(baseline)
    return g


# ===========================================================================
# 1. Initialization
# ===========================================================================

class TestInit:
    def test_default_fs(self, guide):
        assert guide.fs == 256.0

    def test_custom_fs(self):
        g = NeurostimGuide(fs=512.0)
        assert g.fs == 512.0

    def test_no_baseline_initially(self, guide):
        result = guide.recommend(_alpha_dominant_signal())
        assert result["has_baseline"] is False


# ===========================================================================
# 2. Baseline Setting
# ===========================================================================

class TestBaseline:
    def test_set_baseline_returns_dict(self, guide):
        result = guide.set_baseline(_alpha_dominant_signal())
        assert isinstance(result, dict)
        assert result["baseline_set"] is True

    def test_set_baseline_has_iaf(self, guide):
        result = guide.set_baseline(_alpha_dominant_signal())
        assert "iaf" in result
        # Alpha-dominant signal should yield IAF near 10 Hz
        assert 7.0 <= result["iaf"] <= 14.0

    def test_set_baseline_has_alpha_theta(self, guide):
        result = guide.set_baseline(_alpha_dominant_signal())
        assert "baseline_alpha" in result
        assert "baseline_theta" in result
        assert result["baseline_alpha"] > 0.0
        assert result["baseline_theta"] > 0.0

    def test_set_baseline_custom_fs(self, guide):
        result = guide.set_baseline(_alpha_dominant_signal(fs=512, duration=4), fs=512.0)
        assert result["baseline_set"] is True

    def test_baseline_updates_has_baseline(self, guide):
        guide.set_baseline(_alpha_dominant_signal())
        result = guide.recommend(_alpha_dominant_signal())
        assert result["has_baseline"] is True

    def test_baseline_per_user(self, guide):
        guide.set_baseline(_alpha_dominant_signal(), user_id="alice")
        # Default user has no baseline
        result = guide.recommend(_alpha_dominant_signal(), user_id="bob")
        assert result["has_baseline"] is False
        result_alice = guide.recommend(_alpha_dominant_signal(), user_id="alice")
        assert result_alice["has_baseline"] is True

    def test_set_baseline_1d_signal(self, guide):
        """1D signal should work — uses single channel for estimation."""
        sig = _make_signal(freqs=[10, 6, 20], amps=[30, 5, 5])
        result = guide.set_baseline(sig)
        assert result["baseline_set"] is True


# ===========================================================================
# 3. Recommendation Output Structure
# ===========================================================================

class TestRecommendStructure:
    def test_all_keys_present(self, baselined_guide):
        result = baselined_guide.recommend(_alpha_dominant_signal())
        required_keys = [
            "protocol", "target_frequency", "current_alpha_power",
            "current_theta_power", "alpha_deficit", "theta_excess",
            "readiness_score", "contraindication_check", "disclaimer",
            "has_baseline",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_disclaimer_always_present(self, guide):
        result = guide.recommend(_alpha_dominant_signal())
        assert "disclaimer" in result
        assert "advisory" in result["disclaimer"].lower() or "clinical" in result["disclaimer"].lower()

    def test_contraindication_has_safe_and_reasons(self, baselined_guide):
        result = baselined_guide.recommend(_alpha_dominant_signal())
        cc = result["contraindication_check"]
        assert "safe" in cc
        assert "reasons" in cc
        assert isinstance(cc["reasons"], list)

    def test_readiness_score_range(self, baselined_guide):
        result = baselined_guide.recommend(_alpha_dominant_signal())
        assert 0 <= result["readiness_score"] <= 100

    def test_protocol_is_valid_string(self, baselined_guide):
        result = baselined_guide.recommend(_alpha_dominant_signal())
        valid_protocols = {"tacs_alpha", "tdcs_left_dlpfc", "tacs_theta", "no_stimulation"}
        assert result["protocol"] in valid_protocols


# ===========================================================================
# 4. Target State Logic
# ===========================================================================

class TestTargetStates:
    def test_relaxation_recommends_tacs_alpha(self, baselined_guide):
        """When alpha is below baseline, relaxation should recommend tacs_alpha."""
        # Beta-dominant = alpha deficit
        result = baselined_guide.recommend(_beta_dominant_signal(), target_state="relaxation")
        assert result["protocol"] == "tacs_alpha"

    def test_focus_with_theta_excess(self, baselined_guide):
        """Focus with excessive theta should recommend tacs_theta or tdcs."""
        result = baselined_guide.recommend(_theta_dominant_signal(), target_state="focus")
        assert result["protocol"] in {"tacs_theta", "tdcs_left_dlpfc"}

    def test_mood_recommends_tdcs(self, baselined_guide):
        """Mood target should recommend tdcs_left_dlpfc."""
        result = baselined_guide.recommend(_beta_dominant_signal(), target_state="mood")
        assert result["protocol"] == "tdcs_left_dlpfc"

    def test_default_target_is_adaptive(self, baselined_guide):
        """Default target should produce a recommendation without error."""
        result = baselined_guide.recommend(_alpha_dominant_signal(), target_state="default")
        assert result["protocol"] in {"tacs_alpha", "tdcs_left_dlpfc", "tacs_theta", "no_stimulation"}

    def test_target_frequency_is_iaf_for_alpha_protocol(self, baselined_guide):
        """When protocol is tacs_alpha, target_frequency should be near IAF."""
        result = baselined_guide.recommend(_beta_dominant_signal(), target_state="relaxation")
        if result["protocol"] == "tacs_alpha":
            assert 7.0 <= result["target_frequency"] <= 14.0

    def test_alpha_already_high_no_stimulation(self, baselined_guide):
        """If alpha is already very high, relaxation should say no_stimulation."""
        # Strong alpha signal — well above baseline
        very_strong_alpha = _make_multichannel(freqs=[10], amps=[50])
        result = baselined_guide.recommend(very_strong_alpha, target_state="relaxation")
        # Either no_stimulation or contraindication flagged
        cc = result["contraindication_check"]
        if result["alpha_deficit"] is False:
            # No deficit means no need to boost alpha
            assert result["protocol"] == "no_stimulation" or not cc["safe"]


# ===========================================================================
# 5. Contraindications
# ===========================================================================

class TestContraindications:
    def test_seizure_like_flagged(self, baselined_guide):
        """Seizure-like signal (>200 uV) should flag as unsafe."""
        result = baselined_guide.recommend(_seizure_like_signal())
        cc = result["contraindication_check"]
        assert cc["safe"] is False
        assert len(cc["reasons"]) > 0

    def test_normal_signal_is_safe(self, baselined_guide):
        result = baselined_guide.recommend(_alpha_dominant_signal())
        cc = result["contraindication_check"]
        assert cc["safe"] is True
        assert len(cc["reasons"]) == 0

    def test_seizure_forces_no_stimulation(self, baselined_guide):
        """Unsafe contraindication should force no_stimulation protocol."""
        result = baselined_guide.recommend(_seizure_like_signal())
        assert result["protocol"] == "no_stimulation"


# ===========================================================================
# 6. Alpha Deficit and Theta Excess Detection
# ===========================================================================

class TestDeficitDetection:
    def test_alpha_deficit_detected(self, baselined_guide):
        """Beta-dominant signal should show alpha deficit relative to baseline."""
        result = baselined_guide.recommend(_beta_dominant_signal())
        assert result["alpha_deficit"] is True

    def test_no_alpha_deficit_when_alpha_high(self, baselined_guide):
        result = baselined_guide.recommend(_alpha_dominant_signal())
        assert result["alpha_deficit"] is False

    def test_theta_excess_detected(self, baselined_guide):
        """Theta-dominant signal should show theta excess."""
        result = baselined_guide.recommend(_theta_dominant_signal())
        assert result["theta_excess"] is True

    def test_no_theta_excess_when_beta_dominant(self, baselined_guide):
        result = baselined_guide.recommend(_beta_dominant_signal())
        assert result["theta_excess"] is False


# ===========================================================================
# 7. Session Stats
# ===========================================================================

class TestSessionStats:
    def test_initial_stats_empty(self, guide):
        stats = guide.get_session_stats()
        assert stats["n_epochs"] == 0

    def test_stats_count_after_recommendations(self, baselined_guide):
        baselined_guide.recommend(_alpha_dominant_signal())
        baselined_guide.recommend(_beta_dominant_signal())
        stats = baselined_guide.get_session_stats()
        assert stats["n_epochs"] == 2

    def test_stats_has_protocol_distribution(self, baselined_guide):
        baselined_guide.recommend(_alpha_dominant_signal(), target_state="relaxation")
        stats = baselined_guide.get_session_stats()
        assert "recommended_protocols" in stats
        assert isinstance(stats["recommended_protocols"], dict)

    def test_stats_has_mean_readiness(self, baselined_guide):
        baselined_guide.recommend(_alpha_dominant_signal())
        stats = baselined_guide.get_session_stats()
        assert "mean_readiness" in stats
        assert isinstance(stats["mean_readiness"], (int, float))

    def test_stats_per_user(self, guide):
        guide.set_baseline(_alpha_dominant_signal(), user_id="alice")
        guide.recommend(_alpha_dominant_signal(), user_id="alice")
        stats_alice = guide.get_session_stats(user_id="alice")
        stats_default = guide.get_session_stats(user_id="default")
        assert stats_alice["n_epochs"] == 1
        assert stats_default["n_epochs"] == 0


# ===========================================================================
# 8. History
# ===========================================================================

class TestHistory:
    def test_history_initially_empty(self, guide):
        assert guide.get_history() == []

    def test_history_grows_with_recommendations(self, baselined_guide):
        baselined_guide.recommend(_alpha_dominant_signal())
        baselined_guide.recommend(_beta_dominant_signal())
        history = baselined_guide.get_history()
        assert len(history) == 2

    def test_history_last_n(self, baselined_guide):
        for _ in range(5):
            baselined_guide.recommend(_alpha_dominant_signal())
        history = baselined_guide.get_history(last_n=3)
        assert len(history) == 3

    def test_history_cap_500(self, baselined_guide):
        """History should never exceed 500 entries."""
        sig = _alpha_dominant_signal()
        for _ in range(510):
            baselined_guide.recommend(sig)
        history = baselined_guide.get_history()
        assert len(history) <= 500

    def test_history_per_user(self, guide):
        guide.set_baseline(_alpha_dominant_signal(), user_id="alice")
        guide.recommend(_alpha_dominant_signal(), user_id="alice")
        assert len(guide.get_history(user_id="alice")) == 1
        assert len(guide.get_history(user_id="default")) == 0


# ===========================================================================
# 9. Reset
# ===========================================================================

class TestReset:
    def test_reset_clears_history(self, baselined_guide):
        baselined_guide.recommend(_alpha_dominant_signal())
        baselined_guide.reset()
        assert len(baselined_guide.get_history()) == 0

    def test_reset_clears_baseline(self, baselined_guide):
        baselined_guide.reset()
        result = baselined_guide.recommend(_alpha_dominant_signal())
        assert result["has_baseline"] is False

    def test_reset_only_affects_target_user(self, guide):
        guide.set_baseline(_alpha_dominant_signal(), user_id="alice")
        guide.set_baseline(_alpha_dominant_signal(), user_id="bob")
        guide.recommend(_alpha_dominant_signal(), user_id="alice")
        guide.recommend(_alpha_dominant_signal(), user_id="bob")
        guide.reset(user_id="alice")
        assert len(guide.get_history(user_id="alice")) == 0
        assert len(guide.get_history(user_id="bob")) == 1


# ===========================================================================
# 10. Edge Cases
# ===========================================================================

class TestEdgeCases:
    def test_1d_signal_works(self, guide):
        """Single-channel 1D signal should not crash."""
        sig = _make_signal(freqs=[10, 6, 20], amps=[20, 10, 10])
        result = guide.recommend(sig)
        assert "protocol" in result

    def test_short_signal(self, guide):
        """Very short signal (< 1 sec) should still produce a result."""
        short = np.random.randn(4, 64) * 10
        result = guide.recommend(short)
        assert "protocol" in result

    def test_flat_signal(self, guide):
        """Flat-line signal should not crash."""
        flat = np.ones((4, 1024)) * 0.001
        result = guide.recommend(flat)
        assert "protocol" in result

    def test_no_baseline_still_recommends(self, guide):
        """Without baseline, should still produce recommendations with lower confidence."""
        result = guide.recommend(_alpha_dominant_signal())
        assert result["has_baseline"] is False
        assert result["protocol"] in {"tacs_alpha", "tdcs_left_dlpfc", "tacs_theta", "no_stimulation"}

    def test_readiness_lower_without_baseline(self, guide):
        """Readiness score should be lower when no baseline is set."""
        sig = _beta_dominant_signal()
        result_no_bl = guide.recommend(sig, target_state="focus")
        guide.set_baseline(_alpha_dominant_signal())
        result_bl = guide.recommend(sig, target_state="focus")
        assert result_no_bl["readiness_score"] <= result_bl["readiness_score"]

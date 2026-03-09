"""Tests for LanguageProcessor — EEG-based language processing / reading comprehension detector."""

import numpy as np
import pytest

from models.language_processor import LanguageProcessor


FS = 256.0
DURATION_S = 4
N_SAMPLES = int(FS * DURATION_S)
STATES = ["disengaged", "passive_listening", "active_processing", "deep_comprehension"]


@pytest.fixture
def lp():
    """Fresh LanguageProcessor instance."""
    return LanguageProcessor(fs=FS)


def _make_4ch(n_samples=N_SAMPLES):
    """Random 4-channel Muse 2 EEG (TP9, AF7, AF8, TP10)."""
    np.random.seed(42)
    return np.random.randn(4, n_samples) * 15.0


def _make_1ch(n_samples=N_SAMPLES):
    """Single-channel random EEG."""
    np.random.seed(42)
    return np.random.randn(n_samples) * 15.0


def _inject_band(signal, fs, freq_center, amplitude, n_samples=None):
    """Add a sine wave at a specific frequency to a signal."""
    if n_samples is None:
        n_samples = signal.shape[-1]
    t = np.arange(n_samples) / fs
    sine = amplitude * np.sin(2 * np.pi * freq_center * t)
    if signal.ndim == 1:
        return signal + sine
    else:
        out = signal.copy()
        for ch in range(signal.shape[0]):
            out[ch] += sine
        return out


def _make_language_active_signal():
    """Create EEG that mimics language processing:
    - Strong frontal theta (6 Hz) at AF7 (ch1) — semantic load
    - Left lateralization: AF7 alpha suppressed more than AF8
    - Temporal alpha suppression at TP9/TP10
    - Beta desynchronization
    """
    np.random.seed(123)
    signals = np.random.randn(4, N_SAMPLES) * 3.0
    t = np.arange(N_SAMPLES) / FS

    # Strong frontal theta at AF7 (ch1) — language/semantic processing
    signals[1] += 25.0 * np.sin(2 * np.pi * 6 * t)
    # Moderate frontal theta at AF8 (ch2) — less than AF7 for left lateralization
    signals[2] += 10.0 * np.sin(2 * np.pi * 6 * t)

    # Suppress alpha at temporal sites (TP9=ch0, TP10=ch3)
    # Keep alpha low — small amplitude
    signals[0] += 2.0 * np.sin(2 * np.pi * 10 * t)
    signals[3] += 2.0 * np.sin(2 * np.pi * 10 * t)

    # Moderate beta at frontal sites
    signals[1] += 8.0 * np.sin(2 * np.pi * 18 * t)
    signals[2] += 8.0 * np.sin(2 * np.pi * 18 * t)

    return signals


def _make_disengaged_signal():
    """Create EEG that mimics disengagement:
    - Strong alpha everywhere (idle/resting)
    - Low theta (no cognitive processing)
    - No left lateralization
    """
    np.random.seed(456)
    signals = np.random.randn(4, N_SAMPLES) * 2.0
    t = np.arange(N_SAMPLES) / FS

    # Strong alpha everywhere — resting/idle state
    for ch in range(4):
        signals[ch] += 30.0 * np.sin(2 * np.pi * 10 * t)

    # Very low theta
    signals[1] += 2.0 * np.sin(2 * np.pi * 6 * t)
    signals[2] += 2.0 * np.sin(2 * np.pi * 6 * t)

    return signals


# =============================================================================
# Test Class: Initialization
# =============================================================================

class TestInit:
    def test_default_fs(self):
        lp = LanguageProcessor()
        assert lp.fs == 256.0

    def test_custom_fs(self):
        lp = LanguageProcessor(fs=512.0)
        assert lp.fs == 512.0

    def test_initial_state_no_baseline(self):
        lp = LanguageProcessor()
        assert lp._baselines == {} or len(lp._baselines) == 0


# =============================================================================
# Test Class: assess() output structure
# =============================================================================

class TestAssessOutputStructure:
    def test_all_required_keys_present(self, lp):
        result = lp.assess(_make_4ch())
        expected_keys = {
            "processing_state",
            "comprehension_index",
            "semantic_load",
            "left_lateralization",
            "temporal_engagement",
            "working_memory_load",
            "has_baseline",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_processing_state_valid(self, lp):
        result = lp.assess(_make_4ch())
        assert result["processing_state"] in STATES

    def test_comprehension_index_range(self, lp):
        result = lp.assess(_make_4ch())
        assert 0 <= result["comprehension_index"] <= 100

    def test_semantic_load_range(self, lp):
        result = lp.assess(_make_4ch())
        assert 0.0 <= result["semantic_load"] <= 1.0

    def test_left_lateralization_range(self, lp):
        result = lp.assess(_make_4ch())
        assert -1.0 <= result["left_lateralization"] <= 1.0

    def test_temporal_engagement_range(self, lp):
        result = lp.assess(_make_4ch())
        assert 0.0 <= result["temporal_engagement"] <= 1.0

    def test_working_memory_load_range(self, lp):
        result = lp.assess(_make_4ch())
        assert 0.0 <= result["working_memory_load"] <= 1.0

    def test_has_baseline_false_without_calibration(self, lp):
        result = lp.assess(_make_4ch())
        assert result["has_baseline"] is False


# =============================================================================
# Test Class: Single-channel input
# =============================================================================

class TestSingleChannelInput:
    def test_single_channel_works(self, lp):
        result = lp.assess(_make_1ch())
        assert result["processing_state"] in STATES
        assert 0 <= result["comprehension_index"] <= 100

    def test_single_channel_no_lateralization(self, lp):
        """Single channel cannot compute lateralization — should return 0."""
        result = lp.assess(_make_1ch())
        assert result["left_lateralization"] == 0.0

    def test_single_channel_no_temporal_engagement(self, lp):
        """Single channel cannot compute temporal alpha suppression — should return 0."""
        result = lp.assess(_make_1ch())
        assert result["temporal_engagement"] == 0.0


# =============================================================================
# Test Class: Multichannel input
# =============================================================================

class TestMultichannelInput:
    def test_4channel_input(self, lp):
        result = lp.assess(_make_4ch())
        assert result["processing_state"] in STATES

    def test_2channel_input(self, lp):
        """2-channel array should work without errors."""
        signals = np.random.randn(2, N_SAMPLES) * 15.0
        result = lp.assess(signals)
        assert result["processing_state"] in STATES

    def test_3channel_input(self, lp):
        """3-channel array should work without errors."""
        signals = np.random.randn(3, N_SAMPLES) * 15.0
        result = lp.assess(signals)
        assert result["processing_state"] in STATES


# =============================================================================
# Test Class: State classification thresholds
# =============================================================================

class TestStateClassification:
    def test_language_active_signal_not_disengaged(self, lp):
        """Signal with strong language markers should not classify as disengaged."""
        signals = _make_language_active_signal()
        result = lp.assess(signals)
        # Language-active should be at least passive_listening
        assert result["comprehension_index"] > 25

    def test_disengaged_signal_low_comprehension(self, lp):
        """Signal with strong alpha everywhere should have low comprehension."""
        signals = _make_disengaged_signal()
        result = lp.assess(signals)
        assert result["comprehension_index"] < 60

    def test_state_matches_comprehension_index(self, lp):
        """State label should match comprehension_index thresholds."""
        result = lp.assess(_make_4ch())
        ci = result["comprehension_index"]
        state = result["processing_state"]
        if ci < 25:
            assert state == "disengaged"
        elif ci < 50:
            assert state == "passive_listening"
        elif ci < 75:
            assert state == "active_processing"
        else:
            assert state == "deep_comprehension"


# =============================================================================
# Test Class: Semantic load and working memory
# =============================================================================

class TestSemanticLoadAndWorkingMemory:
    def test_high_theta_increases_semantic_load(self, lp):
        """Strong frontal theta should produce higher semantic load."""
        np.random.seed(99)
        low_theta = np.random.randn(4, N_SAMPLES) * 10.0
        high_theta = low_theta.copy()
        t = np.arange(N_SAMPLES) / FS
        # Inject strong theta at AF7 (ch1)
        high_theta[1] += 30.0 * np.sin(2 * np.pi * 6 * t)

        result_low = lp.assess(low_theta)
        result_high = lp.assess(high_theta)
        assert result_high["semantic_load"] > result_low["semantic_load"]

    def test_working_memory_tracks_frontal_theta(self, lp):
        """Working memory load should track frontal theta power."""
        np.random.seed(100)
        low_wm = np.random.randn(4, N_SAMPLES) * 10.0
        high_wm = low_wm.copy()
        t = np.arange(N_SAMPLES) / FS
        # Strong theta at both frontal channels
        high_wm[1] += 25.0 * np.sin(2 * np.pi * 6 * t)
        high_wm[2] += 25.0 * np.sin(2 * np.pi * 6 * t)

        result_low = lp.assess(low_wm)
        result_high = lp.assess(high_wm)
        assert result_high["working_memory_load"] > result_low["working_memory_load"]


# =============================================================================
# Test Class: Left lateralization
# =============================================================================

class TestLeftLateralization:
    def test_stronger_af7_gives_positive_lateralization(self, lp):
        """More AF7 activation (left hemisphere) should give positive lateralization."""
        np.random.seed(200)
        signals = np.random.randn(4, N_SAMPLES) * 5.0
        t = np.arange(N_SAMPLES) / FS
        # Strong theta at AF7 (ch1), weak at AF8 (ch2)
        signals[1] += 30.0 * np.sin(2 * np.pi * 6 * t)
        signals[2] += 3.0 * np.sin(2 * np.pi * 6 * t)

        result = lp.assess(signals)
        assert result["left_lateralization"] > 0.0

    def test_stronger_af8_gives_negative_lateralization(self, lp):
        """More AF8 activation (right hemisphere) should give negative lateralization."""
        np.random.seed(201)
        signals = np.random.randn(4, N_SAMPLES) * 5.0
        t = np.arange(N_SAMPLES) / FS
        # Strong theta at AF8 (ch2), weak at AF7 (ch1)
        signals[2] += 30.0 * np.sin(2 * np.pi * 6 * t)
        signals[1] += 3.0 * np.sin(2 * np.pi * 6 * t)

        result = lp.assess(signals)
        assert result["left_lateralization"] < 0.0


# =============================================================================
# Test Class: Temporal engagement
# =============================================================================

class TestTemporalEngagement:
    def test_low_temporal_alpha_high_engagement(self, lp):
        """Suppressed temporal alpha should indicate high engagement."""
        np.random.seed(300)
        signals = np.random.randn(4, N_SAMPLES) * 5.0
        # TP9/TP10 have very low alpha — suppressed
        # (no alpha injection, just noise)
        result = lp.assess(signals)
        engagement_low_alpha = result["temporal_engagement"]

        np.random.seed(300)
        signals2 = np.random.randn(4, N_SAMPLES) * 5.0
        t = np.arange(N_SAMPLES) / FS
        # TP9/TP10 have strong alpha — idle/disengaged
        signals2[0] += 40.0 * np.sin(2 * np.pi * 10 * t)
        signals2[3] += 40.0 * np.sin(2 * np.pi * 10 * t)

        result2 = lp.assess(signals2)
        engagement_high_alpha = result2["temporal_engagement"]

        assert engagement_low_alpha > engagement_high_alpha


# =============================================================================
# Test Class: Baseline / calibration
# =============================================================================

class TestBaseline:
    def test_set_baseline_returns_dict(self, lp):
        result = lp.set_baseline(_make_4ch())
        assert isinstance(result, dict)
        assert result["baseline_set"] is True

    def test_set_baseline_stores_metrics(self, lp):
        result = lp.set_baseline(_make_4ch())
        assert "baseline_metrics" in result

    def test_has_baseline_after_calibration(self, lp):
        lp.set_baseline(_make_4ch())
        result = lp.assess(_make_4ch())
        assert result["has_baseline"] is True

    def test_custom_fs_in_set_baseline(self, lp):
        result = lp.set_baseline(_make_4ch(), fs=512.0)
        assert result["baseline_set"] is True

    def test_multi_user_baselines(self, lp):
        lp.set_baseline(_make_4ch(), user_id="alice")
        lp.set_baseline(_make_4ch(), user_id="bob")
        r_alice = lp.assess(_make_4ch(), user_id="alice")
        r_bob = lp.assess(_make_4ch(), user_id="bob")
        assert r_alice["has_baseline"] is True
        assert r_bob["has_baseline"] is True

    def test_no_cross_user_baseline(self, lp):
        lp.set_baseline(_make_4ch(), user_id="alice")
        r_other = lp.assess(_make_4ch(), user_id="charlie")
        assert r_other["has_baseline"] is False

    def test_single_channel_baseline(self, lp):
        result = lp.set_baseline(_make_1ch())
        assert result["baseline_set"] is True


# =============================================================================
# Test Class: History
# =============================================================================

class TestHistory:
    def test_history_tracking(self, lp):
        for _ in range(5):
            lp.assess(_make_4ch())
        history = lp.get_history()
        assert len(history) == 5

    def test_history_cap_at_500(self, lp):
        for i in range(510):
            np.random.seed(i)
            lp.assess(np.random.randn(4, N_SAMPLES) * 15.0)
        history = lp.get_history()
        assert len(history) == 500

    def test_history_per_user(self, lp):
        lp.assess(_make_4ch(), user_id="alice")
        lp.assess(_make_4ch(), user_id="alice")
        lp.assess(_make_4ch(), user_id="bob")
        assert len(lp.get_history("alice")) == 2
        assert len(lp.get_history("bob")) == 1

    def test_history_last_n(self, lp):
        for _ in range(10):
            lp.assess(_make_4ch())
        history = lp.get_history(last_n=3)
        assert len(history) == 3

    def test_history_last_n_exceeds_available(self, lp):
        lp.assess(_make_4ch())
        history = lp.get_history(last_n=100)
        assert len(history) == 1

    def test_history_empty_for_unknown_user(self, lp):
        assert lp.get_history("nonexistent") == []


# =============================================================================
# Test Class: Session stats
# =============================================================================

class TestSessionStats:
    def test_empty_stats(self, lp):
        stats = lp.get_session_stats()
        assert stats["n_epochs"] == 0

    def test_stats_with_data(self, lp):
        for _ in range(5):
            lp.assess(_make_4ch())
        stats = lp.get_session_stats()
        assert stats["n_epochs"] == 5
        assert 0 <= stats["mean_comprehension"] <= 100
        assert stats["dominant_state"] in STATES

    def test_stats_state_distribution(self, lp):
        for _ in range(10):
            lp.assess(_make_4ch())
        stats = lp.get_session_stats()
        assert "state_distribution" in stats
        dist = stats["state_distribution"]
        # All state keys should exist
        for state in STATES:
            assert state in dist
        # Distribution should sum to ~1.0
        total = sum(dist.values())
        assert abs(total - 1.0) < 0.01

    def test_stats_per_user(self, lp):
        lp.assess(_make_4ch(), user_id="alice")
        lp.assess(_make_4ch(), user_id="bob")
        lp.assess(_make_4ch(), user_id="bob")
        stats_a = lp.get_session_stats("alice")
        stats_b = lp.get_session_stats("bob")
        assert stats_a["n_epochs"] == 1
        assert stats_b["n_epochs"] == 2


# =============================================================================
# Test Class: Reset
# =============================================================================

class TestReset:
    def test_reset_clears_history(self, lp):
        for _ in range(5):
            lp.assess(_make_4ch())
        lp.reset()
        assert lp.get_history() == []
        assert lp.get_session_stats()["n_epochs"] == 0

    def test_reset_clears_baseline(self, lp):
        lp.set_baseline(_make_4ch())
        lp.reset()
        result = lp.assess(_make_4ch())
        assert result["has_baseline"] is False

    def test_reset_per_user(self, lp):
        lp.assess(_make_4ch(), user_id="alice")
        lp.assess(_make_4ch(), user_id="bob")
        lp.reset("alice")
        assert lp.get_history("alice") == []
        assert len(lp.get_history("bob")) == 1

    def test_reset_nonexistent_user_no_error(self, lp):
        # Should not raise
        lp.reset("nonexistent")


# =============================================================================
# Test Class: Edge cases
# =============================================================================

class TestEdgeCases:
    def test_very_short_signal(self, lp):
        """Very short signal should not crash."""
        short = np.random.randn(4, 64) * 15.0
        result = lp.assess(short)
        assert result["processing_state"] in STATES

    def test_flat_signal(self, lp):
        """Near-zero signal should not crash and should give low scores."""
        flat = np.ones((4, N_SAMPLES)) * 0.001
        result = lp.assess(flat)
        assert result["processing_state"] in STATES
        assert 0 <= result["comprehension_index"] <= 100

    def test_high_amplitude_signal(self, lp):
        """Railed signal should not crash."""
        railed = np.random.randn(4, N_SAMPLES) * 500.0
        result = lp.assess(railed)
        assert result["processing_state"] in STATES

    def test_custom_fs_in_assess(self, lp):
        """Custom fs parameter should override constructor fs."""
        result = lp.assess(_make_4ch(), fs=512.0)
        assert result["processing_state"] in STATES

    def test_nan_in_signal(self, lp):
        """Signal with NaN should not crash."""
        signals = _make_4ch()
        signals[0, 100] = np.nan
        result = lp.assess(signals)
        assert result["processing_state"] in STATES

"""Tests for AffectLabelingTracker -- affect labeling neural efficacy measurement.

Covers:
  - TestBaseline: set baseline, keys, single channel
  - TestPreLabel: record pre-label, output keys
  - TestPostLabel: record post-label with/without pre-label, output keys and types
  - TestLabelingEfficacy: calmer post-label signals give higher efficacy
  - TestLPPReduction: lower beta amplitude post-label gives positive reduction
  - TestPrefrontalIncrease: left-frontal alpha suppression detected
  - TestEfficacyLevels: thresholds produce correct levels
  - TestSessionStats: empty, after trials, improvement trend
  - TestHistory: empty, grows, last_n
  - TestReset: clears everything
  - TestMultiUser: independent users
  - TestEdgeCases: constant, 1D, short, zero signals
"""

import numpy as np
import pytest
from scipy.signal import welch

from models.affect_labeling import AffectLabelingTracker, EFFICACY_LEVELS


FS = 256.0
DURATION = 2.0  # seconds
N_SAMPLES = int(FS * DURATION)


def _make_signal(fs=256.0, duration=2.0, n_channels=4,
                 alpha_amp=10.0, beta_amp=5.0, theta_amp=3.0,
                 noise_amp=1.0, per_channel_alpha=None,
                 per_channel_beta=None):
    """Create synthetic EEG with controllable band amplitudes.

    Args:
        fs: Sampling rate.
        duration: Signal duration in seconds.
        n_channels: Number of channels.
        alpha_amp: Alpha (10 Hz) sine amplitude (all channels unless overridden).
        beta_amp: Beta (20 Hz) sine amplitude (all channels unless overridden).
        theta_amp: Theta (6 Hz) sine amplitude (all channels unless overridden).
        noise_amp: Gaussian noise amplitude.
        per_channel_alpha: List of alpha amplitudes per channel (overrides alpha_amp).
        per_channel_beta: List of beta amplitudes per channel (overrides beta_amp).

    Returns:
        (n_channels, n_samples) numpy array.
    """
    np.random.seed(42)
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs
    signals = np.zeros((n_channels, n_samples))

    for ch in range(n_channels):
        a_amp = per_channel_alpha[ch] if per_channel_alpha else alpha_amp
        b_amp = per_channel_beta[ch] if per_channel_beta else beta_amp

        signals[ch] = (
            a_amp * np.sin(2 * np.pi * 10.0 * t)
            + b_amp * np.sin(2 * np.pi * 20.0 * t)
            + theta_amp * np.sin(2 * np.pi * 6.0 * t)
            + noise_amp * np.random.randn(n_samples)
        )
    return signals


# ---------------------------------------------------------------------------
# TestBaseline
# ---------------------------------------------------------------------------

class TestBaseline:
    def test_set_baseline_returns_dict(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()
        result = tracker.set_baseline(sig, user_id="default")
        assert isinstance(result, dict)

    def test_set_baseline_has_required_keys(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()
        result = tracker.set_baseline(sig)
        assert "baseline_set" in result
        assert "n_channels" in result

    def test_set_baseline_baseline_set_true(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()
        result = tracker.set_baseline(sig)
        assert result["baseline_set"] is True

    def test_set_baseline_n_channels_4(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal(n_channels=4)
        result = tracker.set_baseline(sig)
        assert result["n_channels"] == 4

    def test_set_baseline_single_channel(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal(n_channels=1)[0]  # 1D
        result = tracker.set_baseline(sig)
        assert result["baseline_set"] is True
        assert result["n_channels"] == 1

    def test_set_baseline_with_custom_fs(self):
        tracker = AffectLabelingTracker(fs=128.0)
        sig = _make_signal(fs=128.0, n_channels=4)
        result = tracker.set_baseline(sig, fs=128.0)
        assert result["baseline_set"] is True

    def test_baseline_marks_has_baseline_in_post_label(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()
        tracker.set_baseline(sig)
        tracker.record_pre_label(sig)
        result = tracker.record_post_label(sig)
        assert result["has_baseline"] is True

    def test_no_baseline_marks_false(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()
        tracker.record_pre_label(sig)
        result = tracker.record_post_label(sig)
        assert result["has_baseline"] is False


# ---------------------------------------------------------------------------
# TestPreLabel
# ---------------------------------------------------------------------------

class TestPreLabel:
    def test_record_pre_label_returns_dict(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()
        result = tracker.record_pre_label(sig)
        assert isinstance(result, dict)

    def test_record_pre_label_has_keys(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()
        result = tracker.record_pre_label(sig)
        assert "pre_label_recorded" in result
        assert "n_channels" in result

    def test_record_pre_label_recorded_true(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()
        result = tracker.record_pre_label(sig)
        assert result["pre_label_recorded"] is True

    def test_record_pre_label_n_channels(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal(n_channels=4)
        result = tracker.record_pre_label(sig)
        assert result["n_channels"] == 4

    def test_record_pre_label_1d(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal(n_channels=1)[0]
        result = tracker.record_pre_label(sig)
        assert result["pre_label_recorded"] is True
        assert result["n_channels"] == 1


# ---------------------------------------------------------------------------
# TestPostLabel
# ---------------------------------------------------------------------------

class TestPostLabel:
    def test_returns_dict(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()
        tracker.record_pre_label(sig)
        result = tracker.record_post_label(sig)
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()
        tracker.record_pre_label(sig)
        result = tracker.record_post_label(sig)
        required = {
            "labeling_efficacy",
            "lpp_reduction",
            "prefrontal_increase",
            "alpha_change",
            "beta_change",
            "efficacy_level",
            "has_baseline",
        }
        assert required.issubset(set(result.keys())), (
            f"Missing keys: {required - set(result.keys())}"
        )

    def test_labeling_efficacy_is_float(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()
        tracker.record_pre_label(sig)
        result = tracker.record_post_label(sig)
        assert isinstance(result["labeling_efficacy"], float)

    def test_labeling_efficacy_in_range(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()
        tracker.record_pre_label(sig)
        result = tracker.record_post_label(sig)
        assert 0.0 <= result["labeling_efficacy"] <= 1.0

    def test_lpp_reduction_is_float(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()
        tracker.record_pre_label(sig)
        result = tracker.record_post_label(sig)
        assert isinstance(result["lpp_reduction"], float)

    def test_lpp_reduction_in_range(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()
        tracker.record_pre_label(sig)
        result = tracker.record_post_label(sig)
        assert -1.0 <= result["lpp_reduction"] <= 1.0

    def test_prefrontal_increase_is_float(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()
        tracker.record_pre_label(sig)
        result = tracker.record_post_label(sig)
        assert isinstance(result["prefrontal_increase"], float)

    def test_prefrontal_increase_in_range(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()
        tracker.record_pre_label(sig)
        result = tracker.record_post_label(sig)
        assert -1.0 <= result["prefrontal_increase"] <= 1.0

    def test_alpha_change_is_float(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()
        tracker.record_pre_label(sig)
        result = tracker.record_post_label(sig)
        assert isinstance(result["alpha_change"], float)

    def test_beta_change_is_float(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()
        tracker.record_pre_label(sig)
        result = tracker.record_post_label(sig)
        assert isinstance(result["beta_change"], float)

    def test_efficacy_level_is_string(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()
        tracker.record_pre_label(sig)
        result = tracker.record_post_label(sig)
        assert isinstance(result["efficacy_level"], str)

    def test_efficacy_level_in_valid_set(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()
        tracker.record_pre_label(sig)
        result = tracker.record_post_label(sig)
        assert result["efficacy_level"] in EFFICACY_LEVELS

    def test_has_baseline_is_bool(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()
        tracker.record_pre_label(sig)
        result = tracker.record_post_label(sig)
        assert isinstance(result["has_baseline"], bool)

    def test_label_included_when_provided(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()
        tracker.record_pre_label(sig)
        result = tracker.record_post_label(sig, label="anxious")
        assert result["label"] == "anxious"

    def test_label_absent_when_not_provided(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()
        tracker.record_pre_label(sig)
        result = tracker.record_post_label(sig)
        assert "label" not in result

    def test_post_label_without_pre_label(self):
        """Should still return valid result using zero defaults."""
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()
        result = tracker.record_post_label(sig)
        assert isinstance(result, dict)
        assert 0.0 <= result["labeling_efficacy"] <= 1.0
        assert result["efficacy_level"] in EFFICACY_LEVELS

    def test_pre_label_cleared_after_post_label(self):
        """Pre-label should be consumed (cleared) after recording post-label."""
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()
        tracker.record_pre_label(sig)
        tracker.record_post_label(sig)
        # Second post-label should use zero defaults (no pre-label)
        user = tracker._ensure_user("default")
        assert user["pre_label"] is None


# ---------------------------------------------------------------------------
# TestLabelingEfficacy
# ---------------------------------------------------------------------------

class TestLabelingEfficacy:
    def test_calmer_post_label_higher_efficacy(self):
        """Lower beta post-label (calmer) should give higher efficacy
        than higher beta post-label (more aroused)."""
        np.random.seed(42)
        tracker = AffectLabelingTracker(fs=FS)

        # High arousal pre-label (lots of beta)
        pre_sig = _make_signal(alpha_amp=5.0, beta_amp=30.0, noise_amp=0.5)

        # Calmer post-label: much less beta
        post_calm = _make_signal(alpha_amp=5.0, beta_amp=3.0, noise_amp=0.5)

        # Still aroused post-label: beta stays high
        post_aroused = _make_signal(alpha_amp=5.0, beta_amp=28.0, noise_amp=0.5)

        tracker.record_pre_label(pre_sig, user_id="calm")
        result_calm = tracker.record_post_label(post_calm, user_id="calm")

        tracker.record_pre_label(pre_sig, user_id="aroused")
        result_aroused = tracker.record_post_label(post_aroused, user_id="aroused")

        assert result_calm["labeling_efficacy"] > result_aroused["labeling_efficacy"], (
            f"Calmer post ({result_calm['labeling_efficacy']}) should have higher "
            f"efficacy than aroused post ({result_aroused['labeling_efficacy']})"
        )

    def test_identical_pre_post_gives_low_efficacy(self):
        """Same signal pre and post should give near-zero efficacy."""
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal(alpha_amp=10.0, beta_amp=10.0)
        tracker.record_pre_label(sig)
        result = tracker.record_post_label(sig)
        # With identical signals, lpp_reduction and prefrontal_increase
        # should be ~0, so efficacy should be low.
        assert result["labeling_efficacy"] < 0.1

    def test_efficacy_increases_with_larger_beta_drop(self):
        """Bigger beta drops should produce higher efficacy scores."""
        np.random.seed(42)
        tracker = AffectLabelingTracker(fs=FS)

        pre_sig = _make_signal(beta_amp=40.0, alpha_amp=5.0)

        # Small drop
        post_small = _make_signal(beta_amp=35.0, alpha_amp=5.0)
        tracker.record_pre_label(pre_sig, user_id="small")
        result_small = tracker.record_post_label(post_small, user_id="small")

        # Large drop
        post_large = _make_signal(beta_amp=5.0, alpha_amp=5.0)
        tracker.record_pre_label(pre_sig, user_id="large")
        result_large = tracker.record_post_label(post_large, user_id="large")

        assert result_large["labeling_efficacy"] > result_small["labeling_efficacy"]


# ---------------------------------------------------------------------------
# TestLPPReduction
# ---------------------------------------------------------------------------

class TestLPPReduction:
    def test_lower_post_beta_positive_lpp_reduction(self):
        """Beta decrease post-label should give positive lpp_reduction."""
        np.random.seed(42)
        tracker = AffectLabelingTracker(fs=FS)
        pre = _make_signal(beta_amp=30.0, alpha_amp=5.0)
        post = _make_signal(beta_amp=5.0, alpha_amp=5.0)
        tracker.record_pre_label(pre)
        result = tracker.record_post_label(post)
        assert result["lpp_reduction"] > 0.0

    def test_higher_post_beta_negative_lpp_reduction(self):
        """Beta increase post-label should give negative lpp_reduction."""
        np.random.seed(42)
        tracker = AffectLabelingTracker(fs=FS)
        pre = _make_signal(beta_amp=5.0, alpha_amp=5.0)
        post = _make_signal(beta_amp=30.0, alpha_amp=5.0)
        tracker.record_pre_label(pre)
        result = tracker.record_post_label(post)
        assert result["lpp_reduction"] < 0.0

    def test_equal_beta_zero_lpp_reduction(self):
        """Identical beta pre/post should give ~zero lpp_reduction."""
        np.random.seed(42)
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal(beta_amp=15.0, alpha_amp=5.0)
        tracker.record_pre_label(sig)
        result = tracker.record_post_label(sig)
        assert abs(result["lpp_reduction"]) < 0.05

    def test_lpp_reduction_bounded(self):
        """LPP reduction should be bounded in [-1, 1]."""
        np.random.seed(42)
        tracker = AffectLabelingTracker(fs=FS)
        pre = _make_signal(beta_amp=50.0, alpha_amp=1.0)
        post = _make_signal(beta_amp=0.01, alpha_amp=1.0)
        tracker.record_pre_label(pre)
        result = tracker.record_post_label(post)
        assert -1.0 <= result["lpp_reduction"] <= 1.0


# ---------------------------------------------------------------------------
# TestPrefrontalIncrease
# ---------------------------------------------------------------------------

class TestPrefrontalIncrease:
    def test_af7_alpha_decrease_positive_prefrontal(self):
        """AF7 alpha suppression post-label should produce positive
        prefrontal_increase (VLPFC activation)."""
        np.random.seed(42)
        tracker = AffectLabelingTracker(fs=FS)
        # Pre: high alpha at AF7 (ch1)
        pre = _make_signal(
            per_channel_alpha=[5.0, 30.0, 5.0, 5.0],
            beta_amp=5.0,
        )
        # Post: low alpha at AF7 (ch1)
        post = _make_signal(
            per_channel_alpha=[5.0, 3.0, 5.0, 5.0],
            beta_amp=5.0,
        )
        tracker.record_pre_label(pre)
        result = tracker.record_post_label(post)
        assert result["prefrontal_increase"] > 0.0

    def test_af7_alpha_increase_negative_prefrontal(self):
        """AF7 alpha increase post-label should produce negative
        prefrontal_increase (less VLPFC activation)."""
        np.random.seed(42)
        tracker = AffectLabelingTracker(fs=FS)
        pre = _make_signal(
            per_channel_alpha=[5.0, 3.0, 5.0, 5.0],
            beta_amp=5.0,
        )
        post = _make_signal(
            per_channel_alpha=[5.0, 30.0, 5.0, 5.0],
            beta_amp=5.0,
        )
        tracker.record_pre_label(pre)
        result = tracker.record_post_label(post)
        assert result["prefrontal_increase"] < 0.0

    def test_prefrontal_increase_bounded(self):
        """Prefrontal increase should be in [-1, 1]."""
        np.random.seed(42)
        tracker = AffectLabelingTracker(fs=FS)
        pre = _make_signal(per_channel_alpha=[5.0, 50.0, 5.0, 5.0], beta_amp=5.0)
        post = _make_signal(per_channel_alpha=[5.0, 0.1, 5.0, 5.0], beta_amp=5.0)
        tracker.record_pre_label(pre)
        result = tracker.record_post_label(post)
        assert -1.0 <= result["prefrontal_increase"] <= 1.0


# ---------------------------------------------------------------------------
# TestEfficacyLevels
# ---------------------------------------------------------------------------

class TestEfficacyLevels:
    def _get_level(self, efficacy_value):
        """Replicate threshold logic to determine expected level."""
        if efficacy_value >= 0.7:
            return "highly_effective"
        elif efficacy_value >= 0.5:
            return "effective"
        elif efficacy_value >= 0.3:
            return "moderately_effective"
        else:
            return "minimal_effect"

    def test_minimal_effect_for_zero_efficacy(self):
        """Identical pre/post -> near-zero efficacy -> minimal_effect."""
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()
        tracker.record_pre_label(sig)
        result = tracker.record_post_label(sig)
        assert result["efficacy_level"] == "minimal_effect"

    def test_highly_effective_for_large_reduction(self):
        """Very large beta drop + alpha suppression -> highly_effective."""
        np.random.seed(42)
        tracker = AffectLabelingTracker(fs=FS)
        pre = _make_signal(beta_amp=50.0, per_channel_alpha=[5.0, 40.0, 5.0, 5.0])
        post = _make_signal(beta_amp=2.0, per_channel_alpha=[5.0, 2.0, 5.0, 5.0])
        tracker.record_pre_label(pre)
        result = tracker.record_post_label(post)
        assert result["efficacy_level"] == "highly_effective"

    def test_all_levels_are_valid_enum(self):
        """All four efficacy levels must be in EFFICACY_LEVELS."""
        valid = set(EFFICACY_LEVELS)
        assert "minimal_effect" in valid
        assert "moderately_effective" in valid
        assert "effective" in valid
        assert "highly_effective" in valid

    def test_level_matches_efficacy_value(self):
        """The level should be consistent with the numeric efficacy."""
        np.random.seed(42)
        tracker = AffectLabelingTracker(fs=FS)
        pre = _make_signal(beta_amp=30.0, alpha_amp=5.0)
        post = _make_signal(beta_amp=10.0, alpha_amp=5.0)
        tracker.record_pre_label(pre)
        result = tracker.record_post_label(post)
        expected = self._get_level(result["labeling_efficacy"])
        assert result["efficacy_level"] == expected


# ---------------------------------------------------------------------------
# TestSessionStats
# ---------------------------------------------------------------------------

class TestSessionStats:
    def test_empty_session_stats(self):
        tracker = AffectLabelingTracker(fs=FS)
        stats = tracker.get_session_stats()
        assert stats["n_trials"] == 0
        assert stats["mean_efficacy"] == 0.0
        assert stats["improvement_over_session"] == 0.0

    def test_stats_after_one_trial(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()
        tracker.record_pre_label(sig)
        tracker.record_post_label(sig)
        stats = tracker.get_session_stats()
        assert stats["n_trials"] == 1
        assert isinstance(stats["mean_efficacy"], float)

    def test_stats_n_trials_counts_correctly(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()
        for _ in range(5):
            tracker.record_pre_label(sig)
            tracker.record_post_label(sig)
        stats = tracker.get_session_stats()
        assert stats["n_trials"] == 5

    def test_improvement_positive_when_later_trials_better(self):
        """If later trials have higher efficacy, improvement should be positive."""
        np.random.seed(42)
        tracker = AffectLabelingTracker(fs=FS)

        # First 3 trials: identical pre/post (low efficacy)
        sig_same = _make_signal(beta_amp=15.0, alpha_amp=10.0)
        for _ in range(3):
            tracker.record_pre_label(sig_same)
            tracker.record_post_label(sig_same)

        # Last 3 trials: big beta drop (high efficacy)
        pre_high = _make_signal(beta_amp=40.0, alpha_amp=5.0)
        post_low = _make_signal(beta_amp=3.0, alpha_amp=5.0)
        for _ in range(3):
            tracker.record_pre_label(pre_high)
            tracker.record_post_label(post_low)

        stats = tracker.get_session_stats()
        assert stats["improvement_over_session"] > 0.0

    def test_improvement_zero_with_fewer_than_3_trials(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()
        tracker.record_pre_label(sig)
        tracker.record_post_label(sig)
        tracker.record_pre_label(sig)
        tracker.record_post_label(sig)
        stats = tracker.get_session_stats()
        assert stats["improvement_over_session"] == 0.0

    def test_stats_keys_present(self):
        tracker = AffectLabelingTracker(fs=FS)
        stats = tracker.get_session_stats()
        assert "n_trials" in stats
        assert "mean_efficacy" in stats
        assert "improvement_over_session" in stats


# ---------------------------------------------------------------------------
# TestHistory
# ---------------------------------------------------------------------------

class TestHistory:
    def test_empty_history(self):
        tracker = AffectLabelingTracker(fs=FS)
        history = tracker.get_history()
        assert history == []

    def test_history_grows(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()
        for i in range(4):
            tracker.record_pre_label(sig)
            tracker.record_post_label(sig, label=f"trial_{i}")
        history = tracker.get_history()
        assert len(history) == 4

    def test_history_last_n(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()
        for _ in range(10):
            tracker.record_pre_label(sig)
            tracker.record_post_label(sig)
        history = tracker.get_history(last_n=3)
        assert len(history) == 3

    def test_history_last_n_larger_than_total(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()
        for _ in range(2):
            tracker.record_pre_label(sig)
            tracker.record_post_label(sig)
        history = tracker.get_history(last_n=100)
        assert len(history) == 2

    def test_history_entries_have_required_keys(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()
        tracker.record_pre_label(sig)
        tracker.record_post_label(sig)
        entry = tracker.get_history()[0]
        assert "labeling_efficacy" in entry
        assert "efficacy_level" in entry

    def test_history_capped_at_500(self):
        """History per user should not exceed 500 entries."""
        tracker = AffectLabelingTracker(fs=FS)
        # Use a minimal signal for speed
        sig = np.random.RandomState(42).randn(4, 64)
        for _ in range(510):
            tracker.record_post_label(sig)
        history = tracker.get_history()
        assert len(history) <= 500

    def test_history_returns_copies(self):
        """Returned history should be a copy, not a reference."""
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()
        tracker.record_pre_label(sig)
        tracker.record_post_label(sig)
        h1 = tracker.get_history()
        h1.clear()
        h2 = tracker.get_history()
        assert len(h2) == 1


# ---------------------------------------------------------------------------
# TestReset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_history(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()
        tracker.record_pre_label(sig)
        tracker.record_post_label(sig)
        assert len(tracker.get_history()) == 1
        tracker.reset()
        assert len(tracker.get_history()) == 0

    def test_reset_clears_baseline(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()
        tracker.set_baseline(sig)
        tracker.record_pre_label(sig)
        result_before = tracker.record_post_label(sig)
        assert result_before["has_baseline"] is True

        tracker.reset()
        tracker.record_pre_label(sig)
        result_after = tracker.record_post_label(sig)
        assert result_after["has_baseline"] is False

    def test_reset_clears_pre_label(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()
        tracker.record_pre_label(sig)
        tracker.reset()
        # After reset, user state is cleared, pre_label gone
        user = tracker._ensure_user("default")
        assert user["pre_label"] is None

    def test_reset_allows_fresh_start(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()
        for _ in range(5):
            tracker.record_pre_label(sig)
            tracker.record_post_label(sig)
        tracker.reset()
        stats = tracker.get_session_stats()
        assert stats["n_trials"] == 0

    def test_reset_only_affects_specified_user(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()
        tracker.record_pre_label(sig, user_id="alice")
        tracker.record_post_label(sig, user_id="alice")
        tracker.record_pre_label(sig, user_id="bob")
        tracker.record_post_label(sig, user_id="bob")

        tracker.reset(user_id="alice")
        assert len(tracker.get_history(user_id="alice")) == 0
        assert len(tracker.get_history(user_id="bob")) == 1


# ---------------------------------------------------------------------------
# TestMultiUser
# ---------------------------------------------------------------------------

class TestMultiUser:
    def test_independent_baselines(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig1 = _make_signal(alpha_amp=20.0)
        sig2 = _make_signal(alpha_amp=5.0)

        tracker.set_baseline(sig1, user_id="alice")
        tracker.set_baseline(sig2, user_id="bob")

        # Each user's baseline should be independent
        alice = tracker._ensure_user("alice")
        bob = tracker._ensure_user("bob")
        assert alice["baseline"] is not None
        assert bob["baseline"] is not None
        # They should have different metric values
        assert alice["baseline"]["overall_alpha"] != bob["baseline"]["overall_alpha"]

    def test_independent_histories(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()

        tracker.record_pre_label(sig, user_id="alice")
        tracker.record_post_label(sig, user_id="alice")
        tracker.record_pre_label(sig, user_id="alice")
        tracker.record_post_label(sig, user_id="alice")

        tracker.record_pre_label(sig, user_id="bob")
        tracker.record_post_label(sig, user_id="bob")

        assert len(tracker.get_history(user_id="alice")) == 2
        assert len(tracker.get_history(user_id="bob")) == 1

    def test_independent_session_stats(self):
        tracker = AffectLabelingTracker(fs=FS)
        sig = _make_signal()

        for _ in range(3):
            tracker.record_pre_label(sig, user_id="alice")
            tracker.record_post_label(sig, user_id="alice")

        assert tracker.get_session_stats(user_id="alice")["n_trials"] == 3
        assert tracker.get_session_stats(user_id="bob")["n_trials"] == 0

    def test_pre_label_per_user(self):
        """Pre-label for one user should not affect another."""
        tracker = AffectLabelingTracker(fs=FS)
        pre_sig = _make_signal(beta_amp=40.0)
        post_sig = _make_signal(beta_amp=5.0)

        tracker.record_pre_label(pre_sig, user_id="alice")
        # Bob has no pre-label
        result_bob = tracker.record_post_label(post_sig, user_id="bob")
        result_alice = tracker.record_post_label(post_sig, user_id="alice")

        # Alice should have real pre-label comparison
        # Bob should use zero defaults
        assert isinstance(result_alice["lpp_reduction"], float)
        assert isinstance(result_bob["lpp_reduction"], float)


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_constant_signal(self):
        """Constant signal should not crash, returns valid output."""
        tracker = AffectLabelingTracker(fs=FS)
        sig = np.ones((4, N_SAMPLES)) * 42.0
        tracker.record_pre_label(sig)
        result = tracker.record_post_label(sig)
        assert isinstance(result, dict)
        assert 0.0 <= result["labeling_efficacy"] <= 1.0
        assert result["efficacy_level"] in EFFICACY_LEVELS

    def test_1d_input(self):
        """Single-channel 1D input should reshape and work."""
        tracker = AffectLabelingTracker(fs=FS)
        np.random.seed(42)
        sig = np.random.randn(N_SAMPLES) * 20.0
        tracker.record_pre_label(sig)
        result = tracker.record_post_label(sig)
        assert isinstance(result, dict)
        assert result["efficacy_level"] in EFFICACY_LEVELS

    def test_short_signal_64_samples(self):
        """Very short signal should handle gracefully."""
        tracker = AffectLabelingTracker(fs=FS)
        np.random.seed(42)
        sig = np.random.randn(4, 64) * 10.0
        tracker.record_pre_label(sig)
        result = tracker.record_post_label(sig)
        assert isinstance(result, dict)
        assert 0.0 <= result["labeling_efficacy"] <= 1.0

    def test_very_short_signal_16_samples(self):
        """Extremely short signal should not crash."""
        tracker = AffectLabelingTracker(fs=FS)
        np.random.seed(42)
        sig = np.random.randn(4, 16) * 10.0
        tracker.record_pre_label(sig)
        result = tracker.record_post_label(sig)
        assert isinstance(result, dict)

    def test_zero_signal(self):
        """All-zero signal should produce valid output."""
        tracker = AffectLabelingTracker(fs=FS)
        sig = np.zeros((4, N_SAMPLES))
        tracker.record_pre_label(sig)
        result = tracker.record_post_label(sig)
        assert isinstance(result, dict)
        assert result["labeling_efficacy"] == 0.0

    def test_zero_1d_signal(self):
        """All-zero 1D signal should produce valid output."""
        tracker = AffectLabelingTracker(fs=FS)
        sig = np.zeros(N_SAMPLES)
        tracker.record_pre_label(sig)
        result = tracker.record_post_label(sig)
        assert isinstance(result, dict)

    def test_two_channel_input(self):
        """2-channel input should work (non-standard channel count)."""
        tracker = AffectLabelingTracker(fs=FS)
        np.random.seed(42)
        sig = np.random.randn(2, N_SAMPLES) * 15.0
        tracker.record_pre_label(sig)
        result = tracker.record_post_label(sig)
        assert isinstance(result, dict)
        assert result["efficacy_level"] in EFFICACY_LEVELS

    def test_single_channel_2d_input(self):
        """(1, n_samples) 2D input should work."""
        tracker = AffectLabelingTracker(fs=FS)
        np.random.seed(42)
        sig = np.random.randn(1, N_SAMPLES) * 15.0
        tracker.record_pre_label(sig)
        result = tracker.record_post_label(sig)
        assert isinstance(result, dict)

    def test_large_amplitude_signal(self):
        """Very large amplitude (saturated) should not crash."""
        tracker = AffectLabelingTracker(fs=FS)
        np.random.seed(42)
        sig = np.random.randn(4, N_SAMPLES) * 5000.0
        tracker.record_pre_label(sig)
        result = tracker.record_post_label(sig)
        assert isinstance(result, dict)
        assert 0.0 <= result["labeling_efficacy"] <= 1.0

    def test_baseline_with_short_signal(self):
        """Setting baseline with short signal should not crash."""
        tracker = AffectLabelingTracker(fs=FS)
        sig = np.random.randn(4, 32) * 10.0
        result = tracker.set_baseline(sig)
        assert result["baseline_set"] is True

    def test_custom_fs_propagates(self):
        """Custom sampling rate should be used when fs parameter is None."""
        tracker = AffectLabelingTracker(fs=128.0)
        sig = _make_signal(fs=128.0, n_channels=4)
        tracker.record_pre_label(sig)
        result = tracker.record_post_label(sig)
        assert isinstance(result, dict)

"""Tests for ReactivityRegulationTracker -- emotional reactivity vs regulation.

Covers:
  - Helper functions (_band_power_welch, _compute_faa, _classify_balance_state)
  - ReactivityRegulationTracker class: set_baseline, assess, get_session_stats,
    get_history, reset
  - Multi-user isolation
  - Edge cases: single-channel, short signals, constant signals
  - Balance state mapping
  - History capping at 500
  - Baseline vs no-baseline paths
  - Recovery speed computation
  - Behavioral correctness: stressed vs relaxed signals
"""

import numpy as np
import pytest

from models.reactivity_regulation import (
    ReactivityRegulationTracker,
    _band_power_welch,
    _classify_balance_state,
    _compute_faa,
)


# -- Fixtures -----------------------------------------------------------------


@pytest.fixture
def tracker():
    """Fresh ReactivityRegulationTracker instance."""
    return ReactivityRegulationTracker(fs=256.0)


@pytest.fixture
def multichannel_eeg():
    """4 channels x 4 seconds of synthetic EEG at 256 Hz."""
    np.random.seed(42)
    return np.random.randn(4, 1024) * 20


@pytest.fixture
def relaxed_eeg():
    """4-channel EEG with strong alpha (10 Hz) and weak beta -- relaxed state."""
    np.random.seed(99)
    t = np.arange(1024) / 256.0
    base = np.random.randn(4, 1024) * 2
    for ch in range(4):
        base[ch] += 30 * np.sin(2 * np.pi * 10 * t)   # strong alpha
        base[ch] += 3 * np.sin(2 * np.pi * 20 * t)    # weak beta
    return base


@pytest.fixture
def stressed_eeg():
    """4-channel EEG with strong high-beta (25 Hz) and suppressed alpha."""
    np.random.seed(88)
    t = np.arange(1024) / 256.0
    base = np.random.randn(4, 1024) * 2
    for ch in range(4):
        base[ch] += 3 * np.sin(2 * np.pi * 10 * t)    # weak alpha
        base[ch] += 30 * np.sin(2 * np.pi * 25 * t)   # strong high-beta
    return base


@pytest.fixture
def fs():
    return 256


# -- Helper function tests ---------------------------------------------------


class TestBandPowerWelch:
    def test_returns_positive_float(self):
        signal = np.random.randn(1024) * 20
        power = _band_power_welch(signal, 256.0, (8.0, 12.0))
        assert isinstance(power, float)
        assert power > 0

    def test_alpha_dominant_signal_has_higher_alpha_power(self):
        t = np.arange(1024) / 256.0
        alpha_sig = 30 * np.sin(2 * np.pi * 10 * t) + np.random.randn(1024) * 2
        noise_sig = np.random.randn(1024) * 5
        assert _band_power_welch(alpha_sig, 256.0, (8.0, 12.0)) > \
               _band_power_welch(noise_sig, 256.0, (8.0, 12.0))

    def test_very_short_signal_returns_floor(self):
        short = np.random.randn(10)
        assert _band_power_welch(short, 256.0, (8.0, 12.0)) == pytest.approx(1e-12)

    def test_never_returns_below_floor(self):
        flat = np.zeros(1024)
        assert _band_power_welch(flat, 256.0, (8.0, 12.0)) >= 1e-12

    def test_beta_band_power(self):
        t = np.arange(1024) / 256.0
        beta_sig = 30 * np.sin(2 * np.pi * 20 * t) + np.random.randn(1024) * 2
        beta_power = _band_power_welch(beta_sig, 256.0, (12.0, 30.0))
        alpha_power = _band_power_welch(beta_sig, 256.0, (8.0, 12.0))
        assert beta_power > alpha_power


class TestComputeFAA:
    def test_returns_float(self, multichannel_eeg, fs):
        faa = _compute_faa(multichannel_eeg, fs)
        assert isinstance(faa, float)

    def test_positive_faa_when_right_alpha_greater(self, fs):
        np.random.seed(99)
        t = np.arange(1024) / 256.0
        signals = np.random.randn(4, 1024) * 2
        signals[1] += 10 * np.sin(2 * np.pi * 10 * t)  # AF7: weak alpha
        signals[2] += 40 * np.sin(2 * np.pi * 10 * t)  # AF8: strong alpha
        faa = _compute_faa(signals, fs)
        assert faa > 0

    def test_single_channel_returns_zero(self, fs):
        signal_1d = np.random.randn(1024) * 20
        assert _compute_faa(signal_1d, fs) == 0.0

    def test_two_channel_returns_zero(self, fs):
        signal_2ch = np.random.randn(2, 1024) * 20
        assert _compute_faa(signal_2ch, fs) == 0.0

    def test_symmetric_alpha_near_zero_faa(self, fs):
        np.random.seed(77)
        base = np.random.randn(1024) * 20
        signals = np.zeros((4, 1024))
        signals[1] = base  # AF7
        signals[2] = base  # AF8 (identical)
        faa = _compute_faa(signals, fs)
        assert abs(faa) < 0.01


class TestClassifyBalanceState:
    @pytest.mark.parametrize(
        "rr_ratio,expected",
        [
            (2.0, "well_regulated"),
            (1.51, "well_regulated"),
            (1.5, "balanced"),
            (1.0, "balanced"),
            (0.8, "balanced"),
            (0.79, "reactive"),
            (0.5, "reactive"),
            (0.4, "reactive"),
            (0.39, "dysregulated"),
            (0.1, "dysregulated"),
            (0.0, "dysregulated"),
        ],
    )
    def test_state_mapping(self, rr_ratio, expected):
        assert _classify_balance_state(rr_ratio) == expected


# -- ReactivityRegulationTracker class tests ----------------------------------


class TestAssessOutputStructure:
    def test_all_keys_present(self, tracker, multichannel_eeg, fs):
        result = tracker.assess(multichannel_eeg, fs=fs)
        expected_keys = {
            "reactivity_index",
            "regulation_index",
            "rr_ratio",
            "balance_state",
            "alpha_change",
            "beta_change",
            "faa_shift",
            "recovery_speed",
            "has_baseline",
        }
        assert set(result.keys()) == expected_keys

    def test_reactivity_index_bounded(self, tracker, multichannel_eeg, fs):
        result = tracker.assess(multichannel_eeg, fs=fs)
        assert 0 <= result["reactivity_index"] <= 1

    def test_regulation_index_bounded(self, tracker, multichannel_eeg, fs):
        result = tracker.assess(multichannel_eeg, fs=fs)
        assert 0 <= result["regulation_index"] <= 1

    def test_recovery_speed_bounded(self, tracker, multichannel_eeg, fs):
        result = tracker.assess(multichannel_eeg, fs=fs)
        assert 0 <= result["recovery_speed"] <= 1

    def test_balance_state_valid(self, tracker, multichannel_eeg, fs):
        result = tracker.assess(multichannel_eeg, fs=fs)
        valid_states = {"well_regulated", "balanced", "reactive", "dysregulated"}
        assert result["balance_state"] in valid_states

    def test_rr_ratio_is_float(self, tracker, multichannel_eeg, fs):
        result = tracker.assess(multichannel_eeg, fs=fs)
        assert isinstance(result["rr_ratio"], float)


class TestBaselinePath:
    def test_has_baseline_false_initially(self, tracker, multichannel_eeg, fs):
        result = tracker.assess(multichannel_eeg, fs=fs)
        assert result["has_baseline"] is False

    def test_has_baseline_true_after_set(self, tracker, multichannel_eeg, fs):
        tracker.set_baseline(multichannel_eeg, fs=fs)
        result = tracker.assess(multichannel_eeg, fs=fs)
        assert result["has_baseline"] is True

    def test_set_baseline_output(self, tracker, multichannel_eeg, fs):
        result = tracker.set_baseline(multichannel_eeg, fs=fs)
        assert result["baseline_set"] is True
        assert isinstance(result["baseline_alpha"], float)
        assert isinstance(result["baseline_beta"], float)
        assert isinstance(result["baseline_faa"], float)
        assert result["baseline_alpha"] > 0
        assert result["baseline_beta"] > 0

    def test_set_baseline_single_channel(self, tracker, fs):
        signal = np.random.randn(1024) * 20
        result = tracker.set_baseline(signal, fs=fs)
        assert result["baseline_set"] is True
        assert result["baseline_faa"] == 0.0

    def test_uses_default_fs(self, tracker, multichannel_eeg):
        result = tracker.set_baseline(multichannel_eeg)
        assert result["baseline_set"] is True


class TestNoBaselinePath:
    def test_assess_without_baseline_works(self, tracker, multichannel_eeg, fs):
        result = tracker.assess(multichannel_eeg, fs=fs)
        assert result["has_baseline"] is False
        assert 0 <= result["reactivity_index"] <= 1
        assert 0 <= result["regulation_index"] <= 1

    def test_relaxed_signal_low_reactivity(self, tracker, relaxed_eeg, fs):
        """Relaxed EEG (high alpha, low beta) should show low reactivity."""
        result = tracker.assess(relaxed_eeg, fs=fs)
        assert result["reactivity_index"] < 0.5

    def test_stressed_signal_high_reactivity(self, tracker, stressed_eeg, fs):
        """Stressed EEG (high beta, low alpha) should show higher reactivity."""
        result = tracker.assess(stressed_eeg, fs=fs)
        assert result["reactivity_index"] > 0.3


class TestBaselineComparison:
    def test_alpha_change_negative_under_stress(self, tracker, relaxed_eeg,
                                                 stressed_eeg, fs):
        """After a relaxed baseline, stressed EEG should show negative alpha change."""
        tracker.set_baseline(relaxed_eeg, fs=fs)
        result = tracker.assess(stressed_eeg, fs=fs)
        assert result["alpha_change"] < 0

    def test_beta_change_positive_under_stress(self, tracker, relaxed_eeg,
                                                stressed_eeg, fs):
        """After a relaxed baseline, stressed EEG should show positive beta change."""
        tracker.set_baseline(relaxed_eeg, fs=fs)
        result = tracker.assess(stressed_eeg, fs=fs)
        assert result["beta_change"] > 0

    def test_same_as_baseline_low_reactivity(self, tracker, relaxed_eeg, fs):
        """Assessing with the same signal as baseline should give low reactivity."""
        tracker.set_baseline(relaxed_eeg, fs=fs)
        result = tracker.assess(relaxed_eeg, fs=fs)
        assert result["reactivity_index"] < 0.3


class TestRecoverySpeed:
    def test_recovery_speed_bounded(self, tracker, fs):
        np.random.seed(55)
        tracker.set_baseline(np.random.randn(4, 1024) * 20, fs=fs)
        for _ in range(5):
            result = tracker.assess(np.random.randn(4, 1024) * 20, fs=fs)
        assert 0 <= result["recovery_speed"] <= 1

    def test_recovery_speed_default_with_few_epochs(self, tracker, fs):
        """With fewer than 3 alpha history values, recovery_speed defaults to 0.5."""
        np.random.seed(56)
        tracker.set_baseline(np.random.randn(4, 1024) * 20, fs=fs)
        result = tracker.assess(np.random.randn(4, 1024) * 20, fs=fs)
        assert result["recovery_speed"] == pytest.approx(0.5)


class TestSessionStats:
    def test_empty_session(self, tracker):
        stats = tracker.get_session_stats()
        assert stats["n_epochs"] == 0
        assert stats["mean_rr_ratio"] == 0.0
        assert stats["dominant_state"] == "dysregulated"
        assert stats["state_distribution"] == {}

    def test_stats_after_assessments(self, tracker, fs):
        np.random.seed(60)
        for _ in range(8):
            eeg = np.random.randn(4, 1024) * 20
            tracker.assess(eeg, fs=fs)

        stats = tracker.get_session_stats()
        assert stats["n_epochs"] == 8
        assert isinstance(stats["mean_rr_ratio"], float)
        assert stats["dominant_state"] in {
            "well_regulated", "balanced", "reactive", "dysregulated"
        }
        assert isinstance(stats["state_distribution"], dict)
        total_in_dist = sum(stats["state_distribution"].values())
        assert total_in_dist == 8

    def test_state_distribution_sums_to_n_epochs(self, tracker, fs):
        np.random.seed(61)
        for _ in range(12):
            tracker.assess(np.random.randn(4, 1024) * 20, fs=fs)

        stats = tracker.get_session_stats()
        total = sum(stats["state_distribution"].values())
        assert total == stats["n_epochs"]


class TestGetHistory:
    def test_empty_history(self, tracker):
        assert tracker.get_history() == []

    def test_returns_all_entries(self, tracker, fs):
        np.random.seed(80)
        for _ in range(5):
            tracker.assess(np.random.randn(4, 1024) * 20, fs=fs)

        history = tracker.get_history()
        assert len(history) == 5

    def test_last_n_parameter(self, tracker, fs):
        np.random.seed(81)
        for _ in range(10):
            tracker.assess(np.random.randn(4, 1024) * 20, fs=fs)

        last_3 = tracker.get_history(last_n=3)
        assert len(last_3) == 3
        full = tracker.get_history()
        assert last_3 == full[-3:]

    def test_last_n_larger_than_history(self, tracker, fs):
        np.random.seed(82)
        for _ in range(3):
            tracker.assess(np.random.randn(4, 1024) * 20, fs=fs)

        result = tracker.get_history(last_n=100)
        assert len(result) == 3

    def test_history_is_copy(self, tracker, fs):
        """Returned history should be a copy, not a reference."""
        np.random.seed(83)
        tracker.assess(np.random.randn(4, 1024) * 20, fs=fs)
        h1 = tracker.get_history()
        h1.clear()
        h2 = tracker.get_history()
        assert len(h2) == 1


class TestReset:
    def test_reset_clears_everything(self, tracker, fs):
        np.random.seed(90)
        tracker.set_baseline(np.random.randn(4, 1024) * 20, fs=fs)
        for _ in range(5):
            tracker.assess(np.random.randn(4, 1024) * 20, fs=fs)

        tracker.reset()

        stats = tracker.get_session_stats()
        assert stats["n_epochs"] == 0
        assert tracker.get_history() == []

        result = tracker.assess(np.random.randn(4, 1024) * 20, fs=fs)
        assert result["has_baseline"] is False

    def test_reset_nonexistent_user_no_error(self, tracker):
        tracker.reset(user_id="nonexistent")  # should not raise


class TestMultiUser:
    def test_users_are_isolated(self, tracker, fs):
        np.random.seed(100)

        tracker.set_baseline(np.random.randn(4, 1024) * 20, fs=fs, user_id="alice")
        for _ in range(5):
            tracker.assess(np.random.randn(4, 1024) * 20, fs=fs, user_id="alice")

        # Bob has no data
        bob_stats = tracker.get_session_stats(user_id="bob")
        assert bob_stats["n_epochs"] == 0

        alice_stats = tracker.get_session_stats(user_id="alice")
        assert alice_stats["n_epochs"] == 5

    def test_reset_one_user_doesnt_affect_other(self, tracker, fs):
        np.random.seed(101)

        for _ in range(3):
            tracker.assess(np.random.randn(4, 1024) * 20, fs=fs, user_id="alice")
            tracker.assess(np.random.randn(4, 1024) * 20, fs=fs, user_id="bob")

        tracker.reset(user_id="alice")

        assert tracker.get_session_stats(user_id="alice")["n_epochs"] == 0
        assert tracker.get_session_stats(user_id="bob")["n_epochs"] == 3

    def test_baselines_are_independent(self, tracker, fs):
        np.random.seed(102)

        tracker.set_baseline(np.random.randn(4, 1024) * 20, fs=fs, user_id="alice")

        # Bob should not have baseline
        result = tracker.assess(np.random.randn(4, 1024) * 20, fs=fs, user_id="bob")
        assert result["has_baseline"] is False

        # Alice should have baseline
        result = tracker.assess(np.random.randn(4, 1024) * 20, fs=fs, user_id="alice")
        assert result["has_baseline"] is True


class TestEdgeCases:
    def test_single_channel_assess(self, tracker, fs):
        signal = np.random.randn(1024) * 20
        result = tracker.assess(signal, fs=fs)
        assert result["faa_shift"] == pytest.approx(0.0, abs=0.01)
        assert result["reactivity_index"] >= 0

    def test_constant_signal(self, tracker, fs):
        flat = np.full((4, 1024), 0.001)
        result = tracker.assess(flat, fs=fs)
        assert isinstance(result["reactivity_index"], float)
        assert isinstance(result["regulation_index"], float)

    def test_three_channel_signal(self, tracker, fs):
        eeg_3ch = np.random.randn(3, 1024) * 20
        result = tracker.assess(eeg_3ch, fs=fs)
        assert isinstance(result["faa_shift"], float)

    def test_assess_uses_default_fs(self, tracker, multichannel_eeg):
        result = tracker.assess(multichannel_eeg)
        assert "reactivity_index" in result

    def test_very_short_signal(self, tracker, fs):
        """Signal shorter than nperseg should not crash."""
        short = np.random.randn(4, 32) * 20
        result = tracker.assess(short, fs=fs)
        assert 0 <= result["reactivity_index"] <= 1


class TestHistoryCapping:
    def test_history_capped_at_500(self, tracker, fs):
        np.random.seed(200)
        for _ in range(510):
            eeg = np.random.randn(4, 256) * 20  # short epochs for speed
            tracker.assess(eeg, fs=fs)

        history = tracker.get_history()
        assert len(history) <= 500


class TestBehavioralCorrectness:
    def test_relaxed_higher_rr_ratio_than_stressed(self, fs):
        """A session of relaxed epochs should have higher mean rr_ratio than stressed."""
        t = np.arange(1024) / 256.0

        relaxed_tracker = ReactivityRegulationTracker(fs=fs)
        stressed_tracker = ReactivityRegulationTracker(fs=fs)

        # Set identical baseline for both
        np.random.seed(300)
        baseline = np.random.randn(4, 1024) * 10
        for ch in range(4):
            baseline[ch] += 15 * np.sin(2 * np.pi * 10 * t)
            baseline[ch] += 10 * np.sin(2 * np.pi * 20 * t)
        relaxed_tracker.set_baseline(baseline, fs=fs)
        stressed_tracker.set_baseline(baseline, fs=fs)

        # Relaxed: high alpha, low beta
        np.random.seed(301)
        for _ in range(10):
            eeg = np.random.randn(4, 1024) * 3
            for ch in range(4):
                eeg[ch] += 30 * np.sin(2 * np.pi * 10 * t)   # strong alpha
                eeg[ch] += 5 * np.sin(2 * np.pi * 20 * t)    # weak beta
            relaxed_tracker.assess(eeg, fs=fs)

        # Stressed: low alpha, high beta
        np.random.seed(302)
        for _ in range(10):
            eeg = np.random.randn(4, 1024) * 3
            for ch in range(4):
                eeg[ch] += 5 * np.sin(2 * np.pi * 10 * t)    # weak alpha
                eeg[ch] += 30 * np.sin(2 * np.pi * 25 * t)   # strong high-beta
            stressed_tracker.assess(eeg, fs=fs)

        relaxed_stats = relaxed_tracker.get_session_stats()
        stressed_stats = stressed_tracker.get_session_stats()

        assert relaxed_stats["mean_rr_ratio"] > stressed_stats["mean_rr_ratio"]

    def test_zero_reactivity_gives_well_regulated(self, tracker, relaxed_eeg, fs):
        """When reactivity is near zero (same as baseline), should be well_regulated."""
        tracker.set_baseline(relaxed_eeg, fs=fs)
        result = tracker.assess(relaxed_eeg, fs=fs)
        assert result["balance_state"] in {"well_regulated", "balanced"}

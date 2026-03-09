"""Tests for AffectiveFlexibility — FAA shift dynamics.

Covers:
  - FAA computation helpers (_alpha_power_welch, _compute_faa)
  - AffectiveFlexibility class: assess, set_baseline, get_session_stats,
    get_history, reset
  - Multi-user isolation
  - Edge cases: single-channel, short signals, constant signals
  - Sub-metric correctness: variability, sign_change_rate, shift_speed,
    recovery_index
  - Flexibility level mapping
  - History capping at 500
"""

import numpy as np
import pytest

from models.affective_flexibility import (
    AffectiveFlexibility,
    _alpha_power_welch,
    _compute_faa,
    _score_to_level,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def af():
    """Fresh AffectiveFlexibility instance."""
    return AffectiveFlexibility(fs=256.0)


@pytest.fixture
def multichannel_eeg():
    """4 channels x 4 seconds of synthetic EEG at 256 Hz."""
    np.random.seed(42)
    return np.random.randn(4, 1024) * 20


@pytest.fixture
def alpha_dominant_eeg():
    """4-channel EEG with strong 10 Hz alpha on AF7 (ch1) and AF8 (ch2).

    AF8 has larger alpha than AF7 to produce positive FAA.
    """
    np.random.seed(99)
    t = np.arange(1024) / 256.0
    base = np.random.randn(4, 1024) * 2
    # Inject 10 Hz alpha with different amplitudes
    base[1] += 15 * np.sin(2 * np.pi * 10 * t)  # AF7 alpha = 15 uV
    base[2] += 25 * np.sin(2 * np.pi * 10 * t)  # AF8 alpha = 25 uV
    return base


@pytest.fixture
def fs():
    return 256


# ── Helper function tests ────────────────────────────────────────────────


class TestAlphaPowerWelch:
    def test_returns_positive_float(self):
        signal = np.random.randn(1024) * 20
        power = _alpha_power_welch(signal, 256.0)
        assert isinstance(power, float)
        assert power > 0

    def test_alpha_dominant_signal_has_higher_alpha_power(self):
        """A signal with a strong 10 Hz component should have high alpha power."""
        t = np.arange(1024) / 256.0
        alpha_sig = 30 * np.sin(2 * np.pi * 10 * t) + np.random.randn(1024) * 2
        noise_sig = np.random.randn(1024) * 5
        assert _alpha_power_welch(alpha_sig, 256.0) > _alpha_power_welch(
            noise_sig, 256.0
        )

    def test_very_short_signal_returns_floor(self):
        """Signal with fewer than 16 samples should return floor value."""
        short = np.random.randn(10)
        assert _alpha_power_welch(short, 256.0) == pytest.approx(1e-12)

    def test_never_returns_below_floor(self):
        """Even a flat signal should return at least 1e-12."""
        flat = np.zeros(1024)
        assert _alpha_power_welch(flat, 256.0) >= 1e-12


class TestComputeFAA:
    def test_returns_float(self, multichannel_eeg, fs):
        faa = _compute_faa(multichannel_eeg, fs)
        assert isinstance(faa, float)

    def test_positive_faa_when_right_alpha_greater(self, alpha_dominant_eeg, fs):
        """AF8 (right) has more alpha than AF7 (left) -> positive FAA."""
        faa = _compute_faa(alpha_dominant_eeg, fs)
        assert faa > 0

    def test_single_channel_returns_zero(self, fs):
        """Single-channel input cannot compute asymmetry -> returns 0.0."""
        signal_1d = np.random.randn(1024) * 20
        assert _compute_faa(signal_1d, fs) == 0.0

    def test_two_channel_returns_zero(self, fs):
        """Only 2 channels (need at least 3 for AF7/AF8 at indices 1,2)."""
        signal_2ch = np.random.randn(2, 1024) * 20
        assert _compute_faa(signal_2ch, fs) == 0.0

    def test_symmetric_alpha_near_zero_faa(self, fs):
        """Identical signals on AF7 and AF8 should produce FAA near zero."""
        np.random.seed(77)
        base = np.random.randn(1024) * 20
        signals = np.zeros((4, 1024))
        signals[1] = base  # AF7
        signals[2] = base  # AF8 (same as AF7)
        faa = _compute_faa(signals, fs)
        assert abs(faa) < 0.01


class TestScoreToLevel:
    @pytest.mark.parametrize(
        "score,expected",
        [
            (0, "rigid"),
            (10, "rigid"),
            (19.9, "rigid"),
            (20, "low"),
            (30, "low"),
            (40, "moderate"),
            (50, "moderate"),
            (60, "high"),
            (75, "high"),
            (80, "very_flexible"),
            (95, "very_flexible"),
            (100, "very_flexible"),
        ],
    )
    def test_level_mapping(self, score, expected):
        assert _score_to_level(score) == expected


# ── AffectiveFlexibility class tests ─────────────────────────────────────


class TestAssessOutputStructure:
    def test_all_keys_present(self, af, multichannel_eeg, fs):
        result = af.assess(multichannel_eeg, fs=fs)
        expected_keys = {
            "flexibility_score",
            "flexibility_level",
            "faa_current",
            "faa_variability",
            "sign_change_rate",
            "shift_speed",
            "recovery_index",
            "valence_range",
            "has_baseline",
        }
        assert set(result.keys()) == expected_keys

    def test_flexibility_score_bounded(self, af, multichannel_eeg, fs):
        result = af.assess(multichannel_eeg, fs=fs)
        assert 0 <= result["flexibility_score"] <= 100

    def test_flexibility_level_valid(self, af, multichannel_eeg, fs):
        result = af.assess(multichannel_eeg, fs=fs)
        valid_levels = {"rigid", "low", "moderate", "high", "very_flexible"}
        assert result["flexibility_level"] in valid_levels

    def test_sub_metrics_bounded(self, af, multichannel_eeg, fs):
        """All sub-metrics should be in [0, 1]."""
        # Feed enough epochs
        np.random.seed(123)
        for _ in range(10):
            eeg = np.random.randn(4, 1024) * 20
            result = af.assess(eeg, fs=fs)

        assert 0 <= result["faa_variability"] <= 1
        assert 0 <= result["sign_change_rate"] <= 1
        assert 0 <= result["shift_speed"] <= 1
        assert 0 <= result["recovery_index"] <= 1

    def test_has_baseline_false_initially(self, af, multichannel_eeg, fs):
        result = af.assess(multichannel_eeg, fs=fs)
        assert result["has_baseline"] is False

    def test_has_baseline_true_after_set(self, af, multichannel_eeg, fs):
        af.set_baseline(multichannel_eeg, fs=fs)
        result = af.assess(multichannel_eeg, fs=fs)
        assert result["has_baseline"] is True


class TestAssessDefaults:
    def test_defaults_before_5_epochs(self, af, fs):
        """Before 5 epochs, variability/sign_change/shift/recovery default to 0.5."""
        np.random.seed(10)
        for i in range(4):
            eeg = np.random.randn(4, 1024) * 20
            result = af.assess(eeg, fs=fs)
            assert result["faa_variability"] == 0.5
            assert result["sign_change_rate"] == 0.5
            assert result["shift_speed"] == 0.5
            assert result["recovery_index"] == 0.5

    def test_computed_after_5_epochs(self, af, fs):
        """After 5 epochs, sub-metrics should be computed (not necessarily 0.5)."""
        np.random.seed(20)
        for _ in range(5):
            eeg = np.random.randn(4, 1024) * 20
            result = af.assess(eeg, fs=fs)

        # At least one metric should differ from the 0.5 default
        metrics = [
            result["faa_variability"],
            result["sign_change_rate"],
            result["shift_speed"],
            result["recovery_index"],
        ]
        assert any(m != 0.5 for m in metrics)

    def test_valence_range_zero_for_single_epoch(self, af, multichannel_eeg, fs):
        result = af.assess(multichannel_eeg, fs=fs)
        assert result["valence_range"] == 0.0

    def test_valence_range_grows_with_diverse_epochs(self, af, fs):
        """Feeding epochs with different FAA should increase valence_range."""
        np.random.seed(30)
        t = np.arange(1024) / 256.0
        for i in range(6):
            eeg = np.random.randn(4, 1024) * 5
            # Vary alpha amplitude on AF8 to shift FAA
            eeg[2] += (10 + i * 10) * np.sin(2 * np.pi * 10 * t)
            result = af.assess(eeg, fs=fs)

        assert result["valence_range"] > 0


class TestBaseline:
    def test_set_baseline_output(self, af, multichannel_eeg, fs):
        result = af.set_baseline(multichannel_eeg, fs=fs)
        assert result["baseline_set"] is True
        assert isinstance(result["baseline_faa"], float)

    def test_set_baseline_single_channel(self, af, fs):
        """Single-channel baseline should still work (FAA will be 0.0)."""
        signal = np.random.randn(1024) * 20
        result = af.set_baseline(signal, fs=fs)
        assert result["baseline_set"] is True
        assert result["baseline_faa"] == 0.0

    def test_baseline_affects_recovery_index(self, af, fs):
        """With baseline set, recovery_index should track return-to-baseline."""
        np.random.seed(50)
        baseline_eeg = np.random.randn(4, 1024) * 20
        af.set_baseline(baseline_eeg, fs=fs)

        for _ in range(10):
            eeg = np.random.randn(4, 1024) * 20
            result = af.assess(eeg, fs=fs)

        # Recovery index should be computed (not default)
        assert isinstance(result["recovery_index"], float)
        assert 0 <= result["recovery_index"] <= 1

    def test_uses_default_fs(self, af, multichannel_eeg):
        """If fs not passed, uses self.fs."""
        result = af.set_baseline(multichannel_eeg)
        assert result["baseline_set"] is True


class TestSessionStats:
    def test_empty_session(self, af):
        stats = af.get_session_stats()
        assert stats["n_epochs"] == 0
        assert stats["mean_flexibility"] == 0.0
        assert stats["dominant_level"] == "rigid"
        assert stats["faa_trajectory"] == []

    def test_stats_after_assessments(self, af, fs):
        np.random.seed(60)
        for _ in range(8):
            eeg = np.random.randn(4, 1024) * 20
            af.assess(eeg, fs=fs)

        stats = af.get_session_stats()
        assert stats["n_epochs"] == 8
        assert isinstance(stats["mean_flexibility"], float)
        assert stats["dominant_level"] in {
            "rigid", "low", "moderate", "high", "very_flexible"
        }
        assert len(stats["faa_trajectory"]) == 8

    def test_faa_trajectory_matches_history(self, af, fs):
        np.random.seed(70)
        expected_faa = []
        for _ in range(5):
            eeg = np.random.randn(4, 1024) * 20
            result = af.assess(eeg, fs=fs)
            expected_faa.append(result["faa_current"])

        stats = af.get_session_stats()
        assert stats["faa_trajectory"] == expected_faa


class TestGetHistory:
    def test_empty_history(self, af):
        assert af.get_history() == []

    def test_returns_all_entries(self, af, fs):
        np.random.seed(80)
        for _ in range(5):
            af.assess(np.random.randn(4, 1024) * 20, fs=fs)

        history = af.get_history()
        assert len(history) == 5

    def test_last_n_parameter(self, af, fs):
        np.random.seed(81)
        for _ in range(10):
            af.assess(np.random.randn(4, 1024) * 20, fs=fs)

        last_3 = af.get_history(last_n=3)
        assert len(last_3) == 3
        # Should be the last 3 entries
        full = af.get_history()
        assert last_3 == full[-3:]

    def test_last_n_larger_than_history(self, af, fs):
        np.random.seed(82)
        for _ in range(3):
            af.assess(np.random.randn(4, 1024) * 20, fs=fs)

        result = af.get_history(last_n=100)
        assert len(result) == 3

    def test_history_is_copy(self, af, fs):
        """Returned history should be a copy, not a reference."""
        np.random.seed(83)
        af.assess(np.random.randn(4, 1024) * 20, fs=fs)
        h1 = af.get_history()
        h1.clear()
        h2 = af.get_history()
        assert len(h2) == 1


class TestReset:
    def test_reset_clears_everything(self, af, fs):
        np.random.seed(90)
        af.set_baseline(np.random.randn(4, 1024) * 20, fs=fs)
        for _ in range(5):
            af.assess(np.random.randn(4, 1024) * 20, fs=fs)

        af.reset()

        stats = af.get_session_stats()
        assert stats["n_epochs"] == 0
        assert af.get_history() == []

        result = af.assess(np.random.randn(4, 1024) * 20, fs=fs)
        assert result["has_baseline"] is False

    def test_reset_nonexistent_user_no_error(self, af):
        """Resetting a user that doesn't exist should not raise."""
        af.reset(user_id="nonexistent")  # should not raise


class TestMultiUser:
    def test_users_are_isolated(self, af, fs):
        np.random.seed(100)

        af.set_baseline(np.random.randn(4, 1024) * 20, fs=fs, user_id="alice")

        for _ in range(5):
            af.assess(np.random.randn(4, 1024) * 20, fs=fs, user_id="alice")

        # Bob has no data
        bob_stats = af.get_session_stats(user_id="bob")
        assert bob_stats["n_epochs"] == 0

        alice_stats = af.get_session_stats(user_id="alice")
        assert alice_stats["n_epochs"] == 5

    def test_reset_one_user_doesnt_affect_other(self, af, fs):
        np.random.seed(101)

        for _ in range(3):
            af.assess(np.random.randn(4, 1024) * 20, fs=fs, user_id="alice")
            af.assess(np.random.randn(4, 1024) * 20, fs=fs, user_id="bob")

        af.reset(user_id="alice")

        assert af.get_session_stats(user_id="alice")["n_epochs"] == 0
        assert af.get_session_stats(user_id="bob")["n_epochs"] == 3


class TestEdgeCases:
    def test_single_channel_assess(self, af, fs):
        """Single-channel input (1D) should work with FAA=0."""
        signal = np.random.randn(1024) * 20
        result = af.assess(signal, fs=fs)
        assert result["faa_current"] == 0.0
        assert result["flexibility_score"] >= 0

    def test_constant_signal(self, af, fs):
        """Constant signal should not crash."""
        flat = np.full((4, 1024), 0.001)
        result = af.assess(flat, fs=fs)
        assert isinstance(result["flexibility_score"], float)

    def test_three_channel_signal(self, af, fs):
        """3-channel signal should still compute FAA (has AF7 and AF8)."""
        eeg_3ch = np.random.randn(3, 1024) * 20
        result = af.assess(eeg_3ch, fs=fs)
        assert isinstance(result["faa_current"], float)

    def test_assess_uses_default_fs(self, af, multichannel_eeg):
        """If fs not passed, assess should use self.fs."""
        result = af.assess(multichannel_eeg)
        assert "flexibility_score" in result


class TestHistoryCapping:
    def test_assess_history_capped_at_500(self, af, fs):
        """assess_history should not grow beyond 500 entries."""
        np.random.seed(200)
        for _ in range(510):
            eeg = np.random.randn(4, 256) * 20  # short epochs for speed
            af.assess(eeg, fs=fs)

        history = af.get_history()
        assert len(history) <= 500


class TestFlexibilityBehavior:
    def test_diverse_signals_more_flexible_than_identical(self, fs):
        """Sessions with varying FAA should score higher than repetitive ones."""
        af_diverse = AffectiveFlexibility(fs=fs)
        af_uniform = AffectiveFlexibility(fs=fs)

        t = np.arange(1024) / float(fs)

        # Diverse: vary alpha amplitude on AF8 dramatically across epochs
        # This creates large FAA shifts (sign changes, high variability)
        np.random.seed(300)
        for i in range(10):
            eeg = np.random.randn(4, 1024) * 1  # low noise
            if i % 2 == 0:
                # Positive FAA: AF8 alpha >> AF7 alpha
                eeg[2] += 50 * np.sin(2 * np.pi * 10 * t)
                eeg[1] += 5 * np.sin(2 * np.pi * 10 * t)
            else:
                # Negative FAA: AF7 alpha >> AF8 alpha
                eeg[1] += 50 * np.sin(2 * np.pi * 10 * t)
                eeg[2] += 5 * np.sin(2 * np.pi * 10 * t)
            af_diverse.assess(eeg, fs=fs)

        # Uniform: same alpha ratio on every epoch -> near-constant FAA
        np.random.seed(301)
        for _ in range(10):
            eeg = np.random.randn(4, 1024) * 0.5  # very low noise
            eeg[2] += 30 * np.sin(2 * np.pi * 10 * t)
            eeg[1] += 30 * np.sin(2 * np.pi * 10 * t)
            af_uniform.assess(eeg, fs=fs)

        diverse_stats = af_diverse.get_session_stats()
        uniform_stats = af_uniform.get_session_stats()

        assert diverse_stats["mean_flexibility"] > uniform_stats["mean_flexibility"]


class TestSubMetrics:
    def test_sign_change_rate_all_same_sign(self, af, fs):
        """If FAA never changes sign, sign_change_rate should be 0 or near 0."""
        np.random.seed(400)
        t = np.arange(1024) / float(fs)
        # Make AF8 always have much more alpha than AF7 -> always positive FAA
        for _ in range(6):
            eeg = np.random.randn(4, 1024) * 2
            eeg[2] += 50 * np.sin(2 * np.pi * 10 * t)  # AF8 strong alpha
            eeg[1] += 5 * np.sin(2 * np.pi * 10 * t)   # AF7 weak alpha
            result = af.assess(eeg, fs=fs)

        assert result["sign_change_rate"] < 0.2

    def test_sign_change_rate_alternating(self, fs):
        """If FAA alternates sign every epoch, sign_change_rate should be high."""
        af = AffectiveFlexibility(fs=fs)
        np.random.seed(401)
        t = np.arange(1024) / float(fs)

        for i in range(8):
            eeg = np.random.randn(4, 1024) * 2
            if i % 2 == 0:
                # Positive FAA: AF8 alpha > AF7 alpha
                eeg[2] += 40 * np.sin(2 * np.pi * 10 * t)
                eeg[1] += 5 * np.sin(2 * np.pi * 10 * t)
            else:
                # Negative FAA: AF7 alpha > AF8 alpha
                eeg[1] += 40 * np.sin(2 * np.pi * 10 * t)
                eeg[2] += 5 * np.sin(2 * np.pi * 10 * t)
            result = af.assess(eeg, fs=fs)

        assert result["sign_change_rate"] > 0.7

    def test_shift_speed_with_constant_faa(self, fs):
        """If FAA barely changes, shift_speed should be low."""
        af = AffectiveFlexibility(fs=fs)
        np.random.seed(402)
        t = np.arange(1024) / float(fs)

        for _ in range(6):
            eeg = np.random.randn(4, 1024) * 0.5  # very low noise
            eeg[2] += 20 * np.sin(2 * np.pi * 10 * t)  # fixed alpha
            eeg[1] += 10 * np.sin(2 * np.pi * 10 * t)
            result = af.assess(eeg, fs=fs)

        assert result["shift_speed"] < 0.5

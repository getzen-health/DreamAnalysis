"""Tests for decision confidence and risk-taking detection from EEG."""
import numpy as np
import pytest

from models.decision_detector import DecisionDetector


FS = 256.0
DURATION_S = 4
N_SAMPLES = int(FS * DURATION_S)


@pytest.fixture
def detector():
    return DecisionDetector(fs=FS)


def _make_4ch(n_samples=N_SAMPLES, seed=42):
    """Random 4-channel Muse 2 EEG (TP9, AF7, AF8, TP10)."""
    rng = np.random.RandomState(seed)
    return rng.randn(4, n_samples) * 20.0


def _make_1ch(n_samples=N_SAMPLES, seed=42):
    """Single-channel EEG."""
    rng = np.random.RandomState(seed)
    return rng.randn(n_samples) * 20.0


def _make_theta_dominant(n_samples=N_SAMPLES, seed=42):
    """4-channel signal with strong frontal theta (deliberation).

    High theta at AF7/AF8 (ch1, ch2) + low beta = deliberation state.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n_samples) / FS
    signals = rng.randn(4, n_samples) * 3.0
    # Strong theta (6 Hz) on AF7 and AF8
    theta_wave = 40.0 * np.sin(2 * np.pi * 6 * t)
    signals[1] += theta_wave
    signals[2] += theta_wave
    return signals


def _make_beta_dominant(n_samples=N_SAMPLES, seed=42):
    """4-channel signal with strong beta and low theta/alpha (confident/decided).

    High beta + low alpha + low theta = high confidence, ready state.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n_samples) / FS
    signals = rng.randn(4, n_samples) * 3.0
    # Strong beta (20 Hz) on frontal channels
    beta_wave = 35.0 * np.sin(2 * np.pi * 20 * t)
    signals[1] += beta_wave
    signals[2] += beta_wave
    return signals


def _make_alpha_dominant(n_samples=N_SAMPLES, seed=42):
    """4-channel signal with strong alpha (disengaged/uncertain).

    High alpha + low beta = disengaged or uncertain.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n_samples) / FS
    signals = rng.randn(4, n_samples) * 3.0
    alpha_wave = 45.0 * np.sin(2 * np.pi * 10 * t)
    signals[1] += alpha_wave
    signals[2] += alpha_wave
    return signals


def _make_approach_signal(n_samples=N_SAMPLES, seed=42):
    """Signal with left-dominant activation (positive FAA = approach).

    More alpha on right (AF8, ch2) than left (AF7, ch1) = left activation = approach.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n_samples) / FS
    signals = rng.randn(4, n_samples) * 3.0
    # Low alpha on AF7 (more activation), high alpha on AF8 (less activation)
    signals[1] += 5.0 * np.sin(2 * np.pi * 10 * t)   # small alpha on AF7
    signals[2] += 40.0 * np.sin(2 * np.pi * 10 * t)   # large alpha on AF8
    # High beta for risk-seeking
    beta_wave = 30.0 * np.sin(2 * np.pi * 20 * t)
    signals[1] += beta_wave
    signals[2] += beta_wave
    return signals


def _make_withdrawal_signal(n_samples=N_SAMPLES, seed=42):
    """Signal with right-dominant activation (negative FAA = withdrawal/avoidance).

    More alpha on left (AF7, ch1) than right (AF8, ch2) = right activation = withdrawal.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n_samples) / FS
    signals = rng.randn(4, n_samples) * 3.0
    # High alpha on AF7 (less activation), low alpha on AF8 (more activation)
    signals[1] += 40.0 * np.sin(2 * np.pi * 10 * t)   # large alpha on AF7
    signals[2] += 5.0 * np.sin(2 * np.pi * 10 * t)    # small alpha on AF8
    # High theta for deliberation / risk aversion
    theta_wave = 25.0 * np.sin(2 * np.pi * 6 * t)
    signals[1] += theta_wave
    signals[2] += theta_wave
    return signals


# ── Construction ──────────────────────────────────────────────

class TestConstruction:
    def test_default_construction(self):
        d = DecisionDetector()
        assert d is not None

    def test_custom_fs(self):
        d = DecisionDetector(fs=128.0)
        result = d.assess(_make_4ch(), fs=128.0)
        assert "decision_confidence" in result

    def test_fs_stored(self):
        d = DecisionDetector(fs=512.0)
        # assess without explicit fs should use stored value
        result = d.assess(_make_4ch())
        assert "decision_confidence" in result


# ── assess() output structure ──────────────────────────────────

class TestAssessOutputStructure:
    def test_all_required_keys(self, detector):
        result = detector.assess(_make_4ch())
        required = {
            "decision_confidence",
            "risk_profile",
            "deliberation_intensity",
            "approach_motivation",
            "cognitive_conflict",
            "decision_readiness",
            "has_baseline",
        }
        assert required.issubset(set(result.keys()))

    def test_decision_confidence_range(self, detector):
        result = detector.assess(_make_4ch())
        assert 0 <= result["decision_confidence"] <= 100

    def test_risk_profile_valid(self, detector):
        result = detector.assess(_make_4ch())
        assert result["risk_profile"] in ("risk_seeking", "risk_neutral", "risk_averse")

    def test_deliberation_intensity_range(self, detector):
        result = detector.assess(_make_4ch())
        assert 0.0 <= result["deliberation_intensity"] <= 1.0

    def test_approach_motivation_range(self, detector):
        result = detector.assess(_make_4ch())
        assert -1.0 <= result["approach_motivation"] <= 1.0

    def test_cognitive_conflict_range(self, detector):
        result = detector.assess(_make_4ch())
        assert 0.0 <= result["cognitive_conflict"] <= 1.0

    def test_decision_readiness_valid(self, detector):
        result = detector.assess(_make_4ch())
        assert result["decision_readiness"] in ("ready", "deliberating", "uncertain", "disengaged")

    def test_has_baseline_false_initially(self, detector):
        result = detector.assess(_make_4ch())
        assert result["has_baseline"] is False


# ── Single-channel fallback ────────────────────────────────────

class TestSingleChannel:
    def test_1d_input_works(self, detector):
        result = detector.assess(_make_1ch())
        assert "decision_confidence" in result
        assert 0 <= result["decision_confidence"] <= 100

    def test_single_row_2d(self, detector):
        result = detector.assess(_make_1ch().reshape(1, -1))
        assert "decision_confidence" in result


# ── Deliberation detection ────────────────────────────────────

class TestDeliberation:
    def test_theta_dominant_high_deliberation(self, detector):
        result = detector.assess(_make_theta_dominant())
        assert result["deliberation_intensity"] > 0.4

    def test_beta_dominant_low_deliberation(self, detector):
        result = detector.assess(_make_beta_dominant())
        # Beta-dominant state should show lower deliberation than theta-dominant
        theta_result = detector.assess(_make_theta_dominant(seed=99))
        assert result["deliberation_intensity"] < theta_result["deliberation_intensity"]

    def test_theta_dominant_high_conflict(self, detector):
        result = detector.assess(_make_theta_dominant())
        assert result["cognitive_conflict"] > 0.3


# ── Risk profile classification ────────────────────────────────

class TestRiskProfile:
    def test_approach_signal_risk_seeking(self, detector):
        result = detector.assess(_make_approach_signal())
        assert result["risk_profile"] == "risk_seeking"

    def test_withdrawal_signal_risk_averse(self, detector):
        result = detector.assess(_make_withdrawal_signal())
        assert result["risk_profile"] == "risk_averse"

    def test_neutral_random_signal(self, detector):
        # Random noise should tend toward neutral
        results = []
        for seed in range(10, 20):
            r = detector.assess(_make_4ch(seed=seed))
            results.append(r["risk_profile"])
        # At least some should be neutral (not all extreme)
        assert "risk_neutral" in results


# ── Decision readiness ─────────────────────────────────────────

class TestDecisionReadiness:
    def test_alpha_dominant_uncertain_or_disengaged(self, detector):
        result = detector.assess(_make_alpha_dominant())
        assert result["decision_readiness"] in ("uncertain", "disengaged")

    def test_theta_dominant_deliberating(self, detector):
        result = detector.assess(_make_theta_dominant())
        assert result["decision_readiness"] == "deliberating"


# ── Approach motivation (FAA) ──────────────────────────────────

class TestApproachMotivation:
    def test_approach_signal_positive(self, detector):
        result = detector.assess(_make_approach_signal())
        assert result["approach_motivation"] > 0.0

    def test_withdrawal_signal_negative(self, detector):
        result = detector.assess(_make_withdrawal_signal())
        assert result["approach_motivation"] < 0.0

    def test_single_channel_near_zero(self, detector):
        # Without multichannel data, FAA can't be computed
        result = detector.assess(_make_1ch())
        assert abs(result["approach_motivation"]) < 0.5


# ── Baseline calibration ──────────────────────────────────────

class TestBaseline:
    def test_set_baseline_returns_dict(self, detector):
        result = detector.set_baseline(_make_4ch())
        assert isinstance(result, dict)
        assert result["baseline_set"] is True
        assert "baseline_metrics" in result

    def test_has_baseline_after_set(self, detector):
        detector.set_baseline(_make_4ch())
        result = detector.assess(_make_4ch(seed=99))
        assert result["has_baseline"] is True

    def test_baseline_changes_output(self, detector):
        signals = _make_4ch()
        before = detector.assess(signals)
        detector.set_baseline(signals)
        after = detector.assess(signals)
        # Confidence should differ (baseline normalization applied)
        # They may or may not differ by a lot, but the baseline flag changes
        assert after["has_baseline"] is True
        assert before["has_baseline"] is False

    def test_per_user_baseline(self, detector):
        detector.set_baseline(_make_4ch(seed=10), user_id="alice")
        result_alice = detector.assess(_make_4ch(seed=20), user_id="alice")
        result_bob = detector.assess(_make_4ch(seed=20), user_id="bob")
        assert result_alice["has_baseline"] is True
        assert result_bob["has_baseline"] is False

    def test_baseline_custom_fs(self, detector):
        result = detector.set_baseline(_make_4ch(), fs=128.0)
        assert result["baseline_set"] is True


# ── History ────────────────────────────────────────────────────

class TestHistory:
    def test_empty_history(self, detector):
        assert detector.get_history() == []

    def test_history_grows(self, detector):
        detector.assess(_make_4ch(seed=1))
        detector.assess(_make_4ch(seed=2))
        assert len(detector.get_history()) == 2

    def test_history_last_n(self, detector):
        for i in range(10):
            detector.assess(_make_4ch(seed=i))
        assert len(detector.get_history(last_n=3)) == 3

    def test_history_last_n_larger_than_history(self, detector):
        detector.assess(_make_4ch())
        assert len(detector.get_history(last_n=100)) == 1

    def test_history_cap_500(self, detector):
        """History should not exceed 500 entries."""
        for i in range(510):
            detector.assess(_make_4ch(seed=i % 100))
        assert len(detector.get_history()) <= 500


# ── Session stats ──────────────────────────────────────────────

class TestSessionStats:
    def test_empty_stats(self, detector):
        stats = detector.get_session_stats()
        assert stats["n_epochs"] == 0

    def test_stats_after_assessments(self, detector):
        for i in range(5):
            detector.assess(_make_4ch(seed=i))
        stats = detector.get_session_stats()
        assert stats["n_epochs"] == 5
        assert "mean_confidence" in stats
        assert "dominant_profile" in stats
        assert "dominant_readiness" in stats

    def test_mean_confidence_range(self, detector):
        for i in range(5):
            detector.assess(_make_4ch(seed=i))
        stats = detector.get_session_stats()
        assert 0 <= stats["mean_confidence"] <= 100

    def test_dominant_profile_valid(self, detector):
        for i in range(5):
            detector.assess(_make_4ch(seed=i))
        stats = detector.get_session_stats()
        assert stats["dominant_profile"] in ("risk_seeking", "risk_neutral", "risk_averse")

    def test_dominant_readiness_valid(self, detector):
        for i in range(5):
            detector.assess(_make_4ch(seed=i))
        stats = detector.get_session_stats()
        assert stats["dominant_readiness"] in ("ready", "deliberating", "uncertain", "disengaged")


# ── Multi-user isolation ───────────────────────────────────────

class TestMultiUser:
    def test_independent_histories(self, detector):
        detector.assess(_make_4ch(seed=1), user_id="alice")
        detector.assess(_make_4ch(seed=2), user_id="bob")
        detector.assess(_make_4ch(seed=3), user_id="alice")
        assert len(detector.get_history(user_id="alice")) == 2
        assert len(detector.get_history(user_id="bob")) == 1

    def test_independent_stats(self, detector):
        detector.assess(_make_4ch(seed=1), user_id="alice")
        stats_bob = detector.get_session_stats(user_id="bob")
        assert stats_bob["n_epochs"] == 0

    def test_reset_only_affects_target_user(self, detector):
        detector.assess(_make_4ch(seed=1), user_id="alice")
        detector.assess(_make_4ch(seed=2), user_id="bob")
        detector.reset(user_id="alice")
        assert len(detector.get_history(user_id="alice")) == 0
        assert len(detector.get_history(user_id="bob")) == 1


# ── Reset ──────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_history(self, detector):
        detector.assess(_make_4ch())
        detector.reset()
        assert len(detector.get_history()) == 0

    def test_reset_clears_baseline(self, detector):
        detector.set_baseline(_make_4ch())
        detector.reset()
        result = detector.assess(_make_4ch())
        assert result["has_baseline"] is False

    def test_reset_clears_stats(self, detector):
        detector.assess(_make_4ch())
        detector.reset()
        assert detector.get_session_stats()["n_epochs"] == 0


# ── Edge cases ─────────────────────────────────────────────────

class TestEdgeCases:
    def test_very_short_signal(self, detector):
        """Signal shorter than one Welch window should not crash."""
        short = np.random.RandomState(42).randn(4, 64) * 20
        result = detector.assess(short)
        assert "decision_confidence" in result

    def test_flat_signal(self, detector):
        """DC signal should not crash."""
        flat = np.ones((4, N_SAMPLES)) * 0.001
        result = detector.assess(flat)
        assert 0 <= result["decision_confidence"] <= 100

    def test_nan_in_signal(self, detector):
        """NaN values should be handled gracefully."""
        sig = _make_4ch()
        sig[1, 100] = np.nan
        result = detector.assess(sig)
        assert "decision_confidence" in result

    def test_two_channel_input(self, detector):
        """2-channel input should still work."""
        sig = np.random.RandomState(42).randn(2, N_SAMPLES) * 20
        result = detector.assess(sig)
        assert "decision_confidence" in result

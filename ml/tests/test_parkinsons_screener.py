"""Tests for ParkinsonsScreener — EEG-based Parkinson's tremor screening."""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.parkinsons_screener import ParkinsonsScreener, MEDICAL_DISCLAIMER


# ── Helpers ──────────────────────────────────────────────────────────


def _make_eeg(n_channels=4, duration_s=4, fs=256, rng_seed=42, amplitude=20.0):
    """Generate synthetic multichannel EEG."""
    rng = np.random.default_rng(rng_seed)
    n_samples = int(duration_s * fs)
    return rng.normal(0, amplitude, (n_channels, n_samples)).astype(np.float64)


def _inject_tremor(eeg, freq=5.0, amplitude=60.0, fs=256, channels=None):
    """Inject a sinusoidal tremor signal into temporal channels."""
    n_samples = eeg.shape[1]
    t = np.arange(n_samples) / fs
    tremor = amplitude * np.sin(2 * np.pi * freq * t)
    if channels is None:
        channels = [0, 3]  # TP9, TP10
    out = eeg.copy()
    for ch in channels:
        out[ch] += tremor
    return out


def _inject_theta_excess(eeg, fs=256, amplitude=80.0):
    """Inject strong theta (5 Hz) across all channels."""
    n_samples = eeg.shape[1]
    t = np.arange(n_samples) / fs
    theta = amplitude * np.sin(2 * np.pi * 5.5 * t)
    out = eeg.copy()
    for ch in range(out.shape[0]):
        out[ch] += theta
    return out


# ── Test: Output structure ───────────────────────────────────────────


class TestOutputStructure:
    def test_screen_returns_required_keys(self):
        screener = ParkinsonsScreener(fs=256.0)
        eeg = _make_eeg()
        result = screener.screen(eeg)
        required_keys = {
            "risk_score", "risk_level", "tremor_frequency", "tremor_detected",
            "theta_excess", "beta_deficit", "alpha_peak_freq",
            "asymmetry_index", "biomarkers", "medical_disclaimer", "has_baseline",
        }
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_biomarkers_dict_has_component_scores(self):
        screener = ParkinsonsScreener(fs=256.0)
        eeg = _make_eeg()
        result = screener.screen(eeg)
        biomarkers = result["biomarkers"]
        assert isinstance(biomarkers, dict)
        assert len(biomarkers) > 0

    def test_medical_disclaimer_always_present(self):
        screener = ParkinsonsScreener(fs=256.0)
        eeg = _make_eeg()
        result = screener.screen(eeg)
        assert result["medical_disclaimer"] == MEDICAL_DISCLAIMER
        assert "not a diagnostic" in result["medical_disclaimer"].lower()

    def test_set_baseline_returns_expected_keys(self):
        screener = ParkinsonsScreener(fs=256.0)
        eeg = _make_eeg()
        result = screener.set_baseline(eeg)
        assert result["baseline_set"] is True
        assert "baseline_metrics" in result


# ── Test: Risk score and level ───────────────────────────────────────


class TestRiskScoreAndLevel:
    def test_risk_score_in_range(self):
        screener = ParkinsonsScreener(fs=256.0)
        eeg = _make_eeg()
        result = screener.screen(eeg)
        assert 0 <= result["risk_score"] <= 100

    def test_risk_level_valid(self):
        screener = ParkinsonsScreener(fs=256.0)
        eeg = _make_eeg()
        result = screener.screen(eeg)
        assert result["risk_level"] in {"low", "mild", "moderate", "elevated"}

    def test_risk_level_low_for_clean_eeg(self):
        """Clean random EEG should not trigger elevated risk."""
        screener = ParkinsonsScreener(fs=256.0)
        eeg = _make_eeg(amplitude=10.0, rng_seed=99)
        result = screener.screen(eeg)
        assert result["risk_level"] in {"low", "mild"}

    def test_risk_level_boundaries(self):
        """Verify risk level thresholds: low 0-25, mild 25-50, moderate 50-75, elevated 75-100."""
        screener = ParkinsonsScreener(fs=256.0)
        eeg = _make_eeg()
        result = screener.screen(eeg)
        score = result["risk_score"]
        level = result["risk_level"]
        if score < 25:
            assert level == "low"
        elif score < 50:
            assert level == "mild"
        elif score < 75:
            assert level == "moderate"
        else:
            assert level == "elevated"


# ── Test: Tremor detection ───────────────────────────────────────────


class TestTremorDetection:
    def test_tremor_detected_with_injected_5hz(self):
        """A strong 5 Hz sinusoid on temporal channels should be detected."""
        screener = ParkinsonsScreener(fs=256.0)
        eeg = _make_eeg(amplitude=5.0, duration_s=8, rng_seed=10)
        eeg_tremor = _inject_tremor(eeg, freq=5.0, amplitude=80.0)
        result = screener.screen(eeg_tremor)
        assert result["tremor_detected"] is True
        assert result["tremor_frequency"] is not None
        assert 4.0 <= result["tremor_frequency"] <= 6.0

    def test_no_tremor_in_clean_eeg(self):
        """Clean random EEG should not detect a tremor peak."""
        screener = ParkinsonsScreener(fs=256.0)
        eeg = _make_eeg(amplitude=10.0, rng_seed=77)
        result = screener.screen(eeg)
        assert result["tremor_detected"] is False

    def test_tremor_frequency_is_none_when_not_detected(self):
        screener = ParkinsonsScreener(fs=256.0)
        eeg = _make_eeg(amplitude=10.0, rng_seed=55)
        result = screener.screen(eeg)
        if not result["tremor_detected"]:
            assert result["tremor_frequency"] is None

    def test_tremor_at_4hz_boundary(self):
        """Tremor at 4.5 Hz should be within detection range."""
        screener = ParkinsonsScreener(fs=256.0)
        eeg = _make_eeg(amplitude=5.0, duration_s=8, rng_seed=20)
        eeg_tremor = _inject_tremor(eeg, freq=4.5, amplitude=80.0)
        result = screener.screen(eeg_tremor)
        assert result["tremor_detected"] is True
        assert result["tremor_frequency"] is not None
        assert 4.0 <= result["tremor_frequency"] <= 6.0


# ── Test: Biomarker indices ──────────────────────────────────────────


class TestBiomarkerIndices:
    def test_theta_excess_in_range(self):
        screener = ParkinsonsScreener(fs=256.0)
        eeg = _make_eeg()
        result = screener.screen(eeg)
        assert 0.0 <= result["theta_excess"] <= 1.0

    def test_beta_deficit_in_range(self):
        screener = ParkinsonsScreener(fs=256.0)
        eeg = _make_eeg()
        result = screener.screen(eeg)
        assert 0.0 <= result["beta_deficit"] <= 1.0

    def test_asymmetry_index_non_negative(self):
        screener = ParkinsonsScreener(fs=256.0)
        eeg = _make_eeg()
        result = screener.screen(eeg)
        assert result["asymmetry_index"] >= 0.0

    def test_alpha_peak_freq_physiological(self):
        """Alpha peak should be in a physiological range (1-30 Hz)."""
        screener = ParkinsonsScreener(fs=256.0)
        eeg = _make_eeg(duration_s=4)
        result = screener.screen(eeg)
        assert 1.0 <= result["alpha_peak_freq"] <= 30.0

    def test_theta_excess_higher_with_injected_theta(self):
        """Injecting theta should increase theta_excess score."""
        screener = ParkinsonsScreener(fs=256.0)
        eeg_clean = _make_eeg(amplitude=10.0, rng_seed=30, duration_s=8)
        eeg_theta = _inject_theta_excess(eeg_clean, amplitude=80.0)
        result_clean = screener.screen(eeg_clean)
        result_theta = screener.screen(eeg_theta)
        assert result_theta["theta_excess"] >= result_clean["theta_excess"]


# ── Test: Baseline calibration ───────────────────────────────────────


class TestBaselineCalibration:
    def test_has_baseline_false_initially(self):
        screener = ParkinsonsScreener(fs=256.0)
        eeg = _make_eeg()
        result = screener.screen(eeg)
        assert result["has_baseline"] is False

    def test_has_baseline_true_after_set(self):
        screener = ParkinsonsScreener(fs=256.0)
        eeg = _make_eeg()
        screener.set_baseline(eeg)
        result = screener.screen(eeg)
        assert result["has_baseline"] is True

    def test_baseline_affects_screening(self):
        """Setting a baseline should influence the risk score (baseline normalization)."""
        screener = ParkinsonsScreener(fs=256.0)
        eeg = _make_eeg(rng_seed=50, duration_s=8)
        result_no_bl = screener.screen(eeg)
        screener.set_baseline(eeg)
        result_with_bl = screener.screen(eeg)
        # With baseline from same signal, risk should be lower or equal
        # (comparing to self = normal)
        assert result_with_bl["risk_score"] <= result_no_bl["risk_score"] + 5


# ── Test: Multi-user support ─────────────────────────────────────────


class TestMultiUser:
    def test_separate_baselines_per_user(self):
        screener = ParkinsonsScreener(fs=256.0)
        eeg1 = _make_eeg(rng_seed=1)
        eeg2 = _make_eeg(rng_seed=2)
        screener.set_baseline(eeg1, user_id="alice")
        screener.set_baseline(eeg2, user_id="bob")
        result_a = screener.screen(eeg1, user_id="alice")
        result_b = screener.screen(eeg2, user_id="bob")
        assert result_a["has_baseline"] is True
        assert result_b["has_baseline"] is True

    def test_separate_histories_per_user(self):
        screener = ParkinsonsScreener(fs=256.0)
        eeg = _make_eeg()
        screener.screen(eeg, user_id="alice")
        screener.screen(eeg, user_id="alice")
        screener.screen(eeg, user_id="bob")
        assert len(screener.get_history(user_id="alice")) == 2
        assert len(screener.get_history(user_id="bob")) == 1

    def test_reset_clears_specific_user(self):
        screener = ParkinsonsScreener(fs=256.0)
        eeg = _make_eeg()
        screener.screen(eeg, user_id="alice")
        screener.screen(eeg, user_id="bob")
        screener.reset(user_id="alice")
        assert len(screener.get_history(user_id="alice")) == 0
        assert len(screener.get_history(user_id="bob")) == 1


# ── Test: Session stats ──────────────────────────────────────────────


class TestSessionStats:
    def test_empty_stats(self):
        screener = ParkinsonsScreener(fs=256.0)
        stats = screener.get_session_stats()
        assert stats["n_epochs"] == 0
        assert stats["has_baseline"] is False

    def test_stats_after_screening(self):
        screener = ParkinsonsScreener(fs=256.0)
        eeg = _make_eeg()
        screener.screen(eeg)
        screener.screen(eeg)
        stats = screener.get_session_stats()
        assert stats["n_epochs"] == 2
        assert 0 <= stats["mean_risk"] <= 100

    def test_tremor_detection_rate(self):
        screener = ParkinsonsScreener(fs=256.0)
        eeg_clean = _make_eeg(amplitude=10.0, rng_seed=60, duration_s=8)
        eeg_tremor = _inject_tremor(eeg_clean, freq=5.0, amplitude=80.0)
        screener.screen(eeg_clean)
        screener.screen(eeg_tremor)
        stats = screener.get_session_stats()
        assert 0.0 <= stats["tremor_detection_rate"] <= 1.0


# ── Test: History ─────────────────────────────────────────────────────


class TestHistory:
    def test_history_empty_initially(self):
        screener = ParkinsonsScreener(fs=256.0)
        assert screener.get_history() == []

    def test_history_grows(self):
        screener = ParkinsonsScreener(fs=256.0)
        eeg = _make_eeg()
        for _ in range(3):
            screener.screen(eeg)
        assert len(screener.get_history()) == 3

    def test_history_last_n(self):
        screener = ParkinsonsScreener(fs=256.0)
        eeg = _make_eeg()
        for _ in range(5):
            screener.screen(eeg)
        assert len(screener.get_history(last_n=2)) == 2

    def test_history_capped_at_500(self):
        screener = ParkinsonsScreener(fs=256.0)
        # Use a short signal for speed
        eeg = _make_eeg(duration_s=2)
        for _ in range(510):
            screener.screen(eeg)
        assert len(screener.get_history()) <= 500


# ── Test: Single-channel input ───────────────────────────────────────


class TestSingleChannel:
    def test_single_channel_works(self):
        """Screen should handle 1D input gracefully."""
        screener = ParkinsonsScreener(fs=256.0)
        rng = np.random.default_rng(42)
        eeg = rng.normal(0, 20, 1024)
        result = screener.screen(eeg)
        assert 0 <= result["risk_score"] <= 100
        assert result["tremor_detected"] in (True, False)

    def test_two_channel_input(self):
        """2-channel input should not crash."""
        screener = ParkinsonsScreener(fs=256.0)
        eeg = _make_eeg(n_channels=2)
        result = screener.screen(eeg)
        assert 0 <= result["risk_score"] <= 100


# ── Test: Reset ──────────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_all(self):
        screener = ParkinsonsScreener(fs=256.0)
        eeg = _make_eeg()
        screener.set_baseline(eeg)
        screener.screen(eeg)
        screener.reset()
        assert screener.get_history() == []
        result = screener.screen(eeg)
        assert result["has_baseline"] is False

    def test_reset_specific_user_keeps_others(self):
        screener = ParkinsonsScreener(fs=256.0)
        eeg = _make_eeg()
        screener.set_baseline(eeg, user_id="a")
        screener.set_baseline(eeg, user_id="b")
        screener.screen(eeg, user_id="a")
        screener.screen(eeg, user_id="b")
        screener.reset(user_id="a")
        assert len(screener.get_history(user_id="a")) == 0
        assert len(screener.get_history(user_id="b")) == 1
        # Baseline for b should still exist
        result_b = screener.screen(eeg, user_id="b")
        assert result_b["has_baseline"] is True


# ── Test: Edge cases ──────────────────────────────────────────────────


class TestEdgeCases:
    def test_very_short_signal(self):
        """Short signal (< 1 second) should not crash."""
        screener = ParkinsonsScreener(fs=256.0)
        eeg = _make_eeg(duration_s=0.5)
        result = screener.screen(eeg)
        assert 0 <= result["risk_score"] <= 100

    def test_flat_signal(self):
        """Flat line should not crash."""
        screener = ParkinsonsScreener(fs=256.0)
        eeg = np.ones((4, 1024)) * 0.001
        result = screener.screen(eeg)
        assert 0 <= result["risk_score"] <= 100

    def test_custom_fs(self):
        """Non-default sampling rate should work."""
        screener = ParkinsonsScreener(fs=128.0)
        rng = np.random.default_rng(42)
        eeg = rng.normal(0, 20, (4, 512))
        result = screener.screen(eeg, fs=128.0)
        assert 0 <= result["risk_score"] <= 100

    def test_fs_override_in_screen(self):
        """fs parameter in screen() should override constructor default."""
        screener = ParkinsonsScreener(fs=256.0)
        rng = np.random.default_rng(42)
        eeg = rng.normal(0, 20, (4, 512))
        result = screener.screen(eeg, fs=128.0)
        assert 0 <= result["risk_score"] <= 100

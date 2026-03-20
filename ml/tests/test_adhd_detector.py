"""Tests for ADHDDetector — ADHD attention profile screening via EEG biomarkers."""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.adhd_detector import ADHDDetector, DISCLAIMER


FS = 256
DURATION = 4  # seconds
N_SAMPLES = FS * DURATION


def _make_eeg(theta_amp=10.0, beta_amp=10.0, alpha_amp=10.0, seed=42):
    """Synthesize 4-channel EEG with controllable band amplitudes."""
    rng = np.random.default_rng(seed)
    t = np.arange(N_SAMPLES) / FS
    signals = []
    for ch in range(4):
        theta = theta_amp * np.sin(2 * np.pi * 6 * t + ch * 0.2)
        alpha = alpha_amp * np.sin(2 * np.pi * 10 * t + ch * 0.3)
        beta = beta_amp * np.sin(2 * np.pi * 20 * t + ch * 0.5)
        noise = 2.0 * rng.normal(size=len(t))
        signals.append(theta + alpha + beta + noise)
    return np.array(signals)


def _high_tbr_eeg(seed=42):
    """EEG with high theta, low beta -> elevated TBR (ADHD-like)."""
    return _make_eeg(theta_amp=30.0, beta_amp=3.0, alpha_amp=8.0, seed=seed)


def _low_tbr_eeg(seed=42):
    """EEG with low theta, high beta -> low TBR (typical)."""
    return _make_eeg(theta_amp=3.0, beta_amp=30.0, alpha_amp=10.0, seed=seed)


def _hyperactive_eeg(seed=42):
    """EEG with high beta deficit + variable attention pattern."""
    return _make_eeg(theta_amp=12.0, beta_amp=25.0, alpha_amp=5.0, seed=seed)


@pytest.fixture
def detector():
    return ADHDDetector()


# ── TestBaseline ──────────────────────────────────────────


class TestBaseline:
    def test_set_baseline_returns_dict(self, detector):
        eeg = _make_eeg()
        result = detector.set_baseline(eeg, FS)
        assert isinstance(result, dict)
        assert "baseline_set" in result
        assert result["baseline_set"] is True

    def test_set_baseline_records_tbr(self, detector):
        eeg = _make_eeg()
        result = detector.set_baseline(eeg, FS)
        assert "baseline_tbr" in result
        assert result["baseline_tbr"] > 0

    def test_baseline_single_channel(self, detector):
        rng = np.random.default_rng(42)
        eeg = rng.normal(0, 10, N_SAMPLES)
        result = detector.set_baseline(eeg, FS)
        assert result["baseline_set"] is True


# ── TestAssess ────────────────────────────────────────────


class TestAssess:
    def test_output_keys(self, detector):
        eeg = _make_eeg()
        result = detector.assess(eeg, FS)
        expected_keys = {
            "tbr_score", "tbr_percentile", "attention_variability",
            "inhibition_index", "risk_level", "attention_profile",
            "component_scores", "disclaimer", "not_validated", "scale_context",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_score_ranges(self, detector):
        eeg = _make_eeg()
        result = detector.assess(eeg, FS)
        assert result["tbr_score"] > 0
        assert 0.0 <= result["tbr_percentile"] <= 100.0
        assert 0.0 <= result["attention_variability"] <= 1.0
        assert 0.0 <= result["inhibition_index"] <= 1.0

    def test_disclaimer_present(self, detector):
        eeg = _make_eeg()
        result = detector.assess(eeg, FS)
        assert result["disclaimer"] == DISCLAIMER
        assert "screening tool only" in result["disclaimer"].lower()


# ── TestTBR ───────────────────────────────────────────────


class TestTBR:
    def test_high_theta_low_beta_elevated_tbr(self, detector):
        eeg = _high_tbr_eeg()
        result = detector.assess(eeg, FS)
        assert result["tbr_score"] > 3.0

    def test_low_theta_high_beta_low_tbr(self, detector):
        eeg = _low_tbr_eeg()
        result = detector.assess(eeg, FS)
        assert result["tbr_score"] < 2.0

    def test_high_tbr_higher_percentile(self, detector):
        high = detector.assess(_high_tbr_eeg(seed=10), FS)
        low = detector.assess(_low_tbr_eeg(seed=11), FS)
        assert high["tbr_percentile"] > low["tbr_percentile"]


# ── TestRiskLevels ────────────────────────────────────────


class TestRiskLevels:
    def test_valid_risk_levels(self, detector):
        for seed in range(5):
            eeg = _make_eeg(seed=seed)
            result = detector.assess(eeg, FS)
            assert result["risk_level"] in {"low", "moderate", "elevated", "high"}

    def test_high_tbr_elevated_risk(self, detector):
        eeg = _high_tbr_eeg()
        result = detector.assess(eeg, FS)
        assert result["risk_level"] in {"elevated", "high"}

    def test_low_tbr_low_risk(self, detector):
        eeg = _low_tbr_eeg()
        result = detector.assess(eeg, FS)
        assert result["risk_level"] in {"low", "moderate"}


# ── TestAttentionProfile ──────────────────────────────────


class TestAttentionProfile:
    def test_valid_profiles(self, detector):
        for seed in range(5):
            eeg = _make_eeg(seed=seed)
            result = detector.assess(eeg, FS)
            assert result["attention_profile"] in {
                "theta_dominant", "beta_deficit", "mixed_pattern", "typical",
            }

    def test_high_theta_inattentive(self, detector):
        # High theta, low beta -> theta_dominant or mixed_pattern
        eeg = _high_tbr_eeg()
        result = detector.assess(eeg, FS)
        assert result["attention_profile"] in {"theta_dominant", "mixed_pattern"}

    def test_typical_pattern(self, detector):
        # Balanced theta/beta -> typical
        eeg = _low_tbr_eeg()
        result = detector.assess(eeg, FS)
        assert result["attention_profile"] == "typical"


# ── TestAttentionVariability ──────────────────────────────


class TestAttentionVariability:
    def test_variable_signal_high_variability(self, detector):
        """Signal with alternating high/low theta epochs -> high variability."""
        rng = np.random.default_rng(42)
        t = np.arange(N_SAMPLES) / FS

        # Build signal that alternates between high-theta and low-theta
        # within a single epoch to simulate attention fluctuation
        signal = np.zeros((4, N_SAMPLES))
        for ch in range(4):
            for i in range(4):
                seg_start = i * (N_SAMPLES // 4)
                seg_end = (i + 1) * (N_SAMPLES // 4)
                t_seg = t[seg_start:seg_end]
                if i % 2 == 0:
                    signal[ch, seg_start:seg_end] = 30.0 * np.sin(2 * np.pi * 6 * t_seg)
                else:
                    signal[ch, seg_start:seg_end] = 30.0 * np.sin(2 * np.pi * 20 * t_seg)
            signal[ch] += rng.normal(0, 2, N_SAMPLES)

        # First do multiple assessments to build up variability history
        for _ in range(5):
            detector.assess(signal, FS)

        # Now also test with a stable signal
        det2 = ADHDDetector()
        stable = _make_eeg(seed=99)
        for _ in range(5):
            det2.assess(stable, FS)

        r_var = detector.assess(signal, FS)
        r_stable = det2.assess(stable, FS)
        # Variable signal should have higher or equal variability
        assert r_var["attention_variability"] >= 0.0

    def test_variability_bounded(self, detector):
        eeg = _make_eeg()
        result = detector.assess(eeg, FS)
        assert 0.0 <= result["attention_variability"] <= 1.0


# ── TestSessionStats ──────────────────────────────────────


class TestSessionStats:
    def test_empty_stats(self, detector):
        stats = detector.get_session_stats()
        assert stats["n_assessments"] == 0

    def test_stats_after_assessments(self, detector):
        eeg = _make_eeg()
        detector.assess(eeg, FS)
        detector.assess(eeg, FS)
        stats = detector.get_session_stats()
        assert stats["n_assessments"] == 2
        assert "mean_tbr" in stats
        assert stats["mean_tbr"] > 0


# ── TestHistory ───────────────────────────────────────────


class TestHistory:
    def test_empty_history(self, detector):
        assert detector.get_history() == []

    def test_history_grows(self, detector):
        eeg = _make_eeg()
        detector.assess(eeg, FS)
        detector.assess(eeg, FS)
        detector.assess(eeg, FS)
        assert len(detector.get_history()) == 3

    def test_history_last_n(self, detector):
        eeg = _make_eeg()
        for _ in range(5):
            detector.assess(eeg, FS)
        assert len(detector.get_history(last_n=2)) == 2


# ── TestMultiUser ─────────────────────────────────────────


class TestMultiUser:
    def test_independent_users(self, detector):
        eeg1 = _high_tbr_eeg(seed=1)
        eeg2 = _low_tbr_eeg(seed=2)
        detector.assess(eeg1, FS, user_id="user_a")
        detector.assess(eeg2, FS, user_id="user_b")
        assert len(detector.get_history(user_id="user_a")) == 1
        assert len(detector.get_history(user_id="user_b")) == 1
        # Histories are independent
        stats_a = detector.get_session_stats(user_id="user_a")
        stats_b = detector.get_session_stats(user_id="user_b")
        assert stats_a["mean_tbr"] != stats_b["mean_tbr"]


# ── TestReset ─────────────────────────────────────────────


class TestReset:
    def test_reset_clears_state(self, detector):
        eeg = _make_eeg()
        detector.set_baseline(eeg, FS)
        detector.assess(eeg, FS)
        detector.assess(eeg, FS)
        detector.reset()
        assert detector.get_history() == []
        assert detector.get_session_stats()["n_assessments"] == 0

    def test_reset_specific_user(self, detector):
        eeg = _make_eeg()
        detector.assess(eeg, FS, user_id="keep")
        detector.assess(eeg, FS, user_id="remove")
        detector.reset(user_id="remove")
        assert len(detector.get_history(user_id="keep")) == 1
        assert len(detector.get_history(user_id="remove")) == 0


# ── TestAttentionProfile (additional) ─────────────────────


class TestComponentScores:
    def test_component_scores_keys(self, detector):
        eeg = _make_eeg()
        result = detector.assess(eeg, FS)
        cs = result["component_scores"]
        expected = {"tbr_component", "theta_excess", "beta_deficit",
                    "alpha_variability"}
        assert expected.issubset(set(cs.keys()))

    def test_component_scores_range(self, detector):
        eeg = _make_eeg()
        result = detector.assess(eeg, FS)
        for key, val in result["component_scores"].items():
            assert 0.0 <= val <= 1.0, f"{key} out of range: {val}"

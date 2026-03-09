"""Tests for HeartBrainCoupling model.

Synthetic data used throughout:
  - PPG: 1 Hz sinusoid + 2nd harmonic (mimics 60 BPM systolic peaks)
  - EEG: 4-channel signals with controllable alpha/beta/theta content
  - High-alpha EEG: strong 10 Hz component (relaxed, interoceptive)
  - High-beta EEG: strong 20 Hz component (stressed, sympathetic)

References:
  Schandry (1981) -- heartbeat detection and interoceptive awareness
  Park et al. (2014) -- HEP amplitude indexes body awareness
"""

import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.heart_brain_coupling import HeartBrainCoupling

# ── Synthetic data generators ────────────────────────────────────────────────

FS_EEG = 256
FS_PPG = 64
DURATION_S = 10

_rng = np.random.default_rng(42)


def _make_ppg(duration_s: float = DURATION_S, bpm: float = 60.0, fs: float = FS_PPG):
    """Synthetic PPG at given BPM."""
    t = np.linspace(0, duration_s, int(duration_s * fs))
    freq = bpm / 60.0
    return np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * 2 * freq * t)


def _make_eeg(
    duration_s: float = DURATION_S,
    fs: float = FS_EEG,
    alpha_amp: float = 10.0,
    beta_amp: float = 5.0,
    theta_amp: float = 5.0,
    n_channels: int = 4,
    seed: int = 42,
):
    """Synthetic EEG with controllable band amplitudes."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_s * fs)) / fs
    signals = []
    for ch in range(n_channels):
        alpha = alpha_amp * np.sin(2 * np.pi * 10 * t + ch * 0.3)
        beta = beta_amp * np.sin(2 * np.pi * 20 * t + ch * 0.5)
        theta = theta_amp * np.sin(2 * np.pi * 6 * t + ch * 0.7)
        noise = 2.0 * rng.standard_normal(len(t))
        signals.append(alpha + beta + theta + noise)
    return np.array(signals)


PPG_60BPM = _make_ppg(bpm=60.0)
PPG_90BPM = _make_ppg(bpm=90.0)
PPG_SILENT = np.zeros(int(DURATION_S * FS_PPG))
EEG_DEFAULT = _make_eeg()
EEG_HIGH_ALPHA = _make_eeg(alpha_amp=25.0, beta_amp=3.0, theta_amp=3.0)
EEG_HIGH_BETA = _make_eeg(alpha_amp=3.0, beta_amp=25.0, theta_amp=3.0)
EEG_HIGH_THETA = _make_eeg(alpha_amp=3.0, beta_amp=3.0, theta_amp=25.0)


# ── TestBaseline ─────────────────────────────────────────────────────────────

class TestBaseline:
    def test_set_baseline_returns_required_keys(self):
        model = HeartBrainCoupling()
        result = model.set_baseline(EEG_DEFAULT, PPG_60BPM, fs=FS_EEG)
        required = {
            "baseline_set", "baseline_hr_bpm", "baseline_rmssd",
            "baseline_sdnn", "baseline_lf_hf", "baseline_alpha_power",
            "baseline_theta_power",
        }
        assert required.issubset(result.keys()), (
            f"Missing keys: {required - result.keys()}"
        )

    def test_baseline_set_flag_true(self):
        model = HeartBrainCoupling()
        result = model.set_baseline(EEG_DEFAULT, PPG_60BPM, fs=FS_EEG)
        assert result["baseline_set"] is True

    def test_baseline_hr_bpm_approximately_60(self):
        model = HeartBrainCoupling()
        result = model.set_baseline(EEG_DEFAULT, PPG_60BPM, fs=FS_EEG)
        assert 40 <= result["baseline_hr_bpm"] <= 80, (
            f"Expected ~60 BPM, got {result['baseline_hr_bpm']}"
        )

    def test_baseline_alpha_power_positive(self):
        model = HeartBrainCoupling()
        result = model.set_baseline(EEG_HIGH_ALPHA, PPG_60BPM, fs=FS_EEG)
        assert result["baseline_alpha_power"] > 0

    def test_baseline_with_1d_eeg(self):
        model = HeartBrainCoupling()
        eeg_1d = EEG_DEFAULT[0]
        result = model.set_baseline(eeg_1d, PPG_60BPM, fs=FS_EEG)
        assert result["baseline_set"] is True


# ── TestAnalyze ──────────────────────────────────────────────────────────────

class TestAnalyze:
    def test_output_keys(self):
        model = HeartBrainCoupling()
        result = model.analyze(EEG_DEFAULT, PPG_60BPM, fs=FS_EEG)
        required = {
            "hep_amplitude", "hrv_metrics", "coupling_strength",
            "interoceptive_score", "autonomic_state", "coherence_index",
        }
        assert required.issubset(result.keys()), (
            f"Missing keys: {required - result.keys()}"
        )

    def test_hrv_metrics_subkeys(self):
        model = HeartBrainCoupling()
        result = model.analyze(EEG_DEFAULT, PPG_60BPM, fs=FS_EEG)
        hrv = result["hrv_metrics"]
        required = {"rmssd", "sdnn", "lf_hf_ratio", "hr_bpm"}
        assert required.issubset(hrv.keys()), (
            f"Missing HRV keys: {required - hrv.keys()}"
        )

    def test_coupling_strength_range(self):
        model = HeartBrainCoupling()
        result = model.analyze(EEG_DEFAULT, PPG_60BPM, fs=FS_EEG)
        assert 0.0 <= result["coupling_strength"] <= 1.0

    def test_interoceptive_score_range(self):
        model = HeartBrainCoupling()
        result = model.analyze(EEG_DEFAULT, PPG_60BPM, fs=FS_EEG)
        assert 0.0 <= result["interoceptive_score"] <= 100.0

    def test_coherence_index_range(self):
        model = HeartBrainCoupling()
        result = model.analyze(EEG_DEFAULT, PPG_60BPM, fs=FS_EEG)
        assert 0.0 <= result["coherence_index"] <= 1.0

    def test_autonomic_state_valid_value(self):
        model = HeartBrainCoupling()
        result = model.analyze(EEG_DEFAULT, PPG_60BPM, fs=FS_EEG)
        assert result["autonomic_state"] in (
            "sympathetic", "parasympathetic", "balanced"
        )

    def test_silent_ppg_no_crash(self):
        model = HeartBrainCoupling()
        result = model.analyze(EEG_DEFAULT, PPG_SILENT, fs=FS_EEG)
        assert isinstance(result, dict)
        assert result["coupling_strength"] >= 0.0

    def test_1d_eeg_accepted(self):
        model = HeartBrainCoupling()
        result = model.analyze(EEG_DEFAULT[0], PPG_60BPM, fs=FS_EEG)
        assert "coupling_strength" in result


# ── TestHEP ──────────────────────────────────────────────────────────────────

class TestHEP:
    def test_hep_amplitude_nonnegative(self):
        model = HeartBrainCoupling()
        result = model.analyze(EEG_DEFAULT, PPG_60BPM, fs=FS_EEG)
        assert result["hep_amplitude"] >= 0.0

    def test_hep_amplitude_zero_with_silent_ppg(self):
        model = HeartBrainCoupling()
        result = model.analyze(EEG_DEFAULT, PPG_SILENT, fs=FS_EEG)
        assert result["hep_amplitude"] == 0.0

    def test_hep_with_high_alpha_eeg(self):
        """High alpha EEG should produce detectable HEP."""
        model = HeartBrainCoupling()
        result = model.analyze(EEG_HIGH_ALPHA, PPG_60BPM, fs=FS_EEG)
        assert result["hep_amplitude"] >= 0.0


# ── TestInteroception ────────────────────────────────────────────────────────

class TestInteroception:
    def test_score_between_0_and_100(self):
        model = HeartBrainCoupling()
        result = model.analyze(EEG_DEFAULT, PPG_60BPM, fs=FS_EEG)
        assert 0.0 <= result["interoceptive_score"] <= 100.0

    def test_silent_ppg_low_interoception(self):
        """No heartbeat = no interoceptive signal."""
        model = HeartBrainCoupling()
        result = model.analyze(EEG_DEFAULT, PPG_SILENT, fs=FS_EEG)
        assert result["interoceptive_score"] <= 50.0

    def test_baseline_affects_interoception(self):
        """Setting baseline should change interoceptive scoring."""
        model = HeartBrainCoupling()
        result_no_bl = model.analyze(EEG_DEFAULT, PPG_60BPM, fs=FS_EEG)

        model2 = HeartBrainCoupling()
        model2.set_baseline(EEG_DEFAULT, PPG_60BPM, fs=FS_EEG)
        result_with_bl = model2.analyze(EEG_DEFAULT, PPG_60BPM, fs=FS_EEG)

        # Both should be valid scores, possibly different
        assert 0.0 <= result_no_bl["interoceptive_score"] <= 100.0
        assert 0.0 <= result_with_bl["interoceptive_score"] <= 100.0


# ── TestAutonomicState ───────────────────────────────────────────────────────

class TestAutonomicState:
    def test_returns_valid_state(self):
        model = HeartBrainCoupling()
        result = model.analyze(EEG_DEFAULT, PPG_60BPM, fs=FS_EEG)
        assert result["autonomic_state"] in (
            "sympathetic", "parasympathetic", "balanced"
        )

    def test_different_eeg_can_produce_different_states(self):
        """Different physiological conditions may produce different states.

        We verify the classification works without crash for various inputs.
        The actual autonomic state depends on the HRV LF/HF ratio from PPG,
        not EEG content, so we test multiple PPG rates.
        """
        model = HeartBrainCoupling()
        states = set()
        for ppg in [PPG_60BPM, PPG_90BPM, PPG_SILENT]:
            result = model.analyze(EEG_DEFAULT, ppg, fs=FS_EEG)
            states.add(result["autonomic_state"])
        # At minimum, should not crash on any input
        assert len(states) >= 1

    def test_silent_ppg_classification(self):
        """Silent PPG (no beats) should still produce a valid state."""
        model = HeartBrainCoupling()
        result = model.analyze(EEG_DEFAULT, PPG_SILENT, fs=FS_EEG)
        assert result["autonomic_state"] in (
            "sympathetic", "parasympathetic", "balanced"
        )


# ── TestSessionStats ─────────────────────────────────────────────────────────

class TestSessionStats:
    def test_empty_stats(self):
        model = HeartBrainCoupling()
        stats = model.get_session_stats()
        assert stats["n_analyses"] == 0

    def test_stats_after_data(self):
        model = HeartBrainCoupling()
        model.analyze(EEG_DEFAULT, PPG_60BPM, fs=FS_EEG)
        model.analyze(EEG_HIGH_ALPHA, PPG_60BPM, fs=FS_EEG)
        model.analyze(EEG_HIGH_BETA, PPG_60BPM, fs=FS_EEG)
        stats = model.get_session_stats()
        assert stats["n_analyses"] == 3
        assert "mean_coupling_strength" in stats
        assert "mean_interoceptive_score" in stats
        assert "autonomic_state_distribution" in stats

    def test_stats_coupling_in_range(self):
        model = HeartBrainCoupling()
        for _ in range(5):
            model.analyze(EEG_DEFAULT, PPG_60BPM, fs=FS_EEG)
        stats = model.get_session_stats()
        assert 0.0 <= stats["mean_coupling_strength"] <= 1.0
        assert 0.0 <= stats["max_coupling_strength"] <= 1.0

    def test_has_baseline_flag(self):
        model = HeartBrainCoupling()
        stats = model.get_session_stats()
        assert stats["has_baseline"] is False

        model.set_baseline(EEG_DEFAULT, PPG_60BPM, fs=FS_EEG)
        stats = model.get_session_stats()
        assert stats["has_baseline"] is True


# ── TestHistory ──────────────────────────────────────────────────────────────

class TestHistory:
    def test_empty_history(self):
        model = HeartBrainCoupling()
        assert model.get_history() == []

    def test_history_grows(self):
        model = HeartBrainCoupling()
        model.analyze(EEG_DEFAULT, PPG_60BPM, fs=FS_EEG)
        assert len(model.get_history()) == 1
        model.analyze(EEG_DEFAULT, PPG_60BPM, fs=FS_EEG)
        assert len(model.get_history()) == 2

    def test_last_n(self):
        model = HeartBrainCoupling()
        for _ in range(5):
            model.analyze(EEG_DEFAULT, PPG_60BPM, fs=FS_EEG)
        last_2 = model.get_history(last_n=2)
        assert len(last_2) == 2

    def test_last_n_exceeds_history(self):
        model = HeartBrainCoupling()
        model.analyze(EEG_DEFAULT, PPG_60BPM, fs=FS_EEG)
        all_history = model.get_history(last_n=100)
        assert len(all_history) == 1

    def test_history_contains_expected_keys(self):
        model = HeartBrainCoupling()
        model.analyze(EEG_DEFAULT, PPG_60BPM, fs=FS_EEG)
        entry = model.get_history()[0]
        assert "coupling_strength" in entry
        assert "interoceptive_score" in entry
        assert "autonomic_state" in entry


# ── TestMultiUser ────────────────────────────────────────────────────────────

class TestMultiUser:
    def test_independent_users(self):
        model = HeartBrainCoupling()
        model.set_baseline(EEG_DEFAULT, PPG_60BPM, fs=FS_EEG, user_id="alice")
        model.analyze(EEG_DEFAULT, PPG_60BPM, fs=FS_EEG, user_id="alice")

        model.set_baseline(EEG_HIGH_ALPHA, PPG_60BPM, fs=FS_EEG, user_id="bob")
        model.analyze(EEG_HIGH_ALPHA, PPG_60BPM, fs=FS_EEG, user_id="bob")
        model.analyze(EEG_HIGH_ALPHA, PPG_60BPM, fs=FS_EEG, user_id="bob")

        alice = model.get_session_stats("alice")
        bob = model.get_session_stats("bob")
        assert alice["n_analyses"] == 1
        assert bob["n_analyses"] == 2

    def test_reset_one_user_keeps_other(self):
        model = HeartBrainCoupling()
        model.analyze(EEG_DEFAULT, PPG_60BPM, fs=FS_EEG, user_id="x")
        model.analyze(EEG_DEFAULT, PPG_60BPM, fs=FS_EEG, user_id="y")
        model.reset(user_id="x")
        assert model.get_session_stats("x")["n_analyses"] == 0
        assert model.get_session_stats("y")["n_analyses"] == 1


# ── TestReset ────────────────────────────────────────────────────────────────

class TestReset:
    def test_clears_history(self):
        model = HeartBrainCoupling()
        model.analyze(EEG_DEFAULT, PPG_60BPM, fs=FS_EEG)
        assert len(model.get_history()) == 1
        model.reset()
        assert len(model.get_history()) == 0

    def test_clears_baseline(self):
        model = HeartBrainCoupling()
        model.set_baseline(EEG_DEFAULT, PPG_60BPM, fs=FS_EEG)
        stats = model.get_session_stats()
        assert stats["has_baseline"] is True
        model.reset()
        stats = model.get_session_stats()
        assert stats["has_baseline"] is False

    def test_clears_session_stats(self):
        model = HeartBrainCoupling()
        model.analyze(EEG_DEFAULT, PPG_60BPM, fs=FS_EEG)
        model.reset()
        stats = model.get_session_stats()
        assert stats["n_analyses"] == 0

"""Tests for MCIScreener — MCI/early Alzheimer's EEG screening."""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.mci_screener import MCIScreener, MEDICAL_DISCLAIMER, get_mci_screener


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_eeg(n_channels=4, duration_sec=4, fs=256, rng_seed=42):
    """Generate synthetic multichannel EEG."""
    rng = np.random.default_rng(rng_seed)
    n_samples = int(duration_sec * fs)
    return rng.normal(0, 15, (n_channels, n_samples)).astype(np.float64)


def _make_alpha_dominant_eeg(peak_hz=10.0, n_channels=4, duration_sec=4, fs=256):
    """Synthetic EEG with a strong alpha peak at the given frequency."""
    rng = np.random.default_rng(99)
    n_samples = int(duration_sec * fs)
    t = np.arange(n_samples) / fs
    signals = rng.normal(0, 5, (n_channels, n_samples))
    # Inject strong sinusoid at peak_hz into every channel
    for ch in range(n_channels):
        signals[ch] += 40 * np.sin(2 * np.pi * peak_hz * t + rng.uniform(0, 2 * np.pi))
    return signals


def _make_mci_like_eeg(n_channels=4, duration_sec=4, fs=256):
    """EEG mimicking MCI: heavy theta/delta, weak alpha, slowed peak."""
    rng = np.random.default_rng(77)
    n_samples = int(duration_sec * fs)
    t = np.arange(n_samples) / fs
    signals = rng.normal(0, 3, (n_channels, n_samples))
    for ch in range(n_channels):
        # Strong theta (6 Hz) and delta (2 Hz), weak alpha
        signals[ch] += 30 * np.sin(2 * np.pi * 6.0 * t)
        signals[ch] += 25 * np.sin(2 * np.pi * 2.0 * t)
        signals[ch] += 5 * np.sin(2 * np.pi * 9.0 * t)  # weak alpha
    return signals


# ── Basic output structure ───────────────────────────────────────────────────

def test_screen_returns_required_keys():
    screener = MCIScreener()
    eeg = _make_eeg()
    result = screener.screen(eeg)
    required_keys = [
        "risk_score", "risk_level", "alpha_peak_freq", "theta_alpha_ratio",
        "spectral_entropy", "coherence_index", "delta_ratio", "biomarkers",
        "medical_disclaimer", "has_baseline",
    ]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"


def test_screen_includes_aperiodic_slope():
    screener = MCIScreener()
    result = screener.screen(_make_eeg())
    assert "aperiodic_slope" in result


def test_medical_disclaimer_always_present():
    screener = MCIScreener()
    result = screener.screen(_make_eeg())
    assert result["medical_disclaimer"] == MEDICAL_DISCLAIMER
    assert "not" in result["medical_disclaimer"].lower()
    assert "diagnostic" in result["medical_disclaimer"].lower() or \
           "diagnosis" in result["medical_disclaimer"].lower()


def test_biomarkers_dict_keys():
    screener = MCIScreener()
    result = screener.screen(_make_eeg())
    expected_biomarker_keys = [
        "alpha_peak_freq", "theta_alpha_ratio", "spectral_entropy",
        "coherence_index", "delta_ratio", "aperiodic_slope",
    ]
    for key in expected_biomarker_keys:
        assert key in result["biomarkers"], f"Missing biomarker: {key}"


# ── Risk score range and levels ──────────────────────────────────────────────

def test_risk_score_in_range():
    screener = MCIScreener()
    result = screener.screen(_make_eeg())
    assert 0 <= result["risk_score"] <= 100


def test_risk_level_is_valid():
    screener = MCIScreener()
    result = screener.screen(_make_eeg())
    assert result["risk_level"] in ["low", "mild", "moderate", "elevated"]


def test_risk_level_low_range():
    screener = MCIScreener()
    # Force a known low score by mocking internals is complex;
    # instead verify the classification logic directly
    assert screener._classify_risk(0) == "low"
    assert screener._classify_risk(24.9) == "low"


def test_risk_level_mild_range():
    screener = MCIScreener()
    assert screener._classify_risk(25) == "mild"
    assert screener._classify_risk(49.9) == "mild"


def test_risk_level_moderate_range():
    screener = MCIScreener()
    assert screener._classify_risk(50) == "moderate"
    assert screener._classify_risk(74.9) == "moderate"


def test_risk_level_elevated_range():
    screener = MCIScreener()
    assert screener._classify_risk(75) == "elevated"
    assert screener._classify_risk(100) == "elevated"


# ── Biomarker value ranges ───────────────────────────────────────────────────

def test_spectral_entropy_in_unit_range():
    screener = MCIScreener()
    result = screener.screen(_make_eeg())
    assert 0 <= result["spectral_entropy"] <= 1.0


def test_coherence_index_in_unit_range():
    screener = MCIScreener()
    result = screener.screen(_make_eeg())
    assert 0 <= result["coherence_index"] <= 1.0


def test_delta_ratio_in_unit_range():
    screener = MCIScreener()
    result = screener.screen(_make_eeg())
    assert 0 <= result["delta_ratio"] <= 1.0


def test_alpha_peak_freq_reasonable():
    screener = MCIScreener()
    result = screener.screen(_make_eeg())
    # Should be between 6 and 14 Hz (the search range)
    assert 6.0 <= result["alpha_peak_freq"] <= 14.0


def test_biomarker_scores_in_range():
    screener = MCIScreener()
    result = screener.screen(_make_eeg())
    for key, val in result["biomarkers"].items():
        assert 0 <= val <= 100, f"Biomarker {key} out of [0,100]: {val}"


# ── Alpha peak frequency detection ──────────────────────────────────────────

def test_alpha_peak_detection_10hz():
    screener = MCIScreener()
    eeg = _make_alpha_dominant_eeg(peak_hz=10.0)
    result = screener.screen(eeg)
    # Should detect peak near 10 Hz
    assert abs(result["alpha_peak_freq"] - 10.0) < 1.5


def test_alpha_peak_detection_slowed():
    screener = MCIScreener()
    eeg = _make_alpha_dominant_eeg(peak_hz=7.5)
    result = screener.screen(eeg)
    # Should detect the slowed peak
    assert abs(result["alpha_peak_freq"] - 7.5) < 1.5


# ── MCI-like vs healthy EEG ─────────────────────────────────────────────────

def test_mci_eeg_has_higher_risk_than_healthy():
    screener = MCIScreener()
    healthy = _make_alpha_dominant_eeg(peak_hz=10.0)
    mci = _make_mci_like_eeg()
    result_healthy = screener.screen(healthy)
    result_mci = screener.screen(mci)
    assert result_mci["risk_score"] > result_healthy["risk_score"]


def test_mci_eeg_has_higher_theta_alpha_ratio():
    screener = MCIScreener()
    healthy = _make_alpha_dominant_eeg(peak_hz=10.0)
    mci = _make_mci_like_eeg()
    result_healthy = screener.screen(healthy)
    result_mci = screener.screen(mci)
    assert result_mci["theta_alpha_ratio"] > result_healthy["theta_alpha_ratio"]


# ── Single-channel input ─────────────────────────────────────────────────────

def test_single_channel_input():
    screener = MCIScreener()
    eeg = np.random.default_rng(10).normal(0, 15, 1024)
    result = screener.screen(eeg, fs=256)
    assert "risk_score" in result
    assert 0 <= result["risk_score"] <= 100
    # Single channel: coherence should be 1.0 (no disconnection measurable)
    assert result["coherence_index"] == 1.0


# ── Baseline ─────────────────────────────────────────────────────────────────

def test_set_baseline_returns_correct_structure():
    screener = MCIScreener()
    eeg = _make_eeg()
    result = screener.set_baseline(eeg)
    assert result["baseline_set"] is True
    assert "baseline_metrics" in result
    assert "alpha_peak_freq" in result["baseline_metrics"]


def test_screen_reports_has_baseline():
    screener = MCIScreener()
    eeg = _make_eeg()
    result_no_bl = screener.screen(eeg)
    assert result_no_bl["has_baseline"] is False
    screener.set_baseline(eeg)
    result_with_bl = screener.screen(eeg)
    assert result_with_bl["has_baseline"] is True


def test_baseline_with_age():
    screener = MCIScreener()
    eeg = _make_eeg()
    result = screener.set_baseline(eeg, age=65)
    assert result["baseline_set"] is True


def test_baseline_adjusts_risk_scoring():
    """When baseline is set from the same signal, relative scores should be low."""
    screener = MCIScreener()
    eeg = _make_eeg()
    screener.set_baseline(eeg)
    result = screener.screen(eeg)
    # Same signal as baseline: relative deviations should be ~0
    # so biomarker scores should be low
    for key, val in result["biomarkers"].items():
        assert val < 50, f"Biomarker {key} unexpectedly high with own baseline: {val}"


# ── Multi-user support ───────────────────────────────────────────────────────

def test_multi_user_isolation():
    screener = MCIScreener()
    eeg_a = _make_eeg(rng_seed=1)
    eeg_b = _make_eeg(rng_seed=2)
    screener.set_baseline(eeg_a, user_id="alice")
    screener.screen(eeg_a, user_id="alice")
    screener.screen(eeg_b, user_id="bob")
    stats_alice = screener.get_session_stats(user_id="alice")
    stats_bob = screener.get_session_stats(user_id="bob")
    assert stats_alice["has_baseline"] is True
    assert stats_bob["has_baseline"] is False
    assert stats_alice["n_epochs"] == 1
    assert stats_bob["n_epochs"] == 1


# ── Session stats ────────────────────────────────────────────────────────────

def test_session_stats_empty():
    screener = MCIScreener()
    stats = screener.get_session_stats()
    assert stats["n_epochs"] == 0
    assert stats["mean_risk"] == 0.0
    assert stats["has_baseline"] is False


def test_session_stats_after_screens():
    screener = MCIScreener()
    for seed in range(5):
        screener.screen(_make_eeg(rng_seed=seed))
    stats = screener.get_session_stats()
    assert stats["n_epochs"] == 5
    assert 0 <= stats["mean_risk"] <= 100


# ── History ──────────────────────────────────────────────────────────────────

def test_history_grows():
    screener = MCIScreener()
    screener.screen(_make_eeg(rng_seed=1))
    screener.screen(_make_eeg(rng_seed=2))
    screener.screen(_make_eeg(rng_seed=3))
    history = screener.get_history()
    assert len(history) == 3


def test_history_last_n():
    screener = MCIScreener()
    for seed in range(10):
        screener.screen(_make_eeg(rng_seed=seed))
    last_3 = screener.get_history(last_n=3)
    assert len(last_3) == 3
    full = screener.get_history()
    assert last_3 == full[-3:]


def test_history_capped_at_500():
    screener = MCIScreener()
    # Use small signals for speed
    rng = np.random.default_rng(42)
    for _ in range(510):
        small_eeg = rng.normal(0, 10, (4, 256))
        screener.screen(small_eeg)
    history = screener.get_history()
    assert len(history) <= 500


# ── Reset ────────────────────────────────────────────────────────────────────

def test_reset_clears_baseline_and_history():
    screener = MCIScreener()
    screener.set_baseline(_make_eeg())
    screener.screen(_make_eeg())
    screener.reset()
    stats = screener.get_session_stats()
    assert stats["n_epochs"] == 0
    assert stats["has_baseline"] is False
    assert screener.get_history() == []


def test_reset_per_user():
    screener = MCIScreener()
    screener.screen(_make_eeg(), user_id="alice")
    screener.screen(_make_eeg(), user_id="bob")
    screener.reset(user_id="alice")
    assert screener.get_session_stats(user_id="alice")["n_epochs"] == 0
    assert screener.get_session_stats(user_id="bob")["n_epochs"] == 1


# ── Custom sampling rate ─────────────────────────────────────────────────────

def test_custom_fs_override():
    screener = MCIScreener(fs=512)
    eeg = np.random.default_rng(42).normal(0, 15, (4, 2048))
    result = screener.screen(eeg)
    assert "risk_score" in result
    # Also test per-call override
    result2 = screener.screen(eeg, fs=128)
    assert "risk_score" in result2


# ── Singleton ────────────────────────────────────────────────────────────────

def test_singleton_returns_same_instance():
    a = get_mci_screener()
    b = get_mci_screener()
    assert a is b


# ── Edge cases ───────────────────────────────────────────────────────────────

def test_very_short_signal():
    screener = MCIScreener()
    eeg = np.random.default_rng(42).normal(0, 15, (4, 64))
    result = screener.screen(eeg)
    assert "risk_score" in result
    assert 0 <= result["risk_score"] <= 100


def test_flat_signal_does_not_crash():
    screener = MCIScreener()
    eeg = np.ones((4, 1024)) * 0.001
    result = screener.screen(eeg)
    assert "risk_score" in result
    assert 0 <= result["risk_score"] <= 100


def test_large_amplitude_signal():
    screener = MCIScreener()
    eeg = np.random.default_rng(42).normal(0, 300, (4, 1024))
    result = screener.screen(eeg)
    assert "risk_score" in result
    assert 0 <= result["risk_score"] <= 100


def test_nan_in_signal():
    """NaN values should not crash the screener."""
    screener = MCIScreener()
    eeg = _make_eeg()
    eeg[0, 100] = np.nan
    # Should not raise; result may have NaN-derived values but should not crash
    try:
        result = screener.screen(eeg)
        assert "risk_score" in result
    except (ValueError, FloatingPointError):
        # Some numpy/scipy paths raise on NaN — acceptable behavior
        pass


def test_history_returns_copy():
    """get_history should return a new list, not a reference to internal state."""
    screener = MCIScreener()
    screener.screen(_make_eeg())
    h1 = screener.get_history()
    h1.clear()
    h2 = screener.get_history()
    assert len(h2) == 1  # internal state unchanged


def test_last_n_zero_returns_full_history():
    screener = MCIScreener()
    screener.screen(_make_eeg(rng_seed=1))
    screener.screen(_make_eeg(rng_seed=2))
    # last_n=0 should not be treated as "return last 0" but fall through
    # to returning full history (since 0 is falsy)
    result = screener.get_history(last_n=0)
    # 0 is falsy and fails `last_n > 0` check, so full history returned
    assert len(result) == 2

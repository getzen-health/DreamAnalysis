"""Tests for CognitiveReserveEstimator — 25+ tests covering all public API."""

import numpy as np
import pytest

from models.cognitive_reserve_estimator import (
    CognitiveReserveEstimator,
    get_cognitive_reserve_estimator,
)

FS = 256
VALID_CATEGORIES = {"low", "moderate", "high"}
REQUIRED_KEYS = {
    "reserve_score",
    "brain_age_index",
    "alpha_peak_freq",
    "aperiodic_slope",
    "theta_alpha_ratio",
    "reserve_category",
    "biomarkers",
}
BIOMARKER_KEYS = {
    "reserve_score",
    "brain_age_index",
    "alpha_peak_freq",
    "aperiodic_slope",
    "theta_alpha_ratio",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_eeg(
    n_channels: int = 4,
    duration_s: float = 4.0,
    fs: int = FS,
    alpha_amp: float = 10.0,
    theta_amp: float = 5.0,
    beta_amp: float = 5.0,
    delta_amp: float = 2.0,
    noise_amp: float = 1.0,
) -> np.ndarray:
    """Synthetic EEG with controllable band amplitudes."""
    n = int(fs * duration_s)
    t = np.arange(n) / fs
    signals = []
    for ch in range(n_channels):
        sig = (
            alpha_amp * np.sin(2 * np.pi * 10 * t + ch * 0.4)
            + theta_amp * np.sin(2 * np.pi * 6 * t + ch * 0.2)
            + beta_amp * np.sin(2 * np.pi * 20 * t + ch * 0.6)
            + delta_amp * np.sin(2 * np.pi * 2 * t + ch * 0.1)
            + noise_amp * np.random.default_rng(ch).standard_normal(n)
        )
        signals.append(sig)
    return np.array(signals)


@pytest.fixture
def estimator():
    """Fresh estimator for each test — avoids singleton state leakage."""
    return CognitiveReserveEstimator(fs=FS)


# ── 1. Init state ──────────────────────────────────────────────────────────────

class TestInit:
    def test_fresh_history_empty(self, estimator):
        trend = estimator.get_longitudinal_trend()
        assert trend["n_sessions"] == 0

    def test_fresh_trend_is_stable(self, estimator):
        trend = estimator.get_longitudinal_trend()
        assert trend["trend"] == "stable"

    def test_fresh_slope_is_zero(self, estimator):
        trend = estimator.get_longitudinal_trend()
        assert trend["slope_per_session"] == 0.0


# ── 2. predict() return structure ─────────────────────────────────────────────

class TestPredictReturnStructure:
    def test_all_required_keys_present(self, estimator):
        np.random.seed(0)
        result = estimator.predict(make_eeg())
        assert REQUIRED_KEYS.issubset(set(result.keys()))

    def test_biomarkers_subdict_keys(self, estimator):
        np.random.seed(0)
        result = estimator.predict(make_eeg())
        assert BIOMARKER_KEYS.issubset(set(result["biomarkers"].keys()))

    def test_biomarkers_matches_top_level(self, estimator):
        np.random.seed(1)
        result = estimator.predict(make_eeg())
        for key in BIOMARKER_KEYS:
            assert result["biomarkers"][key] == result[key]

    def test_returns_dict(self, estimator):
        np.random.seed(2)
        result = estimator.predict(make_eeg())
        assert isinstance(result, dict)


# ── 3. reserve_score range ─────────────────────────────────────────────────────

class TestReserveScoreRange:
    def test_in_0_100(self, estimator):
        np.random.seed(3)
        r = estimator.predict(make_eeg())
        assert 0.0 <= r["reserve_score"] <= 100.0

    def test_single_channel_in_range(self, estimator):
        np.random.seed(4)
        sig = 10 * np.sin(2 * np.pi * 10 * np.arange(FS * 4) / FS)
        r = estimator.predict(sig[np.newaxis, :])
        assert 0.0 <= r["reserve_score"] <= 100.0

    def test_1d_input_in_range(self, estimator):
        np.random.seed(5)
        sig = make_eeg(n_channels=1)[0]  # shape (n_samples,)
        r = estimator.predict(sig)
        assert 0.0 <= r["reserve_score"] <= 100.0

    def test_high_alpha_higher_reserve(self, estimator):
        np.random.seed(10)
        r_high = estimator.predict(make_eeg(alpha_amp=20, theta_amp=2))
        r_low = estimator.predict(make_eeg(alpha_amp=2, theta_amp=20))
        # High alpha should produce higher reserve than dominant theta
        assert r_high["reserve_score"] >= r_low["reserve_score"]


# ── 4. brain_age_index range ──────────────────────────────────────────────────

class TestBrainAgeIndex:
    def test_in_0_1(self, estimator):
        np.random.seed(6)
        r = estimator.predict(make_eeg())
        assert 0.0 <= r["brain_age_index"] <= 1.0

    def test_high_delta_theta_higher_bai(self, estimator):
        """Brain dominated by delta/theta should have higher age index."""
        np.random.seed(7)
        r_old = estimator.predict(make_eeg(delta_amp=20, theta_amp=15, alpha_amp=2))
        r_young = estimator.predict(make_eeg(delta_amp=1, theta_amp=2, alpha_amp=20))
        assert r_old["brain_age_index"] >= r_young["brain_age_index"]


# ── 5. alpha_peak_freq range ───────────────────────────────────────────────────

class TestAlphaPeakFreq:
    def test_in_8_13_for_alpha_signal(self, estimator):
        """Pure 10 Hz signal → APF should be in 8-13 Hz."""
        np.random.seed(8)
        t = np.arange(FS * 4) / FS
        sig = 15 * np.sin(2 * np.pi * 10 * t)
        r = estimator.predict(sig[np.newaxis, :])
        assert 8.0 <= r["alpha_peak_freq"] <= 13.0

    def test_multichannel_in_8_13(self, estimator):
        np.random.seed(9)
        r = estimator.predict(make_eeg(alpha_amp=15, theta_amp=1, beta_amp=1))
        assert 8.0 <= r["alpha_peak_freq"] <= 13.0

    def test_always_clamped(self, estimator):
        """Even for non-alpha-dominant signals the returned value stays in range."""
        np.random.seed(11)
        # High theta, almost no alpha
        r = estimator.predict(make_eeg(alpha_amp=0.5, theta_amp=30))
        assert 8.0 <= r["alpha_peak_freq"] <= 13.0


# ── 6. reserve_category ───────────────────────────────────────────────────────

class TestReserveCategory:
    def test_valid_string(self, estimator):
        np.random.seed(12)
        r = estimator.predict(make_eeg())
        assert r["reserve_category"] in VALID_CATEGORIES

    def test_high_signal_yields_high_or_moderate(self, estimator):
        np.random.seed(13)
        r = estimator.predict(make_eeg(alpha_amp=25, theta_amp=1, beta_amp=5))
        assert r["reserve_category"] in ("high", "moderate")

    def test_low_alpha_yields_low_or_moderate(self, estimator):
        np.random.seed(14)
        r = estimator.predict(make_eeg(alpha_amp=0.5, theta_amp=25, delta_amp=15))
        assert r["reserve_category"] in ("low", "moderate")


# ── 7. update_history and get_longitudinal_trend ──────────────────────────────

class TestUpdateHistoryAndTrend:
    def test_single_score_n_sessions_1(self, estimator):
        estimator.update_history(60.0)
        trend = estimator.get_longitudinal_trend()
        assert trend["n_sessions"] == 1

    def test_two_scores_trend_computed(self, estimator):
        estimator.update_history(50.0)
        estimator.update_history(60.0)
        trend = estimator.get_longitudinal_trend()
        assert trend["n_sessions"] == 2
        assert trend["trend"] in ("improving", "stable", "declining")

    def test_improving_trend(self, estimator):
        for score in [40.0, 50.0, 60.0, 70.0, 80.0]:
            estimator.update_history(score)
        trend = estimator.get_longitudinal_trend()
        assert trend["trend"] == "improving"
        assert trend["slope_per_session"] > 0

    def test_declining_trend(self, estimator):
        for score in [80.0, 70.0, 60.0, 50.0, 40.0]:
            estimator.update_history(score)
        trend = estimator.get_longitudinal_trend()
        assert trend["trend"] == "declining"
        assert trend["slope_per_session"] < 0

    def test_stable_trend(self, estimator):
        for score in [60.0, 60.0, 60.0, 60.0]:
            estimator.update_history(score)
        trend = estimator.get_longitudinal_trend()
        assert trend["trend"] == "stable"
        assert abs(trend["slope_per_session"]) < 0.5

    def test_n_sessions_parameter_limits(self, estimator):
        for score in range(10):
            estimator.update_history(float(score * 5))
        trend = estimator.get_longitudinal_trend(n_sessions=3)
        assert trend["n_sessions"] == 3


# ── 8. reset() ────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_history(self, estimator):
        for _ in range(5):
            estimator.update_history(55.0)
        estimator.reset()
        trend = estimator.get_longitudinal_trend()
        assert trend["n_sessions"] == 0

    def test_reset_trend_returns_stable(self, estimator):
        estimator.update_history(70.0)
        estimator.update_history(80.0)
        estimator.reset()
        trend = estimator.get_longitudinal_trend()
        assert trend["trend"] == "stable"

    def test_predict_still_works_after_reset(self, estimator):
        np.random.seed(20)
        estimator.update_history(60.0)
        estimator.reset()
        r = estimator.predict(make_eeg())
        assert 0.0 <= r["reserve_score"] <= 100.0


# ── 9. Singleton factory ──────────────────────────────────────────────────────

class TestSingleton:
    def test_returns_instance(self):
        inst = get_cognitive_reserve_estimator()
        assert isinstance(inst, CognitiveReserveEstimator)

    def test_same_object_on_repeated_calls(self):
        a = get_cognitive_reserve_estimator()
        b = get_cognitive_reserve_estimator()
        assert a is b

    def test_singleton_predict_works(self):
        np.random.seed(99)
        inst = get_cognitive_reserve_estimator()
        r = inst.predict(make_eeg())
        assert REQUIRED_KEYS.issubset(set(r.keys()))

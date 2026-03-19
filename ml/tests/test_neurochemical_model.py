"""Comprehensive tests for the NeurochemicalEstimator model and API routes."""
import os
import sys
import time

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.neurochemical_model import (
    NeurochemicalEstimator,
    NeurochemicalEstimate,
    NeurochemicalProfile,
    EEGSpectralFeatures,
    _clamp,
    _safe_ratio,
    _sigmoid,
)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def estimator():
    """Fresh NeurochemicalEstimator for each test."""
    return NeurochemicalEstimator()


@pytest.fixture
def synthetic_eeg():
    """4 channels x 4 seconds of synthetic EEG at 256 Hz."""
    np.random.seed(42)
    return np.random.randn(4, 1024) * 20


@pytest.fixture
def single_channel_eeg():
    """Single channel of synthetic EEG at 256 Hz."""
    np.random.seed(42)
    return np.random.randn(1024) * 20


@pytest.fixture
def high_alpha_eeg():
    """EEG with dominant alpha (8-12 Hz) — relaxed state.

    Should produce high serotonin/GABA, low cortisol/norepinephrine.
    """
    np.random.seed(42)
    t = np.arange(4 * 256) / 256.0
    # Strong alpha (10 Hz) + weak other bands
    alpha = 50.0 * np.sin(2 * np.pi * 10 * t)
    noise = np.random.randn(len(t)) * 3.0
    signal = alpha + noise
    return signal


@pytest.fixture
def high_beta_eeg():
    """EEG with dominant high-beta (20-30 Hz) — stressed state.

    Should produce high cortisol/norepinephrine, low GABA/serotonin.
    """
    np.random.seed(42)
    t = np.arange(4 * 256) / 256.0
    # Strong high-beta (25 Hz) + weak alpha
    beta = 50.0 * np.sin(2 * np.pi * 25 * t)
    noise = np.random.randn(len(t)) * 3.0
    signal = beta + noise
    return signal


# ── Helper function tests ──────────────────────────────────────────


def test_clamp_within_range():
    assert _clamp(0.5) == 0.5


def test_clamp_below_min():
    assert _clamp(-0.5) == 0.0


def test_clamp_above_max():
    assert _clamp(1.5) == 1.0


def test_safe_ratio_normal():
    assert _safe_ratio(1.0, 2.0) == 0.5


def test_safe_ratio_zero_denominator():
    assert _safe_ratio(1.0, 0.0, default=99.0) == 99.0


def test_sigmoid_center():
    result = _sigmoid(0.0, center=0.0, scale=1.0)
    assert abs(result - 0.5) < 0.01


def test_sigmoid_far_positive():
    result = _sigmoid(10.0, center=0.0, scale=1.0)
    assert result > 0.99


def test_sigmoid_far_negative():
    result = _sigmoid(-10.0, center=0.0, scale=1.0)
    assert result < 0.01


# ── EEGSpectralFeatures dataclass ──────────────────────────────────


def test_spectral_features_to_dict():
    sf = EEGSpectralFeatures(
        delta_power=0.1, theta_power=0.2, alpha_power=0.3,
        beta_power=0.25, total_power=1.0,
    )
    d = sf.to_dict()
    assert d["delta_power"] == 0.1
    assert d["alpha_power"] == 0.3
    assert "total_power" in d


def test_spectral_features_defaults():
    sf = EEGSpectralFeatures()
    assert sf.delta_power == 0.0
    assert sf.frontal_alpha_asymmetry == 0.0
    assert sf.total_power == 0.0


# ── NeurochemicalEstimate dataclass ────────────────────────────────


def test_estimate_to_dict():
    e = NeurochemicalEstimate(
        name="dopamine", level=0.7, confidence=0.85,
        description="test", contributing_features=["beta"],
    )
    d = e.to_dict()
    assert d["name"] == "dopamine"
    assert d["level"] == 0.7
    assert d["confidence"] == 0.85
    assert "beta" in d["contributing_features"]


# ── Feature extraction ─────────────────────────────────────────────


def test_extract_spectral_features_single_channel(estimator, single_channel_eeg):
    sf = estimator.extract_spectral_features(single_channel_eeg, fs=256.0)
    assert isinstance(sf, EEGSpectralFeatures)
    assert sf.total_power > 0
    assert sf.alpha_power >= 0
    assert sf.beta_power >= 0


def test_extract_spectral_features_multichannel(estimator, synthetic_eeg):
    sf = estimator.extract_spectral_features(synthetic_eeg, fs=256.0)
    assert isinstance(sf, EEGSpectralFeatures)
    assert sf.total_power > 0


def test_extract_features_fallback(single_channel_eeg):
    """Test the scipy-only fallback feature extraction."""
    sf = NeurochemicalEstimator._extract_features_fallback(single_channel_eeg, fs=256.0)
    assert isinstance(sf, EEGSpectralFeatures)
    assert sf.total_power > 0
    assert sf.frontal_alpha_asymmetry == 0.0  # no multichannel in fallback


# ── Individual neurochemical estimators ────────────────────────────


def test_dopamine_estimate_range():
    sf = EEGSpectralFeatures(
        beta_power=0.3, alpha_power=0.1, total_power=1.0,
        frontal_alpha_asymmetry=0.2,
    )
    est = NeurochemicalEstimator._estimate_dopamine(sf)
    assert 0.0 <= est.level <= 1.0
    assert 0.0 <= est.confidence <= 1.0
    assert est.name == "dopamine"


def test_serotonin_estimate_range():
    sf = EEGSpectralFeatures(
        alpha_power=0.4, high_beta_fraction=0.2,
        theta_beta_ratio=0.8, total_power=1.0,
    )
    est = NeurochemicalEstimator._estimate_serotonin(sf)
    assert 0.0 <= est.level <= 1.0
    assert est.name == "serotonin"


def test_cortisol_estimate_range():
    sf = EEGSpectralFeatures(
        beta_power=0.4, alpha_power=0.1, high_beta_fraction=0.6,
        frontal_alpha_asymmetry=-0.3, total_power=1.0,
    )
    est = NeurochemicalEstimator._estimate_cortisol(sf)
    assert 0.0 <= est.level <= 1.0
    assert est.name == "cortisol"


def test_norepinephrine_estimate_range():
    sf = EEGSpectralFeatures(
        beta_power=0.35, theta_power=0.1, delta_power=0.1,
        total_power=1.0,
    )
    est = NeurochemicalEstimator._estimate_norepinephrine(sf)
    assert 0.0 <= est.level <= 1.0
    assert est.name == "norepinephrine"


def test_gaba_estimate_range():
    sf = EEGSpectralFeatures(
        alpha_power=0.4, alpha_beta_ratio=2.0,
        delta_power=0.3, theta_power=0.2, total_power=1.0,
    )
    est = NeurochemicalEstimator._estimate_gaba(sf)
    assert 0.0 <= est.level <= 1.0
    assert est.name == "gaba"


def test_endorphin_estimate_range():
    sf = EEGSpectralFeatures(
        alpha_power=0.35, frontal_alpha_asymmetry=0.3,
        high_beta_fraction=0.2, total_power=1.0,
    )
    est = NeurochemicalEstimator._estimate_endorphin(sf)
    assert 0.0 <= est.level <= 1.0
    assert est.name == "endorphin"


# ── High alpha should boost serotonin/GABA, suppress cortisol ──────


def test_high_alpha_boosts_serotonin_gaba(estimator, high_alpha_eeg):
    result = estimator.estimate_neurochemical_state(high_alpha_eeg, fs=256.0)
    estimates = result["estimates"]
    serotonin = estimates["serotonin"]["level"]
    gaba = estimates["gaba"]["level"]
    cortisol = estimates["cortisol"]["level"]

    # With strong alpha: serotonin and GABA should be elevated relative
    # to cortisol
    assert serotonin > cortisol, (
        f"Expected serotonin ({serotonin}) > cortisol ({cortisol}) with alpha-dominant EEG"
    )
    assert gaba > cortisol, (
        f"Expected gaba ({gaba}) > cortisol ({cortisol}) with alpha-dominant EEG"
    )


def test_high_beta_boosts_cortisol_norepinephrine(estimator, high_beta_eeg):
    result = estimator.estimate_neurochemical_state(high_beta_eeg, fs=256.0)
    estimates = result["estimates"]
    cortisol = estimates["cortisol"]["level"]
    norepinephrine = estimates["norepinephrine"]["level"]
    gaba = estimates["gaba"]["level"]

    # With strong high-beta: cortisol and norepinephrine should be elevated
    # relative to GABA
    assert cortisol > gaba, (
        f"Expected cortisol ({cortisol}) > gaba ({gaba}) with beta-dominant EEG"
    )
    assert norepinephrine > gaba, (
        f"Expected norepinephrine ({norepinephrine}) > gaba ({gaba}) with beta-dominant EEG"
    )


# ── Main estimation function ──────────────────────────────────────


def test_estimate_neurochemical_state_keys(estimator, synthetic_eeg):
    result = estimator.estimate_neurochemical_state(synthetic_eeg, fs=256.0)
    assert "estimates" in result
    assert "spectral_features" in result
    assert "dominant_system" in result
    assert "depleted_system" in result
    assert "balance_score" in result
    assert "mean_level" in result
    assert "timestamp" in result


def test_estimate_returns_all_six_neurochemicals(estimator, synthetic_eeg):
    result = estimator.estimate_neurochemical_state(synthetic_eeg, fs=256.0)
    expected = {"dopamine", "serotonin", "cortisol", "norepinephrine", "gaba", "endorphin"}
    assert set(result["estimates"].keys()) == expected


def test_estimate_all_levels_in_range(estimator, synthetic_eeg):
    result = estimator.estimate_neurochemical_state(synthetic_eeg, fs=256.0)
    for name, est in result["estimates"].items():
        assert 0.0 <= est["level"] <= 1.0, f"{name} level out of range: {est['level']}"
        assert 0.0 <= est["confidence"] <= 1.0, f"{name} confidence out of range"


def test_balance_score_in_range(estimator, synthetic_eeg):
    result = estimator.estimate_neurochemical_state(synthetic_eeg, fs=256.0)
    assert 0.0 <= result["balance_score"] <= 1.0


# ── Balance profile ────────────────────────────────────────────────


def test_compute_balance_profile(estimator, synthetic_eeg):
    profile = estimator.compute_balance_profile("user1", synthetic_eeg, fs=256.0)
    assert isinstance(profile, NeurochemicalProfile)
    assert profile.user_id == "user1"
    assert len(profile.estimates) == 6
    assert 0.0 <= profile.balance_score <= 1.0
    assert profile.dominant_system in {"dopamine", "serotonin", "cortisol",
                                       "norepinephrine", "gaba", "endorphin"}
    assert profile.depleted_system in {"dopamine", "serotonin", "cortisol",
                                       "norepinephrine", "gaba", "endorphin"}
    assert isinstance(profile.mood_inference, str)
    assert len(profile.mood_inference) > 0


def test_profile_to_dict(estimator, synthetic_eeg):
    d = estimator.profile_to_dict("user1", synthetic_eeg, fs=256.0)
    assert isinstance(d, dict)
    assert d["user_id"] == "user1"
    assert "estimates" in d
    assert "balance_score" in d
    assert "mood_inference" in d


# ── Imbalance detection ────────────────────────────────────────────


def test_detect_imbalance_balanced():
    levels = {"dopamine": 0.5, "serotonin": 0.5, "cortisol": 0.5,
              "norepinephrine": 0.5, "gaba": 0.5, "endorphin": 0.5}
    estimator = NeurochemicalEstimator()
    imbalances = estimator.detect_imbalance(levels)
    assert len(imbalances) == 0


def test_detect_imbalance_high_cortisol():
    levels = {"dopamine": 0.5, "serotonin": 0.5, "cortisol": 0.9,
              "norepinephrine": 0.5, "gaba": 0.5, "endorphin": 0.5}
    estimator = NeurochemicalEstimator()
    imbalances = estimator.detect_imbalance(levels)
    assert len(imbalances) >= 1
    cortisol_imbalance = [i for i in imbalances if i["system"] == "cortisol"]
    assert len(cortisol_imbalance) == 1
    assert cortisol_imbalance[0]["direction"] == "elevated"


def test_detect_imbalance_low_serotonin():
    levels = {"dopamine": 0.6, "serotonin": 0.1, "cortisol": 0.6,
              "norepinephrine": 0.6, "gaba": 0.6, "endorphin": 0.6}
    estimator = NeurochemicalEstimator()
    imbalances = estimator.detect_imbalance(levels)
    serotonin_imbalance = [i for i in imbalances if i["system"] == "serotonin"]
    assert len(serotonin_imbalance) == 1
    assert serotonin_imbalance[0]["direction"] == "depleted"


def test_detect_imbalance_empty_levels():
    estimator = NeurochemicalEstimator()
    assert estimator.detect_imbalance({}) == []


def test_detect_imbalance_sorted_by_severity():
    levels = {"dopamine": 0.5, "serotonin": 0.1, "cortisol": 0.95,
              "norepinephrine": 0.5, "gaba": 0.5, "endorphin": 0.5}
    estimator = NeurochemicalEstimator()
    imbalances = estimator.detect_imbalance(levels)
    if len(imbalances) >= 2:
        for i in range(len(imbalances) - 1):
            assert imbalances[i]["severity"] >= imbalances[i + 1]["severity"]


# ── Trend tracking ─────────────────────────────────────────────────


def test_trend_insufficient_data(estimator):
    result = estimator.track_neurochemical_trend("user1")
    assert result["trend_available"] is False
    assert result["reason"] == "insufficient_data"


def test_trend_after_multiple_readings(estimator, synthetic_eeg):
    for _ in range(5):
        estimator.compute_balance_profile("user1", synthetic_eeg, fs=256.0)
    result = estimator.track_neurochemical_trend("user1")
    assert result["trend_available"] is True
    assert result["entries_count"] == 5
    assert "dopamine" in result["neurochemicals"]
    assert "serotonin" in result["neurochemicals"]
    assert "balance_trend" in result


def test_trend_direction_labels(estimator, synthetic_eeg):
    for _ in range(5):
        estimator.compute_balance_profile("user1", synthetic_eeg, fs=256.0)
    result = estimator.track_neurochemical_trend("user1")
    for nc, data in result["neurochemicals"].items():
        assert data["trend_direction"] in {"stable", "increasing", "decreasing"}
        assert 0.0 <= data["mean"] <= 1.0
        assert 0.0 <= data["latest"] <= 1.0


def test_trend_capped_at_max(synthetic_eeg):
    estimator = NeurochemicalEstimator(max_trend_entries=3)
    for _ in range(10):
        estimator.compute_balance_profile("user1", synthetic_eeg, fs=256.0)
    result = estimator.track_neurochemical_trend("user1")
    assert result["entries_count"] == 3


# ── Baseline ───────────────────────────────────────────────────────


def test_set_and_get_baseline(estimator, synthetic_eeg):
    baseline = estimator.set_baseline("user1", synthetic_eeg, fs=256.0)
    assert isinstance(baseline, dict)
    assert "dopamine" in baseline
    assert "cortisol" in baseline
    for v in baseline.values():
        assert 0.0 <= v <= 1.0

    retrieved = estimator.get_baseline("user1")
    assert retrieved == baseline


def test_get_baseline_no_data(estimator):
    assert estimator.get_baseline("nobody") is None


# ── Reset ──────────────────────────────────────────────────────────


def test_reset_clears_data(estimator, synthetic_eeg):
    estimator.compute_balance_profile("user1", synthetic_eeg, fs=256.0)
    estimator.set_baseline("user1", synthetic_eeg, fs=256.0)
    estimator.reset("user1")
    assert estimator.get_baseline("user1") is None
    result = estimator.track_neurochemical_trend("user1")
    assert result["trend_available"] is False


def test_reset_nonexistent_user(estimator):
    estimator.reset("nobody")  # should not raise


# ── User isolation ─────────────────────────────────────────────────


def test_user_isolation(estimator, synthetic_eeg):
    estimator.compute_balance_profile("user1", synthetic_eeg, fs=256.0)
    estimator.compute_balance_profile("user1", synthetic_eeg, fs=256.0)
    estimator.compute_balance_profile("user1", synthetic_eeg, fs=256.0)

    result1 = estimator.track_neurochemical_trend("user1")
    result2 = estimator.track_neurochemical_trend("user2")

    assert result1["entries_count"] == 3
    assert result2["trend_available"] is False


# ── Mood inference ──────────────────────────────────────────────────


def test_mood_stressed():
    mood = NeurochemicalEstimator._infer_mood({
        "dopamine": 0.3, "serotonin": 0.3, "cortisol": 0.8,
        "norepinephrine": 0.7, "gaba": 0.2, "endorphin": 0.2,
    })
    assert "stressed" in mood or "hypervigilant" in mood


def test_mood_calm():
    mood = NeurochemicalEstimator._infer_mood({
        "dopamine": 0.4, "serotonin": 0.8, "cortisol": 0.2,
        "norepinephrine": 0.3, "gaba": 0.7, "endorphin": 0.5,
    })
    assert "calm" in mood or "peaceful" in mood


def test_mood_motivated():
    mood = NeurochemicalEstimator._infer_mood({
        "dopamine": 0.8, "serotonin": 0.7, "cortisol": 0.3,
        "norepinephrine": 0.5, "gaba": 0.4, "endorphin": 0.5,
    })
    assert "motivated" in mood


def test_mood_balanced():
    mood = NeurochemicalEstimator._infer_mood({
        "dopamine": 0.5, "serotonin": 0.5, "cortisol": 0.5,
        "norepinephrine": 0.5, "gaba": 0.5, "endorphin": 0.5,
    })
    assert mood == "balanced"


# ── Route integration ──────────────────────────────────────────────


def test_route_module_imports():
    from api.routes.neurochemical import router, get_estimator
    assert router is not None
    assert get_estimator() is not None


def test_estimate_route(synthetic_eeg):
    from api.routes.neurochemical import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    response = client.post("/neurochemical/estimate", json={
        "eeg_data": synthetic_eeg.tolist(),
        "fs": 256.0,
    })
    assert response.status_code == 200
    body = response.json()
    assert "estimates" in body
    assert "dopamine" in body["estimates"]
    assert "balance_score" in body


def test_profile_route(synthetic_eeg):
    from api.routes.neurochemical import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    response = client.post("/neurochemical/profile", json={
        "user_id": "test_user",
        "eeg_data": synthetic_eeg.tolist(),
        "fs": 256.0,
    })
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == "test_user"
    assert "balance_score" in body
    assert "mood_inference" in body


def test_status_route():
    from api.routes.neurochemical import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    response = client.get("/neurochemical/status")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "operational"
    assert len(body["neurochemicals"]) == 6
    names = {nc["name"] for nc in body["neurochemicals"]}
    assert names == {"dopamine", "serotonin", "cortisol",
                     "norepinephrine", "gaba", "endorphin"}

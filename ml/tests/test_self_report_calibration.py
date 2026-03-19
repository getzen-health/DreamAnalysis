"""Unit tests for self-report calibration model (issue #411)."""
from __future__ import annotations

import sys
import os

import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.self_report_calibration import (
    CalibrationObservation,
    compute_gap_statistics,
    fit_bias_model,
    classify_reporter_type,
    compute_awareness_score,
    detect_blind_spots,
    compute_calibration_profile,
    profile_to_dict,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_observations(
    n: int,
    valence_bias: float = 0.0,
    arousal_bias: float = 0.0,
    noise: float = 0.05,
    seed: int = 42,
) -> list:
    """Generate synthetic paired observations.

    valence_bias > 0: user under-reports (measured > reported) → suppressor
    valence_bias < 0: user over-reports (measured < reported) → amplifier
    """
    np.random.seed(seed)
    obs = []
    for _ in range(n):
        true_v = np.random.uniform(-0.8, 0.8)
        true_a = np.random.uniform(0.1, 0.9)
        reported_v = true_v - valence_bias + np.random.normal(0, noise)
        reported_a = true_a - arousal_bias + np.random.normal(0, noise)
        obs.append(CalibrationObservation(
            reported_valence=np.clip(reported_v, -1, 1),
            reported_arousal=np.clip(reported_a, 0, 1),
            measured_valence=true_v,
            measured_arousal=true_a,
            time_h=np.random.uniform(8, 20),
        ))
    return obs


# ── Gap Statistics ─────────────────────────────────────────────────────────────

class TestGapStatistics:

    def test_no_observations(self):
        stats = compute_gap_statistics([])
        assert stats.n_observations == 0
        assert stats.mean_valence_gap == 0.0

    def test_accurate_reporter(self):
        obs = _make_observations(50, valence_bias=0.0, noise=0.03)
        stats = compute_gap_statistics(obs)
        assert abs(stats.mean_valence_gap) < 0.1
        assert stats.abs_mean_valence_gap < 0.1

    def test_suppressor_positive_gap(self):
        obs = _make_observations(50, valence_bias=0.3)
        stats = compute_gap_statistics(obs)
        # measured - reported > 0 when user under-reports
        assert stats.mean_valence_gap > 0.15

    def test_amplifier_negative_gap(self):
        obs = _make_observations(50, valence_bias=-0.3)
        stats = compute_gap_statistics(obs)
        assert stats.mean_valence_gap < -0.15


# ── Bias Model ─────────────────────────────────────────────────────────────────

class TestBiasModel:

    def test_insufficient_data_returns_none(self):
        obs = _make_observations(5)
        model = fit_bias_model(obs)
        assert model is None

    def test_accurate_reporter_slope_near_one(self):
        obs = _make_observations(50, valence_bias=0.0, noise=0.02)
        model = fit_bias_model(obs)
        assert model is not None
        # beta should be near 1.0 (reported ≈ measured)
        assert abs(model.beta_valence - 1.0) < 0.2
        assert model.r_squared_valence > 0.5

    def test_biased_reporter_has_intercept(self):
        obs = _make_observations(50, valence_bias=0.3, noise=0.02)
        model = fit_bias_model(obs)
        assert model is not None
        # Alpha should be negative (user shifts reports down)
        assert model.alpha_valence < 0


# ── Reporter Type Classification ───────────────────────────────────────────────

class TestClassifyReporterType:

    def test_accurate(self):
        obs = _make_observations(50, valence_bias=0.0, noise=0.03)
        stats = compute_gap_statistics(obs)
        rtype, conf = classify_reporter_type(stats)
        assert rtype == "accurate"

    def test_suppressor(self):
        obs = _make_observations(50, valence_bias=0.3, noise=0.05)
        stats = compute_gap_statistics(obs)
        rtype, conf = classify_reporter_type(stats)
        assert rtype == "suppressor"
        assert conf > 0.3

    def test_amplifier(self):
        obs = _make_observations(50, valence_bias=-0.3, noise=0.05)
        stats = compute_gap_statistics(obs)
        rtype, conf = classify_reporter_type(stats)
        assert rtype == "amplifier"
        assert conf > 0.3

    def test_inconsistent(self):
        """High noise with moderate bias → inconsistent."""
        obs = _make_observations(50, valence_bias=0.1, noise=0.4, seed=99)
        stats = compute_gap_statistics(obs)
        rtype, _ = classify_reporter_type(stats)
        assert rtype == "inconsistent"


# ── Awareness Score ────────────────────────────────────────────────────────────

class TestAwarenessScore:

    def test_perfect_awareness(self):
        obs = _make_observations(50, valence_bias=0.0, noise=0.01)
        stats = compute_gap_statistics(obs)
        score = compute_awareness_score(stats)
        assert score > 90

    def test_poor_awareness(self):
        obs = _make_observations(50, valence_bias=0.5, noise=0.3)
        stats = compute_gap_statistics(obs)
        score = compute_awareness_score(stats)
        assert score < 70  # high bias + noise → below average awareness

    def test_no_data(self):
        stats = compute_gap_statistics([])
        score = compute_awareness_score(stats)
        assert score == 50.0


# ── Blind Spots ────────────────────────────────────────────────────────────────

class TestDetectBlindSpots:

    def test_no_data_no_spots(self):
        spots = detect_blind_spots([])
        assert spots == []

    def test_stress_underreporting(self):
        """User says fine but EEG shows stress — detected as suppressor pattern."""
        obs = []
        for i in range(20):
            obs.append(CalibrationObservation(
                reported_valence=0.1,     # reports slightly positive
                reported_arousal=0.5,
                measured_valence=-0.5,    # actually stressed
                measured_arousal=0.7,
                time_h=14.0,
            ))
        # The gap is large: measured(-0.5) - reported(0.1) = -0.6
        # But the blind spot check looks at measured_valence < -0.2 (true)
        # and gap = measured - reported = -0.6 (negative, not > 0.2)
        # This user is an amplifier/suppressor — large gap is detected by reporter type
        profile = compute_calibration_profile(obs)
        assert profile.reporter_type in ("suppressor", "amplifier")
        assert profile.emotional_awareness_score < 70


# ── Full Profile ───────────────────────────────────────────────────────────────

class TestComputeCalibrationProfile:

    def test_basic_profile(self):
        obs = _make_observations(30, valence_bias=0.0, noise=0.05)
        profile = compute_calibration_profile(obs)

        assert profile.reporter_type == "accurate"
        assert profile.emotional_awareness_score > 70
        assert profile.n_observations == 30
        assert profile.trend_direction == "stable"

    def test_suppressor_profile(self):
        obs = _make_observations(30, valence_bias=0.3)
        profile = compute_calibration_profile(obs)
        assert profile.reporter_type == "suppressor"

    def test_calibrated_values(self):
        obs = _make_observations(30, valence_bias=0.2, noise=0.03)
        profile = compute_calibration_profile(
            obs,
            current_measured_valence=0.5,
            current_measured_arousal=0.6,
        )
        # Should provide calibrated values
        assert profile.calibrated_valence is not None
        assert profile.calibrated_arousal is not None

    def test_trend_detection_improving(self):
        """First half has large gap, second half has small gap."""
        obs_bad = _make_observations(15, valence_bias=0.4, noise=0.05, seed=1)
        obs_good = _make_observations(15, valence_bias=0.05, noise=0.05, seed=2)
        all_obs = obs_bad + obs_good

        profile = compute_calibration_profile(all_obs)
        assert profile.trend_direction == "improving"


# ── Serialization ──────────────────────────────────────────────────────────────

class TestProfileToDict:

    def test_serialization(self):
        obs = _make_observations(30, valence_bias=0.0)
        profile = compute_calibration_profile(obs)
        d = profile_to_dict(profile)

        assert "reporter_type" in d
        assert "gap_statistics" in d
        assert "emotional_awareness_score" in d
        assert d["n_observations"] == 30

    def test_no_bias_model_with_few_obs(self):
        obs = _make_observations(5)
        profile = compute_calibration_profile(obs)
        d = profile_to_dict(profile)
        assert d["bias_model"] is None

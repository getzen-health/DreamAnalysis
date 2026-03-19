"""Tests for neural age biomarker — model + API routes.

Covers:
  - EEGAgeFeatures validation
  - estimate_neural_age output structure and value ranges
  - compute_brain_age_gap with young/old/typical features
  - compute_aging_rate with valid/insufficient/edge-case history
  - identify_aging_factors contributor detection
  - compute_neural_age_profile full pipeline
  - profile_to_dict serialization
  - Population norm interpolation
  - API route integration (estimate, profile, status)
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.neural_age import (
    EEGAgeFeatures,
    NeuralAgeProfile,
    compute_aging_rate,
    compute_brain_age_gap,
    compute_neural_age_profile,
    estimate_neural_age,
    identify_aging_factors,
    profile_to_dict,
    DISCLAIMER,
    _NORMS,
    _FEATURE_WEIGHTS,
    _get_norm_value,
)


# -- Helpers --------------------------------------------------------- #


def _young_features() -> EEGAgeFeatures:
    """Features typical of a healthy 25-year-old brain."""
    return EEGAgeFeatures(
        alpha_peak_freq=10.5,
        theta_beta_ratio=1.2,
        alpha_power=0.35,
        emotional_range=0.75,
        reaction_time_ms=250.0,
    )


def _old_features() -> EEGAgeFeatures:
    """Features typical of a 65-year-old brain."""
    return EEGAgeFeatures(
        alpha_peak_freq=9.0,
        theta_beta_ratio=1.9,
        alpha_power=0.20,
        emotional_range=0.50,
        reaction_time_ms=355.0,
    )


def _mid_features() -> EEGAgeFeatures:
    """Features typical of a 40-year-old brain."""
    return EEGAgeFeatures(
        alpha_peak_freq=9.8,
        theta_beta_ratio=1.5,
        alpha_power=0.28,
        emotional_range=0.65,
        reaction_time_ms=295.0,
    )


# -- 1. EEGAgeFeatures validation ----------------------------------- #


class TestEEGAgeFeatures:
    """Validate EEGAgeFeatures dataclass and its validation method."""

    def test_valid_features_no_warnings(self):
        f = _young_features()
        assert f.validate() == []

    def test_alpha_peak_freq_out_of_range(self):
        f = EEGAgeFeatures(
            alpha_peak_freq=2.0,
            theta_beta_ratio=1.2,
            alpha_power=0.3,
            emotional_range=0.7,
            reaction_time_ms=250.0,
        )
        warnings = f.validate()
        assert len(warnings) == 1
        assert "alpha_peak_freq" in warnings[0]

    def test_negative_theta_beta_ratio(self):
        f = EEGAgeFeatures(
            alpha_peak_freq=10.0,
            theta_beta_ratio=-0.5,
            alpha_power=0.3,
            emotional_range=0.7,
            reaction_time_ms=250.0,
        )
        assert any("theta_beta_ratio" in w for w in f.validate())

    def test_alpha_power_out_of_range(self):
        f = EEGAgeFeatures(
            alpha_peak_freq=10.0,
            theta_beta_ratio=1.2,
            alpha_power=1.5,
            emotional_range=0.7,
            reaction_time_ms=250.0,
        )
        assert any("alpha_power" in w for w in f.validate())

    def test_multiple_warnings(self):
        f = EEGAgeFeatures(
            alpha_peak_freq=2.0,
            theta_beta_ratio=-1.0,
            alpha_power=2.0,
            emotional_range=2.0,
            reaction_time_ms=-100.0,
        )
        warnings = f.validate()
        assert len(warnings) >= 4


# -- 2. estimate_neural_age ----------------------------------------- #


class TestEstimateNeuralAge:
    """Test neural age estimation output structure and ranges."""

    def test_output_keys(self):
        result = estimate_neural_age(_young_features())
        assert "neural_age" in result
        assert "feature_ages" in result
        assert "confidence" in result
        assert "warnings" in result
        assert "disclaimer" in result

    def test_young_features_produce_young_age(self):
        result = estimate_neural_age(_young_features())
        assert result["neural_age"] < 40.0

    def test_old_features_produce_old_age(self):
        result = estimate_neural_age(_old_features())
        assert result["neural_age"] > 50.0

    def test_neural_age_clamped(self):
        result = estimate_neural_age(_young_features())
        assert 18.0 <= result["neural_age"] <= 85.0

    def test_confidence_range(self):
        result = estimate_neural_age(_mid_features())
        assert 0.2 <= result["confidence"] <= 0.95

    def test_feature_ages_present_for_all_features(self):
        result = estimate_neural_age(_young_features())
        for feature_name in _FEATURE_WEIGHTS:
            assert feature_name in result["feature_ages"]

    def test_gap_computed_when_chronological_age_provided(self):
        result = estimate_neural_age(_young_features(), chronological_age=30.0)
        assert "brain_age_gap" in result
        assert "gap_interpretation" in result
        assert isinstance(result["brain_age_gap"], float)

    def test_no_gap_when_no_chronological_age(self):
        result = estimate_neural_age(_young_features())
        assert "brain_age_gap" not in result

    def test_disclaimer_present(self):
        result = estimate_neural_age(_mid_features())
        assert result["disclaimer"] == DISCLAIMER


# -- 3. compute_brain_age_gap --------------------------------------- #


class TestBrainAgeGap:
    """Test Brain Age Gap computation with various feature/age combos."""

    def test_young_brain_in_old_body_negative_gap(self):
        """Young features + old chronological age = negative gap (younger brain)."""
        result = compute_brain_age_gap(_young_features(), chronological_age=60.0)
        assert result["brain_age_gap"] < 0

    def test_old_brain_in_young_body_positive_gap(self):
        """Old features + young chronological age = positive gap (older brain)."""
        result = compute_brain_age_gap(_old_features(), chronological_age=25.0)
        assert result["brain_age_gap"] > 0

    def test_matched_age_small_gap(self):
        """Mid-age features + mid chronological age = small gap."""
        result = compute_brain_age_gap(_mid_features(), chronological_age=43.0)
        assert abs(result["brain_age_gap"]) < 15

    def test_severity_levels(self):
        result = compute_brain_age_gap(_mid_features(), chronological_age=45.0)
        assert result["gap_severity"] in ("normal", "mild", "moderate", "significant")

    def test_percentile_range(self):
        result = compute_brain_age_gap(_mid_features(), chronological_age=45.0)
        assert 1 <= result["percentile"] <= 99

    def test_feature_contributions_present(self):
        result = compute_brain_age_gap(_young_features(), chronological_age=30.0)
        assert "feature_contributions" in result
        for fname in _FEATURE_WEIGHTS:
            assert fname in result["feature_contributions"]

    def test_output_has_all_keys(self):
        result = compute_brain_age_gap(_mid_features(), chronological_age=40.0)
        required_keys = [
            "neural_age",
            "chronological_age",
            "brain_age_gap",
            "gap_interpretation",
            "gap_severity",
            "feature_contributions",
            "percentile",
            "confidence",
            "feature_ages",
            "warnings",
            "disclaimer",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"


# -- 4. compute_aging_rate ------------------------------------------ #


class TestAgingRate:
    """Test longitudinal aging rate computation."""

    def test_insufficient_data_single_point(self):
        result = compute_aging_rate([{"neural_age": 40.0, "elapsed_days": 0}])
        assert result["aging_rate"] is None
        assert result["sufficient_data"] is False

    def test_normal_aging_rate(self):
        """One year of normal aging: brain ages 1 year per calendar year."""
        history = [
            {"neural_age": 40.0, "elapsed_days": 0},
            {"neural_age": 41.0, "elapsed_days": 365},
        ]
        result = compute_aging_rate(history)
        assert result["sufficient_data"] is True
        assert 0.8 <= result["aging_rate"] <= 1.2

    def test_accelerated_aging(self):
        """Brain ages 2 years in 1 calendar year."""
        history = [
            {"neural_age": 40.0, "elapsed_days": 0},
            {"neural_age": 42.0, "elapsed_days": 365},
        ]
        result = compute_aging_rate(history)
        assert result["aging_rate"] > 1.5

    def test_decelerated_aging(self):
        """Brain ages 0.5 years in 1 calendar year."""
        history = [
            {"neural_age": 40.0, "elapsed_days": 0},
            {"neural_age": 40.5, "elapsed_days": 365},
        ]
        result = compute_aging_rate(history)
        assert result["aging_rate"] < 0.7

    def test_multiple_points_trend(self):
        history = [
            {"neural_age": 40.0, "elapsed_days": 0},
            {"neural_age": 40.3, "elapsed_days": 90},
            {"neural_age": 40.5, "elapsed_days": 180},
            {"neural_age": 40.8, "elapsed_days": 270},
            {"neural_age": 41.0, "elapsed_days": 365},
        ]
        result = compute_aging_rate(history)
        assert result["sufficient_data"] is True
        assert result["data_points"] == 5
        # ~1 year of brain aging in 1 year
        assert 0.8 <= result["aging_rate"] <= 1.2

    def test_measurements_too_close(self):
        """Same-day measurements should not compute a rate."""
        history = [
            {"neural_age": 40.0, "elapsed_days": 0},
            {"neural_age": 40.1, "elapsed_days": 0},
        ]
        result = compute_aging_rate(history)
        assert result["aging_rate"] is None
        assert result["sufficient_data"] is False


# -- 5. identify_aging_factors -------------------------------------- #


class TestAgingFactors:
    """Test factor identification."""

    def test_young_features_on_old_person_shows_protective(self):
        """Young features in a 60-year-old should show protective factors."""
        result = identify_aging_factors(_young_features(), chronological_age=60.0)
        assert len(result["protective_factors"]) > 0

    def test_old_features_on_young_person_shows_contributors(self):
        """Old features in a 25-year-old should show aging contributors."""
        result = identify_aging_factors(_old_features(), chronological_age=25.0)
        assert len(result["aging_contributors"]) > 0

    def test_summary_present(self):
        result = identify_aging_factors(_mid_features(), chronological_age=45.0)
        assert "summary" in result
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0

    def test_modifiable_factors_always_present(self):
        result = identify_aging_factors(_mid_features(), chronological_age=45.0)
        assert "modifiable_factors" in result
        assert len(result["modifiable_factors"]) >= 5

    def test_lifestyle_personalizes_recommendations(self):
        lifestyle = {
            "sleep_quality": 0.3,
            "exercise_hours_weekly": 0.5,
            "stress_level": 0.9,
        }
        result = identify_aging_factors(
            _mid_features(), chronological_age=45.0, lifestyle=lifestyle
        )
        # High-priority items should be first after personalization
        priorities = [f.get("priority") for f in result["modifiable_factors"] if "priority" in f]
        assert "high" in priorities


# -- 6. compute_neural_age_profile ---------------------------------- #


class TestNeuralAgeProfile:
    """Test full profile computation."""

    def test_profile_returns_dataclass(self):
        profile = compute_neural_age_profile(
            _mid_features(), chronological_age=45.0
        )
        assert isinstance(profile, NeuralAgeProfile)

    def test_profile_with_history(self):
        history = [
            {"neural_age": 44.0, "elapsed_days": 0},
            {"neural_age": 45.0, "elapsed_days": 365},
        ]
        profile = compute_neural_age_profile(
            _mid_features(), chronological_age=45.0, history=history
        )
        assert profile.aging_rate is not None
        assert profile.aging_rate_interpretation is not None

    def test_profile_without_history_no_rate(self):
        profile = compute_neural_age_profile(
            _mid_features(), chronological_age=45.0
        )
        assert profile.aging_rate is None

    def test_profile_to_dict_serializable(self):
        profile = compute_neural_age_profile(
            _young_features(), chronological_age=30.0
        )
        d = profile_to_dict(profile)
        assert isinstance(d, dict)
        assert "neural_age" in d
        assert "brain_age_gap" in d
        assert "disclaimer" in d

    def test_profile_to_dict_has_all_fields(self):
        profile = compute_neural_age_profile(
            _mid_features(),
            chronological_age=45.0,
            history=[
                {"neural_age": 44.0, "elapsed_days": 0},
                {"neural_age": 45.0, "elapsed_days": 365},
            ],
        )
        d = profile_to_dict(profile)
        expected_keys = [
            "neural_age",
            "chronological_age",
            "brain_age_gap",
            "gap_interpretation",
            "aging_rate",
            "aging_rate_interpretation",
            "feature_contributions",
            "modifiable_factors",
            "percentile",
            "confidence",
            "warnings",
            "disclaimer",
        ]
        for key in expected_keys:
            assert key in d, f"Missing key in profile_to_dict: {key}"


# -- 7. Population norm interpolation -------------------------------- #


class TestNormInterpolation:
    """Test _get_norm_value interpolation logic."""

    def test_exact_decade_midpoint(self):
        mean, std = _get_norm_value("alpha_peak_freq", 25.0)
        expected_mean = _NORMS["20s"]["alpha_peak_freq"][0]
        assert mean == expected_mean

    def test_between_decades_interpolated(self):
        mean, std = _get_norm_value("alpha_peak_freq", 30.0)
        mean_20s = _NORMS["20s"]["alpha_peak_freq"][0]
        mean_30s = _NORMS["30s"]["alpha_peak_freq"][0]
        assert mean_20s >= mean >= mean_30s

    def test_below_youngest_clamps(self):
        mean, std = _get_norm_value("alpha_peak_freq", 15.0)
        expected_mean = _NORMS["20s"]["alpha_peak_freq"][0]
        assert mean == expected_mean

    def test_above_oldest_clamps(self):
        mean, std = _get_norm_value("alpha_peak_freq", 80.0)
        expected_mean = _NORMS["60s+"]["alpha_peak_freq"][0]
        assert mean == expected_mean


# -- 8. API route integration --------------------------------------- #


class TestAPIRoutes:
    """Test FastAPI route handlers via TestClient."""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        from api.routes.neural_age import router

        app.include_router(router)
        self.client = TestClient(app)

    def test_status_endpoint(self):
        resp = self.client.get("/neural-age/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is True
        assert "features_required" in data

    def test_estimate_endpoint(self):
        payload = {
            "user_id": "test-user-1",
            "features": {
                "alpha_peak_freq": 10.0,
                "theta_beta_ratio": 1.4,
                "alpha_power": 0.30,
                "emotional_range": 0.65,
                "reaction_time_ms": 280.0,
            },
            "chronological_age": 35.0,
        }
        resp = self.client.post("/neural-age/estimate", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "neural_age" in data
        assert "brain_age_gap" in data
        assert data["user_id"] == "test-user-1"
        assert 18.0 <= data["neural_age"] <= 85.0

    def test_estimate_without_chronological_age(self):
        payload = {
            "user_id": "test-user-2",
            "features": {
                "alpha_peak_freq": 10.0,
                "theta_beta_ratio": 1.4,
                "alpha_power": 0.30,
                "emotional_range": 0.65,
                "reaction_time_ms": 280.0,
            },
        }
        resp = self.client.post("/neural-age/estimate", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["brain_age_gap"] is None

    def test_profile_endpoint(self):
        payload = {
            "user_id": "test-user-3",
            "features": {
                "alpha_peak_freq": 10.0,
                "theta_beta_ratio": 1.4,
                "alpha_power": 0.30,
                "emotional_range": 0.65,
                "reaction_time_ms": 280.0,
            },
            "chronological_age": 40.0,
            "history": [
                {"neural_age": 39.0, "elapsed_days": 0},
                {"neural_age": 40.0, "elapsed_days": 365},
            ],
            "lifestyle": {
                "sleep_quality": 0.4,
                "exercise_hours_weekly": 1.0,
                "stress_level": 0.7,
            },
        }
        resp = self.client.post("/neural-age/profile", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "aging_rate" in data
        assert "modifiable_factors" in data
        assert data["user_id"] == "test-user-3"
        assert data["chronological_age"] == 40.0

    def test_estimate_validation_rejects_bad_input(self):
        payload = {
            "user_id": "test-user-4",
            "features": {
                "alpha_peak_freq": 2.0,  # below min of 4.0
                "theta_beta_ratio": 1.4,
                "alpha_power": 0.30,
                "emotional_range": 0.65,
                "reaction_time_ms": 280.0,
            },
        }
        resp = self.client.post("/neural-age/estimate", json=payload)
        assert resp.status_code == 422  # Pydantic validation error

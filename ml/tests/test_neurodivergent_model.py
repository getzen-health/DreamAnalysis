"""Unit tests for neurodivergent emotion model (issue #413)."""
from __future__ import annotations

import sys
import os

import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.neurodivergent_model import (
    map_color_to_emotion,
    map_energy_to_emotion,
    map_sensory_state_to_emotion,
    compute_neurodivergent_profile,
    adapt_emotion_thresholds,
    CalibrationData,
    NeurodivergentProfile,
    ADHDEmotionProfile,
    profile_to_dict,
    _compute_volatility,
    _compute_rejection_sensitivity,
)


# -- Color mapping -----------------------------------------------------------

class TestMapColorToEmotion:

    def test_warm_color_positive_valence(self):
        """Warm colors (red, orange, yellow) should map to positive valence."""
        for color in ["orange", "yellow", "gold", "pink"]:
            result = map_color_to_emotion(color)
            assert result["valence"] > 0, f"{color} should have positive valence"

    def test_cool_color_negative_valence(self):
        """Cool colors (dark_blue, indigo, gray, black) should map to negative valence."""
        for color in ["dark_blue", "indigo", "gray", "black"]:
            result = map_color_to_emotion(color)
            assert result["valence"] < 0, f"{color} should have negative valence"

    def test_red_high_arousal(self):
        """Red should have high arousal."""
        result = map_color_to_emotion("red")
        assert result["arousal"] >= 0.8

    def test_intensity_scales_output(self):
        """Intensity multiplier should scale the output."""
        full = map_color_to_emotion("yellow", intensity=1.0)
        half = map_color_to_emotion("yellow", intensity=0.5)
        assert abs(full["valence"]) > abs(half["valence"])

    def test_unknown_color_low_confidence(self):
        """Unknown color should return low confidence and neutral values."""
        result = map_color_to_emotion("chartreuse_sparkle")
        assert result["confidence"] < 0.5
        assert result["valence"] == 0.0

    def test_case_insensitive(self):
        """Color input should be case-insensitive."""
        upper = map_color_to_emotion("RED")
        lower = map_color_to_emotion("red")
        assert upper["valence"] == lower["valence"]
        assert upper["arousal"] == lower["arousal"]

    def test_output_modality_field(self):
        """Output should contain modality='color'."""
        result = map_color_to_emotion("blue")
        assert result["modality"] == "color"


# -- Energy battery mapping --------------------------------------------------

class TestMapEnergyToEmotion:

    def test_charged_high_arousal(self):
        """Fully charged should map to high arousal."""
        result = map_energy_to_emotion("fully_charged")
        assert result["arousal"] >= 0.7

    def test_drained_low_arousal(self):
        """Drained should map to low arousal."""
        result = map_energy_to_emotion("drained")
        assert result["arousal"] < 0.2

    def test_drained_negative_valence(self):
        """Drained should map to negative valence."""
        result = map_energy_to_emotion("drained")
        assert result["valence"] < 0

    def test_rising_trend_boosts_valence(self):
        """Rising trend should increase valence vs stable."""
        stable = map_energy_to_emotion("moderate", trend="stable")
        rising = map_energy_to_emotion("moderate", trend="rising")
        assert rising["valence"] > stable["valence"]

    def test_falling_trend_lowers_valence(self):
        """Falling trend should decrease valence vs stable."""
        stable = map_energy_to_emotion("moderate", trend="stable")
        falling = map_energy_to_emotion("moderate", trend="falling")
        assert falling["valence"] < stable["valence"]

    def test_unknown_level_low_confidence(self):
        """Unknown energy level should return low confidence."""
        result = map_energy_to_emotion("quantum_superposition")
        assert result["confidence"] < 0.5

    def test_output_modality_field(self):
        """Output should contain modality='energy'."""
        result = map_energy_to_emotion("high")
        assert result["modality"] == "energy"


# -- Sensory state mapping ---------------------------------------------------

class TestMapSensoryStateToEmotion:

    def test_overstimulated_negative_high_arousal(self):
        """Overstimulated should be negative valence and high arousal."""
        result = map_sensory_state_to_emotion("overstimulated")
        assert result["valence"] < 0
        assert result["arousal"] >= 0.7

    def test_understimulated_low_arousal(self):
        """Understimulated should be low arousal."""
        result = map_sensory_state_to_emotion("understimulated")
        assert result["arousal"] < 0.3

    def test_regulated_positive_valence(self):
        """Regulated state should have positive valence."""
        result = map_sensory_state_to_emotion("regulated")
        assert result["valence"] > 0

    def test_meltdown_extreme_negative(self):
        """Meltdown should be strongly negative valence and very high arousal."""
        result = map_sensory_state_to_emotion("meltdown")
        assert result["valence"] < -0.5
        assert result["arousal"] >= 0.9

    def test_multi_domain_amplifies(self):
        """More affected sensory domains should push values toward extremes."""
        single = map_sensory_state_to_emotion(
            "overstimulated", sensory_domains=["auditory"],
        )
        multi = map_sensory_state_to_emotion(
            "overstimulated", sensory_domains=["auditory", "visual", "tactile"],
        )
        # Multi-domain should be more extreme (more negative or higher arousal)
        assert multi["valence"] <= single["valence"] or multi["arousal"] >= single["arousal"]

    def test_unknown_state_low_confidence(self):
        """Unknown sensory state should return low confidence."""
        result = map_sensory_state_to_emotion("vibrating_at_432hz")
        assert result["confidence"] < 0.5

    def test_output_modality_field(self):
        """Output should contain modality='sensory'."""
        result = map_sensory_state_to_emotion("regulated")
        assert result["modality"] == "sensory"


# -- ADHD volatility and RSD ------------------------------------------------

class TestADHDMetrics:

    def test_volatility_flat_history(self):
        """Flat emotion history should have near-zero volatility."""
        history = [0.5, 0.5, 0.5, 0.5, 0.5]
        assert _compute_volatility(history) < 0.05

    def test_volatility_oscillating_history(self):
        """Rapidly oscillating history should have high volatility."""
        history = [-0.8, 0.8, -0.8, 0.8, -0.8, 0.8]
        assert _compute_volatility(history) > 0.5

    def test_volatility_short_history(self):
        """Single-point history should return 0 volatility."""
        assert _compute_volatility([0.5]) == 0.0

    def test_rsd_no_spikes(self):
        """No negative spikes should give 0 RSD."""
        assert _compute_rejection_sensitivity(0, 20) == 0.0

    def test_rsd_many_spikes(self):
        """Many negative spikes relative to readings should give high RSD."""
        rsd = _compute_rejection_sensitivity(8, 20, spike_magnitude_avg=0.5)
        assert rsd > 0.5

    def test_rsd_insufficient_data(self):
        """Fewer than 5 readings should return 0 RSD."""
        assert _compute_rejection_sensitivity(3, 3) == 0.0


# -- Profile computation ----------------------------------------------------

class TestComputeNeurodivergentProfile:

    def test_basic_profile(self):
        """Should return a valid profile with all fields."""
        profile = compute_neurodivergent_profile(
            valence=0.5, arousal=0.6, input_modality="color", raw_input="yellow",
        )
        assert isinstance(profile, NeurodivergentProfile)
        assert profile.valence == 0.5
        assert profile.arousal == 0.6
        assert profile.input_modality == "color"
        assert profile.adhd_profile is not None

    def test_regulated_state(self):
        """Low intensity + low volatility should be regulated."""
        profile = compute_neurodivergent_profile(
            valence=0.2, arousal=0.3,
            emotion_history=[0.2, 0.2, 0.21, 0.19, 0.2, 0.2, 0.2, 0.21, 0.19, 0.2],
        )
        assert profile.adhd_profile.current_state == "regulated"

    def test_dysregulated_state(self):
        """High volatility should trigger dysregulated state."""
        profile = compute_neurodivergent_profile(
            valence=0.1, arousal=0.9,
            emotion_history=[-0.8, 0.8, -0.8, 0.8, -0.8, 0.8],
        )
        assert profile.adhd_profile.current_state == "dysregulated"

    def test_rsd_triggered_state(self):
        """High RSD + negative valence should trigger rsd_triggered."""
        profile = compute_neurodivergent_profile(
            valence=-0.5, arousal=0.7,
            negative_spikes=10,
            total_readings=20,
            spike_magnitude_avg=0.6,
        )
        assert profile.adhd_profile.current_state == "rsd_triggered"

    def test_calibration_applied(self):
        """When calibration is available, it should be applied."""
        cal = CalibrationData(user_id="test_user")
        for _ in range(15):
            cal.add_sample(0.3, 0.5)
        assert cal.is_calibrated

        profile = compute_neurodivergent_profile(
            valence=0.3, arousal=0.5, calibration=cal,
        )
        assert profile.calibration_applied is True

    def test_uncalibrated_not_applied(self):
        """Uncalibrated data should not affect output."""
        cal = CalibrationData(user_id="test_user")
        cal.add_sample(0.3, 0.5)  # only 1 sample, not enough
        assert not cal.is_calibrated

        profile = compute_neurodivergent_profile(
            valence=0.3, arousal=0.5, calibration=cal,
        )
        assert profile.calibration_applied is False


# -- Threshold adaptation ----------------------------------------------------

class TestAdaptEmotionThresholds:

    def test_uncalibrated_returns_defaults(self):
        """Uncalibrated data should return unmodified thresholds."""
        cal = CalibrationData(user_id="test_user")
        defaults = {"positive_valence": 0.2, "negative_valence": -0.2}
        result = adapt_emotion_thresholds(cal, defaults)
        assert result == defaults

    def test_calibrated_shifts_thresholds(self):
        """Calibrated user with high baseline should shift valence thresholds up."""
        cal = CalibrationData(user_id="test_user")
        # User whose normal valence is 0.4 (happier than average)
        for _ in range(15):
            cal.add_sample(0.4, 0.6)

        result = adapt_emotion_thresholds(cal)
        # Positive threshold should be shifted up by ~0.4
        assert result["positive_valence"] > 0.2
        # Negative threshold should also shift
        assert result["negative_valence"] > -0.2

    def test_arousal_thresholds_shift(self):
        """Calibrated user with high arousal baseline should shift arousal thresholds."""
        cal = CalibrationData(user_id="test_user")
        for _ in range(15):
            cal.add_sample(0.0, 0.8)  # high arousal baseline

        result = adapt_emotion_thresholds(cal)
        # High arousal threshold should shift up
        assert result["high_arousal"] > 0.65


# -- Calibration data --------------------------------------------------------

class TestCalibrationData:

    def test_becomes_calibrated(self):
        """Should become calibrated after enough samples."""
        cal = CalibrationData(user_id="user1", min_samples=10)
        for i in range(10):
            cal.add_sample(0.3, 0.5)
        assert cal.is_calibrated
        assert cal.n_samples == 10

    def test_not_calibrated_before_min(self):
        """Should not be calibrated before min samples."""
        cal = CalibrationData(user_id="user1", min_samples=10)
        for i in range(5):
            cal.add_sample(0.3, 0.5)
        assert not cal.is_calibrated

    def test_baseline_computed_correctly(self):
        """Baseline should reflect the mean of samples."""
        cal = CalibrationData(user_id="user1", min_samples=5)
        for _ in range(5):
            cal.add_sample(0.4, 0.6)
        assert abs(cal.valence_baseline - 0.4) < 0.01
        assert abs(cal.arousal_baseline - 0.6) < 0.01


# -- Serialization -----------------------------------------------------------

class TestProfileToDict:

    def test_serialization_all_fields(self):
        """Serialized profile should contain all expected fields."""
        profile = compute_neurodivergent_profile(
            valence=0.3, arousal=0.5, input_modality="energy", raw_input="high",
        )
        d = profile_to_dict(profile)

        assert "valence" in d
        assert "arousal" in d
        assert "input_modality" in d
        assert "confidence" in d
        assert "adhd_profile" in d
        assert "intensity" in d["adhd_profile"]
        assert "volatility" in d["adhd_profile"]
        assert "rejection_sensitivity" in d["adhd_profile"]
        assert "current_state" in d["adhd_profile"]

    def test_serialization_values_match(self):
        """Serialized values should match the profile."""
        profile = compute_neurodivergent_profile(
            valence=-0.3, arousal=0.8, input_modality="sensory", raw_input="overstimulated",
        )
        d = profile_to_dict(profile)
        assert d["valence"] == profile.valence
        assert d["arousal"] == profile.arousal
        assert d["input_modality"] == "sensory"


# -- Output range validation -------------------------------------------------

class TestOutputRanges:

    def test_valence_in_range(self):
        """All modalities should produce valence in [-1, 1]."""
        for color in ["red", "blue", "black", "yellow", "green"]:
            r = map_color_to_emotion(color)
            assert -1.0 <= r["valence"] <= 1.0

        for level in ["fully_charged", "drained", "moderate"]:
            r = map_energy_to_emotion(level)
            assert -1.0 <= r["valence"] <= 1.0

        for state in ["overstimulated", "regulated", "meltdown", "shutdown"]:
            r = map_sensory_state_to_emotion(state)
            assert -1.0 <= r["valence"] <= 1.0

    def test_arousal_in_range(self):
        """All modalities should produce arousal in [0, 1]."""
        for color in ["red", "blue", "black", "yellow", "green"]:
            r = map_color_to_emotion(color)
            assert 0.0 <= r["arousal"] <= 1.0

        for level in ["fully_charged", "drained", "moderate"]:
            r = map_energy_to_emotion(level)
            assert 0.0 <= r["arousal"] <= 1.0

        for state in ["overstimulated", "regulated", "meltdown", "shutdown"]:
            r = map_sensory_state_to_emotion(state)
            assert 0.0 <= r["arousal"] <= 1.0

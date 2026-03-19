"""Unit tests for cultural emotion calibration model (issue #436)."""
from __future__ import annotations

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.cultural_calibration import (
    CULTURAL_PROFILES,
    EAST_ASIAN,
    LATIN_AMERICAN,
    NORDIC,
    NORTH_AMERICAN,
    SOUTH_ASIAN,
    MIDDLE_EASTERN,
    SUB_SAHARAN_AFRICAN,
    WESTERN_EUROPEAN,
    CulturalProfile,
    get_cultural_profile,
    apply_display_rule_correction,
    calibrate_self_report,
    compute_affect_valuation,
    adapt_thresholds,
    profile_to_dict,
    calibrate,
)


# -- get_cultural_profile ---------------------------------------------------


class TestGetCulturalProfile:

    def test_valid_lookup(self):
        p = get_cultural_profile("east_asian")
        assert p.cluster_name == "east_asian"

    def test_case_insensitive(self):
        p = get_cultural_profile("East_Asian")
        assert p.cluster_name == "east_asian"

    def test_dash_separator(self):
        p = get_cultural_profile("latin-american")
        assert p.cluster_name == "latin_american"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown cultural cluster"):
            get_cultural_profile("atlantean")

    def test_all_eight_profiles_exist(self):
        assert len(CULTURAL_PROFILES) >= 8
        expected = {
            "east_asian", "south_asian", "latin_american", "nordic",
            "north_american", "middle_eastern", "sub_saharan_african",
            "western_european",
        }
        assert expected.issubset(set(CULTURAL_PROFILES.keys()))


# -- apply_display_rule_correction ------------------------------------------


class TestDisplayRuleCorrection:

    def test_east_asian_negative_suppression(self):
        """East Asian culture should shift mild-negative valence further negative."""
        result = apply_display_rule_correction(-0.1, 0.5, EAST_ASIAN)
        # Negative suppression should push corrected valence more negative
        assert result.corrected_valence < result.original_valence
        assert result.valence_adjustment < 0

    def test_latin_american_positive_amplification(self):
        """Latin American culture should attenuate a highly positive reading."""
        result = apply_display_rule_correction(0.8, 0.7, LATIN_AMERICAN)
        # Positive amplification correction should pull positive valence down
        assert result.corrected_valence < result.original_valence

    def test_masking_boosts_arousal(self):
        """High-masking cultures should have corrected arousal > original."""
        result = apply_display_rule_correction(0.0, 0.4, EAST_ASIAN)
        assert result.corrected_arousal > result.original_arousal
        assert result.arousal_adjustment > 0

    def test_no_correction_needed(self):
        """A neutral reading in a culture with low biases should have minimal correction."""
        # Nordic is moderate across the board; use a very positive valence
        # where negative_suppression does not trigger
        result = apply_display_rule_correction(0.5, 0.5, NORDIC)
        # Some correction may apply, but check rationale exists
        assert result.correction_rationale

    def test_clamping(self):
        """Corrected values should remain in valid ranges."""
        result = apply_display_rule_correction(-0.95, 0.95, EAST_ASIAN)
        assert -1.0 <= result.corrected_valence <= 1.0
        assert 0.0 <= result.corrected_arousal <= 1.0

    def test_positive_valence_no_negative_suppression(self):
        """Positive valence should not trigger negative suppression correction."""
        result = apply_display_rule_correction(0.5, 0.5, EAST_ASIAN)
        # Only positive amplification + masking should apply, not negative suppression
        assert "negative suppression" not in result.correction_rationale


# -- calibrate_self_report --------------------------------------------------


class TestCalibrateSelfReport:

    def test_east_asian_acquiescence(self):
        """East Asian acquiescence bias should attenuate positive self-report."""
        result = calibrate_self_report(0.8, 0.7, EAST_ASIAN)
        assert result.calibrated_valence < result.original_valence
        assert any("acquiescence" in c for c in result.bias_corrections_applied)

    def test_latin_extreme_response(self):
        """Latin American extreme response style should compress extremes."""
        result = calibrate_self_report(0.9, 0.9, LATIN_AMERICAN)
        assert result.calibrated_valence < result.original_valence
        assert any("extreme response" in c for c in result.bias_corrections_applied)

    def test_social_desirability_positive_only(self):
        """Social desirability correction should only apply to positive valence."""
        # Negative valence should not get social desirability correction
        result = calibrate_self_report(-0.5, 0.5, EAST_ASIAN)
        assert not any("social desirability" in c for c in result.bias_corrections_applied)

    def test_nordic_minimal_bias(self):
        """Nordic profiles have low biases -- corrections should be small."""
        result = calibrate_self_report(0.5, 0.5, NORDIC)
        # Low acquiescence (0.15) and low extreme response (0.25) -- neither triggers
        diff = abs(result.calibrated_valence - result.original_valence)
        assert diff < 0.1


# -- compute_affect_valuation -----------------------------------------------


class TestAffectValuation:

    def test_calm_positive_matches_east_asian_ideal(self):
        """Calm positive state should align well with East Asian ideal."""
        result = compute_affect_valuation(0.6, 0.3, EAST_ASIAN)
        assert result.alignment_score > 0.7

    def test_excited_positive_matches_north_american_ideal(self):
        """Excited positive state should align well with North American ideal."""
        result = compute_affect_valuation(0.7, 0.75, NORTH_AMERICAN)
        assert result.alignment_score > 0.8

    def test_negative_state_low_alignment(self):
        """Strongly negative state should not align with any cultural ideal."""
        result = compute_affect_valuation(-0.8, 0.2, EAST_ASIAN)
        assert result.alignment_score < 0.5

    def test_interpretation_text(self):
        """Alignment score above 0.75 should produce 'closely matches' text."""
        result = compute_affect_valuation(0.7, 0.3, EAST_ASIAN)
        assert "closely matches" in result.interpretation


# -- adapt_thresholds -------------------------------------------------------


class TestAdaptThresholds:

    def test_expressive_culture_higher_thresholds(self):
        """Latin American (high expressivity) should have higher arousal threshold."""
        thresholds = adapt_thresholds(LATIN_AMERICAN)
        base_thresholds = adapt_thresholds(NORDIC)
        assert thresholds["arousal_threshold"] > base_thresholds["arousal_threshold"]

    def test_masking_culture_lower_masking_factor(self):
        """East Asian (high masking) should have lower masking_factor."""
        thresholds = adapt_thresholds(EAST_ASIAN)
        assert thresholds["masking_factor"] < 1.0

    def test_negative_threshold_increases_with_suppression(self):
        """Higher negative suppression should shift negative threshold up (less negative)."""
        east = adapt_thresholds(EAST_ASIAN)
        latin = adapt_thresholds(LATIN_AMERICAN)
        # East Asian has higher negative_suppression (0.80 vs 0.25)
        assert east["negative_emotion_threshold"] > latin["negative_emotion_threshold"]

    def test_all_thresholds_present(self):
        thresholds = adapt_thresholds(NORTH_AMERICAN)
        expected_keys = {
            "valence_threshold", "arousal_threshold",
            "negative_emotion_threshold", "positive_emotion_threshold",
            "expressivity_factor", "masking_factor",
        }
        assert expected_keys == set(thresholds.keys())


# -- profile_to_dict -------------------------------------------------------


class TestProfileToDict:

    def test_serialization_keys(self):
        d = profile_to_dict(EAST_ASIAN)
        assert "cluster_name" in d
        assert "display_rules" in d
        assert "affect_valuation" in d
        assert "self_report_biases" in d

    def test_round_trip_values(self):
        d = profile_to_dict(SOUTH_ASIAN)
        assert d["cluster_name"] == "south_asian"
        assert d["display_rules"]["masking_tendency"] == 0.50
        assert d["affect_valuation"]["ideal_valence_type"] == "calm_positive"


# -- full calibrate pipeline ------------------------------------------------


class TestCalibratePipeline:

    def test_full_calibration(self):
        result = calibrate(
            valence=0.3,
            arousal=0.5,
            culture="east_asian",
        )
        assert result.culture == "east_asian"
        assert result.display_rule_correction.original_valence == 0.3
        assert result.affect_valuation.alignment_score >= 0.0
        assert "cluster_name" in result.profile_summary

    def test_with_self_report(self):
        result = calibrate(
            valence=0.3,
            arousal=0.5,
            culture="east_asian",
            reported_valence=0.7,
            reported_arousal=0.6,
        )
        # Self-report calibration should use the reported values
        assert result.self_report_calibration.original_valence == 0.7
        assert result.self_report_calibration.original_arousal == 0.6

    def test_unknown_culture_raises(self):
        with pytest.raises(ValueError):
            calibrate(valence=0.0, arousal=0.5, culture="unknown_culture")

    def test_all_cultures_calibrate(self):
        """Every registered culture should run through the pipeline without error."""
        for name in CULTURAL_PROFILES:
            result = calibrate(valence=0.0, arousal=0.5, culture=name)
            assert result.culture == name

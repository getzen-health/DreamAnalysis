"""Tests for cross-cultural emotion recognition and cultural calibration."""

import pytest

from models.multilingual_emotion import (
    CulturalCalibrator,
    CulturalEIAdapter,
    get_culture_group,
)


class TestCultureGroupMapping:
    def test_collectivist_languages(self):
        for lang in ["ja", "zh", "ko", "ar", "hi", "th", "vi"]:
            assert get_culture_group(lang) == "collectivist", f"{lang} should be collectivist"

    def test_individualist_languages(self):
        for lang in ["en", "de", "fr", "es", "it", "pt", "nl"]:
            assert get_culture_group(lang) == "individualist", f"{lang} should be individualist"

    def test_mixed_languages(self):
        for lang in ["ru", "tr", "pl"]:
            assert get_culture_group(lang) == "mixed", f"{lang} should be mixed"

    def test_unknown_defaults_to_individualist(self):
        assert get_culture_group("xx") == "individualist"
        assert get_culture_group("zz") == "individualist"

    def test_case_insensitive(self):
        assert get_culture_group("JA") == "collectivist"
        assert get_culture_group("En") == "individualist"

    def test_handles_longer_codes(self):
        assert get_culture_group("ja-JP") == "collectivist"
        assert get_culture_group("en-US") == "individualist"


class TestCulturalCalibrator:
    def setup_method(self):
        self.cal = CulturalCalibrator()
        self.base_probs = {
            "happy": 0.1, "sad": 0.1, "angry": 0.3,
            "fear": 0.1, "surprise": 0.1, "neutral": 0.3,
        }

    def test_calibrate_returns_required_fields(self):
        result = self.cal.calibrate(self.base_probs, "collectivist")
        assert "emotion" in result
        assert "probabilities" in result
        assert "valence" in result
        assert "arousal" in result
        assert "culture_group" in result
        assert "adjustments_applied" in result
        assert result["calibrated"] is True

    def test_collectivist_boosts_neutral(self):
        result_coll = self.cal.calibrate(self.base_probs, "collectivist")
        result_indv = self.cal.calibrate(self.base_probs, "individualist")
        # Collectivist should have higher neutral probability
        assert result_coll["probabilities"]["neutral"] > result_indv["probabilities"]["neutral"]

    def test_collectivist_suppresses_anger(self):
        high_anger_probs = {
            "happy": 0.05, "sad": 0.05, "angry": 0.6,
            "fear": 0.1, "surprise": 0.1, "neutral": 0.1,
        }
        result = self.cal.calibrate(high_anger_probs, "collectivist")
        # Anger should be reduced from raw 0.6
        assert result["probabilities"]["angry"] < 0.6

    def test_individualist_no_anger_suppression(self):
        result = self.cal.calibrate(self.base_probs, "individualist")
        assert "anger_expression_adjusted" not in result["adjustments_applied"]

    def test_probabilities_sum_to_one(self):
        for culture in ["collectivist", "individualist", "mixed"]:
            result = self.cal.calibrate(self.base_probs, culture)
            total = sum(result["probabilities"].values())
            assert abs(total - 1.0) < 0.01, f"{culture}: probs sum to {total}"

    def test_collectivist_amplifies_negative_valence(self):
        result = self.cal.calibrate(
            self.base_probs, "collectivist", valence=-0.5, arousal=0.6
        )
        # Negative valence should be amplified for collectivist
        assert result["valence"] < -0.5

    def test_individualist_no_valence_change(self):
        result = self.cal.calibrate(
            self.base_probs, "individualist", valence=-0.5, arousal=0.6
        )
        assert result["valence"] == -0.5

    def test_adjustments_tracked(self):
        result = self.cal.calibrate(self.base_probs, "collectivist")
        assert "neutral_prior_boosted" in result["adjustments_applied"]
        assert "anger_expression_adjusted" in result["adjustments_applied"]

    def test_top_emotion_reflects_calibration(self):
        # When anger is high but culture suppresses it
        anger_probs = {
            "happy": 0.05, "sad": 0.05, "angry": 0.4,
            "fear": 0.05, "surprise": 0.05, "neutral": 0.4,
        }
        result = self.cal.calibrate(anger_probs, "collectivist")
        # After collectivist calibration, neutral should likely win
        assert result["emotion"] == "neutral"


class TestCulturalEIAdapter:
    def setup_method(self):
        self.adapter = CulturalEIAdapter()
        self.scores = {
            "self_perception": 50.0,
            "self_expression": 50.0,
            "interpersonal": 50.0,
            "decision_making": 50.0,
            "stress_management": 50.0,
        }

    def test_adjust_returns_required_fields(self):
        result = self.adapter.adjust_ei_scores(self.scores, "collectivist")
        assert "adjusted_scores" in result
        assert "raw_scores" in result
        assert "culture_group" in result
        assert "adjustments" in result

    def test_collectivist_adjustments(self):
        result = self.adapter.adjust_ei_scores(self.scores, "collectivist")
        # Self-expression should be boosted (offset is negative = norms are lower)
        assert result["adjusted_scores"]["self_expression"] < self.scores["self_expression"]
        # Interpersonal should be boosted (collectivist strength)
        assert result["adjusted_scores"]["interpersonal"] > self.scores["interpersonal"]

    def test_individualist_no_adjustments(self):
        result = self.adapter.adjust_ei_scores(self.scores, "individualist")
        for dim in self.scores:
            assert result["adjusted_scores"][dim] == self.scores[dim]

    def test_scores_clamped_to_range(self):
        extreme_scores = {
            "self_perception": 99.0,
            "self_expression": 2.0,
            "interpersonal": 50.0,
            "decision_making": 50.0,
            "stress_management": 50.0,
        }
        result = self.adapter.adjust_ei_scores(extreme_scores, "collectivist")
        for dim, val in result["adjusted_scores"].items():
            assert 0 <= val <= 100, f"{dim} out of range: {val}"

    def test_raw_scores_preserved(self):
        result = self.adapter.adjust_ei_scores(self.scores, "collectivist")
        assert result["raw_scores"] == self.scores

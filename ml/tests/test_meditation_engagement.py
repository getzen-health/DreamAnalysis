"""Tests for meditation engagement metrics (#490) and brain age validation (#489)."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMeditationEngagement:
    """Tests for renamed meditation metrics (#490)."""

    def setup_method(self):
        from models.meditation_classifier import MeditationClassifier
        self.model = MeditationClassifier()

    def test_predict_returns_engagement_key(self, sample_eeg, fs):
        result = self.model.predict(sample_eeg, fs)
        assert "engagement" in result
        assert "engagement_index" in result

    def test_predict_returns_legacy_depth_key(self, sample_eeg, fs):
        """Backward compat: depth keys should still exist."""
        result = self.model.predict(sample_eeg, fs)
        assert "depth" in result
        assert "depth_index" in result

    def test_engagement_matches_depth(self, sample_eeg, fs):
        """engagement and depth should be identical (same value, new name)."""
        result = self.model.predict(sample_eeg, fs)
        assert result["engagement"] == result["depth"]
        assert result["engagement_index"] == result["depth_index"]

    def test_engagement_levels_are_valid(self, sample_eeg, fs):
        from models.meditation_classifier import MEDITATION_LEVELS
        result = self.model.predict(sample_eeg, fs)
        assert result["engagement"] in MEDITATION_LEVELS

    def test_alpha_coherence_present(self, sample_eeg, fs):
        """#490: alpha_coherence is a validated marker."""
        result = self.model.predict(sample_eeg, fs)
        assert "alpha_coherence" in result
        assert isinstance(result["alpha_coherence"], float)

    def test_theta_power_present(self, sample_eeg, fs):
        """#490: theta_power is a validated marker."""
        result = self.model.predict(sample_eeg, fs)
        assert "theta_power" in result
        assert isinstance(result["theta_power"], float)
        assert result["theta_power"] >= 0.0

    def test_multichannel_alpha_coherence(self, multichannel_eeg, fs):
        """Alpha coherence should use real coherence when multichannel available."""
        result = self.model.predict(multichannel_eeg, fs)
        assert "alpha_coherence" in result
        assert isinstance(result["alpha_coherence"], float)

    def test_components_use_theta_elevation(self, sample_eeg, fs):
        """#490: components should use theta_elevation not theta_depth."""
        result = self.model.predict(sample_eeg, fs)
        assert "theta_elevation" in result["components"]

    def test_meditation_levels_alias(self):
        from models.meditation_classifier import MEDITATION_DEPTHS, MEDITATION_LEVELS
        assert MEDITATION_DEPTHS is MEDITATION_LEVELS

    def test_meditation_score_is_float(self, sample_eeg, fs):
        result = self.model.predict(sample_eeg, fs)
        assert isinstance(result["meditation_score"], float)
        assert 0.0 <= result["meditation_score"] <= 1.0


class TestBrainAgeValidation:
    """Tests for brain age validation notes and disclaimer (#489)."""

    def setup_method(self):
        from models.brain_age_estimator import BrainAgeEstimator
        self.model = BrainAgeEstimator()

    def test_predict_includes_disclaimer(self, sample_eeg, fs):
        result = self.model.predict(sample_eeg, fs)
        assert "disclaimer" in result
        assert len(result["disclaimer"]) > 0

    def test_predict_includes_ui_disclaimer(self, sample_eeg, fs):
        """#489: short UI disclaimer for display."""
        result = self.model.predict(sample_eeg, fs)
        assert "ui_disclaimer" in result
        assert "not a medical measurement" in result["ui_disclaimer"].lower()

    def test_predict_includes_accuracy_notes(self, sample_eeg, fs):
        """#489: accuracy range documentation in output."""
        result = self.model.predict(sample_eeg, fs)
        assert "accuracy_notes" in result
        notes = result["accuracy_notes"]
        assert isinstance(notes, dict)
        assert "mae_consumer_eeg" in notes
        assert "mae_research_eeg" in notes
        assert "r2_aperiodic" in notes

    def test_module_level_constants(self):
        from models.brain_age_estimator import DISCLAIMER, UI_DISCLAIMER, ACCURACY_NOTES
        assert "not a medical device" in DISCLAIMER.lower()
        assert "not a medical measurement" in UI_DISCLAIMER.lower()
        assert isinstance(ACCURACY_NOTES, dict)

    def test_predicted_age_is_clamped(self, sample_eeg, fs):
        result = self.model.predict(sample_eeg, fs)
        assert 15.0 <= result["predicted_age"] <= 90.0

    def test_brain_age_gap_with_chronological_age(self, sample_eeg, fs):
        result = self.model.predict(sample_eeg, fs, chronological_age=30.0)
        assert result["brain_age_gap"] is not None
        assert result["gap_interpretation"] is not None

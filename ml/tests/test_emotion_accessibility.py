"""Tests for emotion-aware accessibility layer — issue #460.

Covers: EEG state assessment, accessibility profile computation,
UI adaptation recommendations, overwhelm/fatigue detection,
profile serialization, edge cases, and API routes.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from models.emotion_accessibility import (
    AccessibilityAssessment,
    AccessibilityProfile,
    EEGState,
    UIAdaptation,
    assess_accessibility_needs,
    assessment_to_dict,
    compute_accessibility_profile,
    profile_to_dict,
    recommend_adaptations,
)
from api.routes.emotion_accessibility import router

# ---------------------------------------------------------------------------
# Test client setup
# ---------------------------------------------------------------------------

app = FastAPI()
app.include_router(router)
client = TestClient(app)


# ===========================================================================
# Test assess_accessibility_needs
# ===========================================================================

class TestAssessAccessibilityNeeds:

    def test_calm_state_low_overwhelm(self):
        """Calm EEG state should produce low overwhelm and fatigue."""
        eeg = EEGState(valence=0.5, arousal=0.3, stress=0.1, fatigue=0.1, cognitive_load=0.1)
        assessment = assess_accessibility_needs(eeg)
        assert assessment.overwhelm_score < 0.3
        assert assessment.fatigue_score < 0.3
        assert assessment.needs_intervention is False

    def test_overwhelmed_state_high_score(self):
        """High arousal + stress + cognitive load should trigger overwhelm."""
        eeg = EEGState(valence=-0.5, arousal=0.9, stress=0.9, fatigue=0.3, cognitive_load=0.9)
        assessment = assess_accessibility_needs(eeg)
        assert assessment.overwhelm_score > 0.5
        assert assessment.needs_intervention is True

    def test_fatigued_state_high_fatigue(self):
        """High fatigue + low arousal should produce high fatigue score."""
        eeg = EEGState(valence=0.0, arousal=0.1, stress=0.2, fatigue=0.9, cognitive_load=0.3)
        assessment = assess_accessibility_needs(eeg)
        assert assessment.fatigue_score > 0.4

    def test_default_profile_when_none(self):
        """Should use default profile (no impairments) when none provided."""
        eeg = EEGState()
        assessment = assess_accessibility_needs(eeg, profile=None)
        assert assessment.profile.visual_impairment == 0.0
        assert assessment.profile.motor_difficulty == 0.0

    def test_custom_profile_preserved(self):
        """Custom profile should be preserved in assessment."""
        eeg = EEGState()
        profile = AccessibilityProfile(visual_impairment=0.8, motor_difficulty=0.5)
        assessment = assess_accessibility_needs(eeg, profile)
        assert assessment.profile.visual_impairment == 0.8
        assert assessment.profile.motor_difficulty == 0.5

    def test_stress_threshold_triggers_intervention(self):
        """Stress above threshold should trigger needs_intervention."""
        eeg = EEGState(stress=0.85, arousal=0.4, fatigue=0.2, cognitive_load=0.2)
        assessment = assess_accessibility_needs(eeg)
        assert assessment.needs_intervention is True


# ===========================================================================
# Test recommend_adaptations
# ===========================================================================

class TestRecommendAdaptations:

    def test_visual_impairment_increases_font(self):
        """Visual impairment should increase font size."""
        eeg = EEGState()
        profile = AccessibilityProfile(visual_impairment=0.8)
        assessment = assess_accessibility_needs(eeg, profile)
        adaptation = recommend_adaptations(assessment)
        assert adaptation.font_size > 20
        assert adaptation.high_contrast is True

    def test_sensory_sensitivity_reduces_animation(self):
        """Sensory sensitivity should reduce animation speed."""
        eeg = EEGState()
        profile = AccessibilityProfile(sensory_sensitivity=0.9)
        assessment = assess_accessibility_needs(eeg, profile)
        adaptation = recommend_adaptations(assessment)
        assert adaptation.animation_speed < 0.3
        assert adaptation.reduce_motion is True
        assert adaptation.color_saturation < 0.8

    def test_motor_difficulty_increases_targets(self):
        """Motor difficulty should increase target size multiplier."""
        eeg = EEGState()
        profile = AccessibilityProfile(motor_difficulty=0.8)
        assessment = assess_accessibility_needs(eeg, profile)
        adaptation = recommend_adaptations(assessment)
        assert adaptation.target_size_multiplier > 1.5
        assert adaptation.transition_duration_ms > 300

    def test_cognitive_load_simplifies_layout(self):
        """High cognitive load sensitivity should simplify the layout."""
        eeg = EEGState(cognitive_load=0.8)
        profile = AccessibilityProfile(cognitive_load_sensitivity=0.8)
        assessment = assess_accessibility_needs(eeg, profile)
        adaptation = recommend_adaptations(assessment)
        assert adaptation.simplified_layout is True
        assert adaptation.information_density < 0.5

    def test_overwhelm_reduces_info_density(self):
        """Overwhelmed state should reduce information density."""
        eeg = EEGState(arousal=0.9, stress=0.9, cognitive_load=0.9)
        assessment = assess_accessibility_needs(eeg)
        adaptation = recommend_adaptations(assessment)
        assert adaptation.information_density < 0.6

    def test_sensory_sensitivity_reduces_audio(self):
        """Sensory sensitive users should get reduced audio cues."""
        eeg = EEGState()
        profile = AccessibilityProfile(sensory_sensitivity=0.8)
        assessment = assess_accessibility_needs(eeg, profile)
        adaptation = recommend_adaptations(assessment)
        assert adaptation.audio_cue_volume < 0.3

    def test_visual_impairment_increases_audio(self):
        """Visually impaired users should get enhanced audio cues."""
        eeg = EEGState()
        profile = AccessibilityProfile(visual_impairment=0.8)
        assessment = assess_accessibility_needs(eeg, profile)
        adaptation = recommend_adaptations(assessment)
        assert adaptation.audio_cue_volume > 0.6

    def test_reasons_populated(self):
        """Adaptations should include human-readable reasons."""
        eeg = EEGState(arousal=0.9, stress=0.9, cognitive_load=0.9)
        profile = AccessibilityProfile(visual_impairment=0.8, sensory_sensitivity=0.8)
        assessment = assess_accessibility_needs(eeg, profile)
        adaptation = recommend_adaptations(assessment)
        assert len(adaptation.reasons) > 0
        assert any("font" in r.lower() or "contrast" in r.lower() for r in adaptation.reasons)

    def test_calm_state_minimal_adaptations(self):
        """Calm state with no impairments should produce minimal changes."""
        eeg = EEGState(valence=0.5, arousal=0.4, stress=0.1, fatigue=0.1, cognitive_load=0.1)
        assessment = assess_accessibility_needs(eeg)
        adaptation = recommend_adaptations(assessment)
        assert adaptation.animation_speed > 0.8
        assert adaptation.reduce_motion is False
        assert adaptation.simplified_layout is False


# ===========================================================================
# Test compute_accessibility_profile
# ===========================================================================

class TestComputeAccessibilityProfile:

    def test_empty_history(self):
        """Empty history should produce zero-sensitivity profile."""
        profile = compute_accessibility_profile([])
        assert profile.cognitive_load_sensitivity == 0.0
        assert profile.sensory_sensitivity == 0.0

    def test_high_cognitive_load_history(self):
        """History with frequent high cognitive load should detect sensitivity."""
        history = [
            EEGState(cognitive_load=0.9, arousal=0.5, stress=0.3)
            for _ in range(10)
        ]
        profile = compute_accessibility_profile(history)
        assert profile.cognitive_load_sensitivity > 0.5

    def test_known_impairments_override(self):
        """Known impairments should override computed values."""
        profile = compute_accessibility_profile(
            [],
            known_impairments={"visual_impairment": 0.7, "motor_difficulty": 0.5},
        )
        assert profile.visual_impairment == 0.7
        assert profile.motor_difficulty == 0.5

    def test_overwhelm_history_detects_sensory(self):
        """Frequent overwhelm (high arousal + stress) should flag sensory sensitivity."""
        history = [
            EEGState(arousal=0.9, stress=0.9) for _ in range(8)
        ] + [
            EEGState(arousal=0.3, stress=0.2) for _ in range(2)
        ]
        profile = compute_accessibility_profile(history)
        assert profile.sensory_sensitivity > 0.5


# ===========================================================================
# Test serialization
# ===========================================================================

class TestSerialization:

    def test_profile_to_dict(self):
        """profile_to_dict should produce a dict with all fields."""
        profile = AccessibilityProfile(visual_impairment=0.5, motor_difficulty=0.3)
        d = profile_to_dict(profile)
        assert d["visual_impairment"] == 0.5
        assert d["motor_difficulty"] == 0.3
        assert "cognitive_load_sensitivity" in d
        assert "sensory_sensitivity" in d

    def test_assessment_to_dict(self):
        """assessment_to_dict should contain all nested structures."""
        eeg = EEGState(valence=0.2, arousal=0.6)
        assessment = assess_accessibility_needs(eeg)
        recommend_adaptations(assessment)
        d = assessment_to_dict(assessment)
        assert "eeg_state" in d
        assert "profile" in d
        assert "adaptation" in d
        assert "overwhelm_score" in d
        assert "fatigue_score" in d
        assert "needs_intervention" in d
        assert d["eeg_state"]["valence"] == 0.2


# ===========================================================================
# Test API routes
# ===========================================================================

class TestAPIRoutes:

    def test_status_endpoint(self):
        """GET /accessibility/status should return ready."""
        resp = client.get("/accessibility/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert "supported_profiles" in data

    def test_adapt_endpoint_basic(self):
        """POST /accessibility/adapt should return adaptations."""
        resp = client.post("/accessibility/adapt", json={
            "eeg_state": {
                "valence": 0.0,
                "arousal": 0.5,
                "stress": 0.3,
                "fatigue": 0.3,
                "cognitive_load": 0.3,
            },
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "adaptation" in data
        assert "overwhelm_score" in data
        assert "font_size" in data["adaptation"]

    def test_adapt_with_profile(self):
        """POST /accessibility/adapt with profile should apply profile."""
        resp = client.post("/accessibility/adapt", json={
            "eeg_state": {
                "valence": -0.3,
                "arousal": 0.8,
                "stress": 0.7,
                "fatigue": 0.2,
                "cognitive_load": 0.6,
            },
            "profile": {
                "visual_impairment": 0.7,
                "motor_difficulty": 0.0,
                "cognitive_load_sensitivity": 0.5,
                "sensory_sensitivity": 0.0,
            },
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["profile"]["visual_impairment"] == 0.7
        assert data["adaptation"]["font_size"] > 16

"""Tests for multimodal emotional intelligence integration.

Covers:
  - Multimodal component score computation (voice, health, coherence)
  - EI composite integration with new component types
  - POST /multimodal-ei/assess endpoint with various input combinations
  - Edge cases: missing data, partial inputs, score clamping
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ei_composite import EmotionalIntelligenceIndex, _EXTERNAL_COMPONENTS


# ---------------------------------------------------------------------------
# Unit tests: EI composite accepts new multimodal component names
# ---------------------------------------------------------------------------

class TestEICompositeMultimodalComponents:
    """Verify the EI composite model recognises the 7 new multimodal components."""

    @pytest.fixture
    def model(self):
        return EmotionalIntelligenceIndex()

    def test_new_components_in_external_set(self):
        """All 7 new component names should be recognised."""
        new_names = [
            "voice_valence_clarity",
            "voice_expression_range",
            "voice_cognitive_load",
            "hrv_regulation",
            "sleep_restoration",
            "activity_engagement",
            "health_mood_coherence",
        ]
        for name in new_names:
            assert name in _EXTERNAL_COMPONENTS, f"{name} not in _EXTERNAL_COMPONENTS"

    def test_original_components_still_present(self):
        """Existing 9 components must not be removed."""
        originals = [
            "granularity", "flexibility", "synchrony", "interoception",
            "reactivity_regulation", "affect_labeling", "alexithymia",
            "emotional_memory", "mood_stability",
        ]
        for name in originals:
            assert name in _EXTERNAL_COMPONENTS, f"Original {name} missing"

    def test_update_component_accepts_new_names(self, model):
        """update_component should work for new multimodal names."""
        model.update_component("voice_valence_clarity", 0.8, "user1")
        model.update_component("hrv_regulation", 0.6, "user1")
        # Should not raise; stored components should contain them
        stored = model._components.get("user1", {})
        assert "voice_valence_clarity" in stored
        assert "hrv_regulation" in stored

    def test_compute_eiq_with_voice_components_only(self, model):
        """EIQ should compute with only voice component scores (no EEG)."""
        components = {
            "voice_valence_clarity": 0.75,
            "voice_expression_range": 0.6,
            "voice_cognitive_load": 0.8,
        }
        result = model.compute_eiq(component_scores=components, user_id="voice_user")
        assert result is not None
        assert "eiq_score" in result
        assert 0.0 <= result["eiq_score"] <= 100.0
        assert result["eiq_grade"] in ("A", "B", "C", "D", "F")

    def test_compute_eiq_with_health_components_only(self, model):
        """EIQ should compute with only health component scores (no EEG)."""
        components = {
            "hrv_regulation": 0.7,
            "sleep_restoration": 0.85,
            "activity_engagement": 0.5,
            "health_mood_coherence": 0.6,
        }
        result = model.compute_eiq(component_scores=components, user_id="health_user")
        assert result is not None
        assert 0.0 <= result["eiq_score"] <= 100.0

    def test_compute_eiq_with_all_multimodal(self, model):
        """EIQ should compute with all 7 new components."""
        components = {
            "voice_valence_clarity": 0.8,
            "voice_expression_range": 0.7,
            "voice_cognitive_load": 0.6,
            "hrv_regulation": 0.75,
            "sleep_restoration": 0.9,
            "activity_engagement": 0.55,
            "health_mood_coherence": 0.65,
        }
        result = model.compute_eiq(component_scores=components, user_id="full_user")
        assert result is not None
        assert "dimensions" in result
        for dim in ["self_perception", "self_expression", "interpersonal",
                     "decision_making", "stress_management"]:
            assert dim in result["dimensions"]
            assert 0.0 <= result["dimensions"][dim] <= 100.0

    def test_compute_eiq_blends_eeg_and_multimodal(self, model):
        """When EEG signals AND multimodal components are provided, both contribute."""
        signals = np.random.randn(4, 1024) * 20
        components = {
            "voice_valence_clarity": 0.9,
            "hrv_regulation": 0.8,
        }
        result = model.compute_eiq(
            signals=signals, fs=256.0,
            component_scores=components, user_id="blend_user",
        )
        assert result is not None
        assert result["eiq_score"] > 0

    def test_component_scores_clipped(self, model):
        """Scores outside [0,1] should be clipped before mapping."""
        components = {
            "voice_valence_clarity": 1.5,  # above 1
            "hrv_regulation": -0.3,        # below 0
        }
        result = model.compute_eiq(component_scores=components, user_id="clip_user")
        assert result is not None
        # Should not crash, scores clipped internally

    def test_voice_maps_to_correct_dimensions(self, model):
        """voice_valence_clarity -> self_perception, voice_expression_range -> self_expression."""
        # Only voice_valence_clarity -> self_perception should be boosted
        components_sp = {"voice_valence_clarity": 1.0}
        result_sp = model.compute_eiq(component_scores=components_sp, user_id="sp_user")

        # Only voice_expression_range -> self_expression should be boosted
        components_se = {"voice_expression_range": 1.0}
        result_se = model.compute_eiq(component_scores=components_se, user_id="se_user")

        # self_perception should be high when voice_valence_clarity=1.0
        assert result_sp["dimensions"]["self_perception"] == 100.0
        # self_expression should be high when voice_expression_range=1.0
        assert result_se["dimensions"]["self_expression"] == 100.0

    def test_user_isolation(self, model):
        """Different users should have independent component stores."""
        model.update_component("voice_valence_clarity", 0.9, "user_a")
        model.update_component("voice_valence_clarity", 0.1, "user_b")

        result_a = model.compute_eiq(user_id="user_a")
        result_b = model.compute_eiq(user_id="user_b")

        assert result_a is not None
        assert result_b is not None
        assert result_a["eiq_score"] != result_b["eiq_score"]


# ---------------------------------------------------------------------------
# Unit tests: score computation functions from multimodal_ei route
# ---------------------------------------------------------------------------

class TestMultimodalScoreComputation:
    """Test the score computation logic that will live in the route module."""

    def test_voice_valence_clarity(self):
        """abs(valence) * confidence."""
        from api.routes.multimodal_ei import _compute_voice_scores
        voice = {"valence": 0.8, "arousal": 0.5, "confidence": 0.9}
        scores = _compute_voice_scores(voice)
        expected = abs(0.8) * 0.9  # 0.72
        assert abs(scores["voice_valence_clarity"] - expected) < 0.01

    def test_voice_valence_clarity_negative(self):
        """Negative valence should still produce positive clarity score."""
        from api.routes.multimodal_ei import _compute_voice_scores
        voice = {"valence": -0.6, "arousal": 0.3, "confidence": 1.0}
        scores = _compute_voice_scores(voice)
        assert scores["voice_valence_clarity"] == pytest.approx(0.6, abs=0.01)

    def test_voice_expression_range(self):
        """arousal * 0.6 + abs(valence) * 0.4."""
        from api.routes.multimodal_ei import _compute_voice_scores
        voice = {"valence": 0.5, "arousal": 0.8, "confidence": 0.7}
        scores = _compute_voice_scores(voice)
        expected = 0.8 * 0.6 + abs(0.5) * 0.4  # 0.48 + 0.20 = 0.68
        assert abs(scores["voice_expression_range"] - expected) < 0.01

    def test_voice_cognitive_load_inversion(self):
        """voice_cognitive_load = 1.0 - load_index. High load -> low score."""
        from api.routes.multimodal_ei import _compute_voice_scores
        voice = {"valence": 0.0, "arousal": 0.0, "confidence": 0.5,
                 "cognitive_load_index": 0.7}
        scores = _compute_voice_scores(voice)
        assert scores["voice_cognitive_load"] == pytest.approx(0.3, abs=0.01)

    def test_voice_cognitive_load_default(self):
        """Without cognitive_load_index, should default to 0.5."""
        from api.routes.multimodal_ei import _compute_voice_scores
        voice = {"valence": 0.0, "arousal": 0.0, "confidence": 0.5}
        scores = _compute_voice_scores(voice)
        assert scores["voice_cognitive_load"] == pytest.approx(0.5, abs=0.01)

    def test_health_hrv_regulation_sigmoid(self):
        """HRV sigmoid: 1/(1+exp(-0.1*(hrv-40))). At 40ms -> 0.5."""
        from api.routes.multimodal_ei import _compute_health_scores
        health = {"hrv_sdnn": 40.0}
        scores = _compute_health_scores(health)
        assert scores["hrv_regulation"] == pytest.approx(0.5, abs=0.01)

    def test_health_hrv_regulation_high(self):
        """High HRV (80ms) should give score > 0.5."""
        from api.routes.multimodal_ei import _compute_health_scores
        health = {"hrv_sdnn": 80.0}
        scores = _compute_health_scores(health)
        assert scores["hrv_regulation"] > 0.5

    def test_health_sleep_restoration(self):
        """sleep_score / 100.0."""
        from api.routes.multimodal_ei import _compute_health_scores
        health = {"sleep_score": 85}
        scores = _compute_health_scores(health)
        assert scores["sleep_restoration"] == pytest.approx(0.85, abs=0.01)

    def test_health_sleep_default(self):
        """No sleep_score -> default 0.5."""
        from api.routes.multimodal_ei import _compute_health_scores
        health = {}
        scores = _compute_health_scores(health)
        assert scores["sleep_restoration"] == pytest.approx(0.5, abs=0.01)

    def test_health_activity_engagement_sigmoid(self):
        """Steps sigmoid: 1/(1+exp(-0.001*(steps-5000))). At 5000 -> 0.5."""
        from api.routes.multimodal_ei import _compute_health_scores
        health = {"steps": 5000}
        scores = _compute_health_scores(health)
        assert scores["activity_engagement"] == pytest.approx(0.5, abs=0.01)

    def test_health_activity_engagement_high(self):
        """10000 steps should give score > 0.5."""
        from api.routes.multimodal_ei import _compute_health_scores
        health = {"steps": 10000}
        scores = _compute_health_scores(health)
        assert scores["activity_engagement"] > 0.5

    def test_health_mood_coherence_no_voice(self):
        """Without voice data, coherence should default to 0.5."""
        from api.routes.multimodal_ei import _compute_coherence_score
        health = {"hrv_sdnn": 60.0}
        score = _compute_coherence_score(voice_analysis=None, health_data=health)
        assert score == pytest.approx(0.5, abs=0.01)

    def test_health_mood_coherence_aligned(self):
        """Positive voice + high HRV = high coherence."""
        from api.routes.multimodal_ei import _compute_coherence_score
        voice = {"valence": 0.8}
        health = {"hrv_sdnn": 80.0}
        score = _compute_coherence_score(voice_analysis=voice, health_data=health)
        assert score > 0.5

    def test_health_mood_coherence_misaligned(self):
        """Negative voice + high HRV = lower coherence."""
        from api.routes.multimodal_ei import _compute_coherence_score
        voice = {"valence": -0.8}
        health = {"hrv_sdnn": 80.0}
        score = _compute_coherence_score(voice_analysis=voice, health_data=health)
        assert score < 0.5

    def test_all_scores_in_unit_range(self):
        """All computed scores must be in [0, 1]."""
        from api.routes.multimodal_ei import _compute_voice_scores, _compute_health_scores

        voice = {"valence": -1.0, "arousal": 1.0, "confidence": 1.0,
                 "cognitive_load_index": 0.0}
        v_scores = _compute_voice_scores(voice)
        for k, v in v_scores.items():
            assert 0.0 <= v <= 1.0, f"voice score {k} = {v} out of bounds"

        health = {"hrv_sdnn": 200.0, "sleep_score": 100, "steps": 20000}
        h_scores = _compute_health_scores(health)
        for k, v in h_scores.items():
            assert 0.0 <= v <= 1.0, f"health score {k} = {v} out of bounds"


# ---------------------------------------------------------------------------
# Integration tests: full /multimodal-ei/assess endpoint logic
# ---------------------------------------------------------------------------

class TestMultimodalEIAssess:
    """Test the assess function that ties all modalities together."""

    def test_eeg_only(self):
        """Assessment with only EEG signals should work."""
        from api.routes.multimodal_ei import _assess_multimodal_ei
        signals = np.random.randn(4, 1024) * 20
        result = _assess_multimodal_ei(
            user_id="eeg_user",
            signals=signals,
            fs=256.0,
        )
        assert result is not None
        assert "eiq_score" in result
        assert result["modalities_used"] == ["eeg"]

    def test_voice_only(self):
        """Assessment with only voice analysis should work."""
        from api.routes.multimodal_ei import _assess_multimodal_ei
        voice = {"valence": 0.5, "arousal": 0.6, "confidence": 0.8}
        result = _assess_multimodal_ei(
            user_id="voice_user",
            voice_analysis=voice,
        )
        assert result is not None
        assert "eiq_score" in result
        assert "voice" in result["modalities_used"]

    def test_health_only(self):
        """Assessment with only health data should work."""
        from api.routes.multimodal_ei import _assess_multimodal_ei
        health = {"hrv_sdnn": 55.0, "sleep_score": 80, "steps": 7000}
        result = _assess_multimodal_ei(
            user_id="health_user",
            health_data=health,
        )
        assert result is not None
        assert "eiq_score" in result
        assert "health" in result["modalities_used"]

    def test_all_three_modalities(self):
        """Assessment with EEG + voice + health should use all three."""
        from api.routes.multimodal_ei import _assess_multimodal_ei
        signals = np.random.randn(4, 1024) * 20
        voice = {"valence": 0.7, "arousal": 0.4, "confidence": 0.9}
        health = {"hrv_sdnn": 65.0, "sleep_score": 90, "steps": 8000}
        result = _assess_multimodal_ei(
            user_id="full_user",
            signals=signals,
            fs=256.0,
            voice_analysis=voice,
            health_data=health,
        )
        assert result is not None
        assert "eeg" in result["modalities_used"]
        assert "voice" in result["modalities_used"]
        assert "health" in result["modalities_used"]

    def test_result_has_component_breakdown(self):
        """Result should include a breakdown of component scores."""
        from api.routes.multimodal_ei import _assess_multimodal_ei
        voice = {"valence": 0.5, "arousal": 0.5, "confidence": 0.8}
        health = {"hrv_sdnn": 50.0}
        result = _assess_multimodal_ei(
            user_id="breakdown_user",
            voice_analysis=voice,
            health_data=health,
        )
        assert "component_scores" in result
        assert "voice_valence_clarity" in result["component_scores"]
        assert "hrv_regulation" in result["component_scores"]

    def test_no_modalities_returns_none(self):
        """No inputs should return None (nothing to compute)."""
        from api.routes.multimodal_ei import _assess_multimodal_ei
        result = _assess_multimodal_ei(user_id="empty_user")
        assert result is None

    def test_eiq_score_range(self):
        """EIQ score should always be in [0, 100]."""
        from api.routes.multimodal_ei import _assess_multimodal_ei
        for _ in range(10):
            signals = np.random.randn(4, 1024) * 20
            voice = {
                "valence": np.random.uniform(-1, 1),
                "arousal": np.random.uniform(0, 1),
                "confidence": np.random.uniform(0, 1),
            }
            health = {
                "hrv_sdnn": np.random.uniform(10, 100),
                "sleep_score": np.random.uniform(0, 100),
                "steps": np.random.uniform(0, 15000),
            }
            result = _assess_multimodal_ei(
                user_id="range_user",
                signals=signals,
                voice_analysis=voice,
                health_data=health,
            )
            assert result is not None
            assert 0.0 <= result["eiq_score"] <= 100.0

    def test_user_id_isolation(self):
        """Different user_ids should produce independent results."""
        from api.routes.multimodal_ei import _assess_multimodal_ei
        voice_high = {"valence": 0.9, "arousal": 0.9, "confidence": 1.0}
        voice_low = {"valence": 0.1, "arousal": 0.1, "confidence": 0.2}

        result_a = _assess_multimodal_ei(user_id="iso_a", voice_analysis=voice_high)
        result_b = _assess_multimodal_ei(user_id="iso_b", voice_analysis=voice_low)

        assert result_a is not None and result_b is not None
        # High voice engagement should produce higher score
        assert result_a["eiq_score"] > result_b["eiq_score"]

    def test_partial_voice_data(self):
        """Voice data with missing fields should still work using defaults."""
        from api.routes.multimodal_ei import _assess_multimodal_ei
        voice = {"valence": 0.5}  # missing arousal, confidence
        result = _assess_multimodal_ei(user_id="partial_v", voice_analysis=voice)
        assert result is not None
        assert "eiq_score" in result

    def test_partial_health_data(self):
        """Health data with only some fields should still work."""
        from api.routes.multimodal_ei import _assess_multimodal_ei
        health = {"steps": 3000}  # only steps, no HRV or sleep
        result = _assess_multimodal_ei(user_id="partial_h", health_data=health)
        assert result is not None
        assert "eiq_score" in result

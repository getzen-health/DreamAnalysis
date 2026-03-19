"""Tests for psychedelic integration companion (issue #446).

Covers: session phase detection (preparation, onset, peak, plateau,
comedown, integration), readiness assessment, emotional processing
tracking, integration scoring, safety monitoring, session profile
computation, serialization, and API route integration.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.psychedelic_integration import (
    EEGReading,
    EmotionalEvent,
    EmotionCategory,
    ReadinessLevel,
    SafetyLevel,
    SessionPhase,
    SessionProfile,
    _CLINICAL_DISCLAIMER,
    _SAFETY_RESOURCES,
    assess_readiness,
    check_session_safety,
    compute_integration_score,
    compute_session_profile,
    detect_session_phase,
    profile_to_dict,
    track_emotional_processing,
)


# ---------------------------------------------------------------------------
# Helpers: EEG readings for each phase
# ---------------------------------------------------------------------------


def _preparation_reading(**overrides) -> EEGReading:
    """Normal baseline: good alpha, normal beta, low theta."""
    defaults = dict(
        theta_fraction=0.15, alpha_fraction=0.30,
        beta_fraction=0.30, gamma_fraction=0.05,
        spectral_entropy=0.35, valence=0.1, arousal=0.3,
    )
    defaults.update(overrides)
    return EEGReading(**defaults)


def _onset_reading(**overrides) -> EEGReading:
    """Onset: alpha dropping, theta rising."""
    defaults = dict(
        theta_fraction=0.35, alpha_fraction=0.10,
        beta_fraction=0.20, gamma_fraction=0.10,
        spectral_entropy=0.50, valence=0.0, arousal=0.4,
    )
    defaults.update(overrides)
    return EEGReading(**defaults)


def _peak_reading(**overrides) -> EEGReading:
    """Peak: theta dominant, high gamma, high entropy."""
    defaults = dict(
        theta_fraction=0.50, alpha_fraction=0.05,
        beta_fraction=0.10, gamma_fraction=0.20,
        spectral_entropy=0.75, valence=-0.1, arousal=0.6,
    )
    defaults.update(overrides)
    return EEGReading(**defaults)


def _plateau_reading(**overrides) -> EEGReading:
    """Plateau: sustained theta, partial alpha recovery."""
    defaults = dict(
        theta_fraction=0.35, alpha_fraction=0.15,
        beta_fraction=0.15, gamma_fraction=0.10,
        spectral_entropy=0.55, valence=0.0, arousal=0.4,
    )
    defaults.update(overrides)
    return EEGReading(**defaults)


def _comedown_reading(**overrides) -> EEGReading:
    """Comedown: alpha recovering, theta low, beta moderate, low entropy."""
    defaults = dict(
        theta_fraction=0.18, alpha_fraction=0.22,
        beta_fraction=0.15, gamma_fraction=0.03,
        spectral_entropy=0.30, valence=0.1, arousal=0.3,
    )
    defaults.update(overrides)
    return EEGReading(**defaults)


def _integration_reading(**overrides) -> EEGReading:
    """Integration: alpha restored, theta elevated (0.20-0.30), low beta.

    Entropy at 0.46 — above comedown's 0.45 ceiling but below
    integration's 0.50 ceiling — disambiguates from comedown.
    """
    defaults = dict(
        theta_fraction=0.25, alpha_fraction=0.22,
        beta_fraction=0.22, gamma_fraction=0.03,
        spectral_entropy=0.46, valence=0.2, arousal=0.3,
    )
    defaults.update(overrides)
    return EEGReading(**defaults)


# ---------------------------------------------------------------------------
# 1. Session phase detection
# ---------------------------------------------------------------------------


class TestPhaseDetection:
    def test_detect_preparation(self):
        """Normal baseline should detect preparation phase."""
        result = detect_session_phase(_preparation_reading())
        assert result["phase"] == "preparation"
        assert result["confidence"] > 0.5
        assert "normal_alpha" in result["indicators"]
        assert "normal_beta" in result["indicators"]

    def test_detect_onset(self):
        """Alpha decrease + theta increase should detect onset."""
        result = detect_session_phase(_onset_reading())
        assert result["phase"] == "onset"
        assert result["confidence"] > 0.5
        assert "alpha_decrease" in result["indicators"]
        assert "theta_increase" in result["indicators"]

    def test_detect_peak(self):
        """Theta dominance + high gamma + high entropy should detect peak."""
        result = detect_session_phase(_peak_reading())
        assert result["phase"] == "peak"
        assert result["confidence"] > 0.5
        assert "theta_dominance" in result["indicators"]
        assert "ego_dissolution_markers" in result["indicators"]

    def test_detect_plateau(self):
        """Sustained theta + partial alpha recovery should detect plateau."""
        result = detect_session_phase(_plateau_reading())
        assert result["phase"] == "plateau"
        assert result["confidence"] > 0.5
        assert "sustained_theta" in result["indicators"]

    def test_detect_comedown(self):
        """Alpha recovery + theta decrease should detect comedown."""
        result = detect_session_phase(_comedown_reading())
        assert result["phase"] == "comedown"
        assert result["confidence"] > 0.5
        assert "alpha_recovery" in result["indicators"]

    def test_detect_integration(self):
        """Near-baseline with elevated theta should detect integration."""
        result = detect_session_phase(_integration_reading())
        assert result["phase"] == "integration"
        assert result["confidence"] > 0.5
        assert "elevated_theta" in result["indicators"]

    def test_result_includes_clinical_disclaimer(self):
        """Every phase detection should include clinical disclaimer."""
        result = detect_session_phase(_preparation_reading())
        assert result["clinical_disclaimer"] == _CLINICAL_DISCLAIMER

    def test_all_scores_included(self):
        """Result should include scores for all six phases."""
        result = detect_session_phase(_peak_reading())
        assert "all_scores" in result
        for phase in ["preparation", "onset", "peak", "plateau", "comedown", "integration"]:
            assert phase in result["all_scores"]

    def test_unknown_phase_low_signal(self):
        """All-zero reading should not crash and should return a valid phase."""
        reading = EEGReading()
        result = detect_session_phase(reading)
        assert "phase" in result
        assert result["confidence"] >= 0.0


# ---------------------------------------------------------------------------
# 2. Readiness assessment
# ---------------------------------------------------------------------------


class TestReadinessAssessment:
    def test_ready_when_calm(self):
        """Low stress, anxiety; good sleep and intention should be ready."""
        result = assess_readiness(
            valence=0.2, arousal=0.3,
            stress_index=0.2, anxiety_index=0.2,
            sleep_quality=0.8, intention_clarity=0.7,
        )
        assert result["readiness"] == "ready"
        assert result["score"] >= 0.7
        assert len(result["concerns"]) == 0

    def test_not_ready_high_stress(self):
        """High stress + anxiety + poor sleep should be not ready."""
        result = assess_readiness(
            valence=-0.5, arousal=0.8,
            stress_index=0.9, anxiety_index=0.9,
            sleep_quality=0.1, intention_clarity=0.1,
        )
        assert result["readiness"] == "not_ready"
        assert "elevated_stress" in result["concerns"]
        assert "elevated_anxiety" in result["concerns"]

    def test_caution_moderate_issues(self):
        """Some concerns but not overwhelming should yield caution."""
        result = assess_readiness(
            valence=0.0, arousal=0.4,
            stress_index=0.6, anxiety_index=0.3,
            sleep_quality=0.6, intention_clarity=0.6,
        )
        assert result["readiness"] in ("caution", "ready")
        assert "elevated_stress" in result["concerns"]

    def test_includes_set_assessment(self):
        """Readiness result should include the set assessment details."""
        result = assess_readiness(valence=0.0, arousal=0.4)
        assert "set_assessment" in result
        assert "stress" in result["set_assessment"]
        assert "anxiety" in result["set_assessment"]

    def test_recommendations_for_concerns(self):
        """Each concern should produce a recommendation."""
        result = assess_readiness(
            valence=-0.5, arousal=0.8,
            stress_index=0.8, anxiety_index=0.8,
            sleep_quality=0.1, intention_clarity=0.1,
        )
        assert len(result["recommendations"]) >= 4

    def test_includes_clinical_disclaimer(self):
        """Readiness result should include clinical disclaimer."""
        result = assess_readiness(valence=0.0, arousal=0.3)
        assert result["clinical_disclaimer"] == _CLINICAL_DISCLAIMER


# ---------------------------------------------------------------------------
# 3. Emotional processing tracking
# ---------------------------------------------------------------------------


class TestEmotionalProcessing:
    def test_empty_readings(self):
        """No readings should return empty processing result."""
        result = track_emotional_processing([])
        assert result["processing_depth"] == 0.0
        assert result["events"] == []
        assert result["total_events"] == 0

    def test_tracks_positive_emotions(self):
        """High valence + high arousal readings should surface joy."""
        readings = [
            {"valence": 0.6, "arousal": 0.7, "theta_fraction": 0.4, "spectral_entropy": 0.5},
            {"valence": 0.5, "arousal": 0.6, "theta_fraction": 0.3, "spectral_entropy": 0.4},
        ]
        result = track_emotional_processing(readings)
        assert result["total_events"] == 2
        assert "joy" in result["emotions_surfaced"]

    def test_tracks_negative_emotions(self):
        """Negative valence readings should surface negative categories."""
        readings = [
            {"valence": -0.5, "arousal": 0.7, "theta_fraction": 0.3, "spectral_entropy": 0.3},
            {"valence": -0.4, "arousal": 0.5, "theta_fraction": 0.2, "spectral_entropy": 0.2},
        ]
        result = track_emotional_processing(readings)
        assert result["total_events"] == 2
        surfaced = result["emotions_surfaced"]
        assert any(e in surfaced for e in ["anger", "fear", "sadness", "grief"])

    def test_detects_unresolved_negative(self):
        """Shallow-processed negative emotions should flag unresolved themes."""
        readings = [
            {"valence": -0.5, "arousal": 0.6, "theta_fraction": 0.1, "spectral_entropy": 0.1},
            {"valence": -0.6, "arousal": 0.5, "theta_fraction": 0.1, "spectral_entropy": 0.1},
            {"valence": -0.4, "arousal": 0.5, "theta_fraction": 0.1, "spectral_entropy": 0.1},
        ]
        result = track_emotional_processing(readings)
        assert "unprocessed_negative_emotions" in result["unresolved_themes"]

    def test_processing_depth_increases_with_theta(self):
        """Higher theta should yield deeper processing."""
        shallow = [{"valence": 0.0, "arousal": 0.3, "theta_fraction": 0.1, "spectral_entropy": 0.1}]
        deep = [{"valence": 0.0, "arousal": 0.3, "theta_fraction": 0.7, "spectral_entropy": 0.6}]
        r_shallow = track_emotional_processing(shallow)
        r_deep = track_emotional_processing(deep)
        assert r_deep["processing_depth"] > r_shallow["processing_depth"]

    def test_includes_session_summary(self):
        """Result should include a human-readable session summary."""
        readings = [{"valence": 0.3, "arousal": 0.4, "theta_fraction": 0.3, "spectral_entropy": 0.4}]
        result = track_emotional_processing(readings)
        assert "session_summary" in result
        assert len(result["session_summary"]) > 0


# ---------------------------------------------------------------------------
# 4. Integration scoring
# ---------------------------------------------------------------------------


class TestIntegrationScoring:
    def test_strong_integration(self):
        """Positive shift + deep processing + resolved should score high."""
        result = compute_integration_score(
            pre_session_valence=-0.3,
            post_session_valence=0.3,
            processing_depth=0.8,
            unresolved_count=0,
            emotional_stability=0.8,
        )
        assert result["integration_score"] >= 0.7
        assert result["assessment"] == "strong"

    def test_weak_integration(self):
        """No shift + shallow + unresolved should score low."""
        result = compute_integration_score(
            pre_session_valence=0.0,
            post_session_valence=-0.2,
            processing_depth=0.1,
            unresolved_count=3,
            emotional_stability=0.2,
        )
        assert result["integration_score"] < 0.45
        assert result["assessment"] == "needs_attention"

    def test_time_decay(self):
        """Score should decay slightly after 14+ days."""
        recent = compute_integration_score(
            pre_session_valence=-0.2, post_session_valence=0.3,
            processing_depth=0.6, days_since_session=1,
        )
        distant = compute_integration_score(
            pre_session_valence=-0.2, post_session_valence=0.3,
            processing_depth=0.6, days_since_session=30,
        )
        assert distant["integration_score"] < recent["integration_score"]

    def test_recommendations_for_unresolved(self):
        """Unresolved themes should produce recommendations."""
        result = compute_integration_score(
            pre_session_valence=0.0, post_session_valence=0.0,
            unresolved_count=2,
        )
        assert any("unresolved" in r.lower() for r in result["recommendations"])

    def test_components_included(self):
        """Result should include score components."""
        result = compute_integration_score(
            pre_session_valence=-0.1, post_session_valence=0.2,
        )
        assert "components" in result
        assert "valence_shift" in result["components"]
        assert "processing_depth" in result["components"]
        assert "resolution" in result["components"]
        assert "stability" in result["components"]

    def test_includes_clinical_disclaimer(self):
        """Integration result should include clinical disclaimer."""
        result = compute_integration_score(
            pre_session_valence=0.0, post_session_valence=0.1,
        )
        assert result["clinical_disclaimer"] == _CLINICAL_DISCLAIMER


# ---------------------------------------------------------------------------
# 5. Safety monitoring
# ---------------------------------------------------------------------------


class TestSafetyMonitoring:
    def test_safe_reading(self):
        """Normal reading should return safe level."""
        result = check_session_safety(_preparation_reading())
        assert result["safety_level"] == "safe"
        assert result["grounding_needed"] is False
        assert len(result["notes"]) == 0

    def test_distress_detection(self):
        """Extreme arousal + negative valence should trigger distress."""
        reading = EEGReading(
            arousal=0.90, valence=-0.7,
            theta_fraction=0.3, alpha_fraction=0.1,
        )
        result = check_session_safety(reading)
        assert result["safety_level"] == "distress"
        assert result["grounding_needed"] is True
        assert "Acute distress" in result["notes"][0]

    def test_concern_level(self):
        """High arousal + negative valence (below distress) should be concern."""
        reading = EEGReading(
            arousal=0.80, valence=-0.5,
            theta_fraction=0.3, alpha_fraction=0.1,
        )
        result = check_session_safety(reading)
        assert result["safety_level"] == "concern"
        assert result["grounding_needed"] is True

    def test_sustained_distress_escalation(self):
        """Multiple concerning readings should escalate to concern."""
        current = EEGReading(arousal=0.6, valence=-0.3)
        recent = [
            EEGReading(arousal=0.80, valence=-0.5),
            EEGReading(arousal=0.78, valence=-0.45),
            EEGReading(arousal=0.82, valence=-0.6),
        ]
        result = check_session_safety(current, recent)
        assert result["safety_level"] in ("concern", "distress")
        assert result["grounding_needed"] is True

    def test_always_includes_resources(self):
        """Safety result should always include safety resources."""
        result = check_session_safety(_preparation_reading())
        assert result["safety_resources"] == _SAFETY_RESOURCES
        assert result["clinical_disclaimer"] == _CLINICAL_DISCLAIMER

    def test_grounding_guidance_for_distress(self):
        """Distress level should provide specific grounding guidance."""
        reading = EEGReading(arousal=0.90, valence=-0.7)
        result = check_session_safety(reading)
        assert "GROUNDING NEEDED" in result["guidance"]
        assert "breath" in result["guidance"].lower()


# ---------------------------------------------------------------------------
# 6. Session profile computation
# ---------------------------------------------------------------------------


class TestSessionProfile:
    def test_compute_profile_basic(self):
        """Should compute a complete profile from a single reading."""
        reading = _peak_reading()
        profile = compute_session_profile(reading)
        assert isinstance(profile, SessionProfile)
        assert profile.phase in SessionPhase
        assert profile.safety in SafetyLevel
        assert profile.processing_depth >= 0.0

    def test_profile_to_dict(self):
        """profile_to_dict should serialize all required fields."""
        reading = _onset_reading()
        profile = compute_session_profile(reading)
        d = profile_to_dict(profile)
        assert "phase" in d
        assert "phase_confidence" in d
        assert "safety" in d
        assert "emotional_events" in d
        assert "processing_depth" in d
        assert "unresolved_themes" in d
        assert "grounding_needed" in d
        assert "clinical_disclaimer" in d
        assert "safety_resources" in d

    def test_profile_with_emotional_history(self):
        """Profile with emotional readings should compute processing depth."""
        reading = _plateau_reading()
        emotional = [
            {"valence": 0.3, "arousal": 0.5, "theta_fraction": 0.4, "spectral_entropy": 0.5},
            {"valence": -0.2, "arousal": 0.4, "theta_fraction": 0.3, "spectral_entropy": 0.4},
        ]
        profile = compute_session_profile(reading, emotional_readings=emotional)
        assert len(profile.emotional_events) == 2
        assert profile.processing_depth > 0.0

    def test_profile_integration_with_pre_valence(self):
        """Integration phase with pre_session_valence should compute score."""
        reading = _integration_reading()
        profile = compute_session_profile(
            reading, pre_session_valence=-0.3,
        )
        assert profile.integration_score is not None
        assert profile.integration_score >= 0.0

    def test_profile_no_integration_for_peak(self):
        """Peak phase should NOT compute integration score."""
        reading = _peak_reading()
        profile = compute_session_profile(
            reading, pre_session_valence=-0.3,
        )
        assert profile.integration_score is None


# ---------------------------------------------------------------------------
# 7. Dataclass serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_eeg_reading_to_dict(self):
        """EEGReading.to_dict should include all fields."""
        r = _peak_reading()
        d = r.to_dict()
        assert "theta_fraction" in d
        assert "alpha_fraction" in d
        assert "beta_fraction" in d
        assert "gamma_fraction" in d
        assert "spectral_entropy" in d
        assert "valence" in d
        assert "arousal" in d
        assert "timestamp" in d

    def test_eeg_reading_values_rounded(self):
        """EEGReading.to_dict should round values to 4 decimal places."""
        r = EEGReading(theta_fraction=0.123456789)
        d = r.to_dict()
        assert d["theta_fraction"] == 0.1235

    def test_emotional_event_to_dict(self):
        """EmotionalEvent.to_dict should include all fields."""
        e = EmotionalEvent(
            category=EmotionCategory.JOY,
            intensity=0.8, valence=0.6, arousal=0.7,
            processing_depth=0.5,
        )
        d = e.to_dict()
        assert d["category"] == "joy"
        assert "intensity" in d
        assert "processing_depth" in d


# ---------------------------------------------------------------------------
# 8. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_all_zeros_reading(self):
        """All-zero EEG reading should not crash."""
        reading = EEGReading()
        result = detect_session_phase(reading)
        assert "phase" in result

    def test_extreme_values(self):
        """Extreme but valid values should not crash."""
        reading = EEGReading(
            theta_fraction=1.0, alpha_fraction=0.0,
            beta_fraction=0.0, gamma_fraction=1.0,
            spectral_entropy=1.0, valence=-1.0, arousal=1.0,
        )
        result = detect_session_phase(reading)
        assert "phase" in result

    def test_safety_with_no_recent(self):
        """Safety check with no recent readings should still work."""
        result = check_session_safety(_preparation_reading(), recent_readings=None)
        assert result["safety_level"] == "safe"

    def test_integration_score_clipped(self):
        """Integration score should be clipped to 0-1."""
        result = compute_integration_score(
            pre_session_valence=-1.0, post_session_valence=1.0,
            processing_depth=1.0, unresolved_count=0,
            emotional_stability=1.0,
        )
        assert 0.0 <= result["integration_score"] <= 1.0


# ---------------------------------------------------------------------------
# 9. API route integration
# ---------------------------------------------------------------------------


class TestAPIRoutes:
    def test_status_endpoint(self):
        """GET /psychedelic/status should return ok."""
        from fastapi.testclient import TestClient
        from api.routes.psychedelic_integration import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        resp = client.get("/psychedelic/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model"] == "psychedelic_integration"
        assert "session_phases" in data
        assert len(data["session_phases"]) == 6
        assert "clinical_disclaimer" in data

    def test_phase_endpoint_peak(self):
        """POST /psychedelic/phase with peak EEG should detect peak."""
        from fastapi.testclient import TestClient
        from api.routes.psychedelic_integration import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        resp = client.post("/psychedelic/phase", json={
            "theta_fraction": 0.50,
            "alpha_fraction": 0.05,
            "beta_fraction": 0.10,
            "gamma_fraction": 0.20,
            "spectral_entropy": 0.75,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["phase"] == "peak"
        assert "clinical_disclaimer" in data

    def test_readiness_endpoint(self):
        """POST /psychedelic/readiness should return readiness assessment."""
        from fastapi.testclient import TestClient
        from api.routes.psychedelic_integration import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        resp = client.post("/psychedelic/readiness", json={
            "valence": 0.2,
            "arousal": 0.3,
            "stress_index": 0.2,
            "anxiety_index": 0.1,
            "sleep_quality": 0.8,
            "intention_clarity": 0.7,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["readiness"] == "ready"
        assert "set_assessment" in data

    def test_integration_endpoint(self):
        """POST /psychedelic/integration should return integration score."""
        from fastapi.testclient import TestClient
        from api.routes.psychedelic_integration import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        resp = client.post("/psychedelic/integration", json={
            "pre_session_valence": -0.3,
            "post_session_valence": 0.3,
            "processing_depth": 0.7,
            "unresolved_count": 0,
            "days_since_session": 3,
            "emotional_stability": 0.7,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "integration_score" in data
        assert data["integration_score"] >= 0.0
        assert "assessment" in data
        assert "clinical_disclaimer" in data

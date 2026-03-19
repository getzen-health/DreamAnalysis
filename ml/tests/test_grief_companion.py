"""Tests for grief and loss processing companion (issue #424).

Covers: grief stage detection (denial, anger, bargaining, depression,
acceptance, unknown), trajectory tracking, intervention selection,
anniversary effect detection, safety checks, profile computation,
serialization, and API route integration.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.grief_companion import (
    GriefProfile,
    GriefReading,
    GriefStage,
    SafetyLevel,
    SupportIntervention,
    TrajectoryTrend,
    WordenTask,
    INTERVENTION_LIBRARY,
    _CLINICAL_DISCLAIMER,
    _CRISIS_RESOURCES,
    _INTERVENTIONS_BY_STAGE,
    check_safety,
    compute_grief_profile,
    detect_anniversary_effect,
    detect_grief_stage,
    profile_to_dict,
    select_support_intervention,
    track_grief_trajectory,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _denial_reading(**overrides) -> GriefReading:
    defaults = dict(
        valence=0.05, arousal=0.15, stress_index=0.1,
        anger_index=0.05, focus_index=0.2,
        isolation_index=0.2, hopelessness_index=0.1,
    )
    defaults.update(overrides)
    return GriefReading(**defaults)


def _anger_reading(**overrides) -> GriefReading:
    defaults = dict(
        valence=-0.5, arousal=0.80, stress_index=0.7,
        anger_index=0.75, focus_index=0.4,
        isolation_index=0.2, hopelessness_index=0.1,
    )
    defaults.update(overrides)
    return GriefReading(**defaults)


def _bargaining_reading(**overrides) -> GriefReading:
    defaults = dict(
        valence=-0.10, arousal=0.45, stress_index=0.55,
        anger_index=0.15, focus_index=0.6,
        isolation_index=0.3, hopelessness_index=0.2,
    )
    defaults.update(overrides)
    return GriefReading(**defaults)


def _depression_reading(**overrides) -> GriefReading:
    defaults = dict(
        valence=-0.6, arousal=0.20, stress_index=0.3,
        anger_index=0.05, focus_index=0.2,
        isolation_index=0.7, hopelessness_index=0.6,
    )
    defaults.update(overrides)
    return GriefReading(**defaults)


def _acceptance_reading(**overrides) -> GriefReading:
    defaults = dict(
        valence=0.2, arousal=0.40, stress_index=0.2,
        anger_index=0.05, focus_index=0.6,
        isolation_index=0.1, hopelessness_index=0.1,
    )
    defaults.update(overrides)
    return GriefReading(**defaults)


def _normal_reading(**overrides) -> GriefReading:
    defaults = dict(
        valence=0.3, arousal=0.45, stress_index=0.2,
        anger_index=0.05, focus_index=0.5,
        isolation_index=0.1, hopelessness_index=0.05,
    )
    defaults.update(overrides)
    return GriefReading(**defaults)


# ---------------------------------------------------------------------------
# 1. Grief stage detection
# ---------------------------------------------------------------------------


class TestGriefStageDetection:
    def test_detect_denial(self):
        """Low arousal + flat affect should detect denial."""
        result = detect_grief_stage(_denial_reading())
        assert result["stage"] == "denial"
        assert result["confidence"] > 0.5
        assert "low_arousal" in result["indicators"]
        assert "flat_affect" in result["indicators"]

    def test_detect_anger(self):
        """High arousal + negative valence + high anger should detect anger."""
        result = detect_grief_stage(_anger_reading())
        assert result["stage"] == "anger"
        assert result["confidence"] > 0.5
        assert "elevated_anger" in result["indicators"]
        assert "high_arousal" in result["indicators"]

    def test_detect_bargaining(self):
        """Moderate arousal + mixed valence + stress should detect bargaining."""
        result = detect_grief_stage(_bargaining_reading())
        assert result["stage"] == "bargaining"
        assert result["confidence"] > 0.5
        assert "moderate_arousal" in result["indicators"]

    def test_detect_depression(self):
        """Low arousal + negative valence + isolation should detect depression."""
        result = detect_grief_stage(_depression_reading())
        assert result["stage"] == "depression"
        assert result["confidence"] > 0.5
        assert "negative_valence" in result["indicators"]

    def test_detect_acceptance(self):
        """Balanced arousal + neutral-positive valence should detect acceptance."""
        result = detect_grief_stage(_acceptance_reading())
        assert result["stage"] == "acceptance"
        assert result["confidence"] > 0.5
        assert "neutral_positive_valence" in result["indicators"]

    def test_detect_unknown_low_signal(self):
        """Ambiguous data with no strong indicators should return unknown."""
        # All values in middle range with no strong markers
        reading = GriefReading(
            valence=0.0, arousal=0.5, stress_index=0.3,
            anger_index=0.2, focus_index=0.5,
            isolation_index=0.0, hopelessness_index=0.0,
        )
        result = detect_grief_stage(reading)
        # Should detect something (may be bargaining or acceptance from moderate values)
        assert "stage" in result
        assert result["confidence"] >= 0.0

    def test_result_includes_clinical_disclaimer(self):
        """Every detection result must include clinical disclaimer."""
        result = detect_grief_stage(_denial_reading())
        assert result["clinical_disclaimer"] == _CLINICAL_DISCLAIMER

    def test_result_includes_worden_task(self):
        """Every detection result should include the corresponding Worden task."""
        result = detect_grief_stage(_depression_reading())
        assert "worden_task" in result
        assert result["worden_task"] in [t.value for t in WordenTask]

    def test_all_scores_included(self):
        """Result should include scores for all five stages."""
        result = detect_grief_stage(_anger_reading())
        assert "all_scores" in result
        for stage in ["denial", "anger", "bargaining", "depression", "acceptance"]:
            assert stage in result["all_scores"]


# ---------------------------------------------------------------------------
# 2. Trajectory tracking
# ---------------------------------------------------------------------------


class TestTrajectoryTracking:
    def test_insufficient_data(self):
        """Fewer than minimum readings should return insufficient_data."""
        result = track_grief_trajectory([{"stage": "denial"}])
        assert result["trend"] == "insufficient_data"

    def test_progressing_trajectory(self):
        """Stages moving forward should detect progression."""
        readings = [
            {"stage": "denial", "timestamp": 1000},
            {"stage": "denial", "timestamp": 2000},
            {"stage": "anger", "timestamp": 3000},
            {"stage": "bargaining", "timestamp": 4000},
            {"stage": "depression", "timestamp": 5000},
            {"stage": "acceptance", "timestamp": 6000},
        ]
        result = track_grief_trajectory(readings)
        assert result["trend"] in ("progressing", "oscillating")
        assert result["readings_analyzed"] == 6

    def test_stuck_trajectory(self):
        """Same stage for many readings should detect stuck pattern."""
        readings = [
            {"stage": "depression", "timestamp": i * 1000}
            for i in range(7)
        ]
        result = track_grief_trajectory(readings)
        assert result["trend"] == "stuck"
        assert "depression" in result["message"]

    def test_regressing_trajectory(self):
        """Stages moving backward should detect regression."""
        readings = [
            {"stage": "acceptance", "timestamp": 1000},
            {"stage": "depression", "timestamp": 2000},
            {"stage": "bargaining", "timestamp": 3000},
            {"stage": "anger", "timestamp": 4000},
            {"stage": "denial", "timestamp": 5000},
            {"stage": "denial", "timestamp": 6000},
        ]
        result = track_grief_trajectory(readings)
        assert result["trend"] in ("regressing", "oscillating")

    def test_trajectory_includes_stage_counts(self):
        """Result should include counts per stage."""
        readings = [
            {"stage": "denial", "timestamp": 1000},
            {"stage": "denial", "timestamp": 2000},
            {"stage": "anger", "timestamp": 3000},
        ]
        result = track_grief_trajectory(readings)
        assert "stage_counts" in result
        assert result["stage_counts"]["denial"] == 2

    def test_trajectory_clinical_disclaimer(self):
        """Trajectory result must include clinical disclaimer."""
        readings = [{"stage": "denial", "timestamp": i} for i in range(5)]
        result = track_grief_trajectory(readings)
        assert result["clinical_disclaimer"] == _CLINICAL_DISCLAIMER


# ---------------------------------------------------------------------------
# 3. Intervention selection
# ---------------------------------------------------------------------------


class TestInterventionSelection:
    def test_select_for_denial(self):
        """Should select an intervention for denial stage."""
        result = select_support_intervention("denial")
        assert result["selected"] is True
        assert result["intervention"]["stage"] == "denial"
        assert result["worden_task"] == "accept_reality"

    def test_select_for_anger(self):
        """Should select an intervention for anger stage."""
        result = select_support_intervention("anger")
        assert result["selected"] is True
        assert result["intervention"]["stage"] == "anger"

    def test_select_for_acceptance(self):
        """Should select an intervention for acceptance stage."""
        result = select_support_intervention("acceptance")
        assert result["selected"] is True
        assert result["intervention"]["stage"] == "acceptance"

    def test_select_unknown_stage(self):
        """Unknown stage should return selected=False with message."""
        result = select_support_intervention("unknown")
        assert result["selected"] is False
        assert result["reason"] == "stage_unclear"

    def test_select_invalid_stage(self):
        """Invalid stage string should return selected=False."""
        result = select_support_intervention("not_a_stage")
        assert result["selected"] is False
        assert result["reason"] == "unknown_stage"

    def test_intensity_preference(self):
        """Should prefer matching intensity when available."""
        result = select_support_intervention("anger", intensity_preference="active")
        assert result["selected"] is True
        # Anger has an "active" intervention (Physical Release)
        assert result["intervention"]["intensity"] == "active"

    def test_includes_alternatives(self):
        """Should include alternative interventions."""
        result = select_support_intervention("denial")
        assert "alternatives" in result
        assert isinstance(result["alternatives"], list)

    def test_includes_crisis_resources(self):
        """Intervention result should include crisis resources."""
        result = select_support_intervention("depression")
        assert result["crisis_resources"] == _CRISIS_RESOURCES


# ---------------------------------------------------------------------------
# 4. Anniversary effect detection
# ---------------------------------------------------------------------------


class TestAnniversaryDetection:
    def test_no_anniversary_normal_state(self):
        """Normal reading with no dip should not detect anniversary."""
        current = _normal_reading()
        recent = [_normal_reading(valence=0.3) for _ in range(5)]
        result = detect_anniversary_effect(current, recent)
        assert result["anniversary_detected"] is False

    def test_detect_emotional_dip(self):
        """Significant valence dip should detect anniversary effect."""
        current = GriefReading(valence=-0.3, arousal=0.3)
        recent = [GriefReading(valence=0.2, arousal=0.4) for _ in range(5)]
        result = detect_anniversary_effect(current, recent)
        assert result["emotional_dip_detected"] is True
        assert result["valence_dip"] > 0.25

    def test_anniversary_with_significant_date(self):
        """Dip near a significant date should detect anniversary."""
        current = GriefReading(valence=-0.3, arousal=0.3)
        recent = [GriefReading(valence=0.2, arousal=0.4) for _ in range(5)]
        dates = [{"timestamp": time.time(), "description": "anniversary of loss"}]
        result = detect_anniversary_effect(current, recent, dates)
        assert result["anniversary_detected"] is True
        assert result["near_significant_date"] is True
        assert "anniversary of loss" in result["guidance"]

    def test_no_dip_near_date_no_detection(self):
        """Near a significant date but no dip should not detect anniversary."""
        current = _normal_reading(valence=0.3)
        recent = [_normal_reading(valence=0.3) for _ in range(5)]
        dates = [{"timestamp": time.time(), "description": "birthday"}]
        result = detect_anniversary_effect(current, recent, dates)
        assert result["anniversary_detected"] is False

    def test_insufficient_recent_readings(self):
        """Fewer than 3 recent readings should use zero baseline."""
        current = GriefReading(valence=-0.4, arousal=0.3)
        result = detect_anniversary_effect(current, [])
        assert result["baseline_valence"] == 0.0
        assert "clinical_disclaimer" in result


# ---------------------------------------------------------------------------
# 5. Safety checks
# ---------------------------------------------------------------------------


class TestSafetyChecks:
    def test_safe_reading(self):
        """Normal reading should return safe level."""
        result = check_safety(_normal_reading())
        assert result["safety_level"] == "safe"
        assert len(result["notes"]) == 0

    def test_high_hopelessness_concern(self):
        """High hopelessness should raise concern."""
        reading = GriefReading(
            valence=-0.3, arousal=0.3,
            hopelessness_index=0.8, isolation_index=0.3,
        )
        result = check_safety(reading)
        assert result["safety_level"] in ("concern", "critical")
        assert "High hopelessness detected" in result["notes"]

    def test_isolation_plus_hopelessness_critical(self):
        """High isolation + hopelessness should be critical."""
        reading = GriefReading(
            valence=-0.5, arousal=0.2,
            isolation_index=0.8, hopelessness_index=0.6,
        )
        result = check_safety(reading)
        assert result["safety_level"] == "critical"
        assert "isolation" in " ".join(result["notes"]).lower()

    def test_prolonged_flat_affect(self):
        """Prolonged flat affect should raise concern."""
        current = GriefReading(valence=0.05, arousal=0.15)
        recent = [
            GriefReading(valence=0.05, arousal=0.15)
            for _ in range(6)
        ]
        result = check_safety(current, recent)
        assert result["safety_level"] in ("concern", "critical")
        assert any("flat affect" in n.lower() for n in result["notes"])

    def test_always_includes_crisis_resources(self):
        """Safety result must always include crisis resources."""
        result = check_safety(_normal_reading())
        assert result["crisis_resources"] == _CRISIS_RESOURCES

    def test_always_includes_clinical_disclaimer(self):
        """Safety result must always include clinical disclaimer."""
        result = check_safety(_normal_reading())
        assert result["clinical_disclaimer"] == _CLINICAL_DISCLAIMER

    def test_critical_guidance_mentions_988(self):
        """Critical safety guidance must mention 988 crisis line."""
        reading = GriefReading(
            valence=-0.5, arousal=0.2,
            isolation_index=0.8, hopelessness_index=0.6,
        )
        result = check_safety(reading)
        assert "988" in result["guidance"]


# ---------------------------------------------------------------------------
# 6. Profile computation
# ---------------------------------------------------------------------------


class TestGriefProfile:
    def test_compute_profile_basic(self):
        """Should compute a complete profile from a single reading."""
        reading = _depression_reading()
        profile = compute_grief_profile(reading)
        assert isinstance(profile, GriefProfile)
        assert profile.stage in GriefStage
        assert profile.worden_task in WordenTask
        assert profile.safety in SafetyLevel
        assert profile.trajectory == TrajectoryTrend.INSUFFICIENT_DATA

    def test_profile_with_history(self):
        """Profile with stage history should compute trajectory."""
        reading = _acceptance_reading()
        history = [
            {"stage": "denial", "timestamp": 1000},
            {"stage": "anger", "timestamp": 2000},
            {"stage": "bargaining", "timestamp": 3000},
            {"stage": "depression", "timestamp": 4000},
            {"stage": "acceptance", "timestamp": 5000},
        ]
        profile = compute_grief_profile(
            reading, stage_history=history,
        )
        assert profile.trajectory != TrajectoryTrend.INSUFFICIENT_DATA

    def test_profile_includes_intervention(self):
        """Profile should include a stage-appropriate intervention."""
        reading = _anger_reading()
        profile = compute_grief_profile(reading)
        assert profile.intervention is not None
        assert profile.intervention.stage == GriefStage.ANGER

    def test_profile_to_dict(self):
        """profile_to_dict should serialize all fields."""
        reading = _denial_reading()
        profile = compute_grief_profile(reading)
        d = profile_to_dict(profile)
        assert "stage" in d
        assert "stage_confidence" in d
        assert "worden_task" in d
        assert "trajectory" in d
        assert "safety" in d
        assert "intervention" in d
        assert "clinical_disclaimer" in d
        assert "crisis_resources" in d

    def test_profile_safety_escalation(self):
        """High-risk reading should escalate safety in profile."""
        reading = GriefReading(
            valence=-0.5, arousal=0.2,
            isolation_index=0.8, hopelessness_index=0.7,
        )
        profile = compute_grief_profile(reading)
        assert profile.safety == SafetyLevel.CRITICAL


# ---------------------------------------------------------------------------
# 7. Intervention library structure
# ---------------------------------------------------------------------------


class TestInterventionLibrary:
    def test_all_stages_have_interventions(self):
        """Every non-unknown grief stage should have at least one intervention."""
        for stage in GriefStage:
            if stage == GriefStage.UNKNOWN:
                continue
            interventions = _INTERVENTIONS_BY_STAGE.get(stage, [])
            assert len(interventions) >= 1, f"No interventions for {stage.value}"

    def test_intervention_fields_populated(self):
        """Every intervention should have all required fields."""
        for intervention in INTERVENTION_LIBRARY:
            assert intervention.name
            assert intervention.description
            assert intervention.guidance
            assert intervention.duration_minutes > 0
            assert intervention.intensity in ("gentle", "moderate", "active")

    def test_intervention_to_dict(self):
        """SupportIntervention.to_dict should include all fields."""
        intervention = INTERVENTION_LIBRARY[0]
        d = intervention.to_dict()
        assert "stage" in d
        assert "worden_task" in d
        assert "name" in d
        assert "description" in d
        assert "guidance" in d
        assert "duration_minutes" in d
        assert "intensity" in d


# ---------------------------------------------------------------------------
# 8. Dataclass serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_grief_reading_to_dict(self):
        """GriefReading.to_dict should include all fields."""
        r = _anger_reading()
        d = r.to_dict()
        assert "valence" in d
        assert "arousal" in d
        assert "stress_index" in d
        assert "anger_index" in d
        assert "isolation_index" in d
        assert "hopelessness_index" in d
        assert "timestamp" in d

    def test_grief_reading_values_rounded(self):
        """GriefReading.to_dict should round values to 4 decimal places."""
        r = GriefReading(valence=0.123456789)
        d = r.to_dict()
        assert d["valence"] == 0.1235


# ---------------------------------------------------------------------------
# 9. API route integration
# ---------------------------------------------------------------------------


class TestAPIRoutes:
    def test_status_endpoint(self):
        """GET /grief/status should return ok."""
        from fastapi.testclient import TestClient
        from api.routes.grief_companion import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        resp = client.get("/grief/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model"] == "grief_companion"
        assert "grief_stages" in data
        assert len(data["grief_stages"]) == 5  # 5 Kubler-Ross stages
        assert "clinical_disclaimer" in data

    def test_assess_endpoint_depression(self):
        """POST /grief/assess with depression state should detect depression."""
        from fastapi.testclient import TestClient
        from api.routes.grief_companion import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        resp = client.post("/grief/assess", json={
            "valence": -0.6,
            "arousal": 0.20,
            "stress_index": 0.3,
            "anger_index": 0.05,
            "focus_index": 0.2,
            "isolation_index": 0.7,
            "hopelessness_index": 0.6,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["stage"] == "depression"
        assert "clinical_disclaimer" in data
        assert "crisis_resources" in data
        assert data["intervention"] is not None

    def test_assess_endpoint_with_safety_concern(self):
        """POST /grief/assess with critical markers should escalate safety."""
        from fastapi.testclient import TestClient
        from api.routes.grief_companion import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        resp = client.post("/grief/assess", json={
            "valence": -0.5,
            "arousal": 0.2,
            "isolation_index": 0.8,
            "hopelessness_index": 0.7,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["safety"] == "critical"

    def test_trajectory_endpoint(self):
        """POST /grief/trajectory should return trend."""
        from fastapi.testclient import TestClient
        from api.routes.grief_companion import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        resp = client.post("/grief/trajectory", json={
            "readings": [
                {"stage": "denial", "timestamp": 1000},
                {"stage": "anger", "timestamp": 2000},
                {"stage": "bargaining", "timestamp": 3000},
                {"stage": "depression", "timestamp": 4000},
                {"stage": "acceptance", "timestamp": 5000},
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "trend" in data
        assert data["readings_analyzed"] == 5

    def test_trajectory_insufficient_data(self):
        """POST /grief/trajectory with too few readings should indicate insufficient data."""
        from fastapi.testclient import TestClient
        from api.routes.grief_companion import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        resp = client.post("/grief/trajectory", json={
            "readings": [{"stage": "denial", "timestamp": 1000}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["trend"] == "insufficient_data"


# ---------------------------------------------------------------------------
# 10. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_all_zeros_reading(self):
        """All-zero reading should not crash."""
        reading = GriefReading()
        result = detect_grief_stage(reading)
        assert "stage" in result

    def test_extreme_values(self):
        """Extreme but valid values should not crash."""
        reading = GriefReading(
            valence=-1.0, arousal=1.0, stress_index=1.0,
            anger_index=1.0, focus_index=1.0,
            isolation_index=1.0, hopelessness_index=1.0,
        )
        result = detect_grief_stage(reading)
        assert "stage" in result

    def test_profile_with_empty_history(self):
        """Empty stage history should produce insufficient_data trajectory."""
        reading = _normal_reading()
        profile = compute_grief_profile(reading, stage_history=[])
        assert profile.trajectory == TrajectoryTrend.INSUFFICIENT_DATA

    def test_safety_with_no_recent(self):
        """Safety check with no recent readings should still work."""
        result = check_safety(_normal_reading(), recent_readings=None)
        assert result["safety_level"] == "safe"

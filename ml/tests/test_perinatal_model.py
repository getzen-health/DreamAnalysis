"""Tests for perinatal emotional intelligence (issue #451).

Covers: trimester baselines, PPD risk scoring, blues-vs-PPD distinction,
hormonal emotional pattern tracking, bonding readiness assessment,
partner support detection, profile computation, serialization,
and API route integration.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.perinatal_model import (
    BluesVsPPD,
    BondingLevel,
    HormonalPhase,
    PPDRiskLevel,
    PPDRiskScore,
    PerinatalPhase,
    PerinatalProfile,
    PerinatalReading,
    PerinatalState,
    _CLINICAL_DISCLAIMER,
    _CRISIS_RESOURCES,
    assess_bonding_readiness,
    compute_perinatal_profile,
    compute_trimester_baseline,
    distinguish_blues_vs_ppd,
    profile_to_dict,
    score_ppd_risk,
    track_hormonal_emotional_pattern,
    _detect_partner_support,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _healthy_reading(**overrides) -> PerinatalReading:
    defaults = dict(
        valence=0.2, arousal=0.4, stress_index=0.2,
        anxiety_index=0.2, irritability_index=0.15,
        fatigue_index=0.3, tearfulness_index=0.1,
        bonding_response=0.7,
    )
    defaults.update(overrides)
    return PerinatalReading(**defaults)


def _distressed_reading(**overrides) -> PerinatalReading:
    defaults = dict(
        valence=-0.6, arousal=0.2, stress_index=0.7,
        anxiety_index=0.7, irritability_index=0.6,
        fatigue_index=0.8, tearfulness_index=0.7,
        bonding_response=0.2,
    )
    defaults.update(overrides)
    return PerinatalReading(**defaults)


def _mild_blues_reading(**overrides) -> PerinatalReading:
    defaults = dict(
        valence=-0.1, arousal=0.4, stress_index=0.3,
        anxiety_index=0.35, irritability_index=0.25,
        fatigue_index=0.6, tearfulness_index=0.4,
        bonding_response=0.6,
    )
    defaults.update(overrides)
    return PerinatalReading(**defaults)


# ---------------------------------------------------------------------------
# 1. PerinatalState derivation
# ---------------------------------------------------------------------------


class TestPerinatalState:
    def test_first_trimester(self):
        """Weeks 1-13 should be first trimester."""
        state = PerinatalState(weeks_pregnant=8)
        assert state.trimester == 1
        assert state.phase == PerinatalPhase.FIRST_TRIMESTER

    def test_second_trimester(self):
        """Weeks 14-27 should be second trimester."""
        state = PerinatalState(weeks_pregnant=20)
        assert state.trimester == 2
        assert state.phase == PerinatalPhase.SECOND_TRIMESTER

    def test_third_trimester(self):
        """Weeks 28-40 should be third trimester."""
        state = PerinatalState(weeks_pregnant=35)
        assert state.trimester == 3
        assert state.phase == PerinatalPhase.THIRD_TRIMESTER

    def test_postpartum_early(self):
        """0-2 weeks postpartum should be early postpartum."""
        state = PerinatalState(weeks_postpartum=1)
        assert state.trimester is None
        assert state.phase == PerinatalPhase.POSTPARTUM_EARLY

    def test_postpartum_late(self):
        """3+ weeks postpartum should be late postpartum."""
        state = PerinatalState(weeks_postpartum=6)
        assert state.trimester is None
        assert state.phase == PerinatalPhase.POSTPARTUM_LATE


# ---------------------------------------------------------------------------
# 2. Trimester baselines
# ---------------------------------------------------------------------------


class TestTrimesterBaseline:
    def test_first_trimester_baseline(self):
        """First trimester should report elevated anxiety and fatigue norms."""
        state = PerinatalState(weeks_pregnant=8)
        result = compute_trimester_baseline(state)
        assert result["phase"] == "first_trimester"
        assert "anxiety" in result["baselines"]
        # Expected anxiety mean is 0.50 in first trimester
        assert result["baselines"]["anxiety"]["expected_mean"] == 0.5

    def test_second_trimester_baseline(self):
        """Second trimester should report lower anxiety (honeymoon)."""
        state = PerinatalState(weeks_pregnant=20)
        result = compute_trimester_baseline(state)
        assert result["phase"] == "second_trimester"
        anxiety_mean = result["baselines"]["anxiety"]["expected_mean"]
        assert anxiety_mean < 0.3  # reduced from first trimester

    def test_unknown_phase_returns_empty_baselines(self):
        """Unknown phase should return empty baselines with note."""
        state = PerinatalState()
        result = compute_trimester_baseline(state)
        assert result["baselines"] == {}
        assert "Unknown" in result["note"]


# ---------------------------------------------------------------------------
# 3. PPD risk scoring
# ---------------------------------------------------------------------------


class TestPPDRiskScoring:
    def test_low_risk_healthy_reading(self):
        """Healthy reading should produce low PPD risk."""
        reading = _healthy_reading()
        state = PerinatalState(weeks_postpartum=4)
        risk = score_ppd_risk(reading, state)
        assert risk.risk_level == PPDRiskLevel.LOW
        assert risk.epds_proxy_score < 9

    def test_high_risk_distressed_reading(self):
        """Distressed reading should produce high or critical risk."""
        reading = _distressed_reading()
        state = PerinatalState(weeks_postpartum=6)
        risk = score_ppd_risk(reading, state)
        assert risk.risk_level in (PPDRiskLevel.HIGH, PPDRiskLevel.CRITICAL)
        assert risk.epds_proxy_score >= 13

    def test_persistence_amplifies_score(self):
        """Persistent negative readings should amplify the score."""
        reading = _distressed_reading()
        state = PerinatalState(weeks_postpartum=6)
        recent = [_distressed_reading() for _ in range(5)]

        risk_with = score_ppd_risk(reading, state, recent)
        risk_without = score_ppd_risk(reading, state)
        # Persistence should amplify
        assert risk_with.epds_proxy_score >= risk_without.epds_proxy_score

    def test_indicators_populated(self):
        """Risk score should include relevant indicators."""
        reading = _distressed_reading()
        state = PerinatalState(weeks_postpartum=4)
        risk = score_ppd_risk(reading, state)
        assert len(risk.indicators) > 0
        assert "persistent_negative_mood" in risk.indicators

    def test_clinical_disclaimer_present(self):
        """Risk score must include clinical disclaimer."""
        reading = _healthy_reading()
        state = PerinatalState(weeks_postpartum=4)
        risk = score_ppd_risk(reading, state)
        assert risk.clinical_disclaimer == _CLINICAL_DISCLAIMER

    def test_epds_score_clamped(self):
        """Score should never exceed 30."""
        reading = _distressed_reading(
            valence=-1.0, anxiety_index=1.0, stress_index=1.0,
            tearfulness_index=1.0, fatigue_index=1.0, bonding_response=0.0,
        )
        state = PerinatalState(weeks_postpartum=4)
        risk = score_ppd_risk(reading, state)
        assert risk.epds_proxy_score <= 30.0


# ---------------------------------------------------------------------------
# 4. Blues vs PPD distinction
# ---------------------------------------------------------------------------


class TestBluesVsPPD:
    def test_early_mild_is_blues(self):
        """Mild distress within 2 weeks postpartum should be classified as blues."""
        reading = _mild_blues_reading()
        state = PerinatalState(weeks_postpartum=1)
        result = distinguish_blues_vs_ppd(reading, state)
        assert result["classification"] == "baby_blues"

    def test_late_severe_is_ppd(self):
        """Severe distress after 2 weeks should suggest PPD."""
        reading = _distressed_reading()
        state = PerinatalState(weeks_postpartum=8)
        result = distinguish_blues_vs_ppd(reading, state)
        assert result["classification"] in ("possible_ppd", "likely_ppd")

    def test_not_applicable_during_pregnancy(self):
        """Blues vs PPD should be not_applicable during pregnancy."""
        reading = _healthy_reading()
        state = PerinatalState(weeks_pregnant=20)
        result = distinguish_blues_vs_ppd(reading, state)
        assert result["classification"] == "not_applicable"

    def test_includes_clinical_disclaimer(self):
        """Result must include clinical disclaimer."""
        reading = _mild_blues_reading()
        state = PerinatalState(weeks_postpartum=1)
        result = distinguish_blues_vs_ppd(reading, state)
        assert result["clinical_disclaimer"] == _CLINICAL_DISCLAIMER


# ---------------------------------------------------------------------------
# 5. Hormonal emotional pattern tracking
# ---------------------------------------------------------------------------


class TestHormonalPattern:
    def test_pregnancy_rising(self):
        """During pregnancy, hormonal phase should be pregnancy_rising."""
        state = PerinatalState(weeks_pregnant=20)
        reading = _healthy_reading()
        result = track_hormonal_emotional_pattern(state, reading)
        assert result["hormonal_phase"] == "pregnancy_rising"
        assert result["vulnerability_score"] < 0.5

    def test_acute_drop_postpartum(self):
        """First week postpartum should be acute hormonal drop."""
        state = PerinatalState(weeks_postpartum=0)
        reading = _mild_blues_reading()
        result = track_hormonal_emotional_pattern(state, reading)
        assert result["hormonal_phase"] == "acute_drop"
        assert result["vulnerability_score"] > 0.8

    def test_mood_concern_during_acute_drop(self):
        """Very low valence during acute drop should flag concern."""
        state = PerinatalState(weeks_postpartum=0)
        reading = _distressed_reading()
        result = track_hormonal_emotional_pattern(state, reading)
        assert result["mood_concern"] is True

    def test_stabilized_phase(self):
        """12+ weeks postpartum should be stabilized."""
        state = PerinatalState(weeks_postpartum=14)
        reading = _healthy_reading()
        result = track_hormonal_emotional_pattern(state, reading)
        assert result["hormonal_phase"] == "stabilized"


# ---------------------------------------------------------------------------
# 6. Bonding readiness assessment
# ---------------------------------------------------------------------------


class TestBondingReadiness:
    def test_strong_bonding(self):
        """High bonding response should yield strong level."""
        reading = _healthy_reading(bonding_response=0.8)
        state = PerinatalState(weeks_postpartum=4)
        result = assess_bonding_readiness(reading, state)
        assert result["bonding_level"] == "strong"
        assert result["bonding_score"] >= 0.7

    def test_concern_bonding(self):
        """Very low bonding response should yield concern level."""
        reading = _distressed_reading(bonding_response=0.1, valence=-0.5, arousal=0.2)
        state = PerinatalState(weeks_postpartum=4)
        result = assess_bonding_readiness(reading, state)
        assert result["bonding_level"] == "concern"

    def test_bonding_trend_detection(self):
        """Should detect improving/declining trend from recent readings."""
        reading = _healthy_reading(bonding_response=0.8)
        state = PerinatalState(weeks_postpartum=4)
        recent = [_healthy_reading(bonding_response=0.4) for _ in range(3)]
        result = assess_bonding_readiness(reading, state, recent)
        assert result["trend"] == "improving"


# ---------------------------------------------------------------------------
# 7. Partner support detection
# ---------------------------------------------------------------------------


class TestPartnerSupport:
    def test_no_partner_data(self):
        """No partner data should return default values."""
        reading = _healthy_reading()
        result = _detect_partner_support(reading)
        assert result["partner_data_available"] is False
        assert result["support_score"] == 0.5

    def test_supportive_partner(self):
        """Positive partner valence + low stress should boost support score."""
        reading = _healthy_reading(partner_valence=0.5, partner_stress=0.2)
        result = _detect_partner_support(reading)
        assert result["partner_data_available"] is True
        assert result["support_score"] > 0.7
        assert result["stress_detected"] is False

    def test_stressed_partner(self):
        """Negative partner valence + high stress should detect stress."""
        reading = _healthy_reading(partner_valence=-0.5, partner_stress=0.8)
        result = _detect_partner_support(reading)
        assert result["stress_detected"] is True
        assert result["support_score"] < 0.3


# ---------------------------------------------------------------------------
# 8. Full profile computation
# ---------------------------------------------------------------------------


class TestPerinatalProfile:
    def test_compute_profile_pregnancy(self):
        """Should compute a complete profile during pregnancy."""
        reading = _healthy_reading()
        state = PerinatalState(weeks_pregnant=20)
        profile = compute_perinatal_profile(reading, state)
        assert isinstance(profile, PerinatalProfile)
        assert profile.phase == PerinatalPhase.SECOND_TRIMESTER
        assert profile.ppd_risk.risk_level == PPDRiskLevel.LOW

    def test_compute_profile_postpartum_distressed(self):
        """Distressed postpartum reading should produce high-risk profile."""
        reading = _distressed_reading()
        state = PerinatalState(weeks_postpartum=8)
        profile = compute_perinatal_profile(reading, state)
        assert profile.ppd_risk.risk_level in (PPDRiskLevel.HIGH, PPDRiskLevel.CRITICAL)
        assert profile.blues_vs_ppd in (BluesVsPPD.POSSIBLE_PPD, BluesVsPPD.LIKELY_PPD)

    def test_profile_to_dict_keys(self):
        """profile_to_dict should include all expected keys."""
        reading = _healthy_reading()
        state = PerinatalState(weeks_pregnant=10)
        profile = compute_perinatal_profile(reading, state)
        d = profile_to_dict(profile)
        assert "phase" in d
        assert "trimester" in d
        assert "ppd_risk" in d
        assert "blues_vs_ppd" in d
        assert "hormonal_phase" in d
        assert "bonding" in d
        assert "partner_support" in d
        assert "clinical_disclaimer" in d
        assert "crisis_resources" in d


# ---------------------------------------------------------------------------
# 9. Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_reading_to_dict(self):
        """PerinatalReading.to_dict should include all fields."""
        r = _healthy_reading()
        d = r.to_dict()
        assert "valence" in d
        assert "arousal" in d
        assert "anxiety_index" in d
        assert "bonding_response" in d
        assert "partner_valence" in d  # None is acceptable

    def test_reading_values_rounded(self):
        """PerinatalReading.to_dict should round to 4 decimal places."""
        r = PerinatalReading(valence=0.123456789)
        d = r.to_dict()
        assert d["valence"] == 0.1235


# ---------------------------------------------------------------------------
# 10. API route integration
# ---------------------------------------------------------------------------


class TestAPIRoutes:
    def test_status_endpoint(self):
        """GET /perinatal/status should return ok."""
        from fastapi.testclient import TestClient
        from api.routes.perinatal import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        resp = client.get("/perinatal/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model"] == "perinatal_emotional_intelligence"
        assert "phases" in data
        assert "risk_levels" in data
        assert "clinical_disclaimer" in data

    def test_assess_endpoint_pregnancy(self):
        """POST /perinatal/assess during pregnancy should return profile."""
        from fastapi.testclient import TestClient
        from api.routes.perinatal import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        resp = client.post("/perinatal/assess", json={
            "reading": {
                "valence": 0.2,
                "arousal": 0.4,
                "stress_index": 0.2,
                "anxiety_index": 0.2,
            },
            "weeks_pregnant": 20,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["phase"] == "second_trimester"
        assert data["trimester"] == 2
        assert "ppd_risk" in data
        assert "clinical_disclaimer" in data

    def test_assess_endpoint_postpartum(self):
        """POST /perinatal/assess postpartum should include blues_vs_ppd."""
        from fastapi.testclient import TestClient
        from api.routes.perinatal import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        resp = client.post("/perinatal/assess", json={
            "reading": {
                "valence": -0.5,
                "arousal": 0.2,
                "stress_index": 0.6,
                "anxiety_index": 0.7,
                "tearfulness_index": 0.6,
                "bonding_response": 0.3,
            },
            "weeks_postpartum": 6,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["blues_vs_ppd"] in ("possible_ppd", "likely_ppd")

    def test_ppd_risk_endpoint(self):
        """POST /perinatal/ppd-risk should return risk score."""
        from fastapi.testclient import TestClient
        from api.routes.perinatal import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        resp = client.post("/perinatal/ppd-risk", json={
            "reading": {
                "valence": -0.6,
                "arousal": 0.2,
                "stress_index": 0.7,
                "anxiety_index": 0.7,
            },
            "weeks_postpartum": 8,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "epds_proxy_score" in data
        assert "risk_level" in data
        assert "recommendation" in data
        assert "clinical_disclaimer" in data
        assert "crisis_resources" in data


# ---------------------------------------------------------------------------
# 11. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_all_zeros_reading(self):
        """All-zero reading should not crash."""
        reading = PerinatalReading()
        state = PerinatalState(weeks_postpartum=4)
        risk = score_ppd_risk(reading, state)
        assert risk.epds_proxy_score >= 0

    def test_extreme_values(self):
        """Extreme but valid values should not crash."""
        reading = PerinatalReading(
            valence=-1.0, arousal=1.0, stress_index=1.0,
            anxiety_index=1.0, irritability_index=1.0,
            fatigue_index=1.0, tearfulness_index=1.0,
            bonding_response=0.0,
        )
        state = PerinatalState(weeks_postpartum=4)
        profile = compute_perinatal_profile(reading, state)
        assert profile.ppd_risk.epds_proxy_score <= 30.0

    def test_week_zero_pregnancy(self):
        """Week 0 pregnancy should be first trimester."""
        state = PerinatalState(weeks_pregnant=0)
        assert state.trimester == 1
        assert state.phase == PerinatalPhase.FIRST_TRIMESTER

    def test_profile_with_empty_recent_readings(self):
        """Empty recent_readings list should not crash."""
        reading = _healthy_reading()
        state = PerinatalState(weeks_postpartum=4)
        profile = compute_perinatal_profile(reading, state, recent_readings=[])
        assert isinstance(profile, PerinatalProfile)

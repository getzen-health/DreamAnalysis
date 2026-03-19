"""Tests for clinical bridge: FHIR mapping, consent management, and API routes."""
import os
import sys
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.clinical_bridge import (
    CONSENT_DATA_TYPES,
    NDW_CODES,
    _reset_consent_store,
    check_consent,
    fhir_bundle_to_dict,
    generate_clinical_summary_pdf_data,
    generate_session_prep,
    get_consent_audit,
    manage_consent,
    map_emotion_to_fhir,
    map_mood_to_fhir,
    map_sleep_to_fhir,
    map_voice_to_fhir,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def clean_consent_store():
    """Reset consent store before every test."""
    _reset_consent_store()
    yield
    _reset_consent_store()


@pytest.fixture
def patient_id():
    return "patient-001"


@pytest.fixture
def emotion_session():
    return {
        "valence": 0.35,
        "arousal": 0.6,
        "stress_index": 0.45,
        "focus_index": 0.7,
        "relaxation_index": 0.3,
        "frontal_asymmetry": 0.12,
        "emotion": "happy",
        "timestamp": "2026-03-19T12:00:00Z",
    }


@pytest.fixture
def sleep_session():
    return {
        "sleep_efficiency": 0.88,
        "n3_pct": 0.22,
        "rem_pct": 0.25,
        "hrv_ms": 45.0,
        "quality_score": 72.0,
        "timestamp": "2026-03-19T07:00:00Z",
    }


@pytest.fixture
def voice_screening():
    return {
        "depression_risk": 0.35,
        "anxiety_risk": 0.2,
        "timestamp": "2026-03-19T10:00:00Z",
    }


# ── Emotion mapping tests ────────────────────────────────────────────


def test_map_emotion_to_fhir_returns_observations(patient_id, emotion_session):
    obs = map_emotion_to_fhir(patient_id, emotion_session, session_id="s1")
    assert len(obs) >= 5
    for o in obs:
        assert o["resourceType"] == "Observation"
        assert o["subject"]["reference"] == f"Patient/{patient_id}"
        assert o["status"] == "final"


def test_emotion_fhir_contains_faa(patient_id, emotion_session):
    obs = map_emotion_to_fhir(patient_id, emotion_session, session_id="s1")
    faa_obs = [o for o in obs if "faa" in o["id"]]
    assert len(faa_obs) == 1
    assert faa_obs[0]["valueQuantity"]["value"] == 0.12


def test_emotion_fhir_contains_valence(patient_id, emotion_session):
    obs = map_emotion_to_fhir(patient_id, emotion_session, session_id="s1")
    valence_obs = [o for o in obs if "valence" in o["id"]]
    assert len(valence_obs) == 1
    assert valence_obs[0]["valueQuantity"]["value"] == 0.35


def test_emotion_fhir_contains_emotion_label(patient_id, emotion_session):
    obs = map_emotion_to_fhir(patient_id, emotion_session, session_id="s1")
    label_obs = [o for o in obs if "emotion" in o["id"] and "faa" not in o["id"]
                 and "valence" not in o["id"] and "arousal" not in o["id"]
                 and "stress" not in o["id"] and "focus" not in o["id"]]
    assert len(label_obs) == 1
    assert label_obs[0]["valueString"] == "happy"


def test_emotion_partial_data(patient_id):
    """Only valence provided -- should produce exactly one observation."""
    partial = {"valence": -0.2}
    obs = map_emotion_to_fhir(patient_id, partial, session_id="p1")
    assert len(obs) == 1
    assert obs[0]["valueQuantity"]["value"] == -0.2


# ── Sleep mapping tests ──────────────────────────────────────────────


def test_map_sleep_to_fhir(patient_id, sleep_session):
    obs = map_sleep_to_fhir(patient_id, sleep_session, session_id="sl1")
    assert len(obs) == 5  # efficiency, n3, rem, hrv, quality
    resource_types = {o["resourceType"] for o in obs}
    assert resource_types == {"Observation"}


def test_sleep_partial_data(patient_id):
    partial = {"sleep_efficiency": 0.9}
    obs = map_sleep_to_fhir(patient_id, partial)
    assert len(obs) == 1
    assert obs[0]["valueQuantity"]["value"] == 0.9


# ── Voice mapping tests ──────────────────────────────────────────────


def test_map_voice_to_fhir(patient_id, voice_screening):
    obs = map_voice_to_fhir(patient_id, voice_screening, session_id="v1")
    assert len(obs) == 2
    dep_obs = [o for o in obs if "dep" in o["id"]]
    assert dep_obs[0]["valueQuantity"]["value"] == 0.35


def test_voice_nested_format(patient_id):
    """Voice data with nested depression/anxiety dicts."""
    data = {
        "depression": {"risk_score": 0.5},
        "anxiety": {"risk_score": 0.3},
    }
    obs = map_voice_to_fhir(patient_id, data)
    assert len(obs) == 2


# ── Mood / QuestionnaireResponse tests ────────────────────────────────


def test_map_mood_phq9(patient_id):
    responses = [1, 2, 1, 0, 1, 2, 1, 0, 0]
    qr = map_mood_to_fhir(patient_id, "phq9", responses)
    assert qr["resourceType"] == "QuestionnaireResponse"
    assert qr["status"] == "completed"
    # 9 answer items + 1 total score item
    assert len(qr["item"]) == 10
    total_item = qr["item"][-1]
    assert total_item["answer"][0]["valueInteger"] == sum(responses)


def test_map_mood_gad7(patient_id):
    responses = [2, 1, 1, 2, 0, 1, 1]
    qr = map_mood_to_fhir(patient_id, "gad7", responses, total_score=8)
    assert qr["item"][-1]["answer"][0]["valueInteger"] == 8


# ── FHIR Bundle tests ────────────────────────────────────────────────


def test_fhir_bundle_wraps_resources(patient_id, emotion_session):
    obs = map_emotion_to_fhir(patient_id, emotion_session, session_id="b1")
    bundle = fhir_bundle_to_dict(obs)
    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "collection"
    assert bundle["total"] == len(obs)
    assert len(bundle["entry"]) == len(obs)
    for entry in bundle["entry"]:
        assert "resource" in entry
        assert "fullUrl" in entry


def test_fhir_bundle_empty():
    bundle = fhir_bundle_to_dict([])
    assert bundle["total"] == 0
    assert bundle["entry"] == []


# ── Consent management tests ──────────────────────────────────────────


def test_grant_consent():
    result = manage_consent("u1", "grant", "eeg_emotion", "dr-smith")
    assert result["status"] == "granted"
    assert result["data_type"] == "eeg_emotion"
    assert result["audit_id"]
    assert check_consent("u1", "eeg_emotion", "dr-smith") is True


def test_revoke_consent():
    manage_consent("u1", "grant", "eeg_emotion", "dr-smith")
    result = manage_consent("u1", "revoke", "eeg_emotion", "dr-smith")
    assert result["status"] == "revoked"
    assert check_consent("u1", "eeg_emotion", "dr-smith") is False


def test_query_consent_no_prior():
    result = manage_consent("u1", "query", "sleep_data", "dr-jones")
    assert result["status"] == "none"
    assert result["history"] == []


def test_query_consent_with_history():
    manage_consent("u1", "grant", "sleep_data", "dr-jones")
    manage_consent("u1", "revoke", "sleep_data", "dr-jones")
    result = manage_consent("u1", "query", "sleep_data", "dr-jones")
    assert result["status"] == "none"
    assert len(result["history"]) == 2


def test_consent_invalid_data_type():
    result = manage_consent("u1", "grant", "invalid_type", "dr-x")
    assert "error" in result


def test_consent_invalid_action():
    result = manage_consent("u1", "delete", "eeg_emotion", "dr-x")
    assert "error" in result


def test_consent_audit_trail():
    manage_consent("u1", "grant", "eeg_emotion", "dr-smith", reason="therapy")
    manage_consent("u1", "revoke", "eeg_emotion", "dr-smith", reason="ended")
    audit = get_consent_audit("u1")
    assert len(audit) == 2
    assert audit[0]["action"] == "grant"
    assert audit[0]["reason"] == "therapy"
    assert audit[1]["action"] == "revoke"


def test_consent_per_provider_isolation():
    """Consent for one provider should not affect another."""
    manage_consent("u1", "grant", "eeg_emotion", "dr-A")
    assert check_consent("u1", "eeg_emotion", "dr-A") is True
    assert check_consent("u1", "eeg_emotion", "dr-B") is False


# ── Session prep tests ────────────────────────────────────────────────


def test_session_prep_with_all_data(patient_id, emotion_session, sleep_session, voice_screening):
    prep = generate_session_prep(
        patient_id=patient_id,
        emotion_sessions=[emotion_session],
        sleep_sessions=[sleep_session],
        voice_screenings=[voice_screening],
        mood_scores=[{"phq9_total": 5, "gad7_total": 3}],
    )
    assert prep["patient_id"] == patient_id
    assert "eeg_emotion" in prep["data_sources"]
    assert "sleep_data" in prep["data_sources"]
    assert "voice_biomarkers" in prep["data_sources"]
    assert "mood_checkin" in prep["data_sources"]
    assert prep["emotion_summary"]["avg_valence"] == 0.35
    assert prep["sleep_summary"]["avg_efficiency"] == 0.88


def test_session_prep_empty(patient_id):
    prep = generate_session_prep(patient_id=patient_id)
    assert prep["data_sources"] == []
    assert prep["emotion_summary"] is None
    assert prep["sleep_summary"] is None
    assert len(prep["recommendations"]) > 0


def test_session_prep_clinical_flags_high_stress(patient_id):
    sessions = [{"stress_index": 0.85, "valence": -0.4}]
    prep = generate_session_prep(patient_id=patient_id, emotion_sessions=sessions)
    flags = prep["clinical_flags"]
    assert any("stress" in f.lower() for f in flags)
    assert any("valence" in f.lower() for f in flags)


def test_session_prep_clinical_flags_poor_sleep(patient_id):
    sessions = [{"sleep_efficiency": 0.6, "quality_score": 35}]
    prep = generate_session_prep(patient_id=patient_id, sleep_sessions=sessions)
    flags = prep["clinical_flags"]
    assert any("sleep efficiency" in f.lower() for f in flags)
    assert any("sleep quality" in f.lower() for f in flags)


# ── PDF data tests ────────────────────────────────────────────────────


def test_generate_pdf_data(patient_id, emotion_session):
    prep = generate_session_prep(
        patient_id=patient_id,
        emotion_sessions=[emotion_session],
    )
    pdf = generate_clinical_summary_pdf_data(patient_id, prep)
    assert pdf["title"] == "Clinical Summary Report"
    assert pdf["patient_id"] == patient_id
    assert len(pdf["sections"]) == 7
    assert pdf["disclaimer"]


# ── API route tests ───────────────────────────────────────────────────


@pytest.fixture
def client():
    from api.routes.clinical_bridge import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_status_endpoint(client):
    resp = client.get("/clinical/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "available"
    assert data["fhir_version"] == "R4"
    assert "eeg_emotion" in data["supported_data_types"]


def test_export_fhir_endpoint(client):
    resp = client.post("/clinical/export-fhir", json={
        "patient_id": "p1",
        "emotion_sessions": [
            {"valence": 0.5, "arousal": 0.6, "emotion": "happy"},
        ],
        "sleep_sessions": [
            {"sleep_efficiency": 0.9, "quality_score": 80},
        ],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["resource_count"] > 0
    assert data["bundle"]["resourceType"] == "Bundle"


def test_export_fhir_with_consent_block(client):
    """With provider_id set but no consent granted, data should be skipped."""
    resp = client.post("/clinical/export-fhir", json={
        "patient_id": "p2",
        "provider_id": "dr-nobody",
        "emotion_sessions": [
            {"valence": 0.5, "arousal": 0.6},
        ],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["resource_count"] == 0


def test_session_prep_endpoint(client):
    resp = client.post("/clinical/session-prep", json={
        "patient_id": "p1",
        "emotion_sessions": [
            {"valence": -0.5, "stress_index": 0.8},
        ],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["patient_id"] == "p1"
    assert data["emotion_summary"] is not None
    assert any("stress" in f.lower() for f in data["clinical_flags"])


def test_consent_endpoint_grant(client):
    resp = client.post("/clinical/consent", json={
        "user_id": "u1",
        "action": "grant",
        "data_type": "eeg_emotion",
        "provider_id": "dr-test",
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "granted"


def test_consent_endpoint_invalid_action(client):
    resp = client.post("/clinical/consent", json={
        "user_id": "u1",
        "action": "destroy",
        "data_type": "eeg_emotion",
        "provider_id": "dr-test",
    })
    assert resp.status_code == 400


def test_consent_endpoint_invalid_data_type(client):
    resp = client.post("/clinical/consent", json={
        "user_id": "u1",
        "action": "grant",
        "data_type": "nonexistent",
        "provider_id": "dr-test",
    })
    assert resp.status_code == 400


def test_export_fhir_with_pdf_data(client):
    resp = client.post("/clinical/export-fhir", json={
        "patient_id": "p1",
        "include_pdf_data": True,
        "emotion_sessions": [{"valence": 0.3}],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "pdf_data" in data
    assert data["pdf_data"]["title"] == "Clinical Summary Report"

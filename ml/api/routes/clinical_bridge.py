"""Clinical bridge API routes — FHIR export, session prep, and consent.

POST /clinical/export-fhir   -- export selected data as FHIR R4 bundle
POST /clinical/session-prep  -- generate clinician session prep summary
POST /clinical/consent       -- manage consent grants/revokes/queries
GET  /clinical/status        -- availability and configuration check
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from models.clinical_bridge import (
    CONSENT_DATA_TYPES,
    check_consent,
    fhir_bundle_to_dict,
    generate_clinical_summary_pdf_data,
    generate_session_prep,
    manage_consent,
    map_emotion_to_fhir,
    map_mood_to_fhir,
    map_sleep_to_fhir,
    map_voice_to_fhir,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/clinical", tags=["clinical-bridge"])


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class EmotionSessionData(BaseModel):
    valence: Optional[float] = None
    arousal: Optional[float] = None
    stress_index: Optional[float] = None
    focus_index: Optional[float] = None
    relaxation_index: Optional[float] = None
    frontal_asymmetry: Optional[float] = None
    emotion: Optional[str] = None
    timestamp: Optional[str] = None


class VoiceScreeningData(BaseModel):
    depression_risk: Optional[float] = None
    anxiety_risk: Optional[float] = None
    timestamp: Optional[str] = None


class SleepSessionData(BaseModel):
    sleep_efficiency: Optional[float] = None
    n3_pct: Optional[float] = None
    rem_pct: Optional[float] = None
    hrv_ms: Optional[float] = None
    quality_score: Optional[float] = None
    timestamp: Optional[str] = None


class MoodCheckinData(BaseModel):
    questionnaire_type: str = Field(
        ..., description="phq9 or gad7",
    )
    responses: List[int] = Field(
        ..., description="List of integer responses (0-3 each)",
    )
    total_score: Optional[int] = None


class FHIRExportRequest(BaseModel):
    patient_id: str = Field(..., description="FHIR Patient identifier")
    provider_id: Optional[str] = Field(
        None, description="Provider ID for consent check (skipped if None)",
    )
    emotion_sessions: Optional[List[EmotionSessionData]] = None
    voice_screenings: Optional[List[VoiceScreeningData]] = None
    sleep_sessions: Optional[List[SleepSessionData]] = None
    mood_checkins: Optional[List[MoodCheckinData]] = None
    include_pdf_data: bool = Field(
        False, description="Include structured PDF summary data in response",
    )


class SessionPrepRequest(BaseModel):
    patient_id: str = Field(..., description="FHIR Patient identifier")
    emotion_sessions: Optional[List[EmotionSessionData]] = None
    voice_screenings: Optional[List[VoiceScreeningData]] = None
    sleep_sessions: Optional[List[SleepSessionData]] = None
    mood_scores: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="List of dicts with phq9_total and/or gad7_total keys",
    )


class ConsentRequest(BaseModel):
    user_id: str = Field(..., description="Patient/user identifier")
    action: str = Field(
        ..., description="grant, revoke, or query",
    )
    data_type: str = Field(
        ..., description=f"One of: {sorted(CONSENT_DATA_TYPES)}",
    )
    provider_id: str = Field(..., description="Healthcare provider identifier")
    reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/export-fhir")
async def export_fhir(req: FHIRExportRequest) -> Dict[str, Any]:
    """Export patient data as a FHIR R4 Bundle.

    Collects emotion, voice, sleep, and mood data, maps each to FHIR
    resources (Observation or QuestionnaireResponse), and wraps them
    in a FHIR Bundle. Optionally checks consent before including data.
    """
    all_resources: List[Dict[str, Any]] = []

    # Emotion sessions -> Observations
    if req.emotion_sessions:
        if req.provider_id and not check_consent(req.patient_id, "eeg_emotion", req.provider_id):
            log.info("No consent for eeg_emotion — skipping emotion data")
        else:
            for idx, session in enumerate(req.emotion_sessions):
                obs_list = map_emotion_to_fhir(
                    patient_id=req.patient_id,
                    session_data=session.model_dump(),
                    session_id=f"emo-{idx}",
                )
                all_resources.extend(obs_list)

    # Voice screenings -> Observations
    if req.voice_screenings:
        if req.provider_id and not check_consent(req.patient_id, "voice_biomarkers", req.provider_id):
            log.info("No consent for voice_biomarkers — skipping voice data")
        else:
            for idx, screening in enumerate(req.voice_screenings):
                obs_list = map_voice_to_fhir(
                    patient_id=req.patient_id,
                    voice_data=screening.model_dump(),
                    session_id=f"voice-{idx}",
                )
                all_resources.extend(obs_list)

    # Sleep sessions -> Observations
    if req.sleep_sessions:
        if req.provider_id and not check_consent(req.patient_id, "sleep_data", req.provider_id):
            log.info("No consent for sleep_data — skipping sleep data")
        else:
            for idx, sleep in enumerate(req.sleep_sessions):
                obs_list = map_sleep_to_fhir(
                    patient_id=req.patient_id,
                    sleep_data=sleep.model_dump(),
                    session_id=f"sleep-{idx}",
                )
                all_resources.extend(obs_list)

    # Mood check-ins -> QuestionnaireResponses
    if req.mood_checkins:
        if req.provider_id and not check_consent(req.patient_id, "mood_checkin", req.provider_id):
            log.info("No consent for mood_checkin — skipping mood data")
        else:
            for idx, mood in enumerate(req.mood_checkins):
                qr = map_mood_to_fhir(
                    patient_id=req.patient_id,
                    questionnaire_type=mood.questionnaire_type,
                    responses=mood.responses,
                    total_score=mood.total_score,
                    session_id=f"mood-{idx}",
                )
                all_resources.append(qr)

    bundle = fhir_bundle_to_dict(all_resources)

    result: Dict[str, Any] = {
        "bundle": bundle,
        "resource_count": len(all_resources),
    }

    if req.include_pdf_data:
        prep = generate_session_prep(
            patient_id=req.patient_id,
            emotion_sessions=[s.model_dump() for s in req.emotion_sessions] if req.emotion_sessions else None,
            sleep_sessions=[s.model_dump() for s in req.sleep_sessions] if req.sleep_sessions else None,
            voice_screenings=[s.model_dump() for s in req.voice_screenings] if req.voice_screenings else None,
            mood_scores=None,
        )
        result["pdf_data"] = generate_clinical_summary_pdf_data(
            patient_id=req.patient_id,
            session_prep=prep,
            fhir_bundle=bundle,
        )

    return result


@router.post("/session-prep")
async def session_prep(req: SessionPrepRequest) -> Dict[str, Any]:
    """Generate a session preparation summary for a clinician.

    Aggregates recent biometric data to produce a snapshot of the patient's
    state since last appointment, including clinical flags and recommendations.
    """
    summary = generate_session_prep(
        patient_id=req.patient_id,
        emotion_sessions=[s.model_dump() for s in req.emotion_sessions] if req.emotion_sessions else None,
        sleep_sessions=[s.model_dump() for s in req.sleep_sessions] if req.sleep_sessions else None,
        voice_screenings=[s.model_dump() for s in req.voice_screenings] if req.voice_screenings else None,
        mood_scores=req.mood_scores,
    )
    return summary


@router.post("/consent")
async def consent_endpoint(req: ConsentRequest) -> Dict[str, Any]:
    """Manage consent grants, revocations, and queries.

    Actions:
    - grant: Grant consent for a specific data type and provider.
    - revoke: Revoke previously granted consent.
    - query: Check current consent status and history.
    """
    if req.action not in ("grant", "revoke", "query"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action '{req.action}'. Must be grant, revoke, or query.",
        )

    result = manage_consent(
        user_id=req.user_id,
        action=req.action,
        data_type=req.data_type,
        provider_id=req.provider_id,
        reason=req.reason,
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.get("/status")
async def clinical_status() -> Dict[str, Any]:
    """Check clinical bridge availability and configuration."""
    return {
        "status": "available",
        "fhir_version": "R4",
        "supported_data_types": sorted(CONSENT_DATA_TYPES),
        "supported_exports": [
            "eeg_emotion",
            "voice_biomarkers",
            "sleep_data",
            "mood_checkin",
        ],
        "supported_questionnaires": ["phq9", "gad7"],
        "consent_management": True,
        "session_prep": True,
        "disclaimer": (
            "FHIR resources are generated from consumer-grade biometric devices. "
            "They are intended to support clinical decision-making, not replace "
            "professional assessment or validated clinical instruments."
        ),
    }

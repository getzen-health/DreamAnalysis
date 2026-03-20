"""Clinical bridge: maps NeuralDreamWorkshop data to HL7 FHIR R4 resources.

Provides FHIR-compatible serialization for:
- EEG emotion sessions -> FHIR Observation (custom codes for FAA, valence, arousal)
- Voice biomarkers -> FHIR Observation (depression/anxiety risk scores)
- Mood check-ins -> FHIR QuestionnaireResponse (PHQ-9 / GAD-7 compatible)
- Sleep data -> FHIR Observation (sleep stages, efficiency, HRV)
- Consent management with granular per-data-type, per-provider audit trail
- Session prep summaries for clinician review

Reference: HL7 FHIR R4 (https://hl7.org/fhir/R4/)
"""
from __future__ import annotations

import hashlib
import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FHIR coding system constants
# ---------------------------------------------------------------------------

SYSTEM_LOINC = "http://loinc.org"
SYSTEM_SNOMED = "http://snomed.info/sct"
SYSTEM_NDW = "http://neuraldreamworkshop.local/fhir/codes"
FHIR_STATUS_FINAL = "final"
FHIR_STATUS_PRELIMINARY = "preliminary"

# Custom NDW codes for EEG-derived observations
NDW_CODES = {
    "faa": {"code": "NDW-EEG-001", "display": "Frontal Alpha Asymmetry"},
    "valence": {"code": "NDW-EEG-002", "display": "Emotional Valence (-1 to +1)"},
    "arousal": {"code": "NDW-EEG-003", "display": "Emotional Arousal (0 to 1)"},
    "stress_index": {"code": "NDW-EEG-004", "display": "Stress Index (0 to 1)"},
    "focus_index": {"code": "NDW-EEG-005", "display": "Focus Index (0 to 1)"},
    "relaxation_index": {"code": "NDW-EEG-006", "display": "Relaxation Index (0 to 1)"},
    "emotion": {"code": "NDW-EEG-007", "display": "Classified Emotion Label"},
    "depression_risk": {"code": "NDW-VOICE-001", "display": "Voice Depression Risk Score"},
    "anxiety_risk": {"code": "NDW-VOICE-002", "display": "Voice Anxiety Risk Score"},
    "sleep_efficiency": {"code": "NDW-SLEEP-001", "display": "Sleep Efficiency (0-1)"},
    "sleep_n3_pct": {"code": "NDW-SLEEP-002", "display": "Deep Sleep (N3) Percentage"},
    "sleep_rem_pct": {"code": "NDW-SLEEP-003", "display": "REM Sleep Percentage"},
    "sleep_hrv": {"code": "NDW-SLEEP-004", "display": "Sleep HRV (RMSSD ms)"},
    "sleep_quality": {"code": "NDW-SLEEP-005", "display": "Sleep Quality Score (0-100)"},
}

# LOINC codes for standard observations
LOINC_CODES = {
    "phq9_total": {"code": "44261-6", "display": "PHQ-9 total score"},
    "gad7_total": {"code": "70274-6", "display": "GAD-7 total score"},
    "heart_rate_variability": {"code": "80404-7", "display": "R-R interval.standard deviation (Heart rate variability)"},
}

# Valid data types for consent
CONSENT_DATA_TYPES = frozenset({
    "eeg_emotion", "voice_biomarkers", "mood_checkin",
    "sleep_data", "session_prep", "full_export",
})


# ---------------------------------------------------------------------------
# In-memory consent store
# ---------------------------------------------------------------------------

# user_id -> list of consent records
_consent_store: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

# user_id -> { (data_type, provider_id): bool }
_active_consents: Dict[str, Dict[tuple, bool]] = defaultdict(dict)

# Audit trail: list of all consent mutations
_consent_audit: List[Dict[str, Any]] = []


# ---------------------------------------------------------------------------
# FHIR resource builders
# ---------------------------------------------------------------------------

def _fhir_meta(profile: Optional[str] = None) -> Dict[str, Any]:
    """Build a FHIR Meta element with versionId and lastUpdated."""
    meta: Dict[str, Any] = {
        "versionId": "1",
        "lastUpdated": datetime.now(timezone.utc).isoformat(),
    }
    if profile:
        meta["profile"] = [profile]
    return meta


def _fhir_reference(resource_type: str, identifier: str) -> Dict[str, str]:
    """Build a FHIR Reference element."""
    return {
        "reference": f"{resource_type}/{identifier}",
    }


def _fhir_coding(system: str, code: str, display: str) -> Dict[str, str]:
    """Build a single FHIR Coding entry."""
    return {"system": system, "code": code, "display": display}


def _fhir_codeable_concept(
    system: str, code: str, display: str, text: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a FHIR CodeableConcept."""
    cc: Dict[str, Any] = {
        "coding": [_fhir_coding(system, code, display)],
    }
    if text:
        cc["text"] = text
    return cc


def _fhir_quantity(value: float, unit: str, system: str = "http://unitsofmeasure.org", code: Optional[str] = None) -> Dict[str, Any]:
    """Build a FHIR Quantity element."""
    q: Dict[str, Any] = {"value": round(value, 4), "unit": unit, "system": system}
    if code:
        q["code"] = code
    return q


def _observation_resource(
    obs_id: str,
    patient_id: str,
    code_key: str,
    value: Any,
    effective_dt: Optional[str] = None,
    status: str = FHIR_STATUS_FINAL,
    components: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build a FHIR Observation resource."""
    now_iso = effective_dt or datetime.now(timezone.utc).isoformat()

    ndw = NDW_CODES.get(code_key)
    if ndw:
        code_cc = _fhir_codeable_concept(SYSTEM_NDW, ndw["code"], ndw["display"])
    else:
        code_cc = _fhir_codeable_concept(SYSTEM_NDW, code_key, code_key)

    obs: Dict[str, Any] = {
        "resourceType": "Observation",
        "id": obs_id,
        "meta": _fhir_meta(),
        "status": status,
        "code": code_cc,
        "subject": _fhir_reference("Patient", patient_id),
        "effectiveDateTime": now_iso,
    }

    if isinstance(value, (int, float)):
        obs["valueQuantity"] = _fhir_quantity(float(value), "{score}")
    elif isinstance(value, str):
        obs["valueString"] = value
    elif isinstance(value, dict):
        obs["valueQuantity"] = value

    if components:
        obs["component"] = components

    # Flag non-validated custom NDW scores
    if ndw:
        obs["note"] = [{
            "text": (
                "This observation is derived from consumer-grade EEG hardware "
                "and has not been clinically validated. Research use only."
            )
        }]

    return obs


# ---------------------------------------------------------------------------
# Public mapping functions
# ---------------------------------------------------------------------------

def map_emotion_to_fhir(
    patient_id: str,
    session_data: Dict[str, Any],
    session_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Map an EEG emotion session to a list of FHIR Observation resources.

    Args:
        patient_id: FHIR Patient identifier.
        session_data: Dict with keys like valence, arousal, stress_index,
            focus_index, relaxation_index, frontal_asymmetry, emotion,
            probabilities.
        session_id: Optional session identifier for grouping.

    Returns:
        List of FHIR Observation resource dicts.
    """
    observations: List[Dict[str, Any]] = []
    effective = session_data.get("timestamp") or datetime.now(timezone.utc).isoformat()
    sid = session_id or str(uuid.uuid4())[:8]

    # Frontal Alpha Asymmetry
    if "frontal_asymmetry" in session_data:
        obs = _observation_resource(
            obs_id=f"obs-faa-{sid}",
            patient_id=patient_id,
            code_key="faa",
            value=session_data["frontal_asymmetry"],
            effective_dt=effective,
        )
        observations.append(obs)

    # Valence
    if "valence" in session_data:
        obs = _observation_resource(
            obs_id=f"obs-valence-{sid}",
            patient_id=patient_id,
            code_key="valence",
            value=session_data["valence"],
            effective_dt=effective,
        )
        observations.append(obs)

    # Arousal
    if "arousal" in session_data:
        obs = _observation_resource(
            obs_id=f"obs-arousal-{sid}",
            patient_id=patient_id,
            code_key="arousal",
            value=session_data["arousal"],
            effective_dt=effective,
        )
        observations.append(obs)

    # Stress index
    if "stress_index" in session_data:
        obs = _observation_resource(
            obs_id=f"obs-stress-{sid}",
            patient_id=patient_id,
            code_key="stress_index",
            value=session_data["stress_index"],
            effective_dt=effective,
        )
        observations.append(obs)

    # Focus index
    if "focus_index" in session_data:
        obs = _observation_resource(
            obs_id=f"obs-focus-{sid}",
            patient_id=patient_id,
            code_key="focus_index",
            value=session_data["focus_index"],
            effective_dt=effective,
        )
        observations.append(obs)

    # Classified emotion label
    if "emotion" in session_data:
        obs = _observation_resource(
            obs_id=f"obs-emotion-{sid}",
            patient_id=patient_id,
            code_key="emotion",
            value=session_data["emotion"],
            effective_dt=effective,
        )
        observations.append(obs)

    return observations


def map_voice_to_fhir(
    patient_id: str,
    voice_data: Dict[str, Any],
    session_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Map voice biomarker screening results to FHIR Observations.

    Args:
        patient_id: FHIR Patient identifier.
        voice_data: Dict with depression and anxiety risk scores.
        session_id: Optional session identifier.

    Returns:
        List of FHIR Observation resource dicts.
    """
    observations: List[Dict[str, Any]] = []
    effective = voice_data.get("timestamp") or datetime.now(timezone.utc).isoformat()
    sid = session_id or str(uuid.uuid4())[:8]

    # Depression risk
    dep_score = voice_data.get("depression_risk")
    if dep_score is None and "depression" in voice_data:
        dep_data = voice_data["depression"]
        dep_score = dep_data.get("risk_score") if isinstance(dep_data, dict) else dep_data
    if dep_score is not None:
        obs = _observation_resource(
            obs_id=f"obs-dep-{sid}",
            patient_id=patient_id,
            code_key="depression_risk",
            value=float(dep_score),
            effective_dt=effective,
        )
        observations.append(obs)

    # Anxiety risk
    anx_score = voice_data.get("anxiety_risk")
    if anx_score is None and "anxiety" in voice_data:
        anx_data = voice_data["anxiety"]
        anx_score = anx_data.get("risk_score") if isinstance(anx_data, dict) else anx_data
    if anx_score is not None:
        obs = _observation_resource(
            obs_id=f"obs-anx-{sid}",
            patient_id=patient_id,
            code_key="anxiety_risk",
            value=float(anx_score),
            effective_dt=effective,
        )
        observations.append(obs)

    return observations


def map_sleep_to_fhir(
    patient_id: str,
    sleep_data: Dict[str, Any],
    session_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Map sleep session data to FHIR Observations.

    Args:
        patient_id: FHIR Patient identifier.
        sleep_data: Dict with keys like sleep_efficiency, n3_pct, rem_pct,
            hrv_ms, quality_score.
        session_id: Optional session identifier.

    Returns:
        List of FHIR Observation resource dicts.
    """
    observations: List[Dict[str, Any]] = []
    effective = sleep_data.get("timestamp") or datetime.now(timezone.utc).isoformat()
    sid = session_id or str(uuid.uuid4())[:8]

    field_map = {
        "sleep_efficiency": "sleep_efficiency",
        "n3_pct": "sleep_n3_pct",
        "rem_pct": "sleep_rem_pct",
        "hrv_ms": "sleep_hrv",
        "quality_score": "sleep_quality",
    }

    for data_key, code_key in field_map.items():
        val = sleep_data.get(data_key)
        if val is not None:
            obs = _observation_resource(
                obs_id=f"obs-{code_key.replace('sleep_', 'slp-')}-{sid}",
                patient_id=patient_id,
                code_key=code_key,
                value=float(val),
                effective_dt=effective,
            )
            observations.append(obs)

    return observations


def map_mood_to_fhir(
    patient_id: str,
    questionnaire_type: str,
    responses: List[int],
    total_score: Optional[int] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Map PHQ-9 or GAD-7 mood check-in to a FHIR QuestionnaireResponse.

    Args:
        patient_id: FHIR Patient identifier.
        questionnaire_type: "phq9" or "gad7".
        responses: List of integer responses (0-3 each).
        total_score: Pre-computed total; computed from responses if None.
        session_id: Optional session identifier.

    Returns:
        FHIR QuestionnaireResponse resource dict.
    """
    sid = session_id or str(uuid.uuid4())[:8]
    computed_total = total_score if total_score is not None else sum(responses)

    if questionnaire_type.lower() == "phq9":
        questionnaire_url = "http://loinc.org/vs/LL358-3"
        display = "PHQ-9"
        loinc_code = LOINC_CODES["phq9_total"]
    elif questionnaire_type.lower() == "gad7":
        questionnaire_url = "http://loinc.org/vs/LL358-3"
        display = "GAD-7"
        loinc_code = LOINC_CODES["gad7_total"]
    else:
        questionnaire_url = f"{SYSTEM_NDW}/questionnaire/{questionnaire_type}"
        display = questionnaire_type.upper()
        loinc_code = {"code": questionnaire_type, "display": display}

    items = []
    for idx, resp in enumerate(responses):
        items.append({
            "linkId": f"q{idx + 1}",
            "answer": [{"valueInteger": resp}],
        })

    # Add total score as a final item
    items.append({
        "linkId": "total_score",
        "answer": [{"valueInteger": computed_total}],
        "text": f"{display} Total Score",
    })

    return {
        "resourceType": "QuestionnaireResponse",
        "id": f"qr-{questionnaire_type}-{sid}",
        "meta": _fhir_meta(),
        "status": "completed",
        "questionnaire": questionnaire_url,
        "subject": _fhir_reference("Patient", patient_id),
        "authored": datetime.now(timezone.utc).isoformat(),
        "item": items,
        "disclaimer": (
            "EEG-derived mood scores from NeuralDreamWorkshop are research-grade "
            "wellness estimates based on consumer EEG hardware. This is not a "
            "medical device. Scores are NOT equivalent to clinician-administered "
            "PHQ-9/GAD-7 validated instruments and are for wellness awareness "
            "only, not validated clinical assessments."
        ),
    }


# ---------------------------------------------------------------------------
# FHIR Bundle
# ---------------------------------------------------------------------------

def fhir_bundle_to_dict(
    resources: List[Dict[str, Any]],
    bundle_type: str = "collection",
) -> Dict[str, Any]:
    """Wrap a list of FHIR resources into a FHIR Bundle.

    Args:
        resources: List of FHIR resource dicts.
        bundle_type: Bundle type (collection, document, transaction, etc.).

    Returns:
        FHIR Bundle dict.
    """
    entries = []
    for res in resources:
        entry: Dict[str, Any] = {"resource": res}
        rt = res.get("resourceType", "Unknown")
        rid = res.get("id", "")
        entry["fullUrl"] = f"urn:uuid:{rid}" if rid else f"urn:uuid:{uuid.uuid4()}"
        entries.append(entry)

    return {
        "resourceType": "Bundle",
        "id": str(uuid.uuid4())[:12],
        "meta": _fhir_meta(),
        "type": bundle_type,
        "total": len(entries),
        "entry": entries,
    }


# ---------------------------------------------------------------------------
# Consent management
# ---------------------------------------------------------------------------

def manage_consent(
    user_id: str,
    action: str,
    data_type: str,
    provider_id: str,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Grant, revoke, or query consent for sharing specific data types.

    Args:
        user_id: Patient/user identifier.
        action: "grant", "revoke", or "query".
        data_type: One of CONSENT_DATA_TYPES.
        provider_id: Identifier for the healthcare provider.
        reason: Optional reason for the consent action.

    Returns:
        Dict with consent status and audit info.
    """
    if data_type not in CONSENT_DATA_TYPES:
        return {
            "error": f"Invalid data_type '{data_type}'. Must be one of: {sorted(CONSENT_DATA_TYPES)}",
        }

    consent_key = (data_type, provider_id)
    now_iso = datetime.now(timezone.utc).isoformat()

    audit_entry = {
        "user_id": user_id,
        "action": action,
        "data_type": data_type,
        "provider_id": provider_id,
        "reason": reason,
        "timestamp": now_iso,
        "audit_id": hashlib.sha256(f"{user_id}{action}{data_type}{provider_id}{now_iso}".encode()).hexdigest()[:12],
    }

    if action == "grant":
        _active_consents[user_id][consent_key] = True
        record = {
            "data_type": data_type,
            "provider_id": provider_id,
            "granted": True,
            "granted_at": now_iso,
            "reason": reason,
        }
        _consent_store[user_id].append(record)
        _consent_audit.append(audit_entry)
        return {
            "status": "granted",
            "data_type": data_type,
            "provider_id": provider_id,
            "audit_id": audit_entry["audit_id"],
            "timestamp": now_iso,
        }

    elif action == "revoke":
        _active_consents[user_id][consent_key] = False
        record = {
            "data_type": data_type,
            "provider_id": provider_id,
            "granted": False,
            "revoked_at": now_iso,
            "reason": reason,
        }
        _consent_store[user_id].append(record)
        _consent_audit.append(audit_entry)
        return {
            "status": "revoked",
            "data_type": data_type,
            "provider_id": provider_id,
            "audit_id": audit_entry["audit_id"],
            "timestamp": now_iso,
        }

    elif action == "query":
        is_active = _active_consents[user_id].get(consent_key, False)
        history = [
            r for r in _consent_store[user_id]
            if r["data_type"] == data_type and r["provider_id"] == provider_id
        ]
        return {
            "status": "active" if is_active else "none",
            "data_type": data_type,
            "provider_id": provider_id,
            "history": history,
        }

    else:
        return {"error": f"Invalid action '{action}'. Must be grant, revoke, or query."}


def check_consent(user_id: str, data_type: str, provider_id: str) -> bool:
    """Check whether active consent exists for a specific data sharing scope."""
    return _active_consents.get(user_id, {}).get((data_type, provider_id), False)


def get_consent_audit(user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return the consent audit trail, optionally filtered by user."""
    if user_id:
        return [e for e in _consent_audit if e["user_id"] == user_id]
    return list(_consent_audit)


# ---------------------------------------------------------------------------
# Session prep summary
# ---------------------------------------------------------------------------

def generate_session_prep(
    patient_id: str,
    emotion_sessions: Optional[List[Dict[str, Any]]] = None,
    sleep_sessions: Optional[List[Dict[str, Any]]] = None,
    voice_screenings: Optional[List[Dict[str, Any]]] = None,
    mood_scores: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Generate a session preparation summary for a clinician.

    Aggregates recent biometric data to produce a snapshot of the patient's
    state since last appointment.

    Args:
        patient_id: FHIR Patient identifier.
        emotion_sessions: List of emotion session dicts.
        sleep_sessions: List of sleep data dicts.
        voice_screenings: List of voice screening dicts.
        mood_scores: List of mood questionnaire dicts (phq9/gad7 totals).

    Returns:
        Session prep summary dict.
    """
    summary: Dict[str, Any] = {
        "patient_id": patient_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_sources": [],
        "emotion_summary": None,
        "sleep_summary": None,
        "voice_summary": None,
        "mood_summary": None,
        "clinical_flags": [],
        "recommendations": [],
    }

    # Emotion summary
    if emotion_sessions:
        summary["data_sources"].append("eeg_emotion")
        valences = [s["valence"] for s in emotion_sessions if s.get("valence") is not None]
        arousals = [s["arousal"] for s in emotion_sessions if s.get("arousal") is not None]
        stress_vals = [s["stress_index"] for s in emotion_sessions if s.get("stress_index") is not None]
        emotions = [s["emotion"] for s in emotion_sessions if s.get("emotion") is not None]

        avg_valence = sum(valences) / len(valences) if valences else 0.0
        avg_arousal = sum(arousals) / len(arousals) if arousals else 0.0
        avg_stress = sum(stress_vals) / len(stress_vals) if stress_vals else 0.0

        # Count emotion frequencies
        emotion_freq: Dict[str, int] = defaultdict(int)
        for e in emotions:
            emotion_freq[e] += 1
        dominant = max(emotion_freq, key=emotion_freq.get) if emotion_freq else "unknown"

        summary["emotion_summary"] = {
            "n_sessions": len(emotion_sessions),
            "avg_valence": round(avg_valence, 3),
            "avg_arousal": round(avg_arousal, 3),
            "avg_stress": round(avg_stress, 3),
            "dominant_emotion": dominant,
            "emotion_distribution": dict(emotion_freq),
        }

        if avg_stress > 0.7:
            summary["clinical_flags"].append("Elevated average stress index (>0.7)")
        if avg_valence < -0.3:
            summary["clinical_flags"].append("Persistently negative valence (<-0.3)")

    # Sleep summary
    if sleep_sessions:
        summary["data_sources"].append("sleep_data")
        efficiencies = [s["sleep_efficiency"] for s in sleep_sessions if s.get("sleep_efficiency") is not None]
        qualities = [s["quality_score"] for s in sleep_sessions if s.get("quality_score") is not None]
        hrvs = [s["hrv_ms"] for s in sleep_sessions if s.get("hrv_ms") is not None]

        avg_eff = sum(efficiencies) / len(efficiencies) if efficiencies else 0.0
        avg_qual = sum(qualities) / len(qualities) if qualities else 0.0

        summary["sleep_summary"] = {
            "n_nights": len(sleep_sessions),
            "avg_efficiency": round(avg_eff, 3),
            "avg_quality_score": round(avg_qual, 1),
            "avg_hrv_ms": round(sum(hrvs) / len(hrvs), 1) if hrvs else None,
        }

        if avg_eff < 0.75:
            summary["clinical_flags"].append("Low sleep efficiency (<75%)")
        if avg_qual < 50:
            summary["clinical_flags"].append("Poor sleep quality score (<50)")

    # Voice summary
    if voice_screenings:
        summary["data_sources"].append("voice_biomarkers")
        dep_scores = []
        anx_scores = []
        for v in voice_screenings:
            ds = v.get("depression_risk")
            if ds is None and "depression" in v:
                d = v["depression"]
                ds = d.get("risk_score") if isinstance(d, dict) else d
            if ds is not None:
                dep_scores.append(float(ds))

            ax = v.get("anxiety_risk")
            if ax is None and "anxiety" in v:
                a = v["anxiety"]
                ax = a.get("risk_score") if isinstance(a, dict) else a
            if ax is not None:
                anx_scores.append(float(ax))

        summary["voice_summary"] = {
            "n_screenings": len(voice_screenings),
            "avg_depression_risk": round(sum(dep_scores) / len(dep_scores), 3) if dep_scores else None,
            "avg_anxiety_risk": round(sum(anx_scores) / len(anx_scores), 3) if anx_scores else None,
        }

        avg_dep = sum(dep_scores) / len(dep_scores) if dep_scores else 0
        avg_anx = sum(anx_scores) / len(anx_scores) if anx_scores else 0
        if avg_dep > 0.6:
            summary["clinical_flags"].append("Elevated voice depression risk (>0.6)")
        if avg_anx > 0.6:
            summary["clinical_flags"].append("Elevated voice anxiety risk (>0.6)")

    # Mood summary
    if mood_scores:
        summary["data_sources"].append("mood_checkin")
        phq9_totals = [m["phq9_total"] for m in mood_scores if m.get("phq9_total") is not None]
        gad7_totals = [m["gad7_total"] for m in mood_scores if m.get("gad7_total") is not None]

        summary["mood_summary"] = {
            "n_checkins": len(mood_scores),
            "avg_phq9": round(sum(phq9_totals) / len(phq9_totals), 1) if phq9_totals else None,
            "avg_gad7": round(sum(gad7_totals) / len(gad7_totals), 1) if gad7_totals else None,
        }

        avg_phq9 = sum(phq9_totals) / len(phq9_totals) if phq9_totals else 0
        avg_gad7 = sum(gad7_totals) / len(gad7_totals) if gad7_totals else 0
        if avg_phq9 >= 10:
            summary["clinical_flags"].append("PHQ-9 average >= 10 (moderate depression)")
        if avg_gad7 >= 10:
            summary["clinical_flags"].append("GAD-7 average >= 10 (moderate anxiety)")

    # Recommendations based on flags
    if summary["clinical_flags"]:
        summary["recommendations"].append(
            "Review flagged metrics with patient during session."
        )
    if not summary["data_sources"]:
        summary["recommendations"].append(
            "No biometric data available since last session. "
            "Consider encouraging regular check-ins."
        )

    summary["disclaimer"] = (
        "EEG-derived mood scores from NeuralDreamWorkshop are research-grade "
        "wellness estimates based on consumer EEG hardware. This is not a "
        "medical device. Scores are NOT equivalent to clinician-administered "
        "PHQ-9/GAD-7 validated instruments and are for wellness awareness "
        "only, not validated clinical assessments."
    )

    return summary


# ---------------------------------------------------------------------------
# Clinical summary PDF data
# ---------------------------------------------------------------------------

def generate_clinical_summary_pdf_data(
    patient_id: str,
    session_prep: Dict[str, Any],
    fhir_bundle: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate structured data suitable for rendering a clinical summary PDF.

    This does not produce a PDF file directly -- it returns a structured dict
    that a frontend or PDF renderer can consume.

    Args:
        patient_id: Patient identifier.
        session_prep: Output from generate_session_prep().
        fhir_bundle: Optional FHIR bundle for inclusion.

    Returns:
        Dict with sections for PDF rendering.
    """
    return {
        "title": "Clinical Summary Report",
        "patient_id": patient_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sections": [
            {
                "heading": "Patient Overview",
                "content": {
                    "patient_id": patient_id,
                    "report_period": session_prep.get("generated_at", ""),
                    "data_sources": session_prep.get("data_sources", []),
                },
            },
            {
                "heading": "Emotional State",
                "content": session_prep.get("emotion_summary"),
            },
            {
                "heading": "Sleep Analysis",
                "content": session_prep.get("sleep_summary"),
            },
            {
                "heading": "Voice Biomarkers",
                "content": session_prep.get("voice_summary"),
            },
            {
                "heading": "Mood Assessments",
                "content": session_prep.get("mood_summary"),
            },
            {
                "heading": "Clinical Flags",
                "content": session_prep.get("clinical_flags", []),
            },
            {
                "heading": "Recommendations",
                "content": session_prep.get("recommendations", []),
            },
        ],
        "fhir_bundle_included": fhir_bundle is not None,
        "fhir_resource_count": fhir_bundle.get("total", 0) if fhir_bundle else 0,
        "disclaimer": (
            "This report is generated from consumer-grade biometric devices "
            "and validated screening instruments. It is intended to support "
            "clinical decision-making, not replace professional assessment."
        ),
    }


# ---------------------------------------------------------------------------
# Reset (for testing)
# ---------------------------------------------------------------------------

def _reset_consent_store() -> None:
    """Clear all consent data. For testing only."""
    _consent_store.clear()
    _active_consents.clear()
    _consent_audit.clear()

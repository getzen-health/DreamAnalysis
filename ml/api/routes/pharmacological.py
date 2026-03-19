"""Pharmacological interaction tracking API endpoints.

Endpoints
---------
POST /pharmacological/log
    Log a medication start/stop/change.

POST /pharmacological/analyze
    Analyze medication effects on emotion data.

GET  /pharmacological/status
    Get current pharmacological profile and status.

GET  /pharmacological/medications/{user_id}
    List active medications for a user.

GET  /pharmacological/blunting/{user_id}
    Detect emotional blunting.

GET  /pharmacological/onset/{user_id}/{medication_name}
    Get onset curve for a medication.

GET  /pharmacological/withdrawal/{user_id}
    Detect withdrawal/rebound effects.

POST /pharmacological/emotion-reading
    Log an emotion reading for pharmacological correlation.

DELETE /pharmacological/reset/{user_id}
    Clear all pharmacological data for a user.
"""
from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from models.pharmacological_model import (
    PharmacologicalTracker,
    VALID_CATEGORIES,
    DRUG_EFFECT_DB,
)

router = APIRouter(prefix="/pharmacological", tags=["pharmacological"])

# Module-level singleton
_tracker = PharmacologicalTracker()


def get_tracker() -> PharmacologicalTracker:
    """Return the module-level PharmacologicalTracker singleton."""
    return _tracker


# ── Request models ──────────────────────────────────────────────────


class LogMedicationRequest(BaseModel):
    """Request body for logging a medication."""
    user_id: str = Field(..., description="User identifier")
    name: str = Field(..., description="Medication name (e.g. 'sertraline')")
    category: str = Field(
        ...,
        description=(
            "Medication category: ssri, snri, stimulant, benzodiazepine, "
            "buspirone, mood_stabilizer, beta_blocker, antipsychotic, "
            "gabapentinoid, trazodone, mirtazapine, bupropion"
        ),
    )
    dosage_mg: float = Field(..., description="Dosage in milligrams")
    start_date: Optional[float] = Field(
        default=None,
        description="Unix timestamp of medication start. Defaults to now.",
    )
    end_date: Optional[float] = Field(
        default=None,
        description="Unix timestamp of medication stop. None if still active.",
    )
    notes: Optional[str] = Field(
        default="",
        description="Optional notes about this medication.",
    )


class AnalyzeMedicationRequest(BaseModel):
    """Request body for analyzing medication effects on emotion."""
    user_id: str = Field(..., description="User identifier")
    base_valence: float = Field(
        0.0,
        description="Base emotional valence (-1 to 1)",
    )
    base_arousal: float = Field(
        0.5,
        description="Base arousal level (0 to 1)",
    )
    current_time: Optional[float] = Field(
        default=None,
        description="Unix timestamp. Defaults to now.",
    )


class LogEmotionReadingRequest(BaseModel):
    """Request body for logging an emotion reading."""
    user_id: str = Field(..., description="User identifier")
    timestamp: Optional[float] = Field(
        default=None,
        description="Unix timestamp. Defaults to now.",
    )
    valence: float = Field(0.0, description="Emotional valence (-1 to 1)")
    arousal: float = Field(0.5, description="Arousal level (0 to 1)")
    stress_index: float = Field(0.0, description="Stress index (0 to 1)")
    source: str = Field("eeg", description="Data source: eeg or voice")


# ── Endpoints ───────────────────────────────────────────────────────


@router.post("/log")
async def log_medication(req: LogMedicationRequest):
    """Log a medication start, stop, or dosage change.

    Records the medication with dosage and timestamps for later
    pharmacological analysis against EEG-derived emotion data.
    """
    category = req.category.strip().lower()
    if category not in VALID_CATEGORIES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid medication category '{req.category}'. "
                f"Valid categories: {sorted(VALID_CATEGORIES)}"
            ),
        )

    ts = req.start_date if req.start_date is not None else time.time()

    med = _tracker.log_medication(
        user_id=req.user_id,
        name=req.name,
        category=category,
        dosage_mg=req.dosage_mg,
        start_date=ts,
        end_date=req.end_date,
        notes=req.notes or "",
    )
    return {
        "medication": med.to_dict(),
        "logged_at": ts,
        "category_info": DRUG_EFFECT_DB.get(category, {}),
    }


@router.post("/analyze")
async def analyze_medication_effects(req: AnalyzeMedicationRequest):
    """Analyze how current medications modify emotional baseline.

    Takes a base emotion (valence, arousal) and returns the
    medication-adjusted emotion after accounting for all active
    medications, their onset curves, and range compression effects.
    """
    ct = req.current_time if req.current_time is not None else time.time()

    result = _tracker.compute_medication_effect(
        user_id=req.user_id,
        base_emotion={
            "valence": req.base_valence,
            "arousal": req.base_arousal,
        },
        current_time=ct,
    )
    return result


@router.get("/status")
async def get_pharmacological_status(
    user_id: str,
    current_time: Optional[float] = None,
):
    """Get full pharmacological profile and status for a user.

    Returns active medications, onset status, blunting detection,
    withdrawal risk, and total emotion modifiers.
    """
    ct = current_time if current_time is not None else time.time()
    profile = _tracker.profile_to_dict(user_id, ct)
    return profile


@router.get("/medications/{user_id}")
async def get_medications(user_id: str):
    """List active medications for a user."""
    active = _tracker.get_active_medications(user_id)
    log = _tracker.get_medication_log(user_id)
    return {
        "user_id": user_id,
        "active_count": len(active),
        "active_medications": active,
        "total_logged": len(log),
        "log": log,
    }


@router.get("/blunting/{user_id}")
async def detect_blunting(user_id: str, window_days: float = 14.0):
    """Detect emotional blunting from medication effects.

    Compares emotional range before and after blunting-type medication
    start to determine if emotional responsiveness has narrowed.
    """
    result = _tracker.detect_emotional_blunting(user_id, window_days)
    return result


@router.get("/onset/{user_id}/{medication_name}")
async def get_onset_curve(
    user_id: str,
    medication_name: str,
    bucket_hours: float = 24.0,
):
    """Get the onset curve for a medication.

    Shows how emotion data changes over time buckets since medication start.
    """
    result = _tracker.compute_onset_curve(
        user_id, medication_name, bucket_hours,
    )
    return result


@router.get("/withdrawal/{user_id}")
async def detect_withdrawal(user_id: str, window_days: float = 7.0):
    """Detect withdrawal or rebound effects from stopped medications.

    Checks recently stopped medications for arousal or stress spikes
    in the days following discontinuation.
    """
    result = _tracker.detect_withdrawal(user_id, window_days)
    return result


@router.post("/emotion-reading")
async def log_emotion_reading(req: LogEmotionReadingRequest):
    """Log an emotion reading for pharmacological correlation.

    Called internally from the EEG analysis pipeline to build a
    timeline of emotions for correlation with medication effects.
    """
    ts = req.timestamp if req.timestamp is not None else time.time()
    _tracker.log_emotion_reading(
        user_id=req.user_id,
        timestamp=ts,
        emotion_data={
            "valence": req.valence,
            "arousal": req.arousal,
            "stress_index": req.stress_index,
            "source": req.source,
        },
    )
    return {"stored": True, "timestamp": ts}


@router.delete("/reset/{user_id}")
async def reset_pharmacological_data(user_id: str):
    """Clear all pharmacological data for a user."""
    _tracker.reset(user_id)
    return {"user_id": user_id, "status": "reset"}

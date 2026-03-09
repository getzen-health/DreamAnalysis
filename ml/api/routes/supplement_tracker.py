"""Supplement/Medication/Vitamin Tracker API endpoints.

Endpoints
---------
POST /supplements/log
    Log a supplement intake event.

GET  /supplements/log/{user_id}
    Retrieve supplement log entries, optionally filtered by name.

POST /supplements/brain-state
    Log a brain state snapshot (called internally from /analyze-eeg).

GET  /supplements/correlations/{user_id}
    Analyze correlations between a supplement and brain state changes.

GET  /supplements/report/{user_id}
    Full supplement report with per-supplement correlation verdicts.

GET  /supplements/active/{user_id}
    List supplements taken in the last N hours.

DELETE /supplements/reset/{user_id}
    Clear all supplement data for a user.
"""
from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from models.supplement_tracker import SupplementTracker, VALID_SUPPLEMENT_TYPES

router = APIRouter(prefix="/supplements", tags=["Supplement Tracker"])

# Module-level singleton
_tracker = SupplementTracker()


def get_tracker() -> SupplementTracker:
    """Return the module-level SupplementTracker singleton."""
    return _tracker


# ── Request models ──────────────────────────────────────────────────


class LogSupplementRequest(BaseModel):
    """Request body for logging a supplement intake."""
    user_id: str = Field(..., description="User identifier")
    name: str = Field(..., description="Supplement name (e.g. 'Omega-3', 'Vitamin D')")
    type: str = Field(
        ...,
        description="Type: vitamin, supplement, medication, or food_supplement",
    )
    dosage: float = Field(..., description="Amount taken")
    unit: str = Field(..., description="Unit of measurement (mg, IU, mcg, etc.)")
    timestamp: Optional[float] = Field(
        default=None,
        description="Unix timestamp. Defaults to current time.",
    )
    notes: Optional[str] = Field(
        default="",
        description="Optional notes about this intake.",
    )


class LogBrainStateRequest(BaseModel):
    """Request body for logging a brain state snapshot."""
    user_id: str = Field(..., description="User identifier")
    timestamp: Optional[float] = Field(
        default=None,
        description="Unix timestamp. Defaults to current time.",
    )
    valence: float = Field(0.0, description="Emotional valence (-1 to 1)")
    arousal: float = Field(0.0, description="Arousal level (0 to 1)")
    stress_index: float = Field(0.0, description="Stress index (0 to 1)")
    focus_index: float = Field(0.0, description="Focus index (0 to 1)")
    alpha_beta_ratio: float = Field(0.0, description="Alpha/beta ratio (relaxation)")
    theta_power: float = Field(0.0, description="Theta power (creativity/drowsiness)")
    faa: float = Field(0.0, description="Frontal alpha asymmetry")


# ── Endpoints ───────────────────────────────────────────────────────


@router.post("/log")
async def log_supplement(req: LogSupplementRequest):
    """Log a supplement, vitamin, or medication intake event.

    Records the intake with timestamp for later correlation analysis
    against EEG-derived brain state metrics.
    """
    if req.type not in VALID_SUPPLEMENT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid supplement type '{req.type}'. "
                f"Valid types: {sorted(VALID_SUPPLEMENT_TYPES)}"
            ),
        )

    ts = req.timestamp if req.timestamp is not None else time.time()

    entry_id = _tracker.log_supplement(
        user_id=req.user_id,
        name=req.name,
        supplement_type=req.type,
        dosage=req.dosage,
        unit=req.unit,
        timestamp=ts,
        notes=req.notes or "",
    )
    return {"entry_id": entry_id, "logged_at": ts}


@router.get("/log/{user_id}")
async def get_supplement_log(
    user_id: str,
    last_n: int = 50,
    supplement_name: Optional[str] = None,
):
    """Retrieve supplement log entries for a user.

    Optionally filter by supplement name (case-insensitive).
    Returns the most recent `last_n` entries.
    """
    entries = _tracker.get_log(
        user_id=user_id,
        last_n=last_n,
        supplement_name=supplement_name,
    )
    return {"user_id": user_id, "count": len(entries), "entries": entries}


@router.post("/brain-state")
async def log_brain_state(req: LogBrainStateRequest):
    """Log a brain state snapshot for supplement correlation.

    Called internally by the EEG analysis pipeline to build a
    timeline of brain states for correlation with supplement intake.
    """
    ts = req.timestamp if req.timestamp is not None else time.time()

    _tracker.log_brain_state(
        user_id=req.user_id,
        timestamp=ts,
        emotion_data={
            "valence": req.valence,
            "arousal": req.arousal,
            "stress_index": req.stress_index,
            "focus_index": req.focus_index,
            "alpha_beta_ratio": req.alpha_beta_ratio,
            "theta_power": req.theta_power,
            "faa": req.faa,
        },
    )
    return {"stored": True, "timestamp": ts}


@router.get("/correlations/{user_id}")
async def get_correlations(
    user_id: str,
    supplement_name: str,
    window_hours: float = 4.0,
):
    """Analyze correlations between a supplement and brain state changes.

    Compares brain states in the hours after taking the supplement
    versus control periods when it was not taken. Returns average
    shifts in valence, arousal, stress, focus, and EEG-specific
    metrics (alpha/beta ratio, theta, FAA).
    """
    result = _tracker.analyze_correlations(
        user_id=user_id,
        supplement_name=supplement_name,
        window_hours=window_hours,
    )
    return result


@router.get("/report/{user_id}")
async def get_supplement_report(user_id: str):
    """Generate a full supplement correlation report.

    For each supplement the user takes, computes the overall
    correlation verdict (positive/negative/neutral/insufficient_data)
    against EEG-derived brain metrics.
    """
    return _tracker.get_supplement_report(user_id)


@router.get("/active/{user_id}")
async def get_active_supplements(user_id: str, hours: float = 24.0):
    """List supplements taken within the last N hours.

    Useful for showing which supplements are currently active and
    may be influencing brain state readings.
    """
    active = _tracker.get_active_supplements(user_id, hours=hours)
    return {"user_id": user_id, "hours": hours, "count": len(active), "supplements": active}


@router.delete("/reset/{user_id}")
async def reset_supplement_data(user_id: str):
    """Clear all supplement and brain state data for a user."""
    _tracker.reset(user_id)
    return {"user_id": user_id, "status": "reset"}

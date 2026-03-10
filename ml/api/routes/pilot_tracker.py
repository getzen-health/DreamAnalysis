"""Pilot study session management API (#200).

Endpoints
---------
POST /pilot/session/start        — Start a new session for a participant
POST /pilot/session/complete     — Complete session with EEG quality + survey
GET  /pilot/participants         — List enrolled participants
GET  /pilot/participants/{id}    — Status for one participant
GET  /pilot/metrics              — Study-level feasibility metrics
POST /pilot/reset                — Clear all data (testing only)
"""
from __future__ import annotations

from typing import Dict, Optional

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/pilot", tags=["pilot"])


# ── Schemas ──────────────────────────────────────────────────────────────────

class StartSessionRequest(BaseModel):
    participant_id: str
    eeg_device: str = "Muse 2"


class CompleteSessionRequest(BaseModel):
    participant_id: str
    session_id: str
    eeg_epochs_total: int = 0
    eeg_epochs_usable: int = 0
    survey: Dict = {}


class ResetRequest(BaseModel):
    participant_id: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/session/start")
async def start_session(req: StartSessionRequest):
    """Start a new pilot session for a participant."""
    from models.pilot_tracker import get_tracker
    return get_tracker().start_session(
        participant_id=req.participant_id,
        eeg_device=req.eeg_device,
    )


@router.post("/session/complete")
async def complete_session(req: CompleteSessionRequest):
    """Mark a session complete with EEG quality metrics and survey data."""
    from models.pilot_tracker import get_tracker
    return get_tracker().complete_session(
        participant_id=req.participant_id,
        session_id=req.session_id,
        eeg_epochs_total=req.eeg_epochs_total,
        eeg_epochs_usable=req.eeg_epochs_usable,
        survey=req.survey,
    )


@router.get("/participants")
async def list_participants():
    """List all enrolled participants with session counts."""
    from models.pilot_tracker import get_tracker
    return {"participants": get_tracker().list_participants()}


@router.get("/participants/{participant_id}")
async def participant_status(participant_id: str):
    """Get session status for one participant."""
    from models.pilot_tracker import get_tracker
    return get_tracker().get_participant_status(participant_id)


@router.get("/metrics")
async def study_metrics():
    """Study-level feasibility metrics: completion rate, SQI, verdict."""
    from models.pilot_tracker import get_tracker
    return get_tracker().get_study_metrics()


@router.post("/reset")
async def reset_data(req: ResetRequest):
    """Clear session data. Use participant_id to clear one participant only."""
    from models.pilot_tracker import get_tracker
    return get_tracker().reset(req.participant_id)

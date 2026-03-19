"""Pilot study validation and analysis API routes.

Endpoints:
  POST /pilot/validate-session     -- validate a single session's completeness + SQI
  POST /pilot/participant-summary  -- per-participant summary stats
  GET  /pilot/statistics           -- pilot-wide aggregate statistics
  GET  /pilot/readiness            -- go/no-go readiness report
  GET  /pilot/status               -- lightweight liveness check

GitHub issue: #200
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from models.pilot_validation import (
    compute_participant_summary,
    compute_pilot_statistics,
    compute_session_sqi,
    generate_readiness_report,
    report_to_dict,
    validate_session_completeness,
)

router = APIRouter(tags=["pilot"])


# ── Request schemas ──────────────────────────────────────────────────────────


class ValidateSessionRequest(BaseModel):
    """Payload for validating a single pilot session."""

    session: Dict[str, Any] = Field(
        ..., description="Session record dict (camelCase or snake_case keys)"
    )
    eeg_data: Optional[List[List[float]]] = Field(
        default=None,
        description="Raw EEG data (channels x samples) for SQI computation",
    )
    fs: float = Field(default=256.0, description="Sampling frequency in Hz")
    amplitude_threshold_uv: float = Field(
        default=75.0, description="Amplitude threshold for SQI in micro-volts"
    )


class ParticipantSummaryRequest(BaseModel):
    """Payload for computing per-participant summary."""

    participant_code: str = Field(..., description="Participant code, e.g. P001")
    sessions: List[Dict[str, Any]] = Field(
        ..., description="All session records for this participant"
    )
    fs: float = Field(default=256.0, description="Sampling frequency in Hz")
    amplitude_threshold_uv: float = Field(
        default=75.0, description="Amplitude threshold for SQI in micro-volts"
    )


class PilotStatisticsQuery(BaseModel):
    """Query params wrapped as a model for the GET /pilot/statistics endpoint.

    Since GET endpoints cannot have a JSON body in standard REST, the route
    accepts optional query parameters.  When called from tests or internal
    code, a JSON body with session data can be POSTed to /pilot/statistics
    instead -- but for the spec, we provide a GET that returns stats for an
    empty pilot (or a POST alternative can be added later when wired to DB).
    """

    pass


# ── In-memory store (test / demo -- real deployment reads from DB) ───────────
# Routes accept data via request body so they work without a live database.
# A production version would query pilotSessions / pilotParticipants tables.

_demo_sessions: List[Dict[str, Any]] = []
_demo_participant_codes: List[str] = []


# ── Routes ───────────────────────────────────────────────────────────────────


@router.post("/pilot/validate-session")
async def validate_session(req: ValidateSessionRequest) -> Dict[str, Any]:
    """Validate a single pilot session for completeness and signal quality.

    Returns completeness check and, if eeg_data is provided, SQI metrics.
    """
    completeness = validate_session_completeness(req.session)

    sqi: Optional[Dict[str, Any]] = None
    if req.eeg_data is not None:
        sqi = compute_session_sqi(
            req.eeg_data,
            fs=req.fs,
            amplitude_threshold_uv=req.amplitude_threshold_uv,
        )

    return {
        "completeness": completeness,
        "sqi": sqi,
    }


@router.post("/pilot/participant-summary")
async def participant_summary(req: ParticipantSummaryRequest) -> Dict[str, Any]:
    """Compute summary statistics for a single participant."""
    return compute_participant_summary(
        participant_code=req.participant_code,
        sessions=req.sessions,
        fs=req.fs,
        amplitude_threshold_uv=req.amplitude_threshold_uv,
    )


@router.post("/pilot/statistics")
async def pilot_statistics_post(
    sessions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute pilot-wide statistics from a list of session records.

    Accepts a JSON array of session dicts in the request body.
    """
    return compute_pilot_statistics(sessions)


@router.get("/pilot/statistics")
async def pilot_statistics_get() -> Dict[str, Any]:
    """Return pilot-wide statistics (from in-memory demo store).

    In production this would query the database.  For now returns stats
    computed from whatever has been posted via /pilot/validate-session.
    """
    return compute_pilot_statistics(_demo_sessions, _demo_participant_codes or None)


@router.get("/pilot/readiness")
async def pilot_readiness() -> Dict[str, Any]:
    """Generate a go/no-go readiness report for the pilot.

    Uses the in-memory demo store.  In production, reads from DB.
    """
    report = generate_readiness_report(
        _demo_sessions, _demo_participant_codes or None
    )
    return report_to_dict(report)


@router.get("/pilot/status")
async def pilot_status() -> Dict[str, Any]:
    """Lightweight status/health check for the pilot validation subsystem."""
    return {
        "status": "ok",
        "service": "pilot_validation",
        "version": "1.0.0",
        "demo_sessions_loaded": len(_demo_sessions),
        "demo_participants": len(_demo_participant_codes),
    }

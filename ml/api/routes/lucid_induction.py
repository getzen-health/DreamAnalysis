"""Lucid dream induction endpoints — LuciEntry closed-loop protocol.

Endpoints:
  POST /lucid-induction/start              Start monitoring session
  POST /lucid-induction/process-epoch      Feed EEG epoch, advance state machine
  GET  /lucid-induction/status/{user_id}   Current induction state
  POST /lucid-induction/confirm/{user_id}  Manual lucidity confirmation
  POST /lucid-induction/stop/{user_id}     Stop session
  GET  /lucid-induction/sessions           List active sessions

Reference: Sakaino et al. (2025) LuciEntry, ACM DIS 2025.
"""

from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from models.lucid_dream_inducer import LucidDreamInducer
from ._shared import _numpy_safe

router = APIRouter()

# Global singleton — one inducer handles all user sessions
_inducer = LucidDreamInducer()


# ─── Request / response schemas ────────────────────────────────────────────────

class StartInductionRequest(BaseModel):
    user_id: str = Field("default", description="Unique session identifier")
    fs: float = Field(256.0, description="EEG sampling rate (Hz)")


class ProcessEpochRequest(BaseModel):
    user_id: str = Field("default", description="Session owner")
    signals: List[List[float]] = Field(
        ...,
        description="EEG multichannel data: [[ch0_samples…], [ch1_samples…], …] "
                    "(ch0=TP9, ch1=AF7, ch2=AF8, ch3=TP10)",
    )
    fs: float = Field(256.0, description="Sampling rate (Hz)")
    sleep_stage: Optional[str] = Field(
        None,
        description="Sleep stage label from sleep staging model ('REM', 'N1', 'N2', 'N3', 'Wake')",
    )


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/lucid-induction/start")
async def start_induction(req: StartInductionRequest):
    """Start a lucid dream induction monitoring session.

    Immediately enters REM_MONITORING state. Feed EEG epochs via
    /lucid-induction/process-epoch at ~1 Hz or faster.
    """
    result = _inducer.start_session(req.user_id, req.fs)
    return _numpy_safe(result)


@router.post("/lucid-induction/process-epoch")
async def process_epoch(req: ProcessEpochRequest):
    """Feed one EEG epoch (≥ 1 second) and advance the induction state machine.

    Returns the current state and, crucially, `trigger_cue: true` when the
    system determines that audio cues should be played ("This is a dream" x3).

    The client is responsible for:
    - Playing the audio cue when `trigger_cue` is true (recommended: 3 repetitions
      of a gentle voice recording over 10 seconds at a volume that won't wake the sleeper)
    - Monitoring `state == "lucid_confirmed"` or `state == "lr_detection"` and
      surfacing these to the user

    Channel ordering (Muse 2, BrainFlow):
      ch0 = TP9 (left temporal — EOG left eye reference)
      ch1 = AF7 (left frontal — primary EEG for REM detection)
      ch2 = AF8 (right frontal)
      ch3 = TP10 (right temporal — EOG right eye reference)
    """
    eeg = np.array(req.signals, dtype=np.float32)
    if eeg.ndim == 1:
        eeg = eeg.reshape(1, -1)

    result = _inducer.process_epoch(
        req.user_id,
        eeg,
        fs=req.fs,
        sleep_stage=req.sleep_stage,
    )
    return _numpy_safe(result)


@router.get("/lucid-induction/status/{user_id}")
async def get_induction_status(user_id: str):
    """Get current induction state for a session.

    State machine values:
    - `idle` — not started
    - `rem_monitoring` — watching for REM onset
    - `rem_stable` — sustained REM detected, waiting for cue window
    - `cues_scheduled` — cue delivery imminent (trigger on next epoch)
    - `cues_delivered` — cues played, waiting CUE_DURATION_S
    - `lr_detection` — listening for LRLR eye signal confirmation
    - `lucid_confirmed` — lucid dream confirmed (EOG or user button)
    - `retry` — attempt failed, waiting before retry
    """
    status = _inducer.get_status(user_id)
    if status.get("status") == "no_session":
        raise HTTPException(status_code=404, detail=f"No active session for user '{user_id}'")
    return _numpy_safe(status)


@router.post("/lucid-induction/confirm/{user_id}")
async def confirm_lucidity(user_id: str):
    """Manually confirm lucidity (e.g. user presses in-app button after waking).

    Increments lucid_episodes and moves state to lucid_confirmed.
    Use when LRLR eye movement detection failed but user reports lucidity.
    """
    result = _inducer.confirm_lucidity(user_id)
    if result.get("status") == "no_session":
        raise HTTPException(status_code=404, detail=f"No active session for user '{user_id}'")
    return _numpy_safe(result)


@router.post("/lucid-induction/stop/{user_id}")
async def stop_induction(user_id: str):
    """Stop and tear down the induction session, returning session summary."""
    result = _inducer.stop_session(user_id)
    return _numpy_safe(result)


@router.get("/lucid-induction/sessions")
async def list_induction_sessions():
    """List all active lucid dream induction sessions."""
    return {"sessions": _numpy_safe(_inducer.list_sessions())}

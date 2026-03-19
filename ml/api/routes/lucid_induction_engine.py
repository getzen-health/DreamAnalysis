"""Lucid dream induction engine endpoints — issue #452.

EEG-based REM detection + cue timing + technique tracking + reality test
scheduling. Complements the existing LuciEntry closed-loop routes in
lucid_induction.py with higher-level technique management and profiling.

Endpoints:
  POST /lucid-induction-engine/detect-rem       Check if currently in REM
  POST /lucid-induction-engine/compute-cue      Compute optimal cue timing/type
  POST /lucid-induction-engine/record-attempt    Record induction attempt outcome
  GET  /lucid-induction-engine/profile/{user_id} Get lucid dreaming profile
  GET  /lucid-induction-engine/status            Engine availability check
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
router = APIRouter()

# -- Lazy singleton -----------------------------------------------------------

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        try:
            from models.lucid_induction_engine import get_lucid_induction_engine
            _engine = get_lucid_induction_engine()
        except Exception as exc:
            log.warning("LucidInductionEngine unavailable: %s", exc)
    return _engine


# -- Pydantic schemas ---------------------------------------------------------

class DetectREMRequest(BaseModel):
    user_id: str = Field("default", description="User identifier")
    eeg_data: List[List[float]] = Field(
        ...,
        description=(
            "EEG multichannel data: [[ch0_samples...], [ch1_samples...], ...] "
            "(ch0=TP9, ch1=AF7, ch2=AF8, ch3=TP10)"
        ),
    )
    fs: float = Field(256.0, description="Sampling rate (Hz)")


class ComputeCueRequest(BaseModel):
    user_id: str = Field("default", description="User identifier")
    theta_power: float = Field(0.3, description="Relative theta power 0-1", ge=0.0, le=1.0)
    alpha_power: float = Field(0.1, description="Relative alpha power 0-1", ge=0.0, le=1.0)
    beta_power: float = Field(0.1, description="Relative beta power 0-1", ge=0.0, le=1.0)
    delta_power: float = Field(0.1, description="Relative delta power 0-1", ge=0.0, le=1.0)
    gamma_power: float = Field(0.05, description="Relative gamma power 0-1", ge=0.0, le=1.0)
    emg_amplitude: float = Field(5.0, description="EMG amplitude in uV RMS", ge=0.0)
    spectral_entropy: float = Field(0.7, description="Spectral entropy 0-1", ge=0.0, le=1.0)
    has_k_complex: bool = Field(False, description="K-complex detected in epoch")
    rem_duration_s: float = Field(0.0, description="Cumulative stable REM duration (seconds)", ge=0.0)
    preferred_cue: Optional[str] = Field(
        None, description="Preferred cue type: 'audio', 'haptic', or 'led'"
    )


class RecordAttemptRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    technique: str = Field(
        "external_cue",
        description="Induction technique: 'mild', 'wbtb', or 'external_cue'",
    )
    cue_type: Optional[str] = Field(
        None, description="Cue type used: 'audio', 'haptic', or 'led' (if applicable)"
    )
    cue_intensity: float = Field(0.3, description="Cue intensity 0-1", ge=0.0, le=1.0)
    lucid_reported: bool = Field(False, description="Did user report a lucid dream?")
    dream_recalled: bool = Field(True, description="Did user recall any dream?")
    rem_duration_s: float = Field(0.0, description="REM duration before cue (seconds)", ge=0.0)
    notes: str = Field("", description="Free-text notes about the attempt")


# -- Endpoints ----------------------------------------------------------------

@router.post("/lucid-induction-engine/detect-rem")
async def detect_rem(req: DetectREMRequest):
    """Check if the user is currently in REM based on EEG epoch.

    Analyses EEG features (theta dominance, alpha suppression, low EMG,
    desynchronized EEG, rapid eye movements) and returns a composite
    REM score with state classification.
    """
    engine = _get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="LucidInductionEngine unavailable")

    import numpy as np
    eeg = np.array(req.eeg_data, dtype=np.float32)
    if eeg.ndim == 1:
        eeg = eeg.reshape(1, -1)

    result = engine.detect_rem_state(eeg, fs=req.fs, user_id=req.user_id)
    return result


@router.post("/lucid-induction-engine/compute-cue")
async def compute_cue(req: ComputeCueRequest):
    """Compute optimal cue timing and select cue type.

    Given current EEG features and cumulative REM duration, determines
    whether conditions are optimal for cue delivery and recommends
    cue type and intensity based on user history.
    """
    engine = _get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="LucidInductionEngine unavailable")

    from models.lucid_induction_engine import SleepEEGFeatures, CueType

    features = SleepEEGFeatures(
        theta_power=req.theta_power,
        alpha_power=req.alpha_power,
        beta_power=req.beta_power,
        delta_power=req.delta_power,
        gamma_power=req.gamma_power,
        emg_amplitude=req.emg_amplitude,
        spectral_entropy=req.spectral_entropy,
        has_k_complex=req.has_k_complex,
    )

    timing = engine.compute_cue_timing(
        features, req.rem_duration_s, user_id=req.user_id
    )

    preferred = None
    if req.preferred_cue:
        try:
            preferred = CueType(req.preferred_cue)
        except ValueError:
            pass

    cue_config = engine.select_cue_type(
        user_id=req.user_id, preferred=preferred
    )

    return {
        **timing,
        "recommended_cue": {
            "type": cue_config.cue_type.value,
            "intensity": cue_config.intensity,
            "duration_s": cue_config.duration_s,
            "pattern": cue_config.pattern,
            "repeat_count": cue_config.repeat_count,
        },
    }


@router.post("/lucid-induction-engine/record-attempt")
async def record_attempt(req: RecordAttemptRequest):
    """Record the outcome of a lucid dream induction attempt.

    Tracks success/failure per technique and cue type, rebuilding
    the user's lucid dreaming profile.
    """
    engine = _get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="LucidInductionEngine unavailable")

    from models.lucid_induction_engine import (
        InductionTechnique,
        CueType,
        CueConfig,
    )

    try:
        technique = InductionTechnique(req.technique)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid technique '{req.technique}'. "
                   f"Must be one of: mild, wbtb, external_cue",
        )

    cue_config = None
    if req.cue_type:
        try:
            ct = CueType(req.cue_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid cue_type '{req.cue_type}'. "
                       f"Must be one of: audio, haptic, led",
            )
        cue_config = CueConfig(
            cue_type=ct,
            intensity=req.cue_intensity,
        )

    result = engine.track_induction_success(
        user_id=req.user_id,
        technique=technique,
        cue_config=cue_config,
        lucid_reported=req.lucid_reported,
        dream_recalled=req.dream_recalled,
        rem_duration_s=req.rem_duration_s,
        notes=req.notes,
    )
    return result


@router.get("/lucid-induction-engine/profile/{user_id}")
async def get_profile(user_id: str):
    """Get the lucid dreaming profile for a user.

    Returns aggregated statistics: success rates per technique and
    cue type, best technique/cue, reality test counts, and more.
    """
    engine = _get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="LucidInductionEngine unavailable")

    profile = engine.compute_lucid_profile(user_id)
    if profile is None:
        raise HTTPException(
            status_code=404,
            detail=f"No profile found for user '{user_id}'",
        )

    return engine.profile_to_dict(profile)


@router.get("/lucid-induction-engine/status")
async def engine_status():
    """Check if the lucid induction engine is available."""
    engine = _get_engine()
    return {
        "available": engine is not None,
        "model": "LucidInductionEngine",
        "version": "1.0.0",
        "features": [
            "rem_detection",
            "cue_timing",
            "cue_selection",
            "success_tracking",
            "reality_test_scheduling",
            "lucid_profiling",
        ],
    }

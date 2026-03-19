"""Passive respiratory biometrics API routes.

Endpoints:
  POST /respiratory/analyze    -- analyze audio amplitude envelope for breathing
  POST /respiratory/correlate  -- correlate breathing patterns with emotion data
  GET  /respiratory/status     -- availability check

References:
  - Issue #416: Breath-as-input passive respiratory biometrics
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from models.respiratory_model import (
    classify_breathing_pattern,
    compute_respiratory_emotion_correlation,
    compute_respiratory_features,
    extract_respiratory_rate,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/respiratory", tags=["respiratory"])


# ── Schemas ──────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    amplitude: List[float] = Field(
        ...,
        description="1-D array of audio amplitude values (mono waveform or envelope)",
        min_length=1,
    )
    sample_rate: float = Field(
        16000.0,
        gt=0,
        description="Sample rate in Hz (default 16000)",
    )


class EmotionEntry(BaseModel):
    valence: float = Field(..., ge=-1.0, le=1.0)
    arousal: float = Field(..., ge=0.0, le=1.0)
    timestamp: float = Field(..., description="Unix epoch seconds")


class RespiratoryEntry(BaseModel):
    breaths_per_minute: float = Field(..., ge=0)
    breath_regularity: float = Field(..., ge=0)
    timestamp: float = Field(..., description="Unix epoch seconds")
    inhalation_exhalation_ratio: Optional[float] = None
    sigh_count: Optional[int] = None


class CorrelateRequest(BaseModel):
    respiratory_history: List[RespiratoryEntry] = Field(
        ...,
        min_length=3,
        description="At least 3 respiratory observations with timestamps",
    )
    emotion_history: List[EmotionEntry] = Field(
        ...,
        min_length=3,
        description="At least 3 emotion observations with timestamps",
    )


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/analyze")
def analyze_respiratory(req: AnalyzeRequest) -> Dict[str, Any]:
    """Analyze audio amplitude envelope for breathing patterns.

    Accepts raw audio amplitude values, extracts respiratory features,
    and classifies the breathing pattern into a physiological state.
    """
    import numpy as np

    amplitude = np.array(req.amplitude, dtype=np.float64)

    try:
        features = compute_respiratory_features(amplitude, req.sample_rate)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if "error" in features:
        return {
            "status": "error",
            "error": features["error"],
            **{k: v for k, v in features.items() if k != "error"},
        }

    classification = classify_breathing_pattern(features)

    # Remove the large filtered_envelope from the response
    features.pop("filtered_envelope", None)

    return {
        "status": "ok",
        "features": features,
        "classification": classification,
    }


@router.post("/correlate")
def correlate_respiratory_emotion(req: CorrelateRequest) -> Dict[str, Any]:
    """Correlate breathing patterns with emotion data over time.

    Requires at least 3 time-aligned respiratory + emotion observations.
    Returns Pearson correlations between respiratory features and
    valence/arousal, plus a personal baseline summary.
    """
    resp_history = [entry.model_dump() for entry in req.respiratory_history]
    emo_history = [entry.model_dump() for entry in req.emotion_history]

    result = compute_respiratory_emotion_correlation(resp_history, emo_history)

    if "error" in result:
        return {"status": "error", **result}

    return {"status": "ok", **result}


@router.get("/status")
def respiratory_status() -> Dict[str, Any]:
    """Check whether the respiratory analysis module is available."""
    try:
        from models.respiratory_model import _SCIPY_AVAILABLE

        return {
            "available": _SCIPY_AVAILABLE,
            "module": "respiratory_model",
            "features": [
                "respiratory_rate_extraction",
                "breathing_pattern_classification",
                "emotion_correlation",
                "sigh_detection",
            ],
        }
    except ImportError:
        return {
            "available": False,
            "module": "respiratory_model",
            "error": "module_not_found",
        }

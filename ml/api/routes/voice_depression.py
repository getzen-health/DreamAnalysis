"""Voice biomarker mood/anxiety wellness estimation endpoints (#402).

POST /voice-screening/screen  -- voice-derived mood + anxiety wellness indicators
GET  /voice-screening/status  -- availability check
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
router = APIRouter(prefix="/voice-screening", tags=["voice-depression-screener"])


# -- request / response schemas -----------------------------------------------

class VoiceScreeningInput(BaseModel):
    audio_samples: List[float] = Field(
        ..., description="Raw audio samples (mono, float, 2+ seconds)"
    )
    sample_rate: int = Field(16000, description="Audio sample rate in Hz")
    user_id: str = Field("anonymous", description="User identifier for tracking")
    include_biomarkers: bool = Field(
        False, description="Include raw biomarker values in response"
    )


class DepressionResult(BaseModel):
    phq9_score: float = Field(0.0, description="Voice-derived risk score mapped to PHQ-9 scale (NOT a validated PHQ-9 result)")
    severity: str = "unknown"
    indicators: List[str] = []
    not_validated: bool = True
    scale_context: str = ""
    disclaimer: str = ""


class AnxietyResult(BaseModel):
    gad7_score: float = Field(0.0, description="Voice-derived risk score mapped to GAD-7 scale (NOT a validated GAD-7 result)")
    severity: str = "unknown"
    indicators: List[str] = []
    not_validated: bool = True
    scale_context: str = ""
    disclaimer: str = ""


class VoiceScreeningResponse(BaseModel):
    user_id: str
    depression: DepressionResult
    anxiety: AnxietyResult
    biomarkers: Optional[Dict[str, Any]] = None
    processed_at: float = 0.0
    disclaimer: str = (
        "IMPORTANT: This is not a medical device. These voice-derived wellness "
        "scores are NOT validated PHQ-9 or GAD-7 questionnaire results. They are "
        "experimental estimates for wellness awareness only, not validated clinical "
        "assessments. Do not use for self-assessment or to inform health decisions. "
        "Consult a qualified mental health professional for any concerns."
    )


# -- endpoints ----------------------------------------------------------------

@router.post("/screen", response_model=VoiceScreeningResponse)
async def screen_voice(req: VoiceScreeningInput):
    """Estimate mood and anxiety wellness indicators from voice audio.

    Extracts vocal biomarkers (pitch variability, pause frequency, jitter,
    shimmer, formant stability, energy contour) and produces voice-derived
    wellness scores. Scores are mapped onto PHQ-9 (0-27) and GAD-7 (0-21)
    scales for readability but are NOT validated PHQ-9/GAD-7 results.
    These are wellness estimates only, not validated clinical assessments.

    This is not a medical device. Do not use to inform health decisions.
    Minimum 2 seconds of audio; 10+ seconds recommended.
    """
    from models.voice_depression_screener import (
        compute_screening_profile,
        profile_to_dict,
    )

    audio = np.array(req.audio_samples, dtype=np.float32)

    if len(audio) < req.sample_rate * 2:
        raise HTTPException(422, "Audio too short -- need at least 2 seconds")

    profile = compute_screening_profile(audio, sr=req.sample_rate)
    profile = profile_to_dict(profile)

    depression = profile.get("depression", {})
    anxiety = profile.get("anxiety", {})

    return VoiceScreeningResponse(
        user_id=req.user_id,
        depression=DepressionResult(
            phq9_score=depression.get("phq9_score", 0.0),
            severity=depression.get("severity", "unknown"),
            indicators=depression.get("indicators", []),
            not_validated=depression.get("not_validated", True),
            scale_context=depression.get("scale_context", ""),
            disclaimer=depression.get("disclaimer", ""),
        ),
        anxiety=AnxietyResult(
            gad7_score=anxiety.get("gad7_score", 0.0),
            severity=anxiety.get("severity", "unknown"),
            indicators=anxiety.get("indicators", []),
            not_validated=anxiety.get("not_validated", True),
            scale_context=anxiety.get("scale_context", ""),
            disclaimer=anxiety.get("disclaimer", ""),
        ),
        biomarkers=profile.get("biomarkers") if req.include_biomarkers else None,
        processed_at=time.time(),
    )


@router.get("/status")
async def voice_screening_status() -> Dict[str, Any]:
    """Check availability of voice mood/anxiety wellness estimation."""
    numpy_ok = True
    try:
        import numpy  # noqa: F401
    except ImportError:
        numpy_ok = False

    return {
        "ready": numpy_ok,
        "numpy_available": numpy_ok,
        "screening_models": ["depression_voice_risk", "anxiety_voice_risk"],
        "not_validated": True,
        "wellness_notice": (
            "This is not a medical device. These models produce voice-derived "
            "wellness estimates mapped onto PHQ-9/GAD-7 scales for readability. "
            "They are NOT validated PHQ-9 or GAD-7 questionnaire instruments "
            "and are for wellness awareness only, not validated clinical assessments."
        ),
        "features": [
            "pitch_variability", "f0_mean", "f0_std", "f0_range",
            "jitter_local", "jitter_rap",
            "shimmer_local", "shimmer_apq3",
            "silence_ratio", "pause_count", "mean_pause_duration",
            "speaking_rate", "speaking_rate_variability",
            "formant_stability",
            "energy_mean", "energy_std", "energy_range", "energy_slope",
        ],
        "scales": {
            "phq9": {
                "range": "0-27",
                "note": "Voice-derived risk mapped to PHQ-9 scale — NOT a validated PHQ-9 result",
                "thresholds": {"minimal": 4, "mild": 9, "moderate": 14, "severe": 20},
            },
            "gad7": {
                "range": "0-21",
                "note": "Voice-derived risk mapped to GAD-7 scale — NOT a validated GAD-7 result",
                "thresholds": {"minimal": 4, "mild": 9, "moderate": 14, "severe": 15},
            },
        },
    }

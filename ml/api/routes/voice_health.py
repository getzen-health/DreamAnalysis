"""Voice health API router.

Prefix:  /voice-health
Tag:     Voice Health

Endpoints
---------
POST /voice-health/analyze                   Run full health analysis on audio
GET  /voice-health/baseline/{user_id}        Retrieve stored baseline
POST /voice-health/baseline/{user_id}/reset  Clear baseline for user
GET  /voice-health/indicators                Reference dict of all indicators
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from models.voice_health_analyzer import VoiceHealthAnalyzer

router = APIRouter(prefix="/voice-health", tags=["Voice Health"])

# Module-level singleton so baselines persist across requests
_analyzer = VoiceHealthAnalyzer()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    user_id: str = Field(default="default", description="Per-user baseline key")
    audio_data: List[float] = Field(..., description="Raw audio samples as floats")
    sample_rate: int = Field(default=16000, description="Sample rate in Hz")
    update_baseline: bool = Field(
        default=True,
        description="Whether to update the per-user baseline after this analysis",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/analyze")
async def analyze_voice_health(body: AnalyzeRequest) -> Dict[str, Any]:
    """Analyze voice audio for physical health indicators.

    Computes fatigue index, cognitive sharpness, wellness score, and
    per-user baseline comparison from the provided audio samples.
    """
    if not body.audio_data:
        raise HTTPException(status_code=422, detail="audio_data must not be empty")

    audio = np.array(body.audio_data, dtype=np.float64)
    result = _analyzer.analyze(audio, body.sample_rate, body.user_id)

    if body.update_baseline:
        _analyzer.update_baseline(body.user_id, result["raw_features"])

    return result


@router.get("/baseline/{user_id}")
async def get_baseline(user_id: str) -> Dict[str, Any]:
    """Return the stored per-user voice baseline.

    Returns ``{"status": "no_baseline"}`` when no baseline has been recorded yet.
    """
    bl = _analyzer._baselines.get(user_id)
    if bl is None:
        return {"status": "no_baseline"}
    return {
        "status": "ok",
        "user_id": user_id,
        "n_samples": bl["n"],
        "feature_means": {k: round(v, 6) for k, v in bl["mean"].items()},
        "feature_stds": {k: round(v, 6) for k, v in bl["std"].items()},
    }


@router.post("/baseline/{user_id}/reset")
async def reset_baseline(user_id: str) -> Dict[str, str]:
    """Clear the stored baseline for this user."""
    _analyzer._baselines.pop(user_id, None)
    return {"status": "reset", "user_id": user_id}


@router.get("/indicators")
async def get_indicators() -> Dict[str, Any]:
    """Return a reference dictionary explaining each indicator and its research basis."""
    return {
        "fatigue_index": {
            "range": "0.0 – 1.0",
            "higher_means": "more fatigued",
            "primary_signals": ["f0_variability", "speech_rate", "pause_ratio"],
            "research_basis": (
                "Vocal fatigue is associated with reduced fundamental frequency variability "
                "(Welham & Maclagan 2003), slower speech rate, and increased pause duration "
                "(Klasmeyer & Sendlmeier 1995). These patterns emerge in sleep deprivation "
                "and physical fatigue studies (Harrison & Horne 2000)."
            ),
            "flag_threshold": 0.65,
        },
        "cognitive_sharpness": {
            "range": "0.0 – 1.0",
            "higher_means": "sharper cognitive state",
            "primary_signals": ["articulation_rate", "hnr_estimate"],
            "research_basis": (
                "Articulation rate correlates with working memory load and processing speed "
                "(Goldman-Eisler 1968). Harmonics-to-noise ratio reflects vocal tract control "
                "and is sensitive to fatigue and cognitive load (Baken & Orlikoff 2000)."
            ),
            "flag_threshold": 0.35,
        },
        "voice_wellness_score": {
            "range": "0.0 – 1.0",
            "higher_means": "healthier overall voice",
            "primary_signals": ["fatigue_index", "cognitive_sharpness", "voice_energy"],
            "research_basis": (
                "Composite index combining inverse fatigue, cognitive sharpness, and signal "
                "energy level. Energy level proxies vocal effort and respiratory support "
                "(Titze 1994)."
            ),
        },
        "voice_health_change": {
            "range": "-1.0 – 1.0",
            "higher_means": "improved relative to personal baseline",
            "note": "Requires at least one prior analysis to establish baseline.",
        },
        "raw_features": {
            "speech_rate": {
                "unit": "pseudo-syllables/sec",
                "method": "zero-crossing rate proxy, empirically scaled",
                "typical_range": "2.0 – 6.0 for conversational speech",
            },
            "pause_ratio": {
                "unit": "fraction 0.0 – 1.0",
                "method": "fraction of 20 ms frames below energy threshold",
                "typical_range": "0.15 – 0.35",
            },
            "f0_variability": {
                "unit": "Hz (std dev)",
                "method": "autocorrelation pitch tracking, 40 ms frames",
                "typical_range": "20 – 60 Hz for expressive speech",
            },
            "f0_mean": {
                "unit": "Hz",
                "method": "mean of voiced frame F0 estimates",
                "typical_range": "85 – 255 Hz (sex-dependent)",
            },
            "voice_energy": {
                "unit": "RMS amplitude (normalised)",
                "method": "sqrt(mean(samples^2))",
            },
            "hnr_estimate": {
                "unit": "dB",
                "method": "cepstral autocorrelation peak ratio",
                "typical_range": "15 – 25 dB (healthy voiced speech)",
                "flag_threshold_below": 8.0,
            },
            "articulation_rate": {
                "unit": "pseudo-syllables/sec (speech-only)",
                "method": "speech_rate / (1 - pause_ratio)",
            },
        },
        "wellness_flags": {
            "possible_fatigue": "fatigue_index > 0.65",
            "reduced_vocal_expression": "f0_variability < 15 Hz",
            "high_pause_ratio": "pause_ratio > 0.45",
            "reduced_cognitive_sharpness": "cognitive_sharpness < 0.35",
            "voice_strain": "hnr_estimate < 8.0 dB",
        },
    }

"""Multimodal Emotional Intelligence integration route.

Wires Voice emotion analysis + Apple Health data into the EI composite
scoring system alongside EEG signals.

Endpoints:
  POST /multimodal-ei/assess        -- compute EIQ from EEG + voice + health data
  POST /multimodal-ei/context-prior -- compute and/or apply circadian context priors
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

from models.context_prior import ContextPrior

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/multimodal-ei", tags=["Multimodal EI"])

_ei_model = None
_context_prior = ContextPrior()


def _get_model():
    global _ei_model
    if _ei_model is None:
        from models.ei_composite import get_model
        _ei_model = get_model()
    return _ei_model


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class VoiceAnalysis(BaseModel):
    valence: float = Field(default=0.0, description="Voice emotional valence (-1 to 1)")
    arousal: float = Field(default=0.0, description="Voice emotional arousal (0 to 1)")
    confidence: float = Field(default=0.5, description="Classification confidence (0 to 1)")
    cognitive_load_index: Optional[float] = Field(
        default=None, description="Voice-derived cognitive load (0 to 1)"
    )


class HealthData(BaseModel):
    hrv_sdnn: Optional[float] = Field(default=None, description="HRV SDNN in milliseconds")
    sleep_score: Optional[float] = Field(default=None, description="Sleep quality score (0-100)")
    steps: Optional[float] = Field(default=None, description="Step count")
    heart_rate: Optional[float] = Field(default=None, description="Resting heart rate BPM")


class MultimodalEIRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    signals: Optional[List[List[float]]] = Field(
        default=None, description="EEG signals (channels x samples)"
    )
    fs: float = Field(default=256.0, description="EEG sampling frequency")
    voice_analysis: Optional[VoiceAnalysis] = Field(
        default=None, description="Voice emotion analysis results"
    )
    health_data: Optional[HealthData] = Field(
        default=None, description="Apple Health / Google Fit data"
    )


# ---------------------------------------------------------------------------
# Score computation helpers (exported for testing)
# ---------------------------------------------------------------------------

def _compute_voice_scores(voice: dict) -> Dict[str, float]:
    """Compute voice-derived EI component scores.

    Args:
        voice: Dict with keys valence, arousal, confidence,
               and optionally cognitive_load_index.

    Returns:
        Dict with voice_valence_clarity, voice_expression_range,
        voice_cognitive_load -- all clipped to [0, 1].
    """
    valence = float(voice.get("valence", 0.0))
    arousal = float(voice.get("arousal", 0.0))
    confidence = float(voice.get("confidence", 0.5))
    load_index = voice.get("cognitive_load_index")

    # voice_valence_clarity: abs(valence) * confidence
    clarity = abs(valence) * confidence

    # voice_expression_range: arousal * 0.6 + abs(valence) * 0.4
    expression = arousal * 0.6 + abs(valence) * 0.4

    # voice_cognitive_load: 1.0 - load_index (lower load = better decision capacity)
    if load_index is not None:
        cog_load = 1.0 - float(load_index)
    else:
        cog_load = 0.5  # neutral default

    return {
        "voice_valence_clarity": float(np.clip(clarity, 0.0, 1.0)),
        "voice_expression_range": float(np.clip(expression, 0.0, 1.0)),
        "voice_cognitive_load": float(np.clip(cog_load, 0.0, 1.0)),
    }


def _compute_health_scores(health: dict) -> Dict[str, float]:
    """Compute health-derived EI component scores.

    Args:
        health: Dict with optional keys hrv_sdnn, sleep_score, steps,
                heart_rate.

    Returns:
        Dict with hrv_regulation, sleep_restoration, activity_engagement
        -- all clipped to [0, 1].
    """
    hrv_sdnn = health.get("hrv_sdnn")
    sleep_score = health.get("sleep_score")
    steps = health.get("steps")

    # hrv_regulation: sigmoid with 40ms midpoint
    if hrv_sdnn is not None:
        hrv_val = float(hrv_sdnn)
        hrv_reg = 1.0 / (1.0 + np.exp(-0.1 * (hrv_val - 40.0)))
    else:
        hrv_reg = 0.5  # neutral default

    # sleep_restoration: sleep_score / 100.0
    if sleep_score is not None:
        sleep_rest = float(sleep_score) / 100.0
    else:
        sleep_rest = 0.5  # neutral default

    # activity_engagement: sigmoid of steps with 5000 midpoint
    if steps is not None:
        steps_val = float(steps)
        activity = 1.0 / (1.0 + np.exp(-0.001 * (steps_val - 5000.0)))
    else:
        activity = 0.5  # neutral default

    return {
        "hrv_regulation": float(np.clip(hrv_reg, 0.0, 1.0)),
        "sleep_restoration": float(np.clip(sleep_rest, 0.0, 1.0)),
        "activity_engagement": float(np.clip(activity, 0.0, 1.0)),
    }


def _compute_coherence_score(
    voice_analysis: Optional[dict],
    health_data: Optional[dict],
) -> float:
    """Compute health-mood coherence between voice valence and HRV trend.

    Both positive voice + high HRV = high coherence.
    Misaligned (negative voice + high HRV or positive voice + low HRV) = low.

    Returns:
        Float in [0, 1]. Defaults to 0.5 if either modality is missing.
    """
    if voice_analysis is None or health_data is None:
        return 0.5

    valence = voice_analysis.get("valence")
    hrv_sdnn = health_data.get("hrv_sdnn")

    if valence is None or hrv_sdnn is None:
        return 0.5

    valence = float(valence)
    hrv_sdnn = float(hrv_sdnn)

    # Normalise HRV to [-1, 1] range: <20ms = -1, 40ms = 0, >60ms = +1
    hrv_norm = float(np.clip((hrv_sdnn - 40.0) / 20.0, -1.0, 1.0))

    # Coherence: both same sign = aligned. Product > 0 = coherent.
    # Map product from [-1, 1] to [0, 1] via (product + 1) / 2
    product = valence * hrv_norm
    coherence = (product + 1.0) / 2.0

    return float(np.clip(coherence, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Core assessment function (exported for direct testing)
# ---------------------------------------------------------------------------

def _assess_multimodal_ei(
    user_id: str,
    signals: Optional[np.ndarray] = None,
    fs: float = 256.0,
    voice_analysis: Optional[dict] = None,
    health_data: Optional[dict] = None,
) -> Optional[Dict]:
    """Compute multimodal EIQ assessment.

    Args:
        user_id: User identifier for per-user state.
        signals: EEG array, shape (n_channels, n_samples). Can be None.
        fs: EEG sampling rate.
        voice_analysis: Dict with valence, arousal, confidence keys.
        health_data: Dict with hrv_sdnn, sleep_score, steps keys.

    Returns:
        Dict with eiq_score, dimensions, modalities_used, component_scores.
        Returns None if no modalities are provided.
    """
    has_eeg = signals is not None
    has_voice = voice_analysis is not None
    has_health = health_data is not None

    if not has_eeg and not has_voice and not has_health:
        return None

    # Build the component_scores dict from voice and health
    component_scores: Dict[str, float] = {}
    modalities: list = []

    if has_eeg:
        modalities.append("eeg")

    if has_voice:
        modalities.append("voice")
        voice_scores = _compute_voice_scores(voice_analysis)
        component_scores.update(voice_scores)

    if has_health:
        modalities.append("health")
        health_scores = _compute_health_scores(health_data)
        component_scores.update(health_scores)

    # Coherence requires both voice and health
    if has_voice and has_health:
        coherence = _compute_coherence_score(voice_analysis, health_data)
        component_scores["health_mood_coherence"] = coherence

    # Call the EI composite model
    model = _get_model()
    eeg_array = signals if has_eeg else None
    comp_dict = component_scores if component_scores else None

    result = model.compute_eiq(
        signals=eeg_array,
        fs=fs,
        component_scores=comp_dict,
        user_id=user_id,
    )

    if result is None:
        return None

    result["modalities_used"] = modalities
    result["component_scores"] = component_scores
    result["processed_at"] = time.time()

    return result


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/assess")
async def assess_multimodal_ei(req: MultimodalEIRequest):
    """Compute Emotional Intelligence Quotient from EEG + voice + health data.

    Accepts any combination of modalities. At least one must be provided.
    Voice and health data are transformed into EI component scores and blended
    with EEG-derived dimension scores via the EI composite model.

    Returns EIQ score, grade, dimension breakdown, modalities used, and
    individual component scores.
    """
    # Convert EEG signals if provided
    eeg_array = None
    if req.signals is not None:
        eeg_array = np.array(req.signals, dtype=float)
        if eeg_array.ndim == 1:
            eeg_array = eeg_array[np.newaxis, :]

    # Convert Pydantic models to dicts for the helper functions
    voice_dict = None
    if req.voice_analysis is not None:
        voice_dict = req.voice_analysis.model_dump()

    health_dict = None
    if req.health_data is not None:
        health_dict = req.health_data.model_dump()

    result = _assess_multimodal_ei(
        user_id=req.user_id,
        signals=eeg_array,
        fs=req.fs,
        voice_analysis=voice_dict,
        health_data=health_dict,
    )

    if result is None:
        return {
            "user_id": req.user_id,
            "status": "no_data",
            "message": "At least one modality (EEG, voice, or health) is required.",
        }

    return {"user_id": req.user_id, **result}


# ---------------------------------------------------------------------------
# Context-prior endpoint
# ---------------------------------------------------------------------------


class ContextPriorRequest(BaseModel):
    """Request body for context-aware emotion prior computation."""
    hour: Optional[float] = Field(
        default=None,
        description="Hour of day (0-24). Inferred from server time if omitted.",
    )
    sleep_score: Optional[float] = Field(
        default=None,
        description="Sleep quality score (0-100) from Apple Health / Oura / etc.",
    )
    steps: Optional[float] = Field(
        default=None,
        description="Step count so far today.",
    )
    recent_caffeine: bool = Field(
        default=False,
        description="True if caffeine was consumed in the last 2 hours.",
    )
    day_of_week: Optional[int] = Field(
        default=None,
        description="Day of week (0=Monday … 6=Sunday). Inferred from server time if omitted.",
    )
    # Optional raw prediction to adjust
    raw_valence: Optional[float] = Field(
        default=None,
        description="Raw model valence to adjust (-1 to 1). If provided, returns adjusted values.",
    )
    raw_arousal: Optional[float] = Field(
        default=None,
        description="Raw model arousal to adjust (0 to 1). If provided, returns adjusted values.",
    )
    raw_stress_index: Optional[float] = Field(
        default=None,
        description="Raw stress index to adjust (0 to 1). If provided, returns adjusted values.",
    )
    raw_focus_index: Optional[float] = Field(
        default=None,
        description="Raw focus index to adjust (0 to 1). If provided, returns adjusted values.",
    )
    confidence: float = Field(
        default=0.5,
        description="Model prediction confidence (0-1). Lower = more context weight applied.",
    )


@router.post("/context-prior")
async def compute_context_prior(req: ContextPriorRequest):
    """Compute circadian and contextual emotion priors.

    Returns expected emotion offsets from time-of-day, sleep quality,
    activity, caffeine, and day-of-week signals — all without new hardware.

    If raw emotion values + confidence are supplied, also returns context-
    adjusted predictions. Useful for correcting false positives such as
    'stressed' readings first thing in the morning (just woke up) or
    low-energy voice readings that are normal for the post-lunch dip.

    Research basis: Stone et al. (2006), Wilhelm et al. (2011),
    Monk (2005), Nehlig (2010) caffeine timeline.
    """
    now = datetime.now()
    hour = req.hour if req.hour is not None else (now.hour + now.minute / 60.0)
    day_of_week = req.day_of_week if req.day_of_week is not None else now.weekday()

    prior = _context_prior.compute_prior(
        hour=hour,
        sleep_score=req.sleep_score,
        steps=req.steps,
        recent_caffeine=req.recent_caffeine,
        day_of_week=day_of_week,
    )

    response: dict = {
        "prior": prior,
        "inferred_hour": round(hour, 2),
        "inferred_day_of_week": day_of_week,
    }

    # Apply adjustment if raw values provided
    has_raw = any(v is not None for v in [
        req.raw_valence, req.raw_arousal, req.raw_stress_index, req.raw_focus_index,
    ])
    if has_raw:
        raw = {}
        if req.raw_valence is not None:
            raw["valence"] = req.raw_valence
        if req.raw_arousal is not None:
            raw["arousal"] = req.raw_arousal
        if req.raw_stress_index is not None:
            raw["stress_index"] = req.raw_stress_index
        if req.raw_focus_index is not None:
            raw["focus_index"] = req.raw_focus_index

        adjusted = _context_prior.adjust_prediction(
            raw_prediction=raw,
            prior=prior,
            confidence=req.confidence,
        )
        response["adjusted"] = adjusted

    return response

"""Ambient sound environment profiling API.

Endpoints:
  POST /ambient-sound/classify   -- classify sound environment from features
  POST /ambient-sound/correlate  -- compute sound-emotion correlations
  GET  /ambient-sound/status     -- availability check

GitHub issue: #417
"""

from typing import Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ._shared import _numpy_safe
from models.ambient_sound_model import (
    classify_sound_environment,
    compute_sound_features,
    correlate_sound_emotion,
    generate_sound_insights,
    SOUND_CATEGORIES,
)

router = APIRouter(prefix="/ambient-sound", tags=["ambient-sound"])


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class SoundFeaturesInput(BaseModel):
    """Pre-computed audio features for sound environment classification."""

    spectral_centroid: float = Field(..., description="Centre of spectral mass in Hz")
    spectral_energy: float = Field(..., description="RMS energy of the signal")
    zero_crossing_rate: float = Field(..., description="Fraction of sign-change frames")
    mfcc: Optional[List[float]] = Field(
        default=None,
        description="13-element MFCC vector (optional)",
    )
    spectral_bandwidth: Optional[float] = Field(
        default=None,
        description="Spread of spectrum in Hz (optional)",
    )
    spectral_rolloff: Optional[float] = Field(
        default=None,
        description="Roll-off frequency in Hz (optional)",
    )
    sample_rate: float = Field(
        default=22050.0,
        description="Audio sample rate used during feature extraction",
    )


class SoundRecord(BaseModel):
    category: str = Field(..., description="Sound category label")
    timestamp: float = Field(..., description="Unix timestamp")


class EmotionRecord(BaseModel):
    emotion: str = Field(..., description="Emotion label (happy, sad, etc.)")
    valence: float = Field(..., description="Valence score (-1 to 1)")
    arousal: float = Field(..., description="Arousal score (0 to 1)")
    timestamp: float = Field(..., description="Unix timestamp")


class CorrelateInput(BaseModel):
    """Paired sound and emotion records for correlation analysis."""

    sound_records: List[SoundRecord] = Field(
        ..., description="List of sound classification records"
    )
    emotion_records: List[EmotionRecord] = Field(
        ..., description="List of emotion classification records"
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/classify")
async def classify_sound(data: SoundFeaturesInput):
    """Classify ambient sound environment from pre-computed audio features.

    Categories: silence, nature, urban, social, music, indoor, white_noise.

    Returns the predicted category, confidence score, per-category scores,
    and a feature summary.
    """
    features = compute_sound_features(
        spectral_centroid=data.spectral_centroid,
        spectral_energy=data.spectral_energy,
        zero_crossing_rate=data.zero_crossing_rate,
        mfcc=data.mfcc,
        spectral_bandwidth=data.spectral_bandwidth,
        spectral_rolloff=data.spectral_rolloff,
        sample_rate=data.sample_rate,
    )
    result = classify_sound_environment(features)
    return _numpy_safe(result)


@router.post("/correlate")
async def correlate_sound_and_emotion(data: CorrelateInput):
    """Compute correlations between sound environments and emotional states.

    Pairs sound records with emotion records by nearest timestamp (within 30s).
    Returns per-category emotion statistics, valence/arousal breakdowns,
    and actionable insights.
    """
    sound_dicts = [r.model_dump() for r in data.sound_records]
    emotion_dicts = [r.model_dump() for r in data.emotion_records]

    correlation = correlate_sound_emotion(sound_dicts, emotion_dicts)
    insights = generate_sound_insights(correlation)

    return _numpy_safe({
        "correlation": correlation,
        "insights": insights,
    })


@router.get("/status")
async def ambient_sound_status():
    """Ambient sound profiling availability check.

    Returns supported categories and module readiness.
    """
    return {
        "available": True,
        "categories": SOUND_CATEGORIES,
        "version": "0.1.0",
    }

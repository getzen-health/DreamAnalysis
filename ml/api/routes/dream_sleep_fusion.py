"""Dream-sleep fusion API routes — issue #437.

Endpoints:
  POST /dream-sleep/correlate  — correlate dream entries with sleep data
  POST /dream-sleep/predict    — predict dream type from tonight's sleep data
  GET  /dream-sleep/status     — model availability check
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
router = APIRouter(prefix="/dream-sleep", tags=["dream-sleep-fusion"])

# -- Lazy singleton -----------------------------------------------------------

_model = None


def _get_model():
    global _model
    if _model is None:
        try:
            from models.dream_sleep_fusion import get_dream_sleep_fusion_model  # type: ignore
            _model = get_dream_sleep_fusion_model()
        except Exception as exc:
            log.warning("DreamSleepFusionModel unavailable: %s", exc)
    return _model


# -- Pydantic schemas ---------------------------------------------------------

class DreamEntrySchema(BaseModel):
    text: str = Field(..., description="Dream journal text")
    date: str = Field("", description="Date of the dream (ISO format or free text)")
    lucidity_score: float = Field(
        0.0,
        description="Self-reported lucidity 0-1 (optional, augments text analysis)",
        ge=0.0,
        le=1.0,
    )


class SleepMetricsSchema(BaseModel):
    rem_pct: float = Field(
        0.22, description="REM fraction 0-1", ge=0.0, le=1.0,
    )
    deep_pct: float = Field(
        0.20, description="N3 deep sleep fraction 0-1", ge=0.0, le=1.0,
    )
    efficiency: float = Field(
        0.85, description="Time-asleep / time-in-bed 0-1", ge=0.0, le=1.0,
    )
    duration_h: float = Field(
        7.5, description="Total sleep hours", ge=0.0, le=24.0,
    )
    awakenings: int = Field(
        0, description="Number of awakenings during the night", ge=0, le=50,
    )


class CorrelateRequest(BaseModel):
    dreams: List[DreamEntrySchema] = Field(
        ..., description="Dream journal entries (1 or more)",
    )
    sleep_data: List[SleepMetricsSchema] = Field(
        ..., description="Sleep metrics matching each dream entry (same length)",
    )


class PredictRequest(BaseModel):
    sleep: SleepMetricsSchema = Field(
        ..., description="Tonight's sleep metrics",
    )
    recent_stress: float = Field(
        0.0,
        description="Recent stress level 0-1 (from EEG or self-report)",
        ge=0.0,
        le=1.0,
    )
    recent_emotions: Optional[Dict[str, float]] = Field(
        None,
        description="Recent emotional state dict (e.g. {'anxiety': 0.6, 'joy': 0.2})",
    )


# -- Endpoints ----------------------------------------------------------------

@router.post("/correlate")
def correlate_dream_sleep(req: CorrelateRequest) -> Dict[str, Any]:
    """Correlate dream entries with sleep data.

    Accepts matched lists of dream journal entries and sleep metrics.
    Returns per-entry correlations plus an aggregated profile if 2+ entries.
    """
    model = _get_model()
    if model is None:
        raise HTTPException(
            503,
            "DreamSleepFusionModel unavailable -- check server logs for import errors",
        )

    if len(req.dreams) != len(req.sleep_data):
        raise HTTPException(
            422,
            f"dreams ({len(req.dreams)}) and sleep_data ({len(req.sleep_data)}) "
            "must have the same length",
        )

    if len(req.dreams) == 0:
        raise HTTPException(422, "At least one dream-sleep pair is required")

    if len(req.dreams) > 365:
        raise HTTPException(422, "Maximum 365 entries per request")

    from models.dream_sleep_fusion import (  # type: ignore
        DreamEntry,
        SleepMetrics,
        profile_to_dict,
    )

    dream_entries = [
        DreamEntry(
            text=d.text,
            date=d.date,
            lucidity_score=d.lucidity_score,
        )
        for d in req.dreams
    ]
    sleep_entries = [
        SleepMetrics(
            rem_pct=s.rem_pct,
            deep_pct=s.deep_pct,
            efficiency=s.efficiency,
            duration_h=s.duration_h,
            awakenings=s.awakenings,
        )
        for s in req.sleep_data
    ]

    # Per-entry correlations
    correlations = []
    for dream, sleep in zip(dream_entries, sleep_entries):
        corr = model.correlate(dream, sleep)
        correlations.append({
            "dream_date": corr.dream_date,
            "primary_emotion": corr.primary_emotion,
            "primary_theme": corr.primary_theme,
            "emotions": corr.emotions,
            "themes": corr.themes,
            "sleep_quality_score": corr.sleep_quality_score,
            "correlation_scores": corr.correlation_scores,
            "insights": corr.insights,
            "nightmare_risk": corr.nightmare_risk,
            "lucidity_score": corr.lucidity_score,
        })

    # Aggregated profile when multiple entries
    profile = None
    if len(dream_entries) >= 2:
        p = model.compute_profile(dream_entries, sleep_entries)
        profile = profile_to_dict(p)

    return {
        "status": "ok",
        "count": len(correlations),
        "correlations": correlations,
        "profile": profile,
    }


@router.post("/predict")
def predict_dream_type(req: PredictRequest) -> Dict[str, Any]:
    """Predict dream type distribution from tonight's sleep data.

    Returns probabilities for each dream type (vivid_positive, nightmare,
    lucid, mundane, etc.), plus nightmare risk and recall probability.
    """
    model = _get_model()
    if model is None:
        raise HTTPException(
            503,
            "DreamSleepFusionModel unavailable -- check server logs for import errors",
        )

    from models.dream_sleep_fusion import SleepMetrics  # type: ignore

    sleep = SleepMetrics(
        rem_pct=req.sleep.rem_pct,
        deep_pct=req.sleep.deep_pct,
        efficiency=req.sleep.efficiency,
        duration_h=req.sleep.duration_h,
        awakenings=req.sleep.awakenings,
    )

    result = model.predict_type(
        sleep=sleep,
        recent_stress=req.recent_stress,
        recent_emotions=req.recent_emotions,
    )

    return {"status": "ok", **result}


@router.get("/status")
def dream_sleep_status() -> Dict[str, Any]:
    """Return availability status of the DreamSleepFusionModel."""
    model = _get_model()
    available = model is not None
    return {
        "available": available,
        "model_type": "heuristic dream-sleep correlation (no ML weights required)",
        "description": (
            "Correlates dream content/themes with sleep stage physiology. "
            "Analyzes dream emotional content from text and matches with "
            "sleep architecture metrics to compute correlation scores, "
            "nightmare risk, and dream type predictions."
        ),
        "endpoints": [
            "POST /dream-sleep/correlate",
            "POST /dream-sleep/predict",
            "GET /dream-sleep/status",
        ],
        "inputs": {
            "correlate": [
                "dreams (list of {text, date, lucidity_score})",
                "sleep_data (list of {rem_pct, deep_pct, efficiency, duration_h, awakenings})",
            ],
            "predict": [
                "sleep ({rem_pct, deep_pct, efficiency, duration_h, awakenings})",
                "recent_stress (0-1)",
                "recent_emotions (optional dict)",
            ],
        },
        "outputs": [
            "correlations (per-entry scores and insights)",
            "profile (aggregated when 2+ entries)",
            "dream_probabilities (predict endpoint)",
            "nightmare_risk",
            "recall_probability",
        ],
    }

"""Emotion forecasting endpoints (#406).

POST /emotion-forecast/predict  -- forecast next N days of emotion
GET  /emotion-forecast/status   -- availability check
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
router = APIRouter(prefix="/emotion-forecast", tags=["emotion-forecaster"])


# -- request / response schemas -----------------------------------------------

class DailySummary(BaseModel):
    valence: Optional[float] = Field(None, ge=-1.0, le=1.0)
    arousal: Optional[float] = Field(None, ge=0.0, le=1.0)
    stress: Optional[float] = Field(None, ge=0.0, le=1.0)
    sleep_quality: Optional[float] = Field(None, ge=0.0, le=1.0)
    activity_level: Optional[float] = Field(None, ge=0.0, le=1.0)


class ForecastInput(BaseModel):
    daily_summaries: List[DailySummary] = Field(
        ..., description="7-30 days of daily emotional summaries (most recent last)"
    )
    horizon: int = Field(1, ge=1, le=7, description="Days to forecast (1-7)")
    include_confidence: bool = Field(True, description="Include confidence intervals")
    include_weekly_pattern: bool = Field(False, description="Detect weekly patterns")
    include_feature_importance: bool = Field(False, description="Rank feature importances")


class ForecastDay(BaseModel):
    day: int
    valence: float = 0.0
    arousal: float = 0.0
    stress: float = 0.0
    sleep_quality: float = 0.0
    activity_level: float = 0.0


class ForecastResponse(BaseModel):
    horizon_days: int
    forecasts: List[ForecastDay] = []
    confidence: Optional[Dict[str, Any]] = None
    weekly_pattern: Optional[Dict[str, Any]] = None
    feature_importance: Optional[Dict[str, Any]] = None
    trend_slopes: Optional[Dict[str, float]] = None
    processed_at: float = 0.0


# -- endpoints ----------------------------------------------------------------

@router.post("/predict", response_model=ForecastResponse)
async def predict_emotion(req: ForecastInput):
    """Forecast emotional state for the next N days.

    Uses temporal attention on 7-30 days of historical daily summaries
    (valence, arousal, stress, sleep quality, activity level) to predict
    future emotional states with confidence intervals.

    Minimum 3 days of history required. 14+ days recommended.
    """
    from models.emotion_forecaster import (
        prepare_forecast_input,
        forecast_emotion,
        compute_forecast_confidence,
        detect_weekly_pattern,
        compute_feature_importance,
        forecast_to_dict,
    )

    # Convert pydantic models to dicts
    summaries = [s.model_dump() for s in req.daily_summaries]

    prepared = prepare_forecast_input(summaries)
    if "error" in prepared and prepared["error"] != "insufficient_data":
        raise HTTPException(422, prepared.get("detail", "Invalid input data"))

    if "error" in prepared and prepared["error"] == "insufficient_data":
        raise HTTPException(
            422,
            f"Need at least 3 days of history, got {prepared.get('n_days', 0)}",
        )

    # Main forecast
    result = forecast_emotion(prepared, horizon=req.horizon)
    if "error" in result:
        raise HTTPException(500, f"Forecast failed: {result.get('detail', 'unknown')}")

    result = forecast_to_dict(result)

    # Build response
    forecasts = []
    for fc in result.get("forecasts", []):
        forecasts.append(ForecastDay(**fc))

    response = ForecastResponse(
        horizon_days=result.get("horizon_days", req.horizon),
        forecasts=forecasts,
        trend_slopes=result.get("trend_slopes"),
        processed_at=time.time(),
    )

    # Optional confidence intervals
    if req.include_confidence:
        conf = compute_forecast_confidence(prepared, horizon=req.horizon)
        response.confidence = forecast_to_dict(conf)

    # Optional weekly pattern
    if req.include_weekly_pattern:
        pattern = detect_weekly_pattern(prepared)
        response.weekly_pattern = forecast_to_dict(pattern)

    # Optional feature importance
    if req.include_feature_importance:
        importance = compute_feature_importance(prepared)
        response.feature_importance = forecast_to_dict(importance)

    return response


@router.get("/status")
async def emotion_forecast_status() -> Dict[str, Any]:
    """Check availability of emotion forecasting."""
    return {
        "ready": True,
        "numpy_available": True,
        "features": ["valence", "arousal", "stress", "sleep_quality", "activity_level"],
        "min_days": 3,
        "max_days": 90,
        "max_horizon": 7,
        "methods": [
            "temporal_attention_weighted_mean",
            "linear_trend_extrapolation",
            "weekly_pattern_detection",
            "next_day_valence_correlation",
        ],
    }

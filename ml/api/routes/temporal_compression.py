"""Temporal emotion compression API routes — issue #462.

Endpoints:
  POST /temporal/compress  -- compress emotional timeline
  GET  /temporal/status    -- health check

Issue #462.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from models.temporal_compression import (
    EmotionDataPoint,
    compression_to_dict,
    generate_timelapse,
)

router = APIRouter(prefix="/temporal", tags=["temporal-compression"])


# -- Request / response models -----------------------------------------------

class EmotionDataPointInput(BaseModel):
    timestamp: float = Field(..., description="Unix epoch seconds")
    valence: float = Field(0.0, ge=-1.0, le=1.0, description="Emotional valence")
    arousal: float = Field(0.5, ge=0.0, le=1.0, description="Arousal level")
    stress: float = Field(0.3, ge=0.0, le=1.0, description="Stress level")
    energy: float = Field(0.5, ge=0.0, le=1.0, description="Energy level")
    label: str = Field("", description="Optional contextual label")


class CompressRequest(BaseModel):
    data: List[EmotionDataPointInput] = Field(
        ..., min_length=1, description="Emotional timeline data points"
    )
    num_periods: int = Field(10, ge=1, le=100, description="Number of compressed periods")
    max_key_moments: int = Field(20, ge=1, le=100, description="Maximum key moments to extract")


# -- Endpoints ----------------------------------------------------------------

@router.post("/compress")
async def compress(req: CompressRequest) -> Dict[str, Any]:
    """Compress an emotional timeline into a time-lapse summary.

    Extracts key moments, detects emotional arcs, and generates
    compressed period-by-period summaries suitable for visualization.
    """
    data = [
        EmotionDataPoint(
            timestamp=d.timestamp,
            valence=d.valence,
            arousal=d.arousal,
            stress=d.stress,
            energy=d.energy,
            label=d.label,
        )
        for d in req.data
    ]

    result = generate_timelapse(
        data,
        num_periods=req.num_periods,
        max_key_moments=req.max_key_moments,
    )

    return compression_to_dict(result)


@router.get("/status")
async def status() -> Dict[str, Any]:
    """Health check — confirms the temporal compression module is loaded."""
    return {
        "status": "ready",
        "model_type": "temporal-compression",
        "description": "Compress months of emotional data into time-lapsed summaries",
        "features": [
            "key_moment_extraction",
            "emotional_arc_detection",
            "timeline_compression",
            "timelapse_generation",
        ],
        "timestamp": time.time(),
    }

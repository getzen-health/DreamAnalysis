"""Collective emotional intelligence API routes — issue #461.

Endpoints:
  POST /collective/aggregate  -- submit anonymized emotion data, get collective mood
  GET  /collective/mood       -- current collective mood (from cached data)
  GET  /collective/status     -- health check

Issue #461.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from models.collective_emotion import (
    AnonymousEmotionSample,
    aggregate_anonymous_emotions,
    compute_collective_mood,
    compute_collective_profile,
    detect_collective_events,
    profile_to_dict,
)

router = APIRouter(prefix="/collective", tags=["collective-emotion"])

# Module-level cache for the latest collective mood
_latest_mood: Dict[str, Any] = {}
_latest_samples: List[AnonymousEmotionSample] = []


# -- Request / response models -----------------------------------------------

class AnonymousSampleInput(BaseModel):
    valence: float = Field(0.0, ge=-1.0, le=1.0, description="Emotional valence")
    arousal: float = Field(0.5, ge=0.0, le=1.0, description="Arousal level")
    stress: float = Field(0.3, ge=0.0, le=1.0, description="Stress level")
    energy: float = Field(0.5, ge=0.0, le=1.0, description="Energy level")
    timestamp: float = Field(0.0, description="Unix epoch seconds")
    region: str = Field("unknown", description="Anonymized geographic region")


class AggregateRequest(BaseModel):
    samples: List[AnonymousSampleInput] = Field(
        ..., min_length=1, description="Anonymized emotion samples"
    )
    baseline_samples: Optional[List[AnonymousSampleInput]] = Field(
        None, description="Historical baseline samples for event detection"
    )


# -- Endpoints ----------------------------------------------------------------

@router.post("/aggregate")
async def aggregate(req: AggregateRequest) -> Dict[str, Any]:
    """Submit anonymized emotion data and compute collective intelligence.

    Aggregates the provided samples into a population-level mood snapshot,
    detects collective emotional events, and computes temporal/geographic
    patterns. Enforces minimum group size for privacy.
    """
    global _latest_mood, _latest_samples

    samples = [
        AnonymousEmotionSample(
            valence=s.valence,
            arousal=s.arousal,
            stress=s.stress,
            energy=s.energy,
            timestamp=s.timestamp if s.timestamp > 0 else time.time(),
            region=s.region,
        )
        for s in req.samples
    ]

    baseline = None
    if req.baseline_samples:
        baseline = [
            AnonymousEmotionSample(
                valence=s.valence,
                arousal=s.arousal,
                stress=s.stress,
                energy=s.energy,
                timestamp=s.timestamp,
                region=s.region,
            )
            for s in req.baseline_samples
        ]

    profile = compute_collective_profile(samples, baseline)
    result = profile_to_dict(profile)

    # Cache latest mood
    _latest_mood = result.get("mood", {})
    _latest_samples = samples

    return result


@router.get("/mood")
async def mood() -> Dict[str, Any]:
    """Return the most recent collective mood snapshot.

    Returns cached data from the last /aggregate call. If no data
    has been submitted yet, returns an empty mood with a message.
    """
    if not _latest_mood:
        return {
            "mood": {
                "mean_valence": 0.0,
                "mean_arousal": 0.5,
                "mood_label": "no_data",
                "sample_count": 0,
            },
            "message": "No collective data submitted yet. POST to /collective/aggregate first.",
            "timestamp": time.time(),
        }
    return {
        "mood": _latest_mood,
        "timestamp": time.time(),
    }


@router.get("/status")
async def status() -> Dict[str, Any]:
    """Health check — confirms the collective emotion module is loaded."""
    return {
        "status": "ready",
        "model_type": "collective-emotion",
        "description": "Population-level emotion aggregation with privacy enforcement",
        "min_group_size": 5,
        "cached_sample_count": len(_latest_samples),
        "timestamp": time.time(),
    }

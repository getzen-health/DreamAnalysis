"""Contextual emotion contagion graph API routes — issue #412.

Endpoints:
  POST /emotion-contagion/analyze  -- build influence graph from emotion + context data
  GET  /emotion-contagion/status   -- health check

GitHub issue: #412
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from models.emotion_contagion import (
    EmotionSample,
    ContextEvent,
    detect_state_transitions,
    attribute_transitions,
    build_influence_graph,
    compute_influence_insights,
    graph_to_dict,
)

router = APIRouter(prefix="/emotion-contagion", tags=["emotion-contagion"])


# -- Request / response models -----------------------------------------------

class EmotionSampleInput(BaseModel):
    timestamp: float = Field(..., description="Unix epoch seconds")
    valence: float = Field(..., ge=-1.0, le=1.0, description="Emotional valence")
    arousal: float = Field(..., ge=0.0, le=1.0, description="Arousal level")


class ContextEventInput(BaseModel):
    timestamp: float = Field(..., description="Unix epoch seconds — start of event")
    entity_type: str = Field(..., description="person, place, activity, or media")
    entity_id: str = Field(..., min_length=1, description="Unique entity identifier")
    duration_min: float = Field(..., ge=0.0, description="Event duration in minutes")


class AnalyzeRequest(BaseModel):
    emotion_samples: List[EmotionSampleInput] = Field(
        ..., min_length=2, description="Chronological emotion samples"
    )
    context_events: List[ContextEventInput] = Field(
        default_factory=list, description="Context events to attribute transitions to"
    )
    valence_threshold: float = Field(0.25, ge=0.0, le=2.0, description="Min valence shift for transition")
    arousal_threshold: float = Field(0.20, ge=0.0, le=2.0, description="Min arousal shift for transition")
    sustain_minutes: float = Field(5.0, ge=0.0, description="Min duration a shift must be sustained")
    lookback_min: float = Field(60.0, ge=0.0, description="Lookback window in minutes")


# -- Endpoints ----------------------------------------------------------------

@router.post("/analyze")
async def analyze_contagion(req: AnalyzeRequest):
    """Build an emotional influence graph from emotion samples and context events.

    Detects emotional state transitions, attributes them to preceding context
    events, builds an influence graph, and generates insights (top energizers,
    drainers, latency patterns, recovery times).
    """
    # Convert Pydantic models to domain dataclasses
    samples = [
        EmotionSample(timestamp=s.timestamp, valence=s.valence, arousal=s.arousal)
        for s in req.emotion_samples
    ]
    events = [
        ContextEvent(
            timestamp=e.timestamp,
            entity_type=e.entity_type,
            entity_id=e.entity_id,
            duration_min=e.duration_min,
        )
        for e in req.context_events
    ]

    # Pipeline
    transitions = detect_state_transitions(
        samples,
        valence_threshold=req.valence_threshold,
        arousal_threshold=req.arousal_threshold,
        sustain_minutes=req.sustain_minutes,
    )
    attributions = attribute_transitions(
        transitions, events, lookback_min=req.lookback_min,
    )
    graph = build_influence_graph(attributions, samples)
    insights = compute_influence_insights(graph)

    return {
        "transitions_detected": len(transitions),
        "attributions_made": len(attributions),
        "graph": graph_to_dict(graph),
        "insights": insights,
    }


@router.get("/status")
async def status():
    """Health check — confirms the emotion contagion module is loaded."""
    return {
        "status": "ready",
        "model_type": "temporal-attribution",
        "description": "Contextual emotion contagion graph",
    }

"""Emotional digital twin API routes (issue #419).

POST /emotional-twin/train     -- train/update twin from historical data
POST /emotional-twin/simulate  -- simulate a counterfactual scenario
GET  /emotional-twin/status    -- model & twin availability
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from models.emotional_twin import (
    ContextVector,
    EmotionalState,
    EmotionalTwin,
    ScenarioDefinition,
    compute_calibration_score,
    predict_trajectory,
    simulate_scenario,
    train_twin,
    twin_to_dict,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/emotional-twin", tags=["emotional-twin"])

# In-memory twin store keyed by user_id.
_twins: Dict[str, EmotionalTwin] = {}


# --------------------------------------------------------------------------
# Request / response models
# --------------------------------------------------------------------------

class StateInput(BaseModel):
    valence: float = Field(0.0, ge=-1.0, le=1.0)
    arousal: float = Field(0.5, ge=0.0, le=1.0)
    stress: float = Field(0.3, ge=0.0, le=1.0)
    energy: float = Field(0.5, ge=0.0, le=1.0)


class ContextInput(BaseModel):
    stress_level: float = Field(0.0, ge=0.0, le=1.0)
    novelty: float = Field(0.0, ge=0.0, le=1.0)
    social_change: float = Field(0.0, ge=-1.0, le=1.0)
    routine_disruption: float = Field(0.0, ge=0.0, le=1.0)
    sleep_quality: float = Field(0.5, ge=0.0, le=1.0)
    exercise: float = Field(0.0, ge=0.0, le=1.0)


class TrainRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    contexts: List[ContextInput] = Field(..., min_length=2)
    observed_states: List[StateInput] = Field(..., min_length=2)


class ScenarioInput(BaseModel):
    name: str = "unnamed"
    context_overrides: Optional[Dict[str, float]] = None
    duration_steps: int = Field(5, ge=1, le=100)
    description: str = ""


class SimulateRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    scenario: ScenarioInput
    base_context: Optional[ContextInput] = None


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@router.post("/train")
async def train_emotional_twin(req: TrainRequest) -> Dict[str, Any]:
    """Train or update an emotional digital twin from longitudinal data."""
    if len(req.contexts) != len(req.observed_states):
        raise HTTPException(
            status_code=400,
            detail="contexts and observed_states must have the same length",
        )

    twin = _twins.get(req.user_id, EmotionalTwin(user_id=req.user_id))

    contexts = [
        ContextVector(**c.model_dump()) for c in req.contexts
    ]
    states = [
        EmotionalState(**s.model_dump()) for s in req.observed_states
    ]

    try:
        train_twin(twin, contexts, states)
    except Exception as exc:
        log.exception("train_twin failed for user %s", req.user_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    _twins[req.user_id] = twin
    return {
        "status": "ok",
        "twin": twin_to_dict(twin),
    }


@router.post("/simulate")
async def simulate_emotional_scenario(req: SimulateRequest) -> Dict[str, Any]:
    """Simulate a counterfactual scenario for a trained twin."""
    twin = _twins.get(req.user_id)
    if twin is None:
        raise HTTPException(
            status_code=404,
            detail=f"No twin found for user_id={req.user_id!r}. Train first.",
        )

    scenario = ScenarioDefinition(
        name=req.scenario.name,
        context_overrides=req.scenario.context_overrides,
        duration_steps=req.scenario.duration_steps,
        description=req.scenario.description,
    )
    base_ctx = (
        ContextVector(**req.base_context.model_dump())
        if req.base_context
        else None
    )

    try:
        result = simulate_scenario(twin, scenario, base_ctx)
    except Exception as exc:
        log.exception("simulate_scenario failed for user %s", req.user_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "status": "ok",
        "result": {
            "scenario_name": result.scenario_name,
            "trajectory": result.trajectory,
            "final_state": result.final_state,
            "initial_state": result.initial_state,
            "duration_steps": result.duration_steps,
        },
    }


@router.get("/status")
async def emotional_twin_status() -> Dict[str, Any]:
    """Return model availability and active twin count."""
    return {
        "status": "available",
        "active_twins": len(_twins),
        "twin_ids": list(_twins.keys()),
    }

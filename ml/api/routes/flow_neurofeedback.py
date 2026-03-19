"""Flow state deep work neurofeedback API routes (issue #441).

POST /flow/score                          — compute flow score from EEG features
POST /flow/distraction                    — detect distraction event
POST /flow/session                        — track completed flow session
GET  /flow/optimal-conditions/{user_id}   — learned optimal conditions
GET  /flow/status                         — model availability
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
router = APIRouter(prefix="/flow", tags=["flow-neurofeedback"])


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class FlowScoreRequest(BaseModel):
    alpha: float = Field(..., ge=0.0, description="Alpha band power")
    beta: float = Field(..., ge=0.0, description="Beta band power")
    theta: float = Field(..., ge=0.0, description="Theta band power")
    delta: float = Field(..., ge=0.0, description="Delta band power")
    high_beta: Optional[float] = Field(None, ge=0.0, description="High beta (20-30 Hz)")
    low_beta: Optional[float] = Field(None, ge=0.0, description="Low beta (12-20 Hz)")
    gamma: Optional[float] = Field(None, ge=0.0, description="Gamma band power")


class FlowScoreResponse(BaseModel):
    flow_score: float
    focus_depth: float
    creative_engagement: float
    time_distortion: float
    effortlessness: float
    flow_level: str
    band_contributions: Dict[str, float]


class BaselineStats(BaseModel):
    alpha: Tuple[float, float] = Field(..., description="(mean, std) for alpha")
    beta: Tuple[float, float] = Field(..., description="(mean, std) for beta")
    theta: Tuple[float, float] = Field(..., description="(mean, std) for theta")
    delta: Tuple[float, float] = Field(default=(0.0, 1.0), description="(mean, std) for delta")


class DistractionRequest(BaseModel):
    eeg_features: FlowScoreRequest
    baseline_stats: BaselineStats


class DistractionResponse(BaseModel):
    detected: bool
    distraction_type: Optional[str] = None
    severity: Optional[float] = None
    description: Optional[str] = None
    recovery_suggestion: Optional[str] = None
    timestamp: Optional[float] = None


class SessionConditions(BaseModel):
    time_of_day: Optional[str] = None
    sound_env: Optional[str] = None
    caffeine: Optional[bool] = None
    day_of_week: Optional[str] = None


class FlowSessionRequest(BaseModel):
    user_id: str
    session_id: str
    entry_time: float
    exit_time: float
    readings: List[float] = Field(default_factory=list)
    exit_reason: Optional[str] = "natural"
    distraction_count: int = Field(0, ge=0)
    conditions: Optional[SessionConditions] = None


class FlowSessionResponse(BaseModel):
    user_id: str
    session_id: str
    entry_time: float
    exit_time: float
    duration_minutes: float
    peak_score: float
    mean_score: float
    depth_category: str
    exit_reason: Optional[str]
    distraction_count: int
    distraction_rate_per_hour: float
    productivity_index: float
    conditions: Optional[Dict[str, Any]]


# In-memory session store (per-user list of session summaries)
_user_sessions: Dict[str, List[Dict[str, Any]]] = {}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/score", response_model=FlowScoreResponse)
def flow_score(req: FlowScoreRequest) -> Dict[str, Any]:
    """Compute current flow score from EEG band power features."""
    try:
        from models.flow_neurofeedback import compute_flow_score
    except ImportError as exc:
        log.error("flow_neurofeedback import failed: %s", exc)
        raise HTTPException(503, "Flow neurofeedback model unavailable") from exc

    features: Dict[str, float] = {
        "alpha": req.alpha,
        "beta": req.beta,
        "theta": req.theta,
        "delta": req.delta,
    }
    if req.high_beta is not None:
        features["high_beta"] = req.high_beta
    if req.low_beta is not None:
        features["low_beta"] = req.low_beta
    if req.gamma is not None:
        features["gamma"] = req.gamma

    try:
        result = compute_flow_score(features)
    except Exception as exc:
        log.exception("Flow score computation failed: %s", exc)
        raise HTTPException(500, f"Computation error: {exc}") from exc

    return result


@router.post("/distraction", response_model=DistractionResponse)
def flow_distraction(req: DistractionRequest) -> Dict[str, Any]:
    """Detect distraction event from current EEG relative to baseline."""
    try:
        from models.flow_neurofeedback import detect_distraction
    except ImportError as exc:
        log.error("flow_neurofeedback import failed: %s", exc)
        raise HTTPException(503, "Flow neurofeedback model unavailable") from exc

    features: Dict[str, float] = {
        "alpha": req.eeg_features.alpha,
        "beta": req.eeg_features.beta,
        "theta": req.eeg_features.theta,
        "delta": req.eeg_features.delta,
    }

    baseline: Dict[str, Tuple[float, float]] = {
        "alpha": tuple(req.baseline_stats.alpha),  # type: ignore[arg-type]
        "beta": tuple(req.baseline_stats.beta),  # type: ignore[arg-type]
        "theta": tuple(req.baseline_stats.theta),  # type: ignore[arg-type]
        "delta": tuple(req.baseline_stats.delta),  # type: ignore[arg-type]
    }

    try:
        result = detect_distraction(features, baseline)
    except Exception as exc:
        log.exception("Distraction detection failed: %s", exc)
        raise HTTPException(500, f"Detection error: {exc}") from exc

    if result is None:
        return {"detected": False}

    return {
        "detected": True,
        "distraction_type": result["distraction_type"],
        "severity": result["severity"],
        "description": result["description"],
        "recovery_suggestion": result["recovery_suggestion"],
        "timestamp": result["timestamp"],
    }


@router.post("/session", response_model=FlowSessionResponse)
def flow_session(req: FlowSessionRequest) -> Dict[str, Any]:
    """Track a completed flow session and store for optimal-conditions learning."""
    try:
        from models.flow_neurofeedback import track_flow_session
    except ImportError as exc:
        log.error("flow_neurofeedback import failed: %s", exc)
        raise HTTPException(503, "Flow neurofeedback model unavailable") from exc

    session_data: Dict[str, Any] = {
        "user_id": req.user_id,
        "session_id": req.session_id,
        "entry_time": req.entry_time,
        "exit_time": req.exit_time,
        "readings": req.readings,
        "exit_reason": req.exit_reason,
        "distraction_count": req.distraction_count,
    }

    if req.conditions:
        session_data["conditions"] = {
            "time_of_day": req.conditions.time_of_day,
            "sound_env": req.conditions.sound_env,
            "caffeine": req.conditions.caffeine,
            "day_of_week": req.conditions.day_of_week,
        }

    try:
        result = track_flow_session(session_data)
    except Exception as exc:
        log.exception("Session tracking failed: %s", exc)
        raise HTTPException(500, f"Tracking error: {exc}") from exc

    # Store in memory for optimal-conditions queries
    _user_sessions.setdefault(req.user_id, []).append(result)

    return result


@router.get("/optimal-conditions/{user_id}")
def flow_optimal_conditions(user_id: str) -> Dict[str, Any]:
    """Get learned optimal flow conditions for a user."""
    try:
        from models.flow_neurofeedback import compute_optimal_conditions
    except ImportError as exc:
        log.error("flow_neurofeedback import failed: %s", exc)
        raise HTTPException(503, "Flow neurofeedback model unavailable") from exc

    sessions = _user_sessions.get(user_id, [])

    try:
        result = compute_optimal_conditions(sessions)
    except Exception as exc:
        log.exception("Optimal conditions computation failed: %s", exc)
        raise HTTPException(500, f"Computation error: {exc}") from exc

    result["user_id"] = user_id
    return result


@router.get("/status")
def flow_neurofeedback_status() -> Dict[str, Any]:
    """Return availability status of the flow neurofeedback system."""
    try:
        from models.flow_neurofeedback import compute_flow_score  # noqa: F401
        available = True
    except ImportError:
        available = False

    return {
        "available": available,
        "model_type": "real-time EEG flow neurofeedback",
        "description": (
            "Deep work productivity mode with EEG-driven distraction defense. "
            "Computes flow score (0-100) with sub-dimensions: focus_depth, "
            "creative_engagement, time_distortion, effortlessness. "
            "Detects distractions (beta spike, alpha dropout, theta surge) and "
            "learns optimal flow conditions per user."
        ),
        "endpoints": [
            "POST /flow/score",
            "POST /flow/distraction",
            "POST /flow/session",
            "GET /flow/optimal-conditions/{user_id}",
        ],
        "flow_levels": ["none (<30)", "light (30-49)", "moderate (50-74)", "deep (75+)"],
    }

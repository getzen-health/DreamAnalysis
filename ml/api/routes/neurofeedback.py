"""Neurofeedback session endpoints."""

import asyncio
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ._shared import (
    NeurofeedbackProtocol, PROTOCOLS,
    NeurofeedbackStartRequest, NeurofeedbackEvalRequest,
)
import api.routes._shared as _state

router = APIRouter()

# ── RL agent singleton ────────────────────────────────────────────────────────
_rl_agent = None
_RL_AGENT_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "saved" / "rl_nf_agent.pt"


def _load_rl_agent():
    """Lazy-load the RL agent from disk. Sets module-level _rl_agent."""
    global _rl_agent
    if _rl_agent is not None:
        return
    try:
        from neurofeedback.adaptive_agent import PPOAgent
        agent = PPOAgent()
        if _RL_AGENT_PATH.exists() and agent.load(str(_RL_AGENT_PATH)):
            _rl_agent = agent
    except Exception:
        pass


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class RLTrainRequest(BaseModel):
    protocol_type: str = Field(default="all", description="Protocol or 'all'")
    episodes: int = Field(default=500, ge=1, le=5000, description="Training episodes")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/neurofeedback/protocols")
async def list_protocols():
    """List available neurofeedback protocols."""
    return {
        key: {"name": p["name"], "description": p["description"]}
        for key, p in PROTOCOLS.items()
    }


@router.post("/neurofeedback/start")
async def start_neurofeedback(request: NeurofeedbackStartRequest):
    """Start a neurofeedback session."""
    _state._nf_protocol = NeurofeedbackProtocol(
        protocol_type=request.protocol_type,
        target_band=request.target_band,
        threshold=request.threshold,
    )

    # Attempt to load the RL agent (no-op if already loaded or file absent)
    _load_rl_agent()

    if request.calibrate:
        _state._nf_protocol.start_calibration()
        return {"status": "calibrating", "protocol": request.protocol_type}

    _state._nf_protocol.start()
    return {
        "status": "active",
        "protocol": request.protocol_type,
        "rl_active": _rl_agent is not None and _rl_agent.is_trained,
    }


@router.post("/neurofeedback/evaluate")
async def evaluate_neurofeedback(request: NeurofeedbackEvalRequest):
    """Evaluate current EEG against the active neurofeedback protocol."""
    if _state._nf_protocol is None:
        raise HTTPException(status_code=400, detail="No active neurofeedback session")

    if _state._nf_protocol.is_calibrating:
        done = _state._nf_protocol.add_calibration_sample(request.band_powers)
        progress = len(_state._nf_protocol.baseline_samples) / 30.0
        if done:
            return {
                "status": "calibration_complete",
                "baseline": _state._nf_protocol.baseline,
                "progress": 1.0,
            }
        return {"status": "calibrating", "progress": float(progress)}

    result = _state._nf_protocol.evaluate(request.band_powers, request.channel_powers)

    # ── Adaptive threshold adjustment via RL agent ────────────────────────────
    if _rl_agent is not None and _rl_agent.is_trained:
        rl_state = _state._nf_protocol.get_rl_state(request.band_powers)
        obs = _rl_agent.build_obs(rl_state)
        action, _ = _rl_agent.act(obs)
        delta = (action - 1) * 0.05
        _state._nf_protocol.threshold = float(
            np.clip(_state._nf_protocol.threshold + delta, 0.10, 2.50)
        )
        result["adaptive_threshold"] = round(_state._nf_protocol.threshold, 3)
        result["threshold_action"] = ["easier", "hold", "harder"][action]

    return {"status": "active", **result}


@router.post("/neurofeedback/stop")
async def stop_neurofeedback():
    """Stop the current neurofeedback session and return stats."""
    if _state._nf_protocol is None:
        raise HTTPException(status_code=400, detail="No active neurofeedback session")

    stats = _state._nf_protocol.stop()
    _state._nf_protocol = None
    return {"status": "stopped", "stats": stats}


@router.get("/neurofeedback/rl/status")
async def rl_status():
    """Return RL agent status and current threshold."""
    trained = _rl_agent is not None and _rl_agent.is_trained
    current_threshold = (
        _state._nf_protocol.threshold if _state._nf_protocol is not None else None
    )
    return {
        "trained": trained,
        "is_active": _state._nf_protocol is not None and _state._nf_protocol.is_active,
        "current_threshold": current_threshold,
        "agent_path": str(_RL_AGENT_PATH),
    }


@router.post("/neurofeedback/rl/train")
async def rl_train(request: RLTrainRequest):
    """Trigger offline RL training in a background thread.

    Runs NeurofeedbackEnv + PPO update loop, saves model, returns summary.
    """
    from neurofeedback.protocol_engine import PROTOCOLS as _PROTOCOLS

    valid_types = list(_PROTOCOLS.keys()) + ["all"]
    if request.protocol_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"protocol_type must be one of {valid_types}",
        )

    def _run_training():
        from neurofeedback.rl_environment import NeurofeedbackEnv
        from neurofeedback.adaptive_agent import PPOAgent, Rollout

        if request.protocol_type == "all":
            protocols = ["alpha_up", "smr_up", "theta_beta_ratio"]
        else:
            protocols = [request.protocol_type]

        agent = PPOAgent()
        episodes_run = 0

        for proto in protocols:
            env = NeurofeedbackEnv(protocol_type=proto)
            for _ in range(request.episodes):
                obs = env.reset()
                rollout = Rollout()
                done = False

                while not done:
                    action, log_prob = agent.act(obs)
                    value = agent.value(obs)
                    next_obs, reward, done, _ = env.step(action)
                    rollout.add(obs, action, log_prob, reward, done, value)
                    obs = next_obs

                last_obs = None if done else obs
                agent.update(rollout, last_obs=last_obs)
                episodes_run += 1

        agent.save(str(_RL_AGENT_PATH))
        return agent, episodes_run

    agent, episodes_run = await asyncio.to_thread(_run_training)

    # Refresh global singleton
    global _rl_agent
    _rl_agent = agent

    return {
        "trained": agent.is_trained,
        "episodes_run": episodes_run,
        "model_path": str(_RL_AGENT_PATH),
    }

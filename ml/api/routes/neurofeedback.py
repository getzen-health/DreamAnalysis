"""Neurofeedback session endpoints."""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ._shared import (
    NeurofeedbackProtocol, PROTOCOLS,
    NeurofeedbackStartRequest, NeurofeedbackEvalRequest,
    _get_nf_protocol, _set_nf_protocol,
    sanitize_id,
)
from neurofeedback.progress_tracker import NeurofeedbackProgressTracker

logger = logging.getLogger(__name__)
router = APIRouter()

# ── RL agent singletons ──────────────────────────────────────────────────────
_rl_agent = None
_RL_AGENT_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "saved" / "rl_nf_agent.pt"
_USER_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "user_models"
_USER_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "user_data"

# Per-user RL agents (loaded lazily)
_user_rl_agents: Dict[str, object] = {}

# Per-user trajectory buffers (stored during active neurofeedback sessions)
_trajectory_buffers: Dict[str, list] = {}

# Cooldown tracking: user_id -> timestamp of last session end
_last_session_end: Dict[str, float] = {}


def _load_rl_agent():
    """Lazy-load the global RL agent from disk."""
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


def _reload_rl_agent():
    """Force-reload the global RL agent from disk."""
    global _rl_agent
    try:
        from neurofeedback.adaptive_agent import PPOAgent
        agent = PPOAgent()
        if _RL_AGENT_PATH.exists() and agent.load(str(_RL_AGENT_PATH)):
            _rl_agent = agent
    except Exception:
        pass


def _get_user_rl_agent(user_id: str):
    """Load per-user RL agent if available, else return global agent."""
    if user_id in _user_rl_agents:
        return _user_rl_agents[user_id]
    user_path = _USER_MODELS_DIR / user_id / "rl_nf_agent.pt"
    if user_path.exists():
        try:
            from neurofeedback.adaptive_agent import PPOAgent
            agent = PPOAgent()
            if agent.load(str(user_path)):
                _user_rl_agents[user_id] = agent
                return agent
        except Exception:
            pass
    return _rl_agent


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
    protocol = NeurofeedbackProtocol(
        protocol_type=request.protocol_type,
        target_band=request.target_band,
        threshold=request.threshold,
    )
    _set_nf_protocol(request.user_id, protocol)

    # Attempt to load the RL agent (no-op if already loaded or file absent)
    _load_rl_agent()

    # Check 2-hour cooldown between sessions
    last_end = _last_session_end.get(request.user_id, 0)
    cooldown_remaining = max(0, 7200 - (time.time() - last_end))

    if request.calibrate:
        protocol.start_calibration()
        result: Dict = {"status": "calibrating", "protocol": request.protocol_type}
        if cooldown_remaining > 0:
            result["cooldown_warning"] = {
                "recommended_wait_minutes": round(cooldown_remaining / 60, 1),
                "message": "Recommended 2-hour cooldown between sessions for optimal results.",
            }
        return result

    protocol.start()
    result = {
        "status": "active",
        "protocol": request.protocol_type,
        "rl_active": _rl_agent is not None and _rl_agent.is_trained,
    }
    if cooldown_remaining > 0:
        result["cooldown_warning"] = {
            "recommended_wait_minutes": round(cooldown_remaining / 60, 1),
            "message": "Recommended 2-hour cooldown between sessions for optimal results.",
        }

    # Include dosing alerts from cross-session progress tracker
    tracker = NeurofeedbackProgressTracker(request.user_id, request.protocol_type)
    dosing_alerts = tracker.get_dosing_alerts()
    if dosing_alerts:
        result["dosing_alerts"] = dosing_alerts

    return result


@router.post("/neurofeedback/evaluate")
async def evaluate_neurofeedback(request: NeurofeedbackEvalRequest):
    """Evaluate current EEG against the active neurofeedback protocol."""
    protocol = _get_nf_protocol(request.user_id)
    if protocol is None:
        raise HTTPException(status_code=400, detail="No active neurofeedback session")

    if protocol.is_calibrating:
        done = protocol.add_calibration_sample(request.band_powers)
        progress = len(protocol.baseline_samples) / 30.0
        if done:
            return {
                "status": "calibration_complete",
                "baseline": protocol.baseline,
                "progress": 1.0,
            }
        return {"status": "calibrating", "progress": float(progress)}

    result = protocol.evaluate(request.band_powers, request.channel_powers)

    # ── Adaptive threshold adjustment via RL agent ────────────────────────────
    # Prefer per-user agent if available, else use global agent
    agent = _get_user_rl_agent(request.user_id) or _rl_agent
    if agent is not None and agent.is_trained:
        rl_state = protocol.get_rl_state(request.band_powers)
        obs = agent.build_obs(rl_state)
        action, _ = agent.act(obs)
        delta = (action - 1) * 0.05
        protocol.threshold = float(np.clip(protocol.threshold + delta, 0.10, 2.50))
        result["adaptive_threshold"] = round(protocol.threshold, 3)
        result["threshold_action"] = ["easier", "hold", "harder"][action]

        # Store trajectory step for per-user fine-tuning
        reward = result.get("feedback_value", 0.0)
        step = {
            "obs": obs.tolist(),
            "action": action,
            "reward": float(reward),
            "timestamp": time.time(),
        }
        buf = _trajectory_buffers.setdefault(request.user_id, [])
        buf.append(step)

    # ── Safety checks ────────────────────────────────────────────────────────────────────────────
    result["session_limits"] = protocol.check_session_limits()
    result["fatigue"] = protocol.detect_fatigue()

    return {"status": "active", **result}


@router.post("/neurofeedback/stop")
async def stop_neurofeedback(user_id: str):
    """Stop the current neurofeedback session and return stats."""
    sanitize_id(user_id, "user_id")
    protocol = _get_nf_protocol(user_id)
    if protocol is None:
        raise HTTPException(status_code=400, detail="No active neurofeedback session")

    stats = protocol.stop()
    _set_nf_protocol(user_id, None)

    # Persist trajectory buffer to disk for future RL fine-tuning
    buf = _trajectory_buffers.pop(user_id, [])
    n_steps = len(buf)
    if n_steps > 10:  # only save meaningful sessions
        traj_dir = _USER_DATA_DIR / user_id / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)
        filename = f"traj_{int(time.time())}.json"
        (traj_dir / filename).write_text(json.dumps(buf))
        logger.info(f"[neurofeedback] saved {n_steps} trajectory steps for user={user_id}")
        stats["trajectory_saved"] = True
        stats["trajectory_steps"] = n_steps
    else:
        stats["trajectory_saved"] = False

    # Record session for cross-session progress tracking
    protocol_type = protocol.protocol_type if hasattr(protocol, "protocol_type") else "unknown"
    tracker = NeurofeedbackProgressTracker(user_id, protocol_type)
    tracker.record_session(
        duration_minutes=stats.get("duration_seconds", 0) / 60,
        avg_score=stats.get("avg_score", 0),
        reward_rate=stats.get("reward_rate", 0),
        max_streak=stats.get("max_streak", 0),
        baseline_value=stats.get("baseline", {}).get("alpha") if isinstance(stats.get("baseline"), dict) else None,
    )
    stats["cross_session_progress"] = tracker.get_progress()

    # Record session end for cooldown tracking
    _last_session_end[user_id] = time.time()

    # Post-session safety check prompt
    stats["post_session_check"] = {
        "duration_minutes": protocol.get_session_duration_minutes(),
        "check_items": [
            "Any headache or head pressure?",
            "Any dizziness or lightheadedness?",
            "Any unusual mood changes?",
            "Any difficulty concentrating?",
        ],
    }

    return {"status": "stopped", "stats": stats}


@router.get("/neurofeedback/rl/status")
async def rl_status(user_id: str):
    """Return RL agent status and current threshold."""
    sanitize_id(user_id, "user_id")
    global_trained = _rl_agent is not None and _rl_agent.is_trained
    user_agent = _get_user_rl_agent(user_id)
    user_trained = user_agent is not None and user_agent is not _rl_agent and user_agent.is_trained
    protocol = _get_nf_protocol(user_id)
    current_threshold = protocol.threshold if protocol is not None else None

    # Count stored trajectories for this user
    traj_dir = _USER_DATA_DIR / user_id / "trajectories"
    n_trajectories = len(list(traj_dir.glob("traj_*.json"))) if traj_dir.exists() else 0

    return {
        "global_trained": global_trained,
        "personal_trained": user_trained,
        "is_active": protocol is not None and protocol.is_active,
        "current_threshold": current_threshold,
        "n_trajectories": n_trajectories,
        "agent_path": str(_RL_AGENT_PATH),
    }


@router.get("/neurofeedback/progress/{user_id}")
async def get_neurofeedback_progress(user_id: str, protocol_type: str = "alpha_up"):
    """Return cross-session neurofeedback progress and dosing compliance."""
    tracker = NeurofeedbackProgressTracker(user_id, protocol_type)
    progress = tracker.get_progress()
    progress["dosing_alerts"] = tracker.get_dosing_alerts()
    return progress


@router.post("/neurofeedback/rl/train")
async def rl_train(request: RLTrainRequest):
    """Trigger RL training in an isolated subprocess.

    Runs train_rl_agent.py as a child process to avoid GIL/OpenMP contention
    with the live PyTorch inference running in the event loop.  The event loop
    remains responsive while training proceeds.

    Returns the same {trained, episodes_run, model_path} payload on success,
    or raises HTTP 500 on training failure.
    """
    import sys

    from neurofeedback.protocol_engine import PROTOCOLS as _PROTOCOLS

    valid_types = list(_PROTOCOLS.keys()) + ["all"]
    if request.protocol_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"protocol_type must be one of {valid_types}",
        )

    train_script = (
        Path(__file__).resolve().parent.parent.parent
        / "neurofeedback"
        / "train_rl_agent.py"
    )
    ml_root = Path(__file__).resolve().parent.parent.parent

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        str(train_script),
        "--protocol", request.protocol_type,
        "--episodes", str(request.episodes),
        "--out", str(_RL_AGENT_PATH),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(ml_root),
    )

    _stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        err_tail = stderr.decode(errors="replace")[-600:] if stderr else "unknown error"
        raise HTTPException(
            status_code=500,
            detail=f"RL training subprocess failed (exit {proc.returncode}): {err_tail}",
        )

    # Reload the freshly-trained agent into the module singleton
    _reload_rl_agent()

    protocols = (
        ["alpha_up", "smr_up", "theta_beta_ratio"]
        if request.protocol_type == "all"
        else [request.protocol_type]
    )
    episodes_run = len(protocols) * request.episodes

    return {
        "trained": _rl_agent is not None and _rl_agent.is_trained,
        "episodes_run": episodes_run,
        "model_path": str(_RL_AGENT_PATH),
    }


class RLFineTuneRequest(BaseModel):
    user_id: str = Field(default="default")
    epochs: int = Field(default=10, ge=1, le=100, description="PPO update epochs over stored trajectories")


@router.post("/neurofeedback/rl/fine-tune")
async def rl_fine_tune(request: RLFineTuneRequest):
    """Fine-tune the RL agent on stored user trajectories.

    Loads all trajectory files for the user, builds rollouts from them,
    and runs PPO update. Saves the fine-tuned agent as a per-user model.
    Requires at least 5 trajectory files (sessions).
    """
    sanitize_id(request.user_id, "user_id")
    traj_dir = _USER_DATA_DIR / request.user_id / "trajectories"
    if not traj_dir.exists():
        raise HTTPException(status_code=400, detail="No trajectory data found for this user")

    traj_files = sorted(traj_dir.glob("traj_*.json"))
    if len(traj_files) < 5:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 5 neurofeedback sessions for fine-tuning (have {len(traj_files)})",
        )

    # Load all trajectories
    all_steps = []
    for tf in traj_files:
        try:
            steps = json.loads(tf.read_text())
            all_steps.extend(steps)
        except Exception:
            continue

    if len(all_steps) < 50:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 50 trajectory steps (have {len(all_steps)})",
        )

    # Run fine-tuning in a thread to not block the event loop
    def _do_fine_tune():
        from neurofeedback.adaptive_agent import PPOAgent, Rollout

        # Start from global agent weights (transfer learning)
        agent = PPOAgent()
        if _RL_AGENT_PATH.exists():
            agent.load(str(_RL_AGENT_PATH))

        # Build rollout from stored trajectories
        for _ in range(request.epochs):
            rollout = Rollout()
            for step in all_steps:
                obs = np.array(step["obs"], dtype=np.float32)
                action = step["action"]
                reward = step["reward"]
                log_prob = 0.0  # approximate — we don't have the original log_prob
                value = agent.value(obs)
                rollout.add(obs, action, log_prob, reward, False, value)

            if len(rollout) > 0:
                agent.update(rollout)

        # Save per-user model
        user_model_dir = _USER_MODELS_DIR / request.user_id
        user_model_dir.mkdir(parents=True, exist_ok=True)
        user_model_path = user_model_dir / "rl_nf_agent.pt"
        agent.save(str(user_model_path))

        # Update in-memory cache
        _user_rl_agents[request.user_id] = agent

        return {
            "fine_tuned": True,
            "user_id": request.user_id,
            "n_trajectories": len(traj_files),
            "n_steps": len(all_steps),
            "epochs": request.epochs,
            "model_path": str(user_model_path),
        }

    result = await asyncio.to_thread(_do_fine_tune)
    logger.info(f"[rl-fine-tune] user={request.user_id} steps={result['n_steps']}")
    return result

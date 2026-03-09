"""Neurofeedback protocols for PTSD and anxiety using consumer EEG (#135).

Implements alpha-up / high-beta-down protocol based on Gruzelier (2009) and
van der Kolk (2015) research on EEG-NF for trauma. Tracks sessions, provides
real-time feedback signals, and reports protocol compliance.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/ptsd-nf", tags=["ptsd-neurofeedback"])

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class NFInput(BaseModel):
    signals: List[List[float]]
    fs: float = 256.0
    user_id: str = "default"


class NFStatus(BaseModel):
    user_id: str
    protocol: str
    session_active: bool
    elapsed_seconds: float
    alpha_power: float
    high_beta_power: float
    feedback_signal: float        # -1 (suppress) … +1 (reward)
    compliance_score: float       # 0-1 fraction of frames meeting target
    session_frames: int
    compliant_frames: int
    protocol_note: str


class NFHistory(BaseModel):
    user_id: str
    sessions: List[dict]


# ---------------------------------------------------------------------------
# Protocol parameters (paper-validated)
# ---------------------------------------------------------------------------

_PROTOCOLS = {
    "alpha_up": {
        "target_band": (8, 12),
        "suppress_band": (20, 30),
        "description": "Increase alpha (8-12 Hz), reduce high-beta (20-30 Hz) — Gruzelier SMR/alpha protocol",
    },
    "alpha_theta": {
        "target_band": (4, 12),   # theta + alpha
        "suppress_band": (20, 30),
        "description": "Alpha-theta deepening for trauma desensitisation — Peniston & Kulkosky (1989)",
    },
}

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

class _Session:
    def __init__(self, user_id: str, protocol: str):
        self.user_id = user_id
        self.protocol = protocol
        self.start_time = time.time()
        self.frames: deque = deque(maxlen=2000)
        self.compliant = 0
        self.total = 0
        self.baseline_alpha: Optional[float] = None

_sessions: Dict[str, _Session] = {}
_session_history: Dict[str, list] = defaultdict(list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _band_power(signal: np.ndarray, fs: float, flo: float, fhi: float) -> float:
    from scipy.signal import welch
    nperseg = min(len(signal), int(fs * 2))
    f, psd = welch(signal, fs=fs, nperseg=nperseg)
    idx = np.logical_and(f >= flo, f <= fhi)
    return float(np.mean(psd[idx])) if idx.any() else 0.0


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/start/{user_id}")
async def start_nf_session(user_id: str, protocol: str = "alpha_up"):
    """Start a neurofeedback session for PTSD/anxiety reduction."""
    if protocol not in _PROTOCOLS:
        raise HTTPException(400, f"Unknown protocol. Valid: {list(_PROTOCOLS)}")
    _sessions[user_id] = _Session(user_id, protocol)
    return {
        "user_id": user_id,
        "protocol": protocol,
        "description": _PROTOCOLS[protocol]["description"],
        "started_at": _sessions[user_id].start_time,
        "instructions": (
            "Relax and focus on the feedback tone. "
            "The tone rises when your brain activity meets the therapeutic target. "
            "Session typically runs 20-40 minutes."
        ),
    }


@router.post("/update", response_model=NFStatus)
async def update_nf_session(req: NFInput):
    """Feed one EEG frame into the active NF session; returns real-time feedback."""
    sess = _sessions.get(req.user_id)
    if sess is None:
        raise HTTPException(400, "No active session. Call /ptsd-nf/start/{user_id} first.")

    proto = _PROTOCOLS[sess.protocol]
    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]
    sig = signals[min(1, signals.shape[0] - 1)]  # prefer AF7

    tlo, thi = proto["target_band"]
    slo, shi = proto["suppress_band"]
    alpha   = _band_power(sig, req.fs, tlo, thi) + 1e-9
    hi_beta = _band_power(sig, req.fs, slo, shi) + 1e-9

    # Establish rolling baseline from first 30 frames
    sess.total += 1
    if sess.baseline_alpha is None and sess.total >= 30:
        baseline_frames = list(sess.frames)[-30:]
        if baseline_frames:
            sess.baseline_alpha = float(np.mean([f["alpha"] for f in baseline_frames]))

    baseline = sess.baseline_alpha or alpha

    # Feedback signal: +1 means target met (alpha elevated, hi-beta suppressed)
    alpha_ratio = alpha / (baseline + 1e-9)
    beta_ratio  = hi_beta / (alpha + 1e-9)
    feedback    = float(np.clip(alpha_ratio * 0.7 - beta_ratio * 0.3 - 0.4, -1, 1))

    compliant = alpha_ratio >= 1.05 and beta_ratio <= 1.2
    if compliant:
        sess.compliant += 1

    frame = {"t": time.time(), "alpha": alpha, "hi_beta": hi_beta, "feedback": feedback}
    sess.frames.append(frame)

    compliance = sess.compliant / max(sess.total, 1)
    elapsed    = time.time() - sess.start_time

    note = ""
    if elapsed < 60:
        note = "Warming up — settling into the session"
    elif compliance > 0.6:
        note = "Good — alpha elevation sustained"
    elif compliance > 0.3:
        note = "Progressing — try relaxing your jaw and neck"
    else:
        note = "Struggling — ensure electrode contact and reduce movement"

    return NFStatus(
        user_id=req.user_id,
        protocol=sess.protocol,
        session_active=True,
        elapsed_seconds=elapsed,
        alpha_power=alpha,
        high_beta_power=hi_beta,
        feedback_signal=feedback,
        compliance_score=compliance,
        session_frames=sess.total,
        compliant_frames=sess.compliant,
        protocol_note=note,
    )


@router.post("/stop/{user_id}")
async def stop_nf_session(user_id: str):
    """Stop the active NF session and store summary in history."""
    sess = _sessions.pop(user_id, None)
    if sess is None:
        raise HTTPException(400, "No active session.")
    elapsed = time.time() - sess.start_time
    compliance = sess.compliant / max(sess.total, 1)
    summary = {
        "protocol": sess.protocol,
        "duration_seconds": elapsed,
        "total_frames": sess.total,
        "compliance_score": compliance,
        "ended_at": time.time(),
    }
    _session_history[user_id].append(summary)
    return {"user_id": user_id, "summary": summary}


@router.get("/history/{user_id}", response_model=NFHistory)
async def get_nf_history(user_id: str):
    """Return past NF session summaries for a user."""
    return NFHistory(user_id=user_id, sessions=_session_history[user_id])


@router.get("/protocols")
async def list_nf_protocols():
    """List available neurofeedback protocols."""
    return {"protocols": _PROTOCOLS}

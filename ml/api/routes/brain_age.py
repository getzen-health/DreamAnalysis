"""Brain age estimation API — aperiodic heuristic + SpecParam.

Endpoints:
  POST /brain-age/estimate      — fast heuristic estimate (original #174)
  POST /brain-age/specparam     — full SpecParam decomposition + brain age (#59)
  GET  /brain-age/history/{uid} — per-user session history
  POST /brain-age/reset/{uid}   — clear history
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/brain-age", tags=["brain-age"])

# ──────────────────────────────────────────────────────────────
# Pydantic models
# ──────────────────────────────────────────────────────────────


class BrainAgeInput(BaseModel):
    """Input for both /estimate and /specparam endpoints."""
    # Accept 1-D or 2-D (n_channels × n_samples) signal
    signals: List[List[float]]
    fs: float = 256.0
    chronological_age: Optional[float] = None
    user_id: str


class BrainAgeResult(BaseModel):
    """Response from /estimate (lightweight — compatible with original API)."""
    user_id: str
    predicted_brain_age: Optional[float]
    brain_age_gap: Optional[float]
    aperiodic_exponent: float
    alpha_power: float
    beta_power: float
    delta_power: float
    disclaimer: str
    model_type: str
    processed_at: float


class SpecParamResult(BaseModel):
    """Response from /specparam — full spectral decomposition + brain age."""
    user_id: str
    predicted_brain_age: Optional[float]
    brain_age_gap: Optional[float]
    gap_interpretation: Optional[str]
    gap_severity: Optional[str]
    percentile: Optional[int]
    confidence: float
    # Aperiodic component
    aperiodic_exponent: float
    aperiodic_offset: float
    aperiodic_r2: float
    # Detected oscillatory peaks
    peaks: List[Dict[str, Any]]
    # Alpha peak summary
    alpha_peak_freq: Optional[float]
    alpha_peak_amplitude: Optional[float]
    alpha_peak_bandwidth: Optional[float]
    # Age decomposition
    age_from_exponent: Optional[float]
    age_from_alpha: Optional[float]
    # Spectrogram data for visualization
    spectrogram_freqs: List[float]
    spectrogram_log_psd: List[float]
    spectrogram_aperiodic: List[float]
    spectrogram_residual: List[float]
    # Meta
    disclaimer: str
    model_type: str
    processed_at: float


# ──────────────────────────────────────────────────────────────
# In-memory session history
# ──────────────────────────────────────────────────────────────

_history: dict = defaultdict(lambda: deque(maxlen=200))


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _parse_signals(raw: List[List[float]]) -> np.ndarray:
    """Convert nested list to numpy array; handle 1-D and 2-D inputs."""
    arr = np.array(raw, dtype=float)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]   # (1, n_samples)
    return arr


# ──────────────────────────────────────────────────────────────
# POST /brain-age/estimate
# ──────────────────────────────────────────────────────────────

@router.post("/estimate", response_model=BrainAgeResult)
async def estimate_brain_age(req: BrainAgeInput):
    """Fast brain age estimate via aperiodic heuristic.

    Uses band-power features + aperiodic exponent from a single-pass
    log-log PSD regression. Lightweight — suitable for real-time use.
    """
    from models.brain_age_estimator import get_brain_age_estimator

    signals = _parse_signals(req.signals)

    estimator = get_brain_age_estimator()
    # Pass full multichannel array — estimator averages across channels internally
    result = estimator.predict(
        signals,
        req.fs,
        chronological_age=req.chronological_age,
    )

    out = BrainAgeResult(
        user_id=req.user_id,
        predicted_brain_age=result.get("predicted_age"),
        brain_age_gap=result.get("brain_age_gap"),
        aperiodic_exponent=result.get("aperiodic_exponent", 0.0),
        alpha_power=result.get("alpha_power", 0.0),
        beta_power=result.get("beta_power", 0.0),
        delta_power=result.get("delta_power", 0.0),
        disclaimer=result.get("disclaimer", "Wellness indicator only"),
        model_type=result.get("model_type", "aperiodic_heuristic"),
        processed_at=time.time(),
    )
    _history[req.user_id].append(out.model_dump())
    return out


# ──────────────────────────────────────────────────────────────
# POST /brain-age/specparam
# ──────────────────────────────────────────────────────────────

@router.post("/specparam", response_model=SpecParamResult)
async def specparam_brain_age(req: BrainAgeInput):
    """Full SpecParam decomposition + brain age estimation (issue #59).

    Decomposes the EEG power spectrum into:
      - Aperiodic (1/f) component: offset + exponent * log10(freq)
      - Periodic component: Gaussian peaks above aperiodic baseline

    Brain age is estimated from:
      - Aperiodic exponent (55% weight) — increases ~0.012/year
      - Alpha peak frequency (45% weight) — decreases ~0.03 Hz/year

    Returns the full spectrogram arrays for visualization.
    """
    from models.brain_age_specparam import get_brain_age_specparam

    signals = _parse_signals(req.signals)
    estimator = get_brain_age_specparam(fs=req.fs)
    result = estimator.estimate(signals, chronological_age=req.chronological_age)

    sp = result.get("specparam", {})
    alpha_peak = sp.get("alpha_peak")
    spectrogram = sp.get("spectrogram", {})
    features = result.get("features", {})

    out = SpecParamResult(
        user_id=req.user_id,
        predicted_brain_age=result.get("predicted_age"),
        brain_age_gap=result.get("brain_age_gap"),
        gap_interpretation=result.get("gap_interpretation"),
        gap_severity=result.get("gap_severity"),
        percentile=result.get("percentile"),
        confidence=result.get("confidence", 0.3),
        # Aperiodic
        aperiodic_exponent=features.get("aperiodic_exponent", 0.0),
        aperiodic_offset=features.get("aperiodic_offset", 0.0),
        aperiodic_r2=features.get("aperiodic_r2", 0.0),
        # Peaks
        peaks=sp.get("peaks", []),
        # Alpha peak
        alpha_peak_freq=alpha_peak["center_freq"] if alpha_peak else None,
        alpha_peak_amplitude=alpha_peak["amplitude"] if alpha_peak else None,
        alpha_peak_bandwidth=alpha_peak["bandwidth"] if alpha_peak else None,
        # Age decomposition
        age_from_exponent=features.get("age_from_exponent"),
        age_from_alpha=features.get("age_from_alpha"),
        # Spectrogram
        spectrogram_freqs=spectrogram.get("freqs", []),
        spectrogram_log_psd=spectrogram.get("log_psd", []),
        spectrogram_aperiodic=spectrogram.get("aperiodic_fit", []),
        spectrogram_residual=spectrogram.get("residual", []),
        # Meta
        disclaimer=result.get("disclaimer", ""),
        model_type=result.get("model_type", "specparam_heuristic"),
        processed_at=time.time(),
    )
    _history[req.user_id].append(out.model_dump())
    return out


# ──────────────────────────────────────────────────────────────
# GET /brain-age/history/{user_id}
# ──────────────────────────────────────────────────────────────

@router.get("/history/{user_id}")
async def get_history(user_id: str, limit: int = 50):
    """Return recent brain age estimates for a user (both /estimate and /specparam)."""
    items = list(_history[user_id])[-limit:]
    return {"user_id": user_id, "count": len(items), "history": items}


# ──────────────────────────────────────────────────────────────
# POST /brain-age/reset/{user_id}
# ──────────────────────────────────────────────────────────────

@router.post("/reset/{user_id}")
async def reset_history(user_id: str):
    """Clear brain age history for a user."""
    _history[user_id].clear()
    return {"user_id": user_id, "status": "reset"}

"""Camera rPPG API — contactless HR/HRV from phone camera face ROI.

Prefix : /camera-rppg
Tag    : Camera rPPG

Endpoints
---------
POST /camera-rppg/analyze          — analyse list of [R,G,B] frame means
POST /camera-rppg/analyze-signal   — analyse a pre-extracted green/rPPG channel
GET  /camera-rppg/status           — service capability info
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

router = APIRouter(prefix="/camera-rppg", tags=["Camera rPPG"])

# ---------------------------------------------------------------------------
# Lazy singleton — avoids hard import failure when scipy not installed
# ---------------------------------------------------------------------------
_rppg_instance: Optional[object] = None


def _get_model():
    global _rppg_instance
    if _rppg_instance is None:
        from models.camera_rppg import CameraRPPG

        _rppg_instance = CameraRPPG()
    return _rppg_instance


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------

class FramesPayload(BaseModel):
    """Request body for /analyze — list of per-frame mean RGB values."""

    frames: List[List[float]] = Field(
        ...,
        description=(
            "List of [R, G, B] mean values per frame extracted from face ROI. "
            "Values may be in 0–255 or 0.0–1.0 range. "
            "Minimum 450 frames (15 s at 30 fps)."
        ),
    )
    fps: float = Field(30.0, ge=1.0, le=120.0, description="Camera frame rate (Hz).")


class SignalPayload(BaseModel):
    """Request body for /analyze-signal — raw rPPG or green channel values."""

    signal: List[float] = Field(
        ...,
        description=(
            "1-D list of rPPG signal samples (e.g. mean green channel per frame). "
            "Minimum 450 samples (15 s at 30 fps)."
        ),
    )
    fps: float = Field(30.0, ge=1.0, le=120.0, description="Signal sampling rate (Hz).")


# ---------------------------------------------------------------------------
# Minimum frame count (15 s at given fps, hard floor 450 for 30 fps)
# ---------------------------------------------------------------------------

_MIN_DURATION_S = 15.0
_FLOOR_FRAMES = 450  # 15 s × 30 fps


def _min_frames(fps: float) -> int:
    return max(_FLOOR_FRAMES, int(_MIN_DURATION_S * fps))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/analyze")
async def analyze_rppg_frames(payload: FramesPayload):
    """Extract HR and HRV metrics from per-frame mean RGB values.

    Uses the CHROM algorithm (McDuff et al. 2014):
      - X = 3R – 2G, Y = 1.5R + G – 1.5B
      - Standardise X and Y, combine orthogonally → pulse signal S
      - Bandpass filter 0.7–3 Hz, detect peaks → IBI → HR / RMSSD / pNN50 / LF-HF

    Returns
    -------
    JSON with hr_bpm, rmssd_ms, sdnn_ms, pnn50, lf_hf_ratio, stress_index,
    n_frames, duration_s, n_peaks, algorithm.
    """
    min_f = _min_frames(payload.fps)
    if len(payload.frames) < min_f:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Insufficient frames: got {len(payload.frames)}, "
                f"need at least {min_f} ({_MIN_DURATION_S:.0f}s at {payload.fps} fps)"
            ),
        )

    try:
        model = _get_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Camera rPPG unavailable: {exc}") from exc

    result = model.process_frames(payload.frames, fps=payload.fps)

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return {"status": "ok", **result}


@router.post("/analyze-signal")
async def analyze_rppg_signal(payload: SignalPayload):
    """Extract HR and HRV from a pre-extracted rPPG or green-channel signal.

    Useful when the client performs its own colour extraction before sending.

    Returns
    -------
    Same structure as /analyze.
    """
    min_f = _min_frames(payload.fps)
    if len(payload.signal) < min_f:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Insufficient signal length: got {len(payload.signal)}, "
                f"need at least {min_f} ({_MIN_DURATION_S:.0f}s at {payload.fps} fps)"
            ),
        )

    try:
        model = _get_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Camera rPPG unavailable: {exc}") from exc

    arr = np.array(payload.signal, dtype=np.float64)
    result = model.process_raw_signal(arr, fps=payload.fps)

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return {"status": "ok", **result}


@router.get("/status")
async def rppg_status():
    """Return service capability info.

    Returns
    -------
    JSON with scipy_available (bool), min_frames (int), algorithm (str).
    """
    try:
        import scipy  # noqa: F401

        scipy_available = True
    except ImportError:
        scipy_available = False

    return {
        "scipy_available": scipy_available,
        "min_frames": _FLOOR_FRAMES,
        "min_duration_s": _MIN_DURATION_S,
        "algorithm": "CHROM",
        "reference": "McDuff et al. 2014, IEEE TNSRE",
        "cardiac_band_hz": [0.7, 3.0],
        "hr_range_bpm": [30, 180],
    }

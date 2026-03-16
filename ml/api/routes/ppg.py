"""PPG sensor endpoints for Muse 2 ANCILLARY preset.

Endpoints:
  POST /ppg/analyze  — raw PPG signal → HR, HRV, respiratory rate, emotion estimate
  GET  /ppg/status   — PPG availability and current metrics from connected device
  POST /ppg/stream   — real-time PPG data ingestion (WebSocket-style batched samples)

Muse 2 PPG is on the forehead, 64 Hz. Enable via board.config_board("p50").
"""

import logging
import time
from collections import deque
from typing import Deque, Dict, Optional

import numpy as np
from fastapi import APIRouter, Body, HTTPException

from processing.ppg_processor import (
    FS_PPG,
    extract_hrv_features,
    preprocess_ppg,
    detect_systolic_peaks,
    compute_ibi,
    compute_heart_rate,
    compute_ppg_sqi,
    compute_respiratory_rate,
    _empty_hrv,
)
from models.ppg_emotion import PPGEmotionModel

log = logging.getLogger(__name__)
router = APIRouter()

# ── Module-level singletons ───────────────────────────────────────────────────

_ppg_emotion_model = PPGEmotionModel()

# Rolling buffer for /ppg/stream — accumulates incoming batches.
# Keyed by user_id. Stores at most 30 s of samples (64 Hz × 30 = 1920).
_MAX_BUFFER_SAMPLES: int = int(FS_PPG * 30)
_stream_buffers: Dict[str, Deque[float]] = {}
_stream_last_ts: Dict[str, float] = {}  # last update timestamp per user


def _get_stream_buffer(user_id: str) -> Deque[float]:
    if user_id not in _stream_buffers:
        _stream_buffers[user_id] = deque(maxlen=_MAX_BUFFER_SAMPLES)
    return _stream_buffers[user_id]


# ── POST /ppg/analyze ─────────────────────────────────────────────────────────

@router.post("/ppg/analyze")
async def analyze_ppg(payload: dict = Body(...)):
    """Analyze a raw PPG signal: extract HR, HRV, respiratory rate, and emotion estimate.

    Request body:
      ppg_signal: list[float]  — raw PPG samples from Muse 2 (64 Hz)
      fs: float                — sampling rate in Hz (default 64)

    Response:
      hrv: dict          — all HRV metrics (mean_hr, sdnn, rmssd, pnn50, lf_power,
                           hf_power, lf_hf_ratio, respiratory_rate, stress_index, sqi,
                           n_beats, n_rr_intervals)
      emotion: dict      — arousal, valence, stress, autonomic_state, confidence,
                           explanation
      ppg_samples: int   — number of PPG samples analyzed
      duration_sec: float — analyzed duration in seconds
      status: str
    """
    ppg = payload.get("ppg_signal", [])
    fs = float(payload.get("fs", FS_PPG))

    if not ppg:
        raise HTTPException(status_code=422, detail="ppg_signal is required and must be non-empty")

    arr = np.array(ppg, dtype=np.float64)
    hrv = extract_hrv_features(arr, fs=fs)
    emotion = _ppg_emotion_model.predict(hrv)

    return {
        "status": "ok",
        "hrv": hrv,
        "emotion": emotion,
        "ppg_samples": len(arr),
        "duration_sec": round(len(arr) / fs, 2),
    }


# ── GET /ppg/status ───────────────────────────────────────────────────────────

@router.get("/ppg/status")
async def ppg_status():
    """Return PPG sensor availability and current metrics from the connected device.

    If a Muse 2 is connected and streaming, pulls live PPG data from BrainFlow,
    extracts HRV, and returns current heart rate and signal quality.

    Response:
      available: bool        — True if PPG data could be retrieved
      device_connected: bool — True if a BrainFlow device is active
      is_muse: bool          — True if the connected device is a Muse 2/S
      current_hr: float      — current heart rate in BPM (0 if unavailable)
      sqi: float             — signal quality index 0-1 (0 if unavailable)
      stress_index: float    — current stress level 0-1 (0 if unavailable)
      ppg_enabled: bool      — whether PPG ANCILLARY preset is active
      message: str
    """
    from ._shared import _get_device_manager

    manager = _get_device_manager()

    # No device manager (BrainFlow not installed)
    if manager is None:
        return {
            "available": False,
            "device_connected": False,
            "is_muse": False,
            "ppg_enabled": False,
            "current_hr": 0.0,
            "sqi": 0.0,
            "stress_index": 0.0,
            "message": "BrainFlow not available",
        }

    if not manager.is_connected:
        return {
            "available": False,
            "device_connected": False,
            "is_muse": False,
            "ppg_enabled": False,
            "current_hr": 0.0,
            "sqi": 0.0,
            "stress_index": 0.0,
            "message": "No device connected",
        }

    is_muse = bool(
        manager.current_device_type and manager.current_device_type.startswith("muse_")
    )

    # Try to fetch live PPG data
    ppg_arr: Optional[np.ndarray] = manager.get_ppg_data()

    if ppg_arr is None or len(ppg_arr) < int(FS_PPG * 5):
        return {
            "available": False,
            "device_connected": True,
            "is_muse": is_muse,
            "ppg_enabled": False,
            "current_hr": 0.0,
            "sqi": 0.0,
            "stress_index": 0.0,
            "message": (
                "PPG not enabled — call POST /ppg/stream with ppg_signal data, "
                "or ensure board was configured with p50 preset"
            ),
        }

    hrv = extract_hrv_features(ppg_arr, fs=FS_PPG)
    return {
        "available": True,
        "device_connected": True,
        "is_muse": is_muse,
        "ppg_enabled": True,
        "current_hr": hrv["mean_hr"],
        "sqi": hrv["sqi"],
        "stress_index": hrv["stress_index"],
        "message": "ok",
    }


# ── POST /ppg/stream ──────────────────────────────────────────────────────────

@router.post("/ppg/stream")
async def ppg_stream(payload: dict = Body(...)):
    """Ingest a batch of PPG samples and return running analysis.

    This endpoint accepts batches of PPG samples (e.g., every 0.25-1 s) and
    maintains a 30-second rolling buffer per user. Once the buffer holds at
    least 10 seconds of data, HRV analysis is computed on the full buffer.

    Designed for real-time streaming: call this endpoint once per second with
    the latest PPG samples. The response includes:
      - Current HR and HRV metrics (when buffer has >= 10 s)
      - Emotion estimate (when buffer has >= 10 s)
      - Buffer status

    Request body:
      ppg_samples: list[float]   — new PPG samples to append (64 Hz)
      user_id: str               — user identifier (default "anonymous")
      fs: float                  — sampling rate (default 64)

    Response:
      user_id: str
      buffer_samples: int        — total samples in rolling buffer
      buffer_duration_sec: float
      ready: bool                — True when >= 10 s buffered
      hrv: dict | null           — HRV metrics if ready, else null
      emotion: dict | null       — emotion estimate if ready, else null
      status: str
    """
    samples = payload.get("ppg_samples", [])
    user_id = str(payload.get("user_id", "anonymous"))
    fs = float(payload.get("fs", FS_PPG))

    if not samples:
        raise HTTPException(status_code=422, detail="ppg_samples is required and must be non-empty")

    # Append new samples to the rolling buffer
    buf = _get_stream_buffer(user_id)
    buf.extend(float(s) for s in samples)
    _stream_last_ts[user_id] = time.monotonic()

    buffer_count = len(buf)
    buffer_duration = buffer_count / fs
    min_duration = 10.0  # seconds needed for reliable HRV

    ready = buffer_duration >= min_duration

    if not ready:
        return {
            "user_id": user_id,
            "buffer_samples": buffer_count,
            "buffer_duration_sec": round(buffer_duration, 1),
            "ready": False,
            "hrv": None,
            "emotion": None,
            "status": f"buffering — need {min_duration:.0f}s, have {buffer_duration:.1f}s",
        }

    arr = np.array(list(buf), dtype=np.float64)
    hrv = extract_hrv_features(arr, fs=fs)
    emotion = _ppg_emotion_model.predict(hrv)

    return {
        "user_id": user_id,
        "buffer_samples": buffer_count,
        "buffer_duration_sec": round(buffer_duration, 1),
        "ready": True,
        "hrv": hrv,
        "emotion": emotion,
        "status": "ok",
    }

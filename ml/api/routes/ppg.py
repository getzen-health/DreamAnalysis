"""PPG/HRV stress detection endpoint."""

import logging

import numpy as np
from fastapi import APIRouter, Body, HTTPException

log = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ppg-features")
async def analyze_ppg(payload: dict = Body(...)):
    """Extract HRV features from PPG signal for stress detection.

    Accepts raw PPG samples from Muse 2 ANCILLARY preset.
    Returns HRV metrics: RMSSD, SDNN, LF/HF ratio, stress index.
    """
    from processing.ppg_processor import extract_hrv_features

    ppg = payload.get("ppg_signal", [])
    fs = payload.get("fs", 64)
    if not ppg:
        raise HTTPException(status_code=422, detail="ppg_signal required")
    arr = np.array(ppg, dtype=np.float32)
    features = extract_hrv_features(arr, fs=float(fs))
    return {"status": "ok", "hrv_features": features, "ppg_samples": len(arr)}

"""GRU-based sleep staging API endpoints."""

import logging
from typing import List, Optional

from fastapi import APIRouter, Body, HTTPException

log = logging.getLogger(__name__)
router = APIRouter()


def _get_stager():
    from models.gru_sleep_stager import get_gru_sleep_stager
    return get_gru_sleep_stager()


# ---------------------------------------------------------------------------
# POST /gru-sleep/predict
# ---------------------------------------------------------------------------

@router.post("/gru-sleep/predict")
async def gru_sleep_predict(payload: dict = Body(...)):
    """Predict sleep stage from a raw EEG window.

    Body:
        eeg (list[list[float]] | list[float]): EEG data.
            2-D: shape (n_channels, n_samples)
            1-D: single channel, shape (n_samples,)
        fs (float, optional): Sampling frequency in Hz. Default 256.

    Returns:
        stage (str): Predicted sleep stage (Wake/N1/N2/N3/REM)
        probabilities (dict): Per-stage probability
        confidence (float): Probability of the predicted stage
        channel_ranking (list[int]): Channel indices ranked by delta-power variance
    """
    eeg_raw = payload.get("eeg")
    if eeg_raw is None:
        raise HTTPException(status_code=422, detail="'eeg' field is required")

    fs = float(payload.get("fs", 256.0))

    try:
        import numpy as np
        eeg = np.array(eeg_raw, dtype=float)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot parse 'eeg': {exc}") from exc

    try:
        stager = _get_stager()
        result = stager.predict(eeg, fs=fs)
    except Exception as exc:
        log.exception("gru_sleep predict error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result


# ---------------------------------------------------------------------------
# GET /gru-sleep/status
# ---------------------------------------------------------------------------

@router.get("/gru-sleep/status")
async def gru_sleep_status():
    """Return GRU sleep stager status and buffer info.

    Returns:
        model (str): Model identifier
        buffer_size (int): Current number of frames in temporal buffer
        buffer_capacity (int): Maximum buffer size
        prediction_count (int): Total predictions made since last reset
        ema_initialised (bool): Whether the EMA hidden state is active
        stages (list[str]): Supported sleep stage labels
    """
    stager = _get_stager()
    return stager.get_status()


# ---------------------------------------------------------------------------
# POST /gru-sleep/reset
# ---------------------------------------------------------------------------

@router.post("/gru-sleep/reset")
async def gru_sleep_reset():
    """Reset the temporal buffer and EMA state.

    Call this at the start of a new recording session to prevent
    temporal context from bleeding across sessions.
    """
    stager = _get_stager()
    stager.reset()
    return {"status": "reset", "message": "Temporal buffer and EMA state cleared"}

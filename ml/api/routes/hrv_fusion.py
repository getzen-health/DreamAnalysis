"""HRV-EEG Fusion API endpoints.

POST /hrv/analyze
    Accept EEG signals plus optional HRV biometrics.
    Run the EEG emotion pipeline, then blend the result with HRV data using
    HRVEmotionFusion for a combined stress/valence/arousal prediction.

GET /hrv/status
    Return whether HRV data is available for a given user_id and summarise
    the last fused result stored in the per-user cache.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

router = APIRouter(prefix="/hrv", tags=["hrv_fusion"])

# ── Per-user HRV cache (stores the last received HRV feature dict per user) ───
_hrv_cache: Dict[str, Dict[str, Any]] = {}


# ── Lazy model accessor ───────────────────────────────────────────────────────

def _get_emotion_model():
    """Return the shared EmotionClassifier singleton."""
    try:
        from api.routes._shared import emotion_model
        return emotion_model
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"EmotionClassifier not available: {exc}",
        )


def _get_fusion_model():
    """Return a (stateless) HRVEmotionFusion instance."""
    try:
        from models.hrv_emotion_fusion import HRVEmotionFusion
        return HRVEmotionFusion()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"HRVEmotionFusion not available: {exc}",
        )


# ── Pydantic request / response models ───────────────────────────────────────

class HRVAnalyzeRequest(BaseModel):
    """EEG + optional HRV biometrics for fusion analysis."""

    # EEG — same shape convention as EEGInput in _shared.py
    signals: List[List[float]] = Field(
        ...,
        description=(
            "EEG signals as nested list (channels × samples). "
            "Muse 2 layout: [TP9, AF7, AF8, TP10]."
        ),
    )
    fs: float = Field(
        default=256.0,
        description="EEG sampling rate in Hz.",
    )
    user_id: str = Field(
        default="default",
        description="User identifier for per-user HRV cache.",
    )

    # HRV fields — all optional
    hrv_sdnn: Optional[float] = Field(
        default=None,
        description=(
            "SDNN (standard deviation of NN intervals) in milliseconds. "
            "Typical range: 20–100 ms. < 30 ms → high stress."
        ),
    )
    resting_heart_rate: Optional[float] = Field(
        default=None,
        description=(
            "Resting heart rate in bpm (from wearable overnight average or "
            "Apple Health / Google Fit). > 80 bpm → elevated stress."
        ),
    )
    current_heart_rate: Optional[float] = Field(
        default=None,
        description="Current instantaneous heart rate in bpm.",
    )
    hrv_rmssd: Optional[float] = Field(
        default=None,
        description=(
            "RMSSD (root mean square successive differences) in ms. "
            "Optional — provides a sharper parasympathetic signal than SDNN."
        ),
    )


class HRVAnalyzeResponse(BaseModel):
    """Combined EEG + HRV fusion result."""

    user_id: str
    # EEG-only sub-result
    eeg_emotion: str
    eeg_stress_index: float
    eeg_valence: float
    eeg_arousal: float
    eeg_model_type: str
    # Fused result
    stress_index: float
    valence: float
    arousal: float
    focus_index: float
    hrv_stress_component: float
    hrv_valence_proxy: float
    hrv_contribution: float
    data_quality: Dict[str, Any]
    fusion_weights: Dict[str, float]


class HRVStatusResponse(BaseModel):
    user_id: str
    hrv_available: bool
    last_hrv_fields: List[str]
    message: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/analyze", response_model=HRVAnalyzeResponse)
def hrv_analyze(req: HRVAnalyzeRequest) -> Dict[str, Any]:
    """Run EEG emotion analysis and blend the result with HRV biometrics.

    The endpoint:

    1. Converts ``signals`` to a numpy array and calls
       ``EmotionClassifier.predict()`` to get the EEG-derived emotion result.
    2. Builds an HRV features dict from the optional HRV fields in the request.
       If any HRV fields are present, they are stored in the per-user cache so
       that ``GET /hrv/status`` can report availability.
    3. Calls ``HRVEmotionFusion.predict()`` to blend EEG and HRV signals.
    4. Returns the combined stress / valence / arousal prediction together with
       the raw EEG sub-result and fusion diagnostics.
    """
    # ── Validate and convert EEG ─────────────────────────────────────────────
    try:
        eeg_array = np.array(req.signals, dtype=np.float32)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse signals: {exc}")

    if eeg_array.ndim == 1:
        eeg_array = eeg_array.reshape(1, -1)

    if eeg_array.ndim != 2:
        raise HTTPException(
            status_code=422,
            detail="signals must be 2-D (channels × samples).",
        )

    if eeg_array.shape[1] < 64:
        raise HTTPException(
            status_code=422,
            detail="Each EEG channel must contain at least 64 samples.",
        )

    # ── Run EEG emotion pipeline ─────────────────────────────────────────────
    emotion_model = _get_emotion_model()
    try:
        eeg_result: Dict[str, Any] = emotion_model.predict(eeg_array, req.fs)
    except Exception as exc:
        log.error("EmotionClassifier.predict failed for user %s: %s", req.user_id, exc)
        raise HTTPException(
            status_code=500,
            detail=f"EEG emotion prediction failed: {exc}",
        )

    # ── Build HRV features dict ──────────────────────────────────────────────
    hrv_features: Dict[str, float] = {}
    if req.hrv_sdnn is not None:
        hrv_features["hrv_sdnn"] = req.hrv_sdnn
    if req.resting_heart_rate is not None:
        hrv_features["resting_heart_rate"] = req.resting_heart_rate
    if req.current_heart_rate is not None:
        hrv_features["current_heart_rate"] = req.current_heart_rate
    if req.hrv_rmssd is not None:
        hrv_features["hrv_rmssd"] = req.hrv_rmssd

    # Update per-user HRV cache whenever at least one HRV field is present
    if hrv_features:
        _hrv_cache[req.user_id] = hrv_features

    # ── Fuse EEG + HRV ──────────────────────────────────────────────────────
    fusion_model = _get_fusion_model()
    try:
        fusion_result: Dict[str, Any] = fusion_model.predict(eeg_result, hrv_features)
    except Exception as exc:
        log.error("HRVEmotionFusion.predict failed for user %s: %s", req.user_id, exc)
        raise HTTPException(
            status_code=500,
            detail=f"HRV fusion failed: {exc}",
        )

    return {
        "user_id":              req.user_id,
        # EEG sub-result
        "eeg_emotion":          eeg_result.get("emotion", "unknown"),
        "eeg_stress_index":     float(eeg_result.get("stress_index", 0.5)),
        "eeg_valence":          float(eeg_result.get("valence", 0.0)),
        "eeg_arousal":          float(eeg_result.get("arousal", 0.5)),
        "eeg_model_type":       eeg_result.get("model_type", "feature-based"),
        # Fused result
        "stress_index":         fusion_result["stress_index"],
        "valence":              fusion_result["valence"],
        "arousal":              fusion_result["arousal"],
        "focus_index":          fusion_result["focus_index"],
        "hrv_stress_component": fusion_result["hrv_stress_component"],
        "hrv_valence_proxy":    fusion_result["hrv_valence_proxy"],
        "hrv_contribution":     fusion_result["hrv_contribution"],
        "data_quality":         fusion_result["data_quality"],
        "fusion_weights":       fusion_result["fusion_weights"],
    }


@router.get("/status")
def hrv_status(user_id: str = "default") -> Dict[str, Any]:
    """Return HRV data availability for a user.

    Checks the per-user HRV cache populated by ``POST /hrv/analyze``.
    A ``hrv_available=true`` response means the last ``/analyze`` request
    for this user included at least one HRV field and the fusion was able
    to use real HRV data rather than falling back to EEG-only.

    Args:
        user_id: The user identifier (query parameter).

    Returns:
        Dict with ``hrv_available`` (bool), the list of HRV fields present
        in the last call, and a human-readable status message.
    """
    cached = _hrv_cache.get(user_id)

    if cached is None:
        return {
            "user_id":         user_id,
            "hrv_available":   False,
            "last_hrv_fields": [],
            "message": (
                "No HRV data received for this user yet. "
                "Send hrv_sdnn, resting_heart_rate, or current_heart_rate "
                "in POST /hrv/analyze to enable HRV fusion."
            ),
        }

    fields = list(cached.keys())
    has_core = "hrv_sdnn" in cached or "resting_heart_rate" in cached

    return {
        "user_id":         user_id,
        "hrv_available":   has_core,
        "last_hrv_fields": fields,
        "message": (
            f"HRV data available ({len(fields)} field(s): {', '.join(fields)}). "
            "Fusion uses EEG 70% + HRV 30% for stress; EEG 60% + HRV 40% for valence."
            if has_core else
            f"Partial HRV data ({', '.join(fields)}) — core fields hrv_sdnn / "
            "resting_heart_rate missing. Fusion contribution may be limited."
        ),
    }

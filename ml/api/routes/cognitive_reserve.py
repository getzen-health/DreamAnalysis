"""Cognitive reserve estimation API endpoints.

Endpoints:
    POST /cognitive-reserve/analyze         — analyze EEG epoch → reserve biomarkers
    POST /cognitive-reserve/update-history  — append a score to longitudinal history
    GET  /cognitive-reserve/trend           — get longitudinal trend
    POST /cognitive-reserve/reset           — reset longitudinal history

GitHub issue: #125 (EEG Biomarkers for Cognitive Reserve Estimation in Aging)
"""

from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(tags=["Cognitive Reserve"])


# ── Request / response models ─────────────────────────────────────────────────

class CognitiveReserveAnalyzeRequest(BaseModel):
    eeg_data: List[List[float]] = Field(
        ...,
        description="EEG data as [[ch0_samples...], [ch1_samples...]] "
                    "or [[single_channel_samples...]]",
    )
    fs: float = Field(default=256.0, gt=0, description="Sampling rate in Hz")


class UpdateHistoryRequest(BaseModel):
    score: float = Field(..., ge=0.0, le=100.0, description="Reserve score (0-100)")


class TrendRequest(BaseModel):
    n_sessions: Optional[int] = Field(
        default=None, gt=0,
        description="Limit trend to the last n sessions. Omit to use all history.",
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/cognitive-reserve/analyze")
async def analyze_cognitive_reserve(request: CognitiveReserveAnalyzeRequest):
    """Estimate cognitive reserve from an EEG epoch.

    Computes four spectral biomarkers:
    - Alpha peak frequency (8-13 Hz): higher = more reserve
    - Aperiodic 1/f slope: flatter (less negative) = more reserve
    - Theta/alpha ratio: lower = more reserve
    - Brain age index: 0=young brain, 1=aged brain

    Returns reserve_score (0-100), brain_age_index, alpha_peak_freq,
    aperiodic_slope, theta_alpha_ratio, reserve_category, and a biomarkers
    sub-dict with all numeric values.
    """
    try:
        from models.cognitive_reserve_estimator import get_cognitive_reserve_estimator
        estimator = get_cognitive_reserve_estimator()
        eeg = np.array(request.eeg_data, dtype=np.float64)
        result = estimator.predict(eeg, fs=request.fs)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/cognitive-reserve/update-history")
async def update_cognitive_reserve_history(request: UpdateHistoryRequest):
    """Append a reserve score to the longitudinal session history.

    Call after each session's analyze result to build a trend over time.
    """
    try:
        from models.cognitive_reserve_estimator import get_cognitive_reserve_estimator
        estimator = get_cognitive_reserve_estimator()
        estimator.update_history(request.score)
        return {"status": "ok", "score_added": request.score}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/cognitive-reserve/trend")
async def get_cognitive_reserve_trend(n_sessions: Optional[int] = None):
    """Get longitudinal cognitive reserve trend.

    Returns slope_per_session (points/session), trend ("improving" | "stable" |
    "declining"), and n_sessions (number of data points used).

    Requires at least 2 scores in history for a meaningful trend.
    """
    try:
        from models.cognitive_reserve_estimator import get_cognitive_reserve_estimator
        estimator = get_cognitive_reserve_estimator()
        return estimator.get_longitudinal_trend(n_sessions=n_sessions)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/cognitive-reserve/reset")
async def reset_cognitive_reserve():
    """Reset the longitudinal score history.

    Does not affect the underlying model — only clears stored session scores.
    """
    try:
        from models.cognitive_reserve_estimator import get_cognitive_reserve_estimator
        estimator = get_cognitive_reserve_estimator()
        estimator.reset()
        return {"status": "ok", "message": "Cognitive reserve history cleared."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

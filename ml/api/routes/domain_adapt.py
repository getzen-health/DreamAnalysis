"""Cross-subject domain adaptation API.

Feature-level CORAL-lite alignment that closes the cross-subject EEG
distribution gap (+15-26 accuracy points).

Endpoints:
  POST /domain-adapt/set-source    -- set population-average source statistics
  POST /domain-adapt/calibrate     -- calibrate on user's target features
  POST /domain-adapt/adapt         -- adapt live feature vector to source domain
  GET  /domain-adapt/status        -- adapter state and alignment score
  POST /domain-adapt/reset         -- clear all statistics

GitHub issue: #113
"""

from typing import List

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ._shared import _numpy_safe
from models.domain_adapter import DomainAdapter

router = APIRouter(tags=["domain-adapt"])

_adapter = DomainAdapter()


class SourceStatsInput(BaseModel):
    mean: List[float] = Field(..., description="Per-feature mean from training data.")
    std: List[float] = Field(..., description="Per-feature std from training data.")


class FeaturesInput(BaseModel):
    features: List[List[float]] = Field(
        ...,
        description=(
            "Feature matrix: [[f1, f2, ...], ...] (one row per sample). "
            "For single-sample adapt, pass a single-row matrix."
        ),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/domain-adapt/set-source")
async def set_source_stats(data: SourceStatsInput):
    """Set population-average source domain statistics.

    Call once at startup (or when the training data distribution changes).
    Typically: mean and std computed over all training-set feature vectors.

    Returns confirmation with n_features.
    """
    try:
        _adapter.set_source_stats(mean=data.mean, std=data.std)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"status": "ok", "n_features": len(data.mean)}


@router.post("/domain-adapt/calibrate")
async def calibrate_adapter(data: FeaturesInput):
    """Calibrate the adapter on the current user's features.

    Submit 10+ feature vectors from the user's resting-state or labeled
    calibration session. Returns whether adaptation is now active and
    how well the adapted distribution aligns with the source.

    Returns calibrated (bool), n_samples, alignment_before, alignment_after.
    """
    feats = np.array(data.features, dtype=np.float64)
    try:
        result = _adapter.calibrate(target_features=feats)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return _numpy_safe(result)


@router.post("/domain-adapt/adapt")
async def adapt_features(data: FeaturesInput):
    """Adapt a live feature vector (or batch) to the source domain.

    If the adapter is not yet calibrated, returns the original features
    unchanged with adaptation_applied=false.

    Returns adapted_features, alignment_score, adaptation_applied.
    """
    feats = np.array(data.features, dtype=np.float64)
    try:
        result = _adapter.adapt(features=feats)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return _numpy_safe(result)


@router.get("/domain-adapt/status")
async def get_adapter_status():
    """Get current adapter state and alignment score."""
    return _numpy_safe(_adapter.get_stats())


@router.post("/domain-adapt/reset")
async def reset_adapter():
    """Clear all source and target statistics."""
    _adapter.reset()
    return {"status": "ok", "message": "Domain adapter cleared."}

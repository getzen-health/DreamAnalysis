"""Federated learning API routes for privacy-preserving emotion model improvement.

Endpoints:
  POST /federated-emotion/local-update         -- submit a local model update
  POST /federated-emotion/aggregate            -- trigger aggregation round
  GET  /federated-emotion/status               -- global model status
  GET  /federated-emotion/privacy-budget/{user_id} -- per-user privacy budget

GitHub issue: #434
"""
from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from models.federated_emotion import (
    LocalUpdate,
    aggregate_deltas,
    apply_differential_privacy,
    compute_privacy_budget,
    compute_weight_delta,
    federated_to_dict,
    get_global_model_status,
    get_global_weights,
    privacy_budget_to_dict,
    submit_local_update,
    train_local_model,
    DEFAULT_EPSILON,
    DEFAULT_DELTA,
    DEFAULT_SENSITIVITY,
    N_FEATURES,
    N_CLASSES,
    MIN_SAMPLES_FOR_TRAINING,
)

import numpy as np

router = APIRouter(prefix="/federated-emotion", tags=["federated-emotion"])


# -- Request / response models ------------------------------------------------

class LocalUpdateRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="User ID submitting the update")
    features: List[List[float]] = Field(
        ...,
        min_length=MIN_SAMPLES_FOR_TRAINING,
        description=f"EEG feature matrix: list of {N_FEATURES}-element vectors",
    )
    labels: List[int] = Field(
        ...,
        min_length=MIN_SAMPLES_FOR_TRAINING,
        description=f"Emotion labels (0-{N_CLASSES - 1})",
    )
    quality_score: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Data quality score (0-1, higher = cleaner signal)",
    )
    epsilon: float = Field(
        default=DEFAULT_EPSILON, gt=0.0, le=10.0,
        description="Differential privacy epsilon for this round",
    )
    delta: float = Field(
        default=DEFAULT_DELTA, gt=0.0, lt=1.0,
        description="Differential privacy delta for this round",
    )


class AggregateRequest(BaseModel):
    min_updates: int = Field(
        default=2, ge=1,
        description="Minimum number of pending updates required to aggregate",
    )


# -- Endpoints ----------------------------------------------------------------

@router.post("/local-update")
async def federated_local_update(request: LocalUpdateRequest):
    """Submit a local model update trained on the user's private EEG data.

    The user's raw EEG features are used to train a local model, compute
    weight deltas against the global model, apply differential privacy noise,
    and submit the protected delta to the aggregation queue.

    Raw features are never stored server-side -- only the noised weight delta
    is retained.
    """
    try:
        features = np.array(request.features, dtype=np.float32)
        labels = np.array(request.labels, dtype=np.int32)

        # Validate shapes
        if features.ndim != 2 or features.shape[1] != N_FEATURES:
            raise HTTPException(
                status_code=400,
                detail=f"Features must be (n_samples, {N_FEATURES}), got {features.shape}",
            )
        if labels.shape[0] != features.shape[0]:
            raise HTTPException(
                status_code=400,
                detail=f"Labels length ({labels.shape[0]}) must match features rows ({features.shape[0]})",
            )
        if not np.all((labels >= 0) & (labels < N_CLASSES)):
            raise HTTPException(
                status_code=400,
                detail=f"Labels must be in [0, {N_CLASSES})",
            )

        # Get current global weights as starting point
        global_weights = get_global_weights()

        # Train locally
        trained = train_local_model(
            features=features,
            labels=labels,
            initial_weights=global_weights,
        )

        # Compute delta
        if global_weights is not None:
            delta = compute_weight_delta(trained, global_weights)
        else:
            delta = trained

        # Apply differential privacy
        noised_delta = apply_differential_privacy(
            weight_delta=delta,
            epsilon=request.epsilon,
            delta=request.delta,
            sensitivity=DEFAULT_SENSITIVITY,
        )

        # Submit to aggregation queue
        result = submit_local_update(
            user_id=request.user_id,
            weight_delta=noised_delta,
            n_samples=features.shape[0],
            quality_score=request.quality_score,
            epsilon=request.epsilon,
            delta=request.delta,
        )

        return result

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/aggregate")
async def federated_aggregate(request: AggregateRequest):
    """Trigger a federated aggregation round.

    Aggregates all pending local updates via weighted FedAvg and updates
    the global model.  Requires at least *min_updates* pending submissions.
    """
    try:
        status = get_global_model_status()

        if status.pending_updates < request.min_updates:
            return {
                "success": False,
                "reason": "insufficient_updates",
                "pending": status.pending_updates,
                "required": request.min_updates,
                "message": (
                    f"Need at least {request.min_updates} pending updates, "
                    f"have {status.pending_updates}."
                ),
            }

        result = aggregate_deltas()

        if result is None:
            return {
                "success": False,
                "reason": "no_updates",
                "message": "No pending updates to aggregate.",
            }

        new_status = get_global_model_status()
        return {
            "success": True,
            "round": new_status.current_round,
            "status": federated_to_dict(new_status),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def federated_status():
    """Return the current federated learning status.

    Includes: current round, total participants, contribution counts,
    pending update count, and round history.
    """
    try:
        status = get_global_model_status()
        return federated_to_dict(status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/privacy-budget/{user_id}")
async def federated_privacy_budget(user_id: str):
    """Return the privacy budget status for a specific user.

    Shows cumulative epsilon spent, remaining budget, contribution count,
    and whether the budget is exhausted.
    """
    try:
        budget = compute_privacy_budget(user_id)
        return privacy_budget_to_dict(budget)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

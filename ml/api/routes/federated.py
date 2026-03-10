"""Federated learning API endpoints.

Privacy architecture:
    - Raw EEG never sent to server — only weight deltas
    - Each user must opt in explicitly before updates are accepted
    - Local differential privacy (Gaussian noise) applied on client side
    - Server aggregates via FedAvg / Fuzzy-FedAvg

Endpoints:
    POST /federated/opt-in             — User opts in to FL
    POST /federated/opt-out            — User opts out
    POST /federated/submit-update      — Client submits weight delta
    GET  /federated/global-model       — Client fetches global weights
    POST /federated/add-local-sample   — Add one labeled EEG sample to local buffer
    POST /federated/local-train        — Trigger local training, return delta
    GET  /federated/status             — FL training progress + stats
    POST /federated/force-aggregate    — Force aggregation (admin/debug)
"""

import logging
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Federated Learning"])


# ── Request / Response models ─────────────────────────────────────────────────

class OptInRequest(BaseModel):
    user_id: str = Field(..., description="User ID opting in to federated learning")


class SubmitUpdateRequest(BaseModel):
    user_id: str
    delta: Dict[str, List] = Field(
        ..., description="Weight delta dict: {layer_name: flat_list_of_floats}"
    )
    n_samples: int = Field(..., ge=1, description="Number of local samples used in training")


class AddSampleRequest(BaseModel):
    user_id: str
    features: List[float] = Field(
        ..., min_length=1, description="17-element EEG feature vector"
    )
    label: int = Field(
        ..., ge=0, le=5,
        description="Emotion label: 0=happy,1=sad,2=angry,3=fear,4=surprise,5=neutral"
    )


class LocalTrainRequest(BaseModel):
    user_id: str
    submit_to_server: bool = Field(
        default=True,
        description="If True, automatically submit the computed delta to the server"
    )
    dp_epsilon: Optional[float] = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Differential privacy epsilon (smaller = more private)"
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/federated/opt-in")
async def federated_opt_in(request: OptInRequest):
    """User opts in to federated learning.

    Required before any weight updates are accepted from this user.
    Consent is stored in-memory; UI should confirm before calling.
    """
    try:
        from models.federated_trainer import get_federated_trainer
        trainer = get_federated_trainer()
        trainer.opt_in(request.user_id)
        return {
            "success": True,
            "user_id": request.user_id,
            "message": "Opted in to federated learning. Your EEG data stays on your device — only model weight updates are shared.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/federated/opt-out")
async def federated_opt_out(request: OptInRequest):
    """User opts out of federated learning. No further updates accepted."""
    try:
        from models.federated_trainer import get_federated_trainer
        trainer = get_federated_trainer()
        trainer.opt_out(request.user_id)
        return {"success": True, "user_id": request.user_id, "message": "Opted out."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/federated/submit-update")
async def federated_submit_update(request: SubmitUpdateRequest):
    """Client submits a weight delta to the server.

    The server collects deltas from multiple opted-in users, then runs FedAvg
    once enough updates arrive. Raw EEG is never included in the request.
    """
    try:
        from models.federated_trainer import get_federated_trainer
        trainer = get_federated_trainer()

        result = trainer.receive_update(
            client_id=request.user_id,
            delta_dict=request.delta,
            n_samples=request.n_samples,
        )

        # Auto-aggregate if threshold reached
        if result.get("will_aggregate") and trainer.should_aggregate():
            trainer.aggregate()
            result["aggregated"] = True
            result["message"] = "Aggregation complete — global model updated."
        else:
            result["aggregated"] = False

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/federated/global-model")
async def federated_get_global_model(user_id: str):
    """Fetch the current global model weights.

    Returns the aggregated model that clients should apply before local training.
    """
    try:
        from models.federated_trainer import get_federated_trainer
        trainer = get_federated_trainer()
        weights = trainer.get_global_weights()

        if weights is None:
            return {
                "available": False,
                "message": "No global model yet — need at least 2 clients to complete a round.",
            }

        return {
            "available": True,
            "weights": weights,
            "round_num": trainer.get_status()["round_num"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/federated/add-local-sample")
async def federated_add_local_sample(request: AddSampleRequest):
    """Add one labeled EEG feature vector to the user's local training buffer.

    Called once per analyzed EEG epoch when the user confirms the emotion label.
    Data never leaves the device — it's stored in the server process memory
    keyed by user_id, but raw features are discarded after local training runs.
    """
    try:
        from models.federated_client import get_federated_client
        client = get_federated_client(request.user_id)
        features = np.array(request.features, dtype=np.float32)
        client.add_sample(features, request.label)

        return {
            "success": True,
            "user_id": request.user_id,
            "n_local_samples": client.n_local_samples(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/federated/local-train")
async def federated_local_train(request: LocalTrainRequest):
    """Trigger local training on the user's buffered EEG samples.

    Returns the weight delta. If submit_to_server=True (default), the delta
    is automatically submitted to /federated/submit-update.

    Requires at least 10 local samples to run.
    """
    try:
        from models.federated_client import get_federated_client, FederatedEEGClient
        from models.federated_trainer import get_federated_trainer

        client = get_federated_client(request.user_id)

        # Sync with latest global model first
        trainer = get_federated_trainer()
        global_weights = trainer.get_global_weights()
        if global_weights:
            client.apply_global_weights(global_weights)

        # Require minimum local data
        n = client.n_local_samples()
        if n < 10:
            return {
                "success": False,
                "reason": f"Need at least 10 local samples, have {n}. Add more labeled EEG epochs first.",
                "n_local_samples": n,
            }

        # Override DP epsilon if specified
        if request.dp_epsilon is not None:
            client.dp_epsilon = request.dp_epsilon

        # Run local training
        delta_dict, n_samples = client.local_train()

        if not delta_dict:
            return {"success": False, "reason": "Local training produced no output."}

        result: Dict = {
            "success": True,
            "user_id": request.user_id,
            "n_samples": n_samples,
            "dp_enabled": client.use_dp,
            "dp_epsilon": client.dp_epsilon,
        }

        if request.submit_to_server:
            submit_result = trainer.receive_update(
                client_id=request.user_id,
                delta_dict=delta_dict,
                n_samples=n_samples,
            )
            if trainer.should_aggregate():
                trainer.aggregate()
                submit_result["aggregated"] = True
            result["submitted"] = submit_result
        else:
            # Return delta for manual submission
            result["delta"] = delta_dict

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/federated/status")
async def federated_status():
    """Federated learning training status: rounds, participants, accuracy."""
    try:
        from models.federated_trainer import get_federated_trainer
        trainer = get_federated_trainer()
        status = trainer.get_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/federated/force-aggregate")
async def federated_force_aggregate():
    """Force aggregation of pending updates (admin/debug use only).

    Runs regardless of whether min_clients threshold is met.
    """
    try:
        from models.federated_trainer import get_federated_trainer
        trainer = get_federated_trainer()
        result = trainer.aggregate()

        if result is None:
            return {"success": False, "reason": "No pending updates to aggregate."}

        return {
            "success": True,
            "round_num": trainer.get_status()["round_num"],
            "message": "Aggregation complete.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

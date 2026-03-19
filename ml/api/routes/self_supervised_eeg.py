"""Self-supervised EEG pretraining API (#387).

Endpoints:
  POST /self-supervised/create-task    -- create pretext task samples
  POST /self-supervised/few-shot-config -- generate few-shot calibration config
  GET  /self-supervised/status          -- service readiness check
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ._shared import _numpy_safe

log = logging.getLogger(__name__)

router = APIRouter(prefix="/self-supervised", tags=["self-supervised-eeg"])


# ── Pydantic request / response schemas ──────────────────────────────────────

class CreateTaskRequest(BaseModel):
    signal: List[List[float]] = Field(
        ...,
        description="EEG signal as list of channels, each a list of floats. "
                    "Shape: (n_channels, n_samples).",
    )
    task_type: str = Field(
        default="masked_prediction",
        description="Pretext task type: masked_prediction, contrastive, temporal_order",
    )
    negative_signal: Optional[List[List[float]]] = Field(
        default=None,
        description="Negative example signal for contrastive task",
    )
    mask_ratio: float = Field(
        default=0.15, ge=0.0, le=0.5,
        description="Fraction of timepoints to mask (masked_prediction only)",
    )
    seed: Optional[int] = Field(default=None, description="Random seed")


class CreateTaskResponse(BaseModel):
    task_type: str
    task_data: Dict[str, Any]
    signal_shape: List[int]
    created_at: float


class FewShotConfigRequest(BaseModel):
    n_samples_per_class: int = Field(default=5, ge=1, le=100, description="Labeled samples per class")
    n_classes: int = Field(default=6, ge=2, le=20, description="Number of emotion classes")
    n_channels: int = Field(default=4, ge=1, le=64, description="Number of EEG channels")
    fs: int = Field(default=256, ge=64, le=1024, description="Sampling rate (Hz)")
    epoch_duration: float = Field(default=4.0, ge=0.5, le=30.0, description="Epoch duration (seconds)")
    class_names: Optional[List[str]] = Field(default=None, description="Emotion class names")


class FewShotConfigResponse(BaseModel):
    config: Dict[str, Any]
    created_at: float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/status")
async def self_supervised_status() -> Dict[str, Any]:
    """Return service readiness and supported pretext tasks."""
    from models.self_supervised_eeg import SUPPORTED_TASKS, EMOTION_CLASSES

    return {
        "status": "ready",
        "supported_tasks": list(SUPPORTED_TASKS),
        "emotion_classes": list(EMOTION_CLASSES),
        "default_mask_ratio": 0.15,
        "default_n_channels": 4,
        "default_fs": 256,
    }


@router.post("/create-task", response_model=CreateTaskResponse)
async def create_task(req: CreateTaskRequest) -> Dict[str, Any]:
    """Create a pretext task sample for self-supervised pretraining.

    Supports three task types:
      - masked_prediction: mask 15% of timepoints, predict from context
      - contrastive: create positive/negative pairs for contrastive learning
      - temporal_order: predict if segment A is before or after segment B
    """
    if not req.signal or not req.signal[0]:
        raise HTTPException(status_code=400, detail="Signal must not be empty")

    try:
        from models.self_supervised_eeg import create_pretext_task

        signal = np.array(req.signal, dtype=np.float64)
        negative = None
        if req.negative_signal is not None:
            negative = np.array(req.negative_signal, dtype=np.float64)

        result = create_pretext_task(
            signal=signal,
            task_type=req.task_type,
            negative_signal=negative,
            seed=req.seed,
            mask_ratio=req.mask_ratio,
        )

        # Convert numpy arrays to lists for JSON serialization
        task_data = {}
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                task_data[key] = _numpy_safe(value.tolist())
            else:
                task_data[key] = _numpy_safe(value)

        return {
            "task_type": req.task_type,
            "task_data": task_data,
            "signal_shape": list(signal.shape),
            "created_at": time.time(),
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        log.exception("Pretext task creation failed")
        raise HTTPException(status_code=500, detail=f"Task creation failed: {exc}")


@router.post("/few-shot-config", response_model=FewShotConfigResponse)
async def few_shot_config(req: FewShotConfigRequest) -> Dict[str, Any]:
    """Generate optimal few-shot calibration configuration.

    Given the number of labeled samples available per emotion class,
    computes recommended hyperparameters for fine-tuning a pretrained
    EEG encoder with minimal data.
    """
    try:
        from models.self_supervised_eeg import compute_few_shot_config

        result = compute_few_shot_config(
            n_samples_per_class=req.n_samples_per_class,
            n_classes=req.n_classes,
            n_channels=req.n_channels,
            fs=req.fs,
            epoch_duration=req.epoch_duration,
            class_names=req.class_names,
        )

        # Remove the dataclass object, keep serializable fields
        config_dict = {k: v for k, v in result.items() if k != "config"}

        return {
            "config": _numpy_safe(config_dict),
            "created_at": time.time(),
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        log.exception("Few-shot config computation failed")
        raise HTTPException(status_code=500, detail=f"Config computation failed: {exc}")

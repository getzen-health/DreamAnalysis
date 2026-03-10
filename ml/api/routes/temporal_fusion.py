"""Multi-temporal emotion fusion API endpoints.

Implements hierarchical 4-level temporal buffering per user + modality:
  Level 1 (fast):    0.5–2 s   — micro-changes, startle flashes
  Level 2 (medium):  2–10 s    — current emotional state
  Level 3 (slow):    10–60 s   — mood / trend
  Level 4 (context): 1–24 h    — daily pattern

Endpoints
---------
POST /temporal-fusion/push
    Push one emotion reading into the buffer.

GET  /temporal-fusion/fuse/{user_id}
    Return attention-weighted fused state across all 4 temporal levels.

GET  /temporal-fusion/stats/{user_id}
    Return per-level descriptive statistics.

DELETE /temporal-fusion/clear/{user_id}
    Clear all buffers for a user.

GET  /temporal-fusion/buffers
    List all active buffers in the registry.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from processing.temporal_buffer import get_buffer, list_buffers

router = APIRouter(prefix="/temporal-fusion", tags=["Multi-Temporal Fusion"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class EmotionReading(BaseModel):
    """A single emotion snapshot to push into the temporal buffer."""
    user_id: str = Field(default="default", description="User identifier")
    modality: str = Field(
        default="voice",
        description="Signal source: voice | eeg | hrv | multimodal",
    )
    valence: Optional[float] = Field(
        default=None, description="Emotional valence (-1 to 1)"
    )
    arousal: Optional[float] = Field(
        default=None, description="Emotional arousal (0 to 1)"
    )
    stress_index: Optional[float] = Field(
        default=None, description="Stress index (0 to 1)"
    )
    focus_index: Optional[float] = Field(
        default=None, description="Focus index (0 to 1)"
    )
    extra: Optional[Dict[str, float]] = Field(
        default=None, description="Any additional numeric emotion dimensions"
    )
    timestamp: Optional[float] = Field(
        default=None, description="Unix timestamp (defaults to server time)"
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/push")
async def push_emotion_reading(req: EmotionReading):
    """Push one emotion reading into the multi-temporal buffer.

    The reading is stored at all 4 temporal levels simultaneously.
    Older samples outside each level's rolling window are automatically evicted.

    Typical usage: call after every voice check-in, EEG epoch, or HRV poll.
    Returns the current fused state immediately after ingestion.
    """
    values: Dict[str, float] = {}
    if req.valence is not None:
        values["valence"] = req.valence
    if req.arousal is not None:
        values["arousal"] = req.arousal
    if req.stress_index is not None:
        values["stress_index"] = req.stress_index
    if req.focus_index is not None:
        values["focus_index"] = req.focus_index
    if req.extra:
        values.update({k: v for k, v in req.extra.items() if isinstance(v, (int, float))})

    if not values:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=422,
            detail="At least one emotion dimension is required (valence, arousal, stress_index, focus_index, or extra).",
        )

    buf = get_buffer(user_id=req.user_id, modality=req.modality)
    buf.push(values, ts=req.timestamp)

    fused = buf.fuse(emotion_keys=list(values.keys()))
    return {
        "status": "pushed",
        "pushed_values": values,
        "pushed_at": req.timestamp or time.time(),
        "fused": fused,
    }


@router.get("/fuse/{user_id}")
async def get_fused_state(user_id: str, modality: str = "voice"):
    """Return attention-weighted fused emotion state across all 4 temporal levels.

    The fused value for each dimension is a weighted average of the per-level
    means, where attention weights follow EmotionTFN (MDPI 2025) defaults:
      - valence:      [fast=0.10, medium=0.35, slow=0.40, context=0.15]
      - arousal:      [fast=0.20, medium=0.40, slow=0.30, context=0.10]
      - stress_index: [fast=0.15, medium=0.40, slow=0.35, context=0.10]
      - focus_index:  [fast=0.10, medium=0.35, slow=0.40, context=0.15]

    Levels with no samples are zeroed-out and the remaining weights renormalized.

    Args:
        user_id:  User identifier.
        modality: voice | eeg | hrv | multimodal (default: voice).
    """
    buf = get_buffer(user_id=user_id, modality=modality)
    if buf._push_count == 0:
        return {
            "user_id": user_id,
            "modality": modality,
            "status": "no_data",
            "message": "No readings pushed yet. Call POST /temporal-fusion/push first.",
        }
    fused = buf.fuse()
    fused["status"] = "ok"
    return fused


@router.get("/stats/{user_id}")
async def get_temporal_stats(user_id: str, modality: str = "voice"):
    """Return descriptive statistics per temporal level.

    For each of the 4 levels, returns sample count, window size, and
    per-dimension mean/std/min/max within that window.

    Args:
        user_id:  User identifier.
        modality: voice | eeg | hrv | multimodal (default: voice).
    """
    buf = get_buffer(user_id=user_id, modality=modality)
    stats = buf.level_stats()
    return {
        "user_id": user_id,
        "modality": modality,
        "n_samples_total": buf._push_count,
        "levels": stats,
        "level_windows": {
            "fast":    "0–2 seconds (micro-expressions, startle)",
            "medium":  "0–10 seconds (current emotional state)",
            "slow":    "0–60 seconds (mood / trend)",
            "context": "0–24 hours (daily pattern)",
        },
    }


@router.delete("/clear/{user_id}")
async def clear_user_buffers(user_id: str, modality: Optional[str] = None):
    """Clear temporal buffer(s) for a user.

    Args:
        user_id:  User identifier.
        modality: If provided, clears only that modality's buffer.
                  If omitted, clears all modalities for this user.
    """
    from processing.temporal_buffer import _registry

    cleared = []
    if modality:
        key = f"{user_id}:{modality}"
        if key in _registry:
            _registry[key].clear()
            cleared.append(f"{user_id}:{modality}")
    else:
        for key in list(_registry.keys()):
            if key.startswith(f"{user_id}:"):
                _registry[key].clear()
                cleared.append(key)

    return {
        "status": "cleared",
        "cleared_buffers": cleared,
        "n_cleared": len(cleared),
    }


@router.get("/buffers")
async def list_active_buffers():
    """List all active temporal buffers across all users and modalities."""
    buffers = list_buffers()
    return {
        "n_buffers": len(buffers),
        "buffers": buffers,
    }

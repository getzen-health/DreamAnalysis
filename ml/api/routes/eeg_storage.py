"""EEG session data storage endpoints for model training pipeline.

Accepts raw EEG epochs from live sessions (BLE or WebSocket) and stores them
as numpy compressed files for future model training and personalization.
"""

import logging
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

USER_DATA_DIR = Path(__file__).parent.parent.parent / "user_data"


class SaveEEGRequest(BaseModel):
    signals: List[List[float]] = Field(..., description="EEG signals (channels x samples)")
    user_id: str = Field(..., description="User identifier for stored training data")
    session_id: Optional[str] = Field(default=None, description="Session ID (auto-generated if omitted)")
    sample_rate: float = Field(default=256.0)
    device_type: str = Field(default="muse_2")
    predicted_emotion: Optional[str] = Field(default=None)
    user_correction: Optional[str] = Field(default=None, description="User's corrected emotion label")
    band_powers: Optional[Dict[str, float]] = Field(default=None)
    frontal_asymmetry: Optional[float] = Field(default=None)
    valence: Optional[float] = Field(default=None)
    arousal: Optional[float] = Field(default=None)
    signal_quality: Optional[float] = Field(default=None, description="SQI 0-1")


@router.post("/sessions/save-eeg")
async def save_eeg(data: SaveEEGRequest):
    """Save raw EEG epoch with metadata for future model training.

    Stores each epoch as a compressed .npz file under
    user_data/{user_id}/sessions/{session_id}/epoch_{timestamp}.npz
    """
    signals = np.array(data.signals, dtype=np.float32)

    if signals.ndim != 2 or signals.shape[0] < 1:
        raise HTTPException(status_code=422, detail="signals must be 2D (channels x samples)")

    if signals.shape[1] < 32:
        raise HTTPException(status_code=422, detail="signals too short — need at least 32 samples")

    session_id = data.session_id or str(uuid.uuid4())[:8]
    ts = int(time.time() * 1000)

    # Create directory structure
    session_dir = USER_DATA_DIR / data.user_id / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    filename = f"epoch_{ts}.npz"
    filepath = session_dir / filename

    # Build metadata dict (only include non-None values)
    metadata = {
        "user_id": data.user_id,
        "session_id": session_id,
        "sample_rate": data.sample_rate,
        "device_type": data.device_type,
        "timestamp": time.time(),
        "n_channels": int(signals.shape[0]),
        "n_samples": int(signals.shape[1]),
        "duration_sec": float(signals.shape[1] / data.sample_rate),
    }
    if data.predicted_emotion is not None:
        metadata["predicted_emotion"] = data.predicted_emotion
    if data.user_correction is not None:
        metadata["user_correction"] = data.user_correction
    if data.valence is not None:
        metadata["valence"] = float(data.valence)
    if data.arousal is not None:
        metadata["arousal"] = float(data.arousal)
    if data.frontal_asymmetry is not None:
        metadata["frontal_asymmetry"] = float(data.frontal_asymmetry)
    if data.signal_quality is not None:
        metadata["signal_quality"] = float(data.signal_quality)

    # Save signals + band_powers + metadata
    save_kwargs: dict = {"signals": signals, "metadata": np.array([metadata])}
    if data.band_powers:
        bp_keys = sorted(data.band_powers.keys())
        save_kwargs["band_powers"] = np.array([data.band_powers[k] for k in bp_keys], dtype=np.float32)
        save_kwargs["band_power_names"] = np.array(bp_keys)

    np.savez_compressed(str(filepath), **save_kwargs)

    # Count total epochs for this user
    total_epochs = sum(1 for _ in (USER_DATA_DIR / data.user_id / "sessions").rglob("epoch_*.npz"))

    logger.info(
        f"[save-eeg] user={data.user_id} session={session_id} "
        f"shape={signals.shape} emotion={data.predicted_emotion} "
        f"total_epochs={total_epochs}"
    )

    return {
        "saved": filename,
        "session_id": session_id,
        "user_id": data.user_id,
        "total_epochs": total_epochs,
        "shape": list(signals.shape),
    }


@router.get("/sessions/training-data/stats")
async def training_data_stats(user_id: str):
    """Get statistics on stored EEG training data."""
    if not USER_DATA_DIR.exists():
        return {"total_epochs": 0, "users": {}, "ready_to_train": False}

    users_data: Dict[str, dict] = {}
    total = 0

    user_dirs = [USER_DATA_DIR / user_id]

    for user_dir in user_dirs:
        sessions_dir = user_dir / "sessions"
        if not sessions_dir.exists():
            continue

        uid = user_dir.name
        epochs = list(sessions_dir.rglob("epoch_*.npz"))
        n_epochs = len(epochs)
        total += n_epochs

        # Count labeled epochs (those with user_correction)
        n_labeled = 0
        emotion_counts: Dict[str, int] = {}
        for ep in epochs:
            try:
                data = np.load(str(ep), allow_pickle=True)
                meta = data["metadata"][0] if "metadata" in data else {}
                label = meta.get("user_correction") or meta.get("predicted_emotion")
                if label:
                    n_labeled += 1
                    emotion_counts[label] = emotion_counts.get(label, 0) + 1
            except Exception:
                continue

        n_sessions = len(list(sessions_dir.iterdir())) if sessions_dir.exists() else 0
        users_data[uid] = {
            "total_epochs": n_epochs,
            "labeled_epochs": n_labeled,
            "n_sessions": n_sessions,
            "emotion_distribution": emotion_counts,
        }

    return {
        "total_epochs": total,
        "users": users_data,
        "ready_to_train": total >= 50,
    }

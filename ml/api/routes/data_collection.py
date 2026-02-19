"""Muse 2 training data collection endpoints."""

import json
import time
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException

from ._shared import CollectRequest

router = APIRouter()

_COLLECT_DIR = Path("data/muse2_collected")


@router.post("/collect-training-data")
async def collect_training_data(data: CollectRequest):
    """Save labeled EEG data from live Muse 2 sessions for future training."""
    valid_labels = ["happy", "sad", "angry", "fearful", "relaxed", "focused"]
    if data.label not in valid_labels:
        raise HTTPException(status_code=422, detail=f"Invalid label. Must be one of: {valid_labels}")

    signals = np.array(data.signals)
    if signals.shape[0] < 1 or signals.shape[1] < 34:
        raise HTTPException(status_code=422, detail="Signals too short. Need at least 34 samples per channel.")

    _COLLECT_DIR.mkdir(parents=True, exist_ok=True)

    filename = f"{data.label}_{int(time.time() * 1000)}.json"
    filepath = _COLLECT_DIR / filename

    record = {
        "signals": [ch.tolist() if isinstance(ch, np.ndarray) else ch for ch in data.signals],
        "sample_rate": data.sample_rate,
        "label": data.label,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_channels": signals.shape[0],
        "n_samples": signals.shape[1],
    }
    filepath.write_text(json.dumps(record))

    total_files = len(list(_COLLECT_DIR.glob("*.json")))
    return {
        "saved": filename,
        "total_collected": total_files,
        "label": data.label,
    }


@router.get("/collected-data/stats")
async def collected_data_stats():
    """Get statistics on collected Muse 2 training data."""
    if not _COLLECT_DIR.exists():
        return {"total": 0, "per_label": {}, "ready_to_train": False}

    files = list(_COLLECT_DIR.glob("*.json"))
    per_label: dict = {}
    for f in files:
        label = f.stem.rsplit("_", 1)[0]
        per_label[label] = per_label.get(label, 0) + 1

    return {
        "total": len(files),
        "per_label": per_label,
        "ready_to_train": len(files) >= 30,
    }

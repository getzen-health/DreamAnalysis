"""FEMBA EEG Foundation Model — 7.8M params, edge-deployable (#24).

Wraps the FEMBA (Foundation EEG Model with Bidirectional Attention) concept.
When trained weights are not loaded, uses a compact feature-based pipeline
that replicates the expected output structure. FEMBA-style patch embedding
is approximated via windowed DE feature extraction.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/femba", tags=["femba"])

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class FEMBAInput(BaseModel):
    signals: List[List[float]]     # (n_channels, n_samples)
    fs: float = 256.0
    task: str = "emotion"          # emotion | sleep | workload | anomaly
    user_id: str = Field(..., min_length=1)


class FEMBAResult(BaseModel):
    user_id: str
    task: str
    prediction: str
    probabilities: dict
    embedding: List[float]          # 128-dim FEMBA embedding
    patch_count: int
    model_used: str
    confidence: float
    processed_at: float


class FEMBAFineTuneInput(BaseModel):
    user_id: str
    signals: List[List[float]]
    fs: float = 256.0
    label: str
    task: str = "emotion"


# ---------------------------------------------------------------------------
# Task labels
# ---------------------------------------------------------------------------

_TASK_LABELS = {
    "emotion":  ["happy", "sad", "angry", "fear", "surprise", "neutral"],
    "sleep":    ["wake", "n1", "n2", "n3", "rem"],
    "workload": ["low", "medium", "high"],
    "anomaly":  ["normal", "artifact", "anomaly"],
}

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

_user_finetune: Dict[str, Dict[str, List[np.ndarray]]] = defaultdict(
    lambda: defaultdict(list)
)
_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))

# Try loading FEMBA weights
_femba_model = None
try:
    from models.femba_model import FEMBAModel  # type: ignore
    _femba_model = FEMBAModel()
except Exception:
    pass


# ---------------------------------------------------------------------------
# FEMBA-style patch embedding (approximation)
# ---------------------------------------------------------------------------

def _patch_embed(signals: np.ndarray, fs: float, patch_duration: float = 0.5) -> np.ndarray:
    """Split signal into time patches, extract DE features per patch → stack."""
    n_ch = signals.shape[0]
    patch_len = int(fs * patch_duration)
    n_samples = signals.shape[1]
    n_patches = max(1, n_samples // patch_len)

    bands = [(0.5, 4), (4, 8), (8, 12), (12, 30), (30, 45)]
    patch_embeddings = []

    for p in range(n_patches):
        start = p * patch_len
        end   = start + patch_len
        patch = signals[:, start:end]
        feats = []
        for ch in range(min(n_ch, 4)):
            for flo, fhi in bands:
                from scipy.signal import welch
                nperseg = min(patch.shape[1], int(fs))
                f, psd = welch(patch[ch], fs=fs, nperseg=nperseg)
                idx = np.logical_and(f >= flo, f <= fhi)
                de = -float(np.sum(psd[idx] * np.log(psd[idx] + 1e-12))) if idx.any() else 0.0
                feats.append(de)
        patch_embeddings.append(feats)

    # Mean-pool patches → 128-d embedding via repetition/truncation
    mean_patch = np.mean(patch_embeddings, axis=0)
    # Normalize
    norm = np.linalg.norm(mean_patch)
    mean_patch = mean_patch / (norm + 1e-9)
    # Expand/trim to 128 dims
    emb = np.resize(mean_patch, 128)
    return emb, n_patches


def _classify_from_embedding(emb: np.ndarray, task: str,
                              user_feats: Dict[str, List[np.ndarray]]) -> dict:
    labels = _TASK_LABELS.get(task, ["unknown"])
    if user_feats and any(user_feats.values()):
        # k-NN from user fine-tune samples
        sims = {}
        for label in labels:
            if label in user_feats and user_feats[label]:
                proto = np.mean(user_feats[label], axis=0)
                norm_proto = proto / (np.linalg.norm(proto) + 1e-9)
                sims[label] = float(np.dot(emb, norm_proto))
            else:
                sims[label] = 0.0
        total = sum(max(0, v) for v in sims.values()) + 1e-9
        probs = {k: max(0, v) / total for k, v in sims.items()}
    else:
        # Feature heuristics by task
        alpha_proxy = float(abs(emb[8])) + 1e-9
        beta_proxy  = float(abs(emb[12])) + 1e-9
        theta_proxy = float(abs(emb[4]))  + 1e-9
        if task == "emotion":
            v = float(np.tanh((alpha_proxy / beta_proxy - 0.7) * 2))
            probs = {
                "happy":   max(0.0, 0.25 + 0.15 * v),
                "sad":     max(0.0, 0.15 - 0.10 * v),
                "angry":   0.10,
                "fear":    0.10,
                "surprise": 0.10,
                "neutral": max(0.0, 0.30 - 0.05 * abs(v)),
            }
        elif task == "sleep":
            probs = {
                "wake": 0.30, "n1": 0.20, "n2": 0.25, "n3": 0.15, "rem": 0.10
            }
        elif task == "workload":
            load = float(np.clip(beta_proxy / (alpha_proxy + 1e-9) - 0.5, 0, 1))
            probs = {
                "low": max(0, 0.5 - load),
                "medium": float(np.clip(1 - abs(load - 0.5) * 2, 0.1, 0.5)),
                "high": max(0, load - 0.3),
            }
        else:
            probs = {l: 1.0 / len(labels) for l in labels}

        total = sum(probs.values()) + 1e-9
        probs = {k: v / total for k, v in probs.items()}

    best = max(probs, key=lambda k: probs[k])
    return {"best": best, "probs": probs}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/predict", response_model=FEMBAResult)
async def femba_predict(req: FEMBAInput):
    """Predict brain state using FEMBA foundation model (or approximation)."""
    if req.task not in _TASK_LABELS:
        from fastapi import HTTPException
        raise HTTPException(400, f"task must be one of {list(_TASK_LABELS)}")

    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    if _femba_model is not None:
        try:
            pred = _femba_model.predict(signals, req.fs, req.task)
            emb = pred.get("embedding", np.zeros(128))
            probs = pred.get("probabilities", {})
            best  = pred.get("prediction", "unknown")
            n_patches = pred.get("patch_count", 1)
            model_used = "femba_loaded"
        except Exception:
            emb, n_patches = _patch_embed(signals, req.fs)
            cls = _classify_from_embedding(emb, req.task, _user_finetune[req.user_id])
            best, probs = cls["best"], cls["probs"]
            model_used = "femba_approximation"
    else:
        emb, n_patches = _patch_embed(signals, req.fs)
        cls = _classify_from_embedding(emb, req.task, _user_finetune[req.user_id])
        best, probs = cls["best"], cls["probs"]
        model_used = "femba_approximation"

    confidence = float(probs.get(best, 0.5))
    result = FEMBAResult(
        user_id=req.user_id,
        task=req.task,
        prediction=best,
        probabilities=probs,
        embedding=emb.tolist() if hasattr(emb, "tolist") else list(emb),
        patch_count=n_patches,
        model_used=model_used,
        confidence=confidence,
        processed_at=time.time(),
    )
    _history[req.user_id].append(result.dict())
    return result


@router.post("/finetune")
async def femba_finetune(req: FEMBAFineTuneInput):
    """Add a labeled sample to user's FEMBA fine-tune store (few-shot adaptation)."""
    labels = _TASK_LABELS.get(req.task, [])
    if req.label not in labels:
        from fastapi import HTTPException
        raise HTTPException(400, f"For task '{req.task}', label must be one of {labels}")

    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    emb, _ = _patch_embed(signals, req.fs)
    _user_finetune[req.user_id][req.label].append(emb)
    n_total = sum(len(v) for v in _user_finetune[req.user_id].values())
    return {
        "user_id": req.user_id,
        "task": req.task,
        "label": req.label,
        "n_finetune_samples": n_total,
        "status": "added",
    }


@router.get("/status")
async def femba_status():
    """Return FEMBA model status."""
    return {
        "model_loaded": _femba_model is not None,
        "model_type": "femba_loaded" if _femba_model is not None else "femba_approximation",
        "embedding_dim": 128,
        "supported_tasks": list(_TASK_LABELS),
        "parameters_approx": "7.8M (full model) / ~500 (approximation)",
    }


@router.post("/reset/{user_id}")
async def femba_reset(user_id: str):
    """Clear user fine-tune samples and history."""
    _user_finetune.pop(user_id, None)
    _history[user_id].clear()
    return {"user_id": user_id, "status": "reset"}

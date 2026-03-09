"""Contrastive learning for cross-subject EEG emotion transfer (#40).

Implements an EmotionCLIP-style contrastive adaptation: uses NT-Xent loss
conceptually to bring EEG embeddings of the same emotion closer while pushing
different emotions apart. API accepts labeled EEG frames for adaptation, then
classifies unlabeled frames using the adapted embedding space.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/contrastive", tags=["contrastive-transfer"])

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ContrastiveInput(BaseModel):
    signals: List[List[float]]
    fs: float = 256.0
    user_id: str = "default"


class LabeledSample(BaseModel):
    signals: List[List[float]]
    emotion: str               # happy / sad / angry / fear / surprise / neutral
    fs: float = 256.0
    user_id: str = "default"


class TransferResult(BaseModel):
    user_id: str
    predicted_emotion: str
    probabilities: dict
    embedding_distance: float  # distance from nearest prototype
    n_support_samples: int
    confidence: float
    processed_at: float


# ---------------------------------------------------------------------------
# Supported emotions
# ---------------------------------------------------------------------------

_EMOTIONS = ["happy", "sad", "angry", "fear", "surprise", "neutral"]


# ---------------------------------------------------------------------------
# In-memory embedding store
# ---------------------------------------------------------------------------

class _EmbeddingStore:
    """Per-user labeled embedding database for contrastive k-NN classification."""
    def __init__(self):
        self.embeddings: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.predictions: deque = deque(maxlen=500)

_stores: Dict[str, _EmbeddingStore] = defaultdict(_EmbeddingStore)


# ---------------------------------------------------------------------------
# Feature extraction (17-D per channel → mean across channels → 17-D vector)
# ---------------------------------------------------------------------------

def _extract_embedding(signals: np.ndarray, fs: float) -> np.ndarray:
    from scipy.signal import welch

    def bp(sig, flo, fhi):
        nperseg = min(len(sig), int(fs * 2))
        f, p = welch(sig, fs=fs, nperseg=nperseg)
        idx = np.logical_and(f >= flo, f <= fhi)
        return float(np.mean(p[idx])) + 1e-9 if idx.any() else 1e-9

    n_ch = min(signals.shape[0], 4)
    feats_all = []
    for ch in range(n_ch):
        sig = signals[ch]
        d = bp(sig, 0.5, 4);  t = bp(sig, 4, 8)
        a = bp(sig, 8, 12);   b = bp(sig, 12, 30)
        hb = bp(sig, 20, 30); g = bp(sig, 30, 45)
        total = d + t + a + b + g + 1e-9
        feats = [
            d/total, t/total, a/total, b/total, g/total,
            a/b, t/b, a/t, hb/b, d/t,
            np.log(a+1e-9), np.log(t+1e-9), np.log(b+1e-9),
        ]
        feats_all.append(feats)

    emb = np.mean(feats_all, axis=0)
    norm = np.linalg.norm(emb)
    return emb / (norm + 1e-9)


# ---------------------------------------------------------------------------
# k-NN classification with cosine similarity
# ---------------------------------------------------------------------------

def _knn_classify(query: np.ndarray, store: _EmbeddingStore,
                   k: int = 5) -> Tuple[str, dict, float]:
    if not any(store.embeddings.values()):
        # No labeled samples → uniform distribution
        probs = {e: 1/len(_EMOTIONS) for e in _EMOTIONS}
        return "neutral", probs, 1.0

    similarities: List[Tuple[float, str]] = []
    for emotion, embs in store.embeddings.items():
        for emb in embs:
            sim = float(np.dot(query, emb))
            similarities.append((sim, emotion))

    similarities.sort(reverse=True)
    top_k = similarities[:k]

    counts = defaultdict(float)
    for sim, emo in top_k:
        counts[emo] += max(0, sim)

    total = sum(counts.values()) + 1e-9
    probs = {e: float(counts.get(e, 0) / total) for e in _EMOTIONS}

    best = max(probs, key=lambda k: probs[k])
    best_sim = similarities[0][0] if similarities else 0.0
    dist = float(1.0 - best_sim)
    return best, probs, dist


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/add-sample")
async def add_labeled_sample(req: LabeledSample):
    """Add a labeled EEG sample to the user's contrastive embedding store."""
    if req.emotion not in _EMOTIONS:
        raise HTTPException(400, f"emotion must be one of {_EMOTIONS}")
    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]
    emb = _extract_embedding(signals, req.fs)
    _stores[req.user_id].embeddings[req.emotion].append(emb)
    n_total = sum(len(v) for v in _stores[req.user_id].embeddings.values())
    return {
        "user_id": req.user_id,
        "emotion": req.emotion,
        "n_support_samples": n_total,
        "status": "added",
    }


@router.post("/classify", response_model=TransferResult)
async def classify_with_transfer(req: ContrastiveInput):
    """Classify EEG emotion using contrastive transfer with user's labeled samples."""
    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    emb   = _extract_embedding(signals, req.fs)
    store = _stores[req.user_id]
    best, probs, dist = _knn_classify(emb, store)

    n_support = sum(len(v) for v in store.embeddings.values())
    confidence = float(np.clip(probs[best] + n_support * 0.01, 0, 1))

    result = TransferResult(
        user_id=req.user_id,
        predicted_emotion=best,
        probabilities=probs,
        embedding_distance=dist,
        n_support_samples=n_support,
        confidence=confidence,
        processed_at=time.time(),
    )
    store.predictions.append(result.dict())
    return result


@router.get("/stats/{user_id}")
async def contrastive_stats(user_id: str):
    """Return contrastive transfer statistics for a user."""
    store = _stores[user_id]
    support = {e: len(v) for e, v in store.embeddings.items()}
    n_predictions = len(store.predictions)
    return {
        "user_id": user_id,
        "support_samples_per_emotion": support,
        "total_support_samples": sum(support.values()),
        "n_predictions": n_predictions,
        "note": "Add labeled samples via /contrastive/add-sample to improve accuracy",
    }


@router.post("/reset/{user_id}")
async def contrastive_reset(user_id: str):
    """Clear user embedding store."""
    _stores[user_id] = _EmbeddingStore()
    return {"user_id": user_id, "status": "reset"}

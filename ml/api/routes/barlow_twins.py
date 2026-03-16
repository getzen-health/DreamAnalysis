"""Barlow Twins EEG representation learning API (#66).

Exposes endpoints for:
  - Extracting learned EEG embeddings from a pre-trained Barlow Twins encoder
  - Checking encoder status (loaded / not loaded)
  - Nearest-neighbour emotion classification using embeddings (no fine-tuning needed)

The encoder is loaded lazily on first request from:
  models/saved/barlow_twins_eeg.pt      (full checkpoint)
  models/saved/barlow_twins_encoder.pt  (encoder-only checkpoint — preferred)

If no checkpoint exists the endpoints still work but return a feature-based
fallback embedding (17-dim band-power features zero-padded to 128).

Routes
──────
  GET  /barlow-twins/status
  POST /barlow-twins/embed
  POST /barlow-twins/classify
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

router = APIRouter(prefix="/barlow-twins", tags=["barlow-twins"])

# ── Model paths ───────────────────────────────────────────────────────────────

_SAVED_DIR = Path("models/saved")
_FULL_CKPT = _SAVED_DIR / "barlow_twins_eeg.pt"
_ENC_CKPT  = _SAVED_DIR / "barlow_twins_encoder.pt"

# ── Lazy-loaded encoder singleton ─────────────────────────────────────────────

_encoder = None         # EEGEncoder | None
_encoder_dim: int = 128
_encoder_loaded: bool = False
_encoder_error: str = ""


def _load_encoder() -> None:
    """Load the Barlow Twins encoder on first request."""
    global _encoder, _encoder_loaded, _encoder_error, _encoder_dim

    if _encoder_loaded:
        return  # already attempted (success or failure)

    _encoder_loaded = True   # mark as attempted regardless of outcome

    # Try encoder-only checkpoint first (smaller, preferred)
    for ckpt_path in (_ENC_CKPT, _FULL_CKPT):
        if not ckpt_path.exists():
            continue
        try:
            from models.barlow_twins_eeg import BarlowTwinsEEG
            _encoder = BarlowTwinsEEG.load_encoder_only(ckpt_path)
            _encoder_dim = _encoder.enc_dim
            log.info("Barlow Twins encoder loaded ← %s  (dim=%d)", ckpt_path, _encoder_dim)
            _encoder_error = ""
            return
        except Exception as exc:
            _encoder_error = str(exc)
            log.warning("Barlow Twins encoder load failed (%s): %s", ckpt_path, exc)

    _encoder_error = (
        "No checkpoint found. Train with: "
        "python -m training.train_barlow_twins --use-synthetic"
    )
    log.info("Barlow Twins encoder not available — using fallback embeddings. %s", _encoder_error)


# ── Fallback: band-power feature embedding ────────────────────────────────────

def _fallback_embedding(signals: np.ndarray, fs: float) -> np.ndarray:
    """Return a 128-dim embedding derived from band-power features.

    Used when no Barlow Twins checkpoint exists.  The 17-dim feature vector
    (delta/theta/alpha/beta/gamma powers + ratios) is zero-padded to 128 dims
    and L2-normalised so it occupies the same embedding space geometry as the
    trained encoder (roughly).

    Args:
        signals: (n_channels, n_samples) EEG array
        fs:      sampling rate in Hz

    Returns:
        (128,) float32 embedding.
    """
    try:
        from processing.eeg_processor import extract_features, preprocess
        sig = signals[0] if signals.ndim == 2 else signals
        proc = preprocess(sig, fs)
        feats = extract_features(proc, fs)
        vec = np.array(list(feats.values()), dtype=np.float32)
    except Exception:
        vec = np.zeros(17, dtype=np.float32)

    # Zero-pad to 128 dims
    emb = np.zeros(128, dtype=np.float32)
    n = min(len(vec), 128)
    emb[:n] = vec[:n]

    # L2-normalise
    norm = np.linalg.norm(emb)
    if norm > 1e-8:
        emb /= norm
    return emb


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class EmbedRequest(BaseModel):
    signals: List[List[float]] = Field(
        ...,
        description="EEG signals: list of channels, each a list of samples. "
                    "Shape (n_channels, n_samples). For Muse 2: 4 × 1024.",
    )
    fs: float = Field(default=256.0, description="Sampling frequency in Hz")
    user_id: str = Field(default="default", description="User identifier")


class EmbedResponse(BaseModel):
    embedding: List[float]
    embedding_dim: int
    model_type: str          # "barlow_twins" | "fallback_band_power"
    user_id: str
    processed_at: float


class ClassifyRequest(BaseModel):
    signals: List[List[float]] = Field(
        ...,
        description="EEG signals (channels × samples)",
    )
    fs: float = Field(default=256.0, description="Sampling frequency in Hz")
    user_id: str = Field(default="default")
    k: int = Field(default=5, ge=1, le=50, description="k for k-NN classification")


class ClassifyResponse(BaseModel):
    predicted_emotion: str
    probabilities: Dict[str, float]
    embedding: List[float]
    embedding_dim: int
    model_type: str
    n_support_samples: int
    user_id: str
    processed_at: float


# ── Per-user k-NN support store ────────────────────────────────────────────────

from collections import defaultdict, deque

_EMOTIONS = ["happy", "sad", "angry", "fear", "surprise", "neutral"]

# Per-user labeled embeddings for zero-shot / few-shot classification
_user_embeddings: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))


# ── Shared embedding extraction helper ────────────────────────────────────────

def _extract_embedding(signals_raw: list, fs: float) -> tuple:
    """Convert raw signals list to a (128,) numpy embedding.

    Returns:
        (embedding, model_type_str) where model_type_str is "barlow_twins"
        or "fallback_band_power".
    """
    _load_encoder()

    arr = np.array(signals_raw, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]     # (1, n_samples)

    if _encoder is not None:
        try:
            emb = _encoder_extract(arr, fs)
            return emb, "barlow_twins"
        except Exception as exc:
            log.warning("Barlow Twins encoder inference failed: %s  — using fallback", exc)

    return _fallback_embedding(arr, fs), "fallback_band_power"


def _encoder_extract(signals: np.ndarray, fs: float) -> np.ndarray:
    """Run the loaded EEGEncoder on the signals array."""
    import torch
    n_ch, n_t = signals.shape
    n_samples = _encoder.n_samples

    # Pad / trim
    if n_t < n_samples:
        signals = np.pad(signals, ((0, 0), (0, n_samples - n_t)), mode="edge")
    elif n_t > n_samples:
        signals = signals[:, :n_samples]

    # Per-channel z-score
    mu = signals.mean(axis=-1, keepdims=True)
    sd = signals.std(axis=-1, keepdims=True) + 1e-7
    signals = (signals - mu) / sd

    # Match channel count expected by encoder
    enc_ch = _encoder.n_channels
    if n_ch < enc_ch:
        pad = np.zeros((enc_ch - n_ch, signals.shape[1]), dtype=np.float32)
        signals = np.vstack([signals, pad])
    elif n_ch > enc_ch:
        signals = signals[:enc_ch]

    x = torch.from_numpy(signals.astype(np.float32)).unsqueeze(0)   # (1, ch, t)
    _encoder.eval()
    with torch.no_grad():
        emb = _encoder(x)
    return emb.squeeze(0).cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/status")
async def barlow_twins_status():
    """Return encoder loading status and checkpoint availability."""
    _load_encoder()
    return {
        "encoder_loaded": _encoder is not None,
        "embedding_dim": _encoder_dim if _encoder is not None else 128,
        "checkpoint_enc_exists":  _ENC_CKPT.exists(),
        "checkpoint_full_exists": _FULL_CKPT.exists(),
        "model_type": "barlow_twins" if _encoder is not None else "fallback_band_power",
        "error": _encoder_error if _encoder is None else None,
        "note": (
            "Train the encoder with: python -m training.train_barlow_twins --use-synthetic"
            if _encoder is None else
            "Encoder ready. Use POST /barlow-twins/embed to extract representations."
        ),
    }


@router.post("/embed", response_model=EmbedResponse)
async def extract_embedding_endpoint(req: EmbedRequest):
    """Extract a Barlow Twins EEG embedding.

    Encodes raw EEG signals into a 128-dimensional representation learned by
    the self-supervised Barlow Twins pre-training objective.  The embedding can
    be used for:
      - Transfer learning (fine-tune on a small labeled dataset)
      - Nearest-neighbour classification without any fine-tuning
      - Similarity search across sessions

    If no Barlow Twins checkpoint exists a fallback band-power embedding is
    returned instead.  The embedding has the same 128-dim shape in both cases.
    """
    if not req.signals:
        raise HTTPException(status_code=400, detail="signals must not be empty")

    emb, model_type = _extract_embedding(req.signals, req.fs)

    return EmbedResponse(
        embedding=emb.tolist(),
        embedding_dim=int(emb.shape[0]),
        model_type=model_type,
        user_id=req.user_id,
        processed_at=time.time(),
    )


@router.post("/classify", response_model=ClassifyResponse)
async def classify_with_embedding(req: ClassifyRequest):
    """Classify EEG emotion using Barlow Twins embeddings + k-NN.

    Uses the learned embedding space to perform nearest-neighbour classification.
    Add labeled reference samples via POST /barlow-twins/add-sample to improve
    accuracy beyond the default uniform-prior baseline.

    Returns:
        predicted_emotion, probabilities, embedding, and classification metadata.
    """
    if not req.signals:
        raise HTTPException(status_code=400, detail="signals must not be empty")

    emb, model_type = _extract_embedding(req.signals, req.fs)
    user_store = _user_embeddings[req.user_id]

    # k-NN over labeled embeddings for this user
    n_support = sum(len(v) for v in user_store.values())

    if n_support == 0:
        # No labeled samples — return uniform distribution
        probs = {e: round(1.0 / len(_EMOTIONS), 4) for e in _EMOTIONS}
        best_emotion = "neutral"
    else:
        sims: list = []
        for emotion, embs_list in user_store.items():
            for ref_emb in embs_list:
                ref = np.array(ref_emb, dtype=np.float32)
                sim = float(np.dot(emb, ref) / (np.linalg.norm(emb) * np.linalg.norm(ref) + 1e-9))
                sims.append((sim, emotion))

        sims.sort(reverse=True)
        top_k = sims[: req.k]

        counts: Dict[str, float] = defaultdict(float)
        for sim, emo in top_k:
            counts[emo] += max(0.0, sim)

        total_w = sum(counts.values()) + 1e-9
        probs = {e: round(float(counts.get(e, 0.0) / total_w), 4) for e in _EMOTIONS}
        best_emotion = max(probs, key=lambda k: probs[k])

    return ClassifyResponse(
        predicted_emotion=best_emotion,
        probabilities=probs,
        embedding=emb.tolist(),
        embedding_dim=int(emb.shape[0]),
        model_type=model_type,
        n_support_samples=n_support,
        user_id=req.user_id,
        processed_at=time.time(),
    )


class AddSampleRequest(BaseModel):
    signals: List[List[float]] = Field(..., description="EEG signals (channels × samples)")
    emotion: str = Field(..., description="Emotion label: happy/sad/angry/fear/surprise/neutral")
    fs: float = Field(default=256.0)
    user_id: str = Field(default="default")


@router.post("/add-sample")
async def add_labeled_sample(req: AddSampleRequest):
    """Add a labeled EEG sample to the user's k-NN reference store.

    Embeddings are extracted from raw EEG and stored per-user per-emotion.
    More samples → better k-NN classification in POST /barlow-twins/classify.
    """
    if req.emotion not in _EMOTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"emotion must be one of {_EMOTIONS}",
        )

    if not req.signals:
        raise HTTPException(status_code=400, detail="signals must not be empty")

    emb, model_type = _extract_embedding(req.signals, req.fs)
    _user_embeddings[req.user_id][req.emotion].append(emb.tolist())

    n_total = sum(len(v) for v in _user_embeddings[req.user_id].values())
    return {
        "user_id":          req.user_id,
        "emotion":          req.emotion,
        "n_samples_added":  1,
        "total_support":    n_total,
        "model_type":       model_type,
        "status":           "added",
    }


@router.get("/user-stats/{user_id}")
async def user_embedding_stats(user_id: str):
    """Return k-NN support sample counts per emotion for a user."""
    store = _user_embeddings[user_id]
    support = {e: len(store.get(e, [])) for e in _EMOTIONS}
    return {
        "user_id":                  user_id,
        "support_per_emotion":      support,
        "total_support":            sum(support.values()),
        "encoder_loaded":           _encoder is not None,
        "model_type":               "barlow_twins" if _encoder is not None else "fallback_band_power",
    }


@router.post("/reset/{user_id}")
async def reset_user_store(user_id: str):
    """Clear the k-NN sample store for a user."""
    _user_embeddings[user_id] = defaultdict(list)
    return {"user_id": user_id, "status": "reset"}

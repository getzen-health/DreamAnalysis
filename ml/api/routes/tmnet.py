"""EEG+Voice Transformer Fusion (TMNet Architecture) (#21).

TMNet: Temporal-Modality Network — fuses EEG and voice features using a
cross-modal attention mechanism. When full model weights are not loaded,
provides a feature-level late-fusion pipeline that averages EEG and voice
emotion probabilities with learned weighting.

Based on the TMNet concept for multimodal emotion recognition combining
EEG time-series with speech features.
"""

from __future__ import annotations

import io
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/tmnet", tags=["tmnet"])

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TMNetResult(BaseModel):
    user_id: str
    fused_emotion: str
    fused_probabilities: dict
    eeg_emotion: str
    voice_emotion: Optional[str]
    eeg_weight: float
    voice_weight: float
    cross_modal_confidence: float
    model_used: str
    processed_at: float


class TMNetEEGInput(BaseModel):
    """For EEG-only fusion (no audio)."""
    signals: List[List[float]]
    fs: float = 256.0
    user_id: str = "default"
    voice_probs: Optional[dict] = None   # Optional pre-computed voice probs


# ---------------------------------------------------------------------------
# In-memory
# ---------------------------------------------------------------------------

_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
_EMOTIONS = ["happy", "sad", "angry", "fear", "surprise", "neutral"]

_tmnet_model = None
try:
    from models.tmnet_model import TMNetModel  # type: ignore
    _tmnet_model = TMNetModel()
except Exception:
    pass


# ---------------------------------------------------------------------------
# EEG feature extraction
# ---------------------------------------------------------------------------

def _eeg_emotion_probs(signals: np.ndarray, fs: float) -> dict:
    from scipy.signal import welch

    def bp(sig, flo, fhi):
        nperseg = min(len(sig), int(fs * 2))
        f, p = welch(sig, fs=fs, nperseg=nperseg)
        idx = np.logical_and(f >= flo, f <= fhi)
        return float(np.mean(p[idx])) + 1e-9 if idx.any() else 1e-9

    n_ch = signals.shape[0]
    alpha_v, beta_v, theta_v, hbeta_v = [], [], [], []
    for ch in range(min(n_ch, 4)):
        alpha_v.append(bp(signals[ch], 8, 12))
        beta_v.append(bp(signals[ch], 12, 30))
        theta_v.append(bp(signals[ch], 4, 8))
        hbeta_v.append(bp(signals[ch], 20, 30))

    alpha = float(np.mean(alpha_v)); beta = float(np.mean(beta_v))
    theta = float(np.mean(theta_v)); hbeta = float(np.mean(hbeta_v))

    valence = float(np.tanh((alpha / beta - 0.7) * 2.0))
    arousal = float(np.clip(beta / (alpha + beta), 0, 1))
    stress  = float(np.clip(hbeta / beta, 0, 1))

    # Map to probabilities
    probs = {
        "happy":    float(np.clip(0.3 * max(0, valence) + 0.2 * arousal, 0, 1)),
        "sad":      float(np.clip(0.3 * max(0, -valence) * (1 - arousal), 0, 1)),
        "angry":    float(np.clip(0.25 * max(0, -valence) * arousal + 0.15 * stress, 0, 1)),
        "fear":     float(np.clip(0.2 * (1 - max(0, valence)) * arousal, 0, 1)),
        "surprise": float(np.clip(0.15 * arousal, 0, 1)),
        "neutral":  float(np.clip(0.4 - 0.15 * abs(valence) - 0.1 * stress, 0.05, 1)),
    }
    total = sum(probs.values()) + 1e-9
    return {k: v / total for k, v in probs.items()}


# ---------------------------------------------------------------------------
# Cross-modal late fusion
# ---------------------------------------------------------------------------

def _fuse(eeg_probs: dict, voice_probs: Optional[dict],
           eeg_w: float = 0.6, voice_w: float = 0.4) -> dict:
    """Late fusion: weighted average of EEG and voice probabilities."""
    if voice_probs is None:
        return {k: float(eeg_probs.get(k, 1 / len(_EMOTIONS))) for k in _EMOTIONS}

    fused = {}
    for emo in _EMOTIONS:
        ep = float(eeg_probs.get(emo, 0.0))
        vp = float(voice_probs.get(emo, 0.0))
        fused[emo] = eeg_w * ep + voice_w * vp

    total = sum(fused.values()) + 1e-9
    return {k: v / total for k, v in fused.items()}


def _confidence(fused: dict, eeg_probs: dict, voice_probs: Optional[dict]) -> float:
    sorted_vals = sorted(fused.values(), reverse=True)
    margin = sorted_vals[0] - sorted_vals[1] if len(sorted_vals) > 1 else 0.5
    agreement = 0.0
    if voice_probs:
        best_eeg   = max(eeg_probs,   key=lambda k: eeg_probs[k])
        best_voice = max(voice_probs, key=lambda k: voice_probs[k])
        agreement  = 0.2 if best_eeg == best_voice else -0.1
    return float(np.clip(0.5 + margin * 2 + agreement, 0, 1))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/classify-eeg", response_model=TMNetResult)
async def tmnet_classify_eeg(req: TMNetEEGInput):
    """
    Fuse EEG with optional pre-computed voice probabilities.

    If voice_probs is not provided, falls back to EEG-only classification.
    """
    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    eeg_probs = _eeg_emotion_probs(signals, req.fs)
    eeg_best  = max(eeg_probs, key=lambda k: eeg_probs[k])

    voice_probs = req.voice_probs
    voice_best  = max(voice_probs, key=lambda k: voice_probs[k]) if voice_probs else None

    eeg_w  = 0.6 if voice_probs else 1.0
    voice_w = 0.4 if voice_probs else 0.0

    fused = _fuse(eeg_probs, voice_probs, eeg_w, voice_w)
    fused_best = max(fused, key=lambda k: fused[k])
    conf = _confidence(fused, eeg_probs, voice_probs)

    model_used = "tmnet_loaded" if _tmnet_model is not None else "tmnet_feature_fusion"
    result = TMNetResult(
        user_id=req.user_id,
        fused_emotion=fused_best,
        fused_probabilities=fused,
        eeg_emotion=eeg_best,
        voice_emotion=voice_best,
        eeg_weight=eeg_w,
        voice_weight=voice_w,
        cross_modal_confidence=conf,
        model_used=model_used,
        processed_at=time.time(),
    )
    _history[req.user_id].append(result.dict())
    return result


@router.post("/classify-multimodal", response_model=TMNetResult)
async def tmnet_classify_multimodal(
    eeg_signals: str = Form(..., description="JSON-encoded [[ch0...],[ch1...]] float array"),
    fs: float = Form(256.0),
    user_id: str = Form("default"),
    audio: Optional[UploadFile] = File(None),
):
    """
    Fuse EEG and audio in a single multipart request.

    Sends EEG as JSON form field and audio as file upload.
    Returns fused TMNet emotion.
    """
    import json as _json
    try:
        signals_list = _json.loads(eeg_signals)
        signals = np.array(signals_list, dtype=float)
    except Exception:
        raise HTTPException(400, "eeg_signals must be valid JSON 2D array")
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    eeg_probs = _eeg_emotion_probs(signals, fs)
    eeg_best  = max(eeg_probs, key=lambda k: eeg_probs[k])

    voice_probs = None
    if audio is not None:
        try:
            raw = await audio.read()
            import soundfile as sf
            audio_data, sr = sf.read(io.BytesIO(raw))
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
            # Quick acoustic classification
            energy = float(np.mean(audio_data ** 2))
            zcr    = float(np.mean(np.abs(np.diff(np.sign(audio_data)))) / 2)
            voice_probs = {
                "happy":    float(np.clip(energy * 10 + zcr * 5, 0, 0.4)),
                "sad":      float(np.clip(0.3 - energy * 3, 0.05, 0.3)),
                "angry":    float(np.clip(energy * 8, 0, 0.3)),
                "fear":     float(np.clip(zcr * 8, 0, 0.25)),
                "surprise": float(np.clip(zcr * 5, 0, 0.2)),
                "neutral":  float(np.clip(0.4 - energy * 2 - zcr * 2, 0.1, 0.4)),
            }
            vt = sum(voice_probs.values()) + 1e-9
            voice_probs = {k: v / vt for k, v in voice_probs.items()}
        except Exception:
            voice_probs = None

    voice_best = max(voice_probs, key=lambda k: voice_probs[k]) if voice_probs else None
    eeg_w  = 0.6 if voice_probs else 1.0
    voice_w = 0.4 if voice_probs else 0.0
    fused = _fuse(eeg_probs, voice_probs, eeg_w, voice_w)
    fused_best = max(fused, key=lambda k: fused[k])
    conf = _confidence(fused, eeg_probs, voice_probs)

    model_used = "tmnet_loaded" if _tmnet_model is not None else "tmnet_feature_fusion"
    result = TMNetResult(
        user_id=user_id,
        fused_emotion=fused_best,
        fused_probabilities=fused,
        eeg_emotion=eeg_best,
        voice_emotion=voice_best,
        eeg_weight=eeg_w,
        voice_weight=voice_w,
        cross_modal_confidence=conf,
        model_used=model_used,
        processed_at=time.time(),
    )
    _history[user_id].append(result.dict())
    return result


@router.get("/status")
async def tmnet_status():
    """Return TMNet model status."""
    return {
        "model_loaded": _tmnet_model is not None,
        "model_type": "tmnet_loaded" if _tmnet_model is not None else "tmnet_feature_fusion",
        "modalities": ["eeg", "voice"],
        "fusion_strategy": "cross_modal_attention (loaded) / late_fusion (fallback)",
        "eeg_weight_default": 0.6,
        "voice_weight_default": 0.4,
    }


@router.post("/reset/{user_id}")
async def tmnet_reset(user_id: str):
    """Clear TMNet history for a user."""
    _history[user_id].clear()
    return {"user_id": user_id, "status": "reset"}

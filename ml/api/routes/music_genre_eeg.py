"""Music-specific EEG features for genre-aware emotion classification (#136).

Implements beta-band entrainment analysis, rhythmic entrainment index, and
genre-specific frequency templates to improve emotion classification when
the listener's genre/BPM context is known.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/music-eeg", tags=["music-eeg"])

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class MusicEEGInput(BaseModel):
    signals: List[List[float]]       # (n_channels, n_samples)
    fs: float = 256.0
    genre: Optional[str] = None      # rock, pop, classical, jazz, electronic, ambient
    bpm: Optional[float] = None      # beats-per-minute of current track
    user_id: str = "default"


class GenreEmotionResult(BaseModel):
    user_id: str
    genre: Optional[str]
    bpm: Optional[float]
    genre_adjusted_valence: float    # -1..1
    genre_adjusted_arousal: float    # 0..1
    rhythmic_entrainment_index: float  # 0..1 — how locked the EEG is to the beat
    beat_frequency_power: float      # power at BPM/60 Hz
    alpha_suppression: float         # task-relevant alpha ERD
    beta_entrainment: float          # beta locked to rhythmic stimulus
    genre_emotion_label: str
    confidence: float
    processed_at: float


# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

_GENRE_VALENCE_BIAS = {
    "classical": 0.15,
    "jazz": 0.10,
    "ambient": 0.20,
    "pop": 0.05,
    "rock": -0.05,
    "electronic": 0.0,
}

_GENRE_AROUSAL_BIAS = {
    "classical": -0.05,
    "jazz": 0.05,
    "ambient": -0.15,
    "pop": 0.10,
    "rock": 0.20,
    "electronic": 0.15,
}


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _band_power(signal: np.ndarray, fs: float, flo: float, fhi: float) -> float:
    from scipy.signal import welch
    nperseg = min(len(signal), int(fs * 2))
    f, psd = welch(signal, fs=fs, nperseg=nperseg)
    idx = np.logical_and(f >= flo, f <= fhi)
    return float(np.mean(psd[idx])) if idx.any() else 0.0


def _compute_genre_features(signals: np.ndarray, fs: float,
                              bpm: Optional[float]) -> dict:
    n_channels = signals.shape[0]
    alpha_vals, beta_vals, theta_vals = [], [], []
    for ch in range(n_channels):
        alpha_vals.append(_band_power(signals[ch], fs, 8, 12))
        beta_vals.append(_band_power(signals[ch], fs, 12, 30))
        theta_vals.append(_band_power(signals[ch], fs, 4, 8))

    alpha = float(np.mean(alpha_vals)) + 1e-9
    beta  = float(np.mean(beta_vals))  + 1e-9
    theta = float(np.mean(theta_vals)) + 1e-9

    alpha_suppression = float(np.clip(beta / (alpha + beta) - 0.5, 0, 0.5) * 2)
    base_valence  = float(np.tanh((alpha / beta - 0.7) * 2.0))
    base_arousal  = float(np.clip(beta / (alpha + beta), 0, 1))

    # Beat-locked analysis
    beat_freq_power = 0.0
    rhythmic_idx    = 0.0
    beta_entrainment = 0.0
    if bpm is not None and bpm > 0:
        beat_hz = bpm / 60.0
        beat_freq_power = float(np.mean([
            _band_power(signals[ch], fs, beat_hz * 0.9, beat_hz * 1.1)
            for ch in range(n_channels)
        ]))
        total_power = float(np.mean([
            _band_power(signals[ch], fs, 0.5, 45)
            for ch in range(n_channels)
        ])) + 1e-9
        rhythmic_idx = float(np.clip(beat_freq_power / total_power * 20, 0, 1))
        beta_entrainment = float(np.clip(
            _band_power(signals[0], fs, max(1, beat_hz - 2), beat_hz + 2)
            / (beta + 1e-9), 0, 1
        ))

    return {
        "base_valence": base_valence,
        "base_arousal": base_arousal,
        "alpha_suppression": alpha_suppression,
        "beat_freq_power": beat_freq_power,
        "rhythmic_entrainment_index": rhythmic_idx,
        "beta_entrainment": beta_entrainment,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/extract-features", response_model=GenreEmotionResult)
async def extract_genre_eeg_features(req: MusicEEGInput):
    """Extract genre-aware EEG features and return emotion classification."""
    signals = np.array(req.signals, dtype=float)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]

    feats = _compute_genre_features(signals, req.fs, req.bpm)

    # Apply genre bias if genre known
    genre_key = (req.genre or "").lower()
    v_bias = _GENRE_VALENCE_BIAS.get(genre_key, 0.0)
    a_bias = _GENRE_AROUSAL_BIAS.get(genre_key, 0.0)

    adj_valence = float(np.clip(feats["base_valence"] + v_bias, -1, 1))
    adj_arousal = float(np.clip(feats["base_arousal"] + a_bias, 0, 1))

    # Map to label
    if adj_valence > 0.2 and adj_arousal > 0.5:
        label = "excited"
    elif adj_valence > 0.2 and adj_arousal <= 0.5:
        label = "content"
    elif adj_valence <= -0.2 and adj_arousal > 0.5:
        label = "tense"
    elif adj_valence <= -0.2 and adj_arousal <= 0.5:
        label = "sad"
    else:
        label = "neutral"

    entrainment = feats["rhythmic_entrainment_index"]
    confidence = float(np.clip(0.5 + abs(adj_valence) * 0.3 + entrainment * 0.2, 0, 1))

    result = GenreEmotionResult(
        user_id=req.user_id,
        genre=req.genre,
        bpm=req.bpm,
        genre_adjusted_valence=adj_valence,
        genre_adjusted_arousal=adj_arousal,
        rhythmic_entrainment_index=entrainment,
        beat_frequency_power=feats["beat_freq_power"],
        alpha_suppression=feats["alpha_suppression"],
        beta_entrainment=feats["beta_entrainment"],
        genre_emotion_label=label,
        confidence=confidence,
        processed_at=time.time(),
    )
    _history[req.user_id].append(result.dict())
    return result


@router.get("/stats/{user_id}")
async def get_music_eeg_stats(user_id: str = "default"):
    """Return aggregate music-EEG statistics for a user."""
    h = list(_history[user_id])
    if not h:
        return {"user_id": user_id, "n_sessions": 0}
    valences = [r["genre_adjusted_valence"] for r in h]
    arousals = [r["genre_adjusted_arousal"] for r in h]
    entrainments = [r["rhythmic_entrainment_index"] for r in h]
    genres = [r["genre"] for r in h if r["genre"]]
    return {
        "user_id": user_id,
        "n_sessions": len(h),
        "mean_valence": float(np.mean(valences)),
        "mean_arousal": float(np.mean(arousals)),
        "mean_rhythmic_entrainment": float(np.mean(entrainments)),
        "top_genre": max(set(genres), key=genres.count) if genres else None,
    }


@router.post("/reset/{user_id}")
async def reset_music_eeg(user_id: str = "default"):
    """Clear stored music-EEG history for a user."""
    _history[user_id].clear()
    return {"user_id": user_id, "status": "reset"}

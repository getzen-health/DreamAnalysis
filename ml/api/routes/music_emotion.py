"""Music-induced emotion detection API.

Endpoints:
  POST /music-emotion/baseline  -- record neutral listening baseline
  POST /music-emotion/assess    -- assess emotion during music playback
  POST /music-emotion/frisson   -- detect frisson (musical chills) event
  GET  /music-emotion/stats     -- session statistics
  POST /music-emotion/reset     -- clear baseline and history

GitHub issue: #127
"""

import numpy as np
from fastapi import APIRouter

from ._shared import EEGInput, _numpy_safe
from models.music_emotion import MusicEmotionDetector

router = APIRouter(tags=["music-emotion"])

_detector = MusicEmotionDetector()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/music-emotion/baseline")
async def set_music_baseline(data: EEGInput):
    """Record neutral (non-music) baseline for temporal alpha asymmetry.

    Call during 2 minutes of silence or white noise before music playback.
    Baseline normalises subsequent assessments for inter-individual differences
    in resting temporal alpha amplitude.

    Returns confirmation that baseline was stored.
    """
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    _detector.set_baseline(signals, fs=data.fs)
    return _numpy_safe({"status": "ok", "baseline_set": True, "user_id": data.user_id})


@router.post("/music-emotion/assess")
async def assess_music_emotion(data: EEGInput):
    """Assess emotional response to music from EEG.

    Key EEG markers:
    - Temporal alpha asymmetry (TP9/TP10): music-valence (Sammler et al. 2007)
    - Frontal theta burst: frisson / musical chills (Sachs et al. 2016)
    - Alpha/beta ratio: engagement vs passive listening

    Returns emotion quadrant (energetic_positive, calm_positive,
    energetic_negative, calm_negative), valence, arousal, frisson flag,
    temporal_asymmetry, engagement_level, and recommendations.
    """
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    result = _detector.assess(signals, fs=data.fs)
    return _numpy_safe(result)


@router.post("/music-emotion/frisson")
async def detect_frisson(data: EEGInput):
    """Detect frisson (musical chills) from frontal theta burst + alpha drop.

    Frisson correlates with dopaminergic reward activation (Blood 1999,
    Craig 2005). Returns frisson_detected flag, frisson_score (0-1),
    theta_burst, and alpha_drop indicators.
    """
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    result = _detector.detect_frisson(signals, fs=data.fs)
    return _numpy_safe(result)


@router.get("/music-emotion/stats")
async def get_music_emotion_stats():
    """Get session statistics for music emotion assessments.

    Returns n_assessments, mean_valence, mean_arousal, frisson_count,
    dominant_quadrant, and emotion_distribution across the session.
    """
    result = _detector.get_session_stats()
    return _numpy_safe(result)


@router.post("/music-emotion/reset")
async def reset_music_emotion():
    """Clear baseline and session history."""
    _detector.reset()
    return {
        "status": "ok",
        "message": "Music emotion session reset. Record a new baseline before assessing.",
    }

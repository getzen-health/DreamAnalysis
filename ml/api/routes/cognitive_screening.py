"""Voice-based cognitive decline screening and elderly emotional monitoring.

POST /cognitive/screen              — Run cognitive screening from voice audio
POST /cognitive/track-longitudinal  — Add data point for trajectory tracking
GET  /cognitive/trajectory/{user_id} — Get cognitive trajectory over time
POST /elderly/emotion-check         — Age-adapted emotion assessment
POST /elderly/loneliness-risk       — Passive loneliness screening
GET  /elderly/wellbeing-summary/{user_id} — Combined cognitive + emotional summary
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict, Optional

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

from ml.models.cognitive_screening import (
    ElderlyEmotionMonitor,
    VoiceCognitiveScreener,
)

log = logging.getLogger(__name__)
router = APIRouter(tags=["cognitive-screening"])

_screener = VoiceCognitiveScreener()
_elderly_monitor = ElderlyEmotionMonitor()

_DEFAULT_SR = 16000


# ── Request / Response models ─────────────────────────────────────────────────


class CognitiveScreenRequest(BaseModel):
    audio_b64: str = Field(..., description="Base64-encoded audio (mono WAV or PCM)")
    sample_rate: int = Field(_DEFAULT_SR, description="Sample rate in Hz")
    user_id: str = Field(default="default")
    age: Optional[int] = Field(default=None, ge=18, le=120)


class LongitudinalRequest(BaseModel):
    user_id: str = Field(default="default")
    cognitive_risk_score: float = Field(..., ge=0, le=1)
    risk_level: str = Field(default="normal")
    feature_flags: list = Field(default_factory=list)


class ElderlyEmotionRequest(BaseModel):
    audio_b64: str = Field(..., description="Base64-encoded audio")
    sample_rate: int = Field(_DEFAULT_SR)
    user_id: str = Field(default="default")
    age: Optional[int] = Field(default=None, ge=18, le=120)
    emotion_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional voice emotion prediction (valence, arousal, etc.)",
    )


class LonelinessRequest(BaseModel):
    audio_b64: str = Field(..., description="Base64-encoded audio")
    sample_rate: int = Field(_DEFAULT_SR)
    user_id: str = Field(default="default")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _decode_audio(audio_b64: str, sr: int) -> np.ndarray:
    """Decode base64 audio to numpy array."""
    raw = base64.b64decode(audio_b64)

    # Try WAV first
    try:
        import wave

        with wave.open(io.BytesIO(raw), "rb") as wf:
            n_frames = wf.getnframes()
            data = wf.readframes(n_frames)
            dtype = np.int16 if wf.getsampwidth() == 2 else np.float32
            audio = np.frombuffer(data, dtype=dtype).astype(float)
            if dtype == np.int16:
                audio = audio / 32768.0
            return audio
    except Exception:
        pass

    # Fall back to raw PCM (16-bit signed)
    audio = np.frombuffer(raw, dtype=np.int16).astype(float) / 32768.0
    return audio


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("/cognitive/screen")
async def cognitive_screen(req: CognitiveScreenRequest):
    """Run cognitive screening from voice audio.

    Returns risk score, risk level, feature flags, and extracted features.
    IMPORTANT: This is a screening tool only, NOT diagnostic.
    """
    try:
        audio = _decode_audio(req.audio_b64, req.sample_rate)
    except Exception as e:
        return {"error": f"Failed to decode audio: {e}"}

    result = _screener.screen(audio, req.sample_rate, req.age)

    # Auto-track if user_id provided
    if req.user_id != "default":
        _screener.add_longitudinal_point(req.user_id, result)

    return result


@router.post("/cognitive/track-longitudinal")
async def track_longitudinal(req: LongitudinalRequest):
    """Add a data point for cognitive trajectory tracking."""
    _screener.add_longitudinal_point(req.user_id, {
        "cognitive_risk_score": req.cognitive_risk_score,
        "risk_level": req.risk_level,
        "feature_flags": req.feature_flags,
    })
    return {"ok": True, "user_id": req.user_id}


@router.get("/cognitive/trajectory/{user_id}")
async def get_trajectory(user_id: str, last_n: Optional[int] = None):
    """Get cognitive trajectory over time for a user."""
    return _screener.get_trajectory(user_id, last_n)


@router.post("/elderly/emotion-check")
async def elderly_emotion_check(req: ElderlyEmotionRequest):
    """Age-adapted emotion assessment accounting for positivity bias."""
    try:
        audio = _decode_audio(req.audio_b64, req.sample_rate)
    except Exception as e:
        return {"error": f"Failed to decode audio: {e}"}

    features = _screener.extract_cognitive_features(audio, req.sample_rate)
    result = _elderly_monitor.assess(features, req.emotion_result, req.age)
    result["features"] = features
    return result


@router.post("/elderly/loneliness-risk")
async def loneliness_risk(req: LonelinessRequest):
    """Passive loneliness screening from voice prosodic markers."""
    try:
        audio = _decode_audio(req.audio_b64, req.sample_rate)
    except Exception as e:
        return {"error": f"Failed to decode audio: {e}"}

    features = _screener.extract_cognitive_features(audio, req.sample_rate)
    result = _elderly_monitor.assess_loneliness_risk(features)
    result["features"] = features
    return result


@router.get("/elderly/wellbeing-summary/{user_id}")
async def wellbeing_summary(user_id: str):
    """Combined cognitive + emotional wellbeing summary for a user."""
    trajectory = _screener.get_trajectory(user_id)

    latest_risk = 0.0
    latest_level = "unknown"
    if trajectory["trajectory"]:
        latest = trajectory["trajectory"][-1]
        latest_risk = latest["risk_score"]
        latest_level = latest["risk_level"]

    return {
        "user_id": user_id,
        "cognitive": {
            "latest_risk_score": latest_risk,
            "latest_risk_level": latest_level,
            "trend": trajectory["trend"],
            "n_assessments": trajectory["n_assessments"],
        },
        "disclaimer": (
            "This is a screening tool only and does NOT constitute a medical "
            "diagnosis. Consult a healthcare professional for clinical evaluation."
        ),
    }

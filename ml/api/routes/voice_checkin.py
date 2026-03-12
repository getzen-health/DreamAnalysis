"""Voice micro check-in — lightweight emotional pulse logging.

POST /voice-checkin/submit           — analyse audio + auto-log to supplement tracker
GET  /voice-checkin/history/{user_id} — last N check-in records
GET  /voice-checkin/daily-summary/{user_id} — morning/noon/evening mood trajectory
"""
from __future__ import annotations

import base64
import io
import json
import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from models.context_prior import ContextPrior, blend_with_prior  # type: ignore

# Session storage directory (shared with SessionRecorder)
_SESSIONS_DIR = Path(__file__).parent.parent.parent / "sessions"
_SESSIONS_DIR.mkdir(exist_ok=True)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/voice-checkin", tags=["voice-checkin"])

# ── In-memory check-in store  (user_id → list of records) ────────────────────
_CHECKIN_HISTORY: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

VALID_CHECKIN_TYPES = {"morning", "noon", "evening"}

# ── Lazy model handles ────────────────────────────────────────────────────────
_voice_model = None
_context_prior = ContextPrior()


def _get_voice_model():
    global _voice_model
    if _voice_model is None:
        try:
            from models.voice_emotion_model import get_voice_model  # type: ignore
            _voice_model = get_voice_model()
        except ImportError as exc:
            log.warning("VoiceEmotionModel unavailable: %s", exc)
    return _voice_model


def _get_tracker():
    """Return the shared SupplementTracker singleton from the supplement_tracker route."""
    try:
        from .supplement_tracker import get_tracker  # type: ignore
        return get_tracker()
    except Exception as exc:
        log.warning("Shared SupplementTracker unavailable: %s", exc)
        return None


# ── Helper: derive stress/focus proxies from voice biomarkers ─────────────────

def _derive_stress_index(biomarkers: Dict[str, Any]) -> float:
    """Derive 0-1 stress proxy from jitter, shimmer, and HNR."""
    jitter = float(biomarkers.get("jitter_local", 0.0) or 0.0)
    shimmer = float(biomarkers.get("shimmer_local", 0.0) or 0.0)
    hnr_db = float(biomarkers.get("hnr_db", 20.0) or 20.0)
    # Normalise: jitter ~0-0.05, shimmer ~0-0.15, HNR 0-20 dB
    jitter_norm = float(np.clip(jitter / 0.05, 0.0, 1.0))
    shimmer_norm = float(np.clip(shimmer / 0.15, 0.0, 1.0))
    hnr_stress = float(np.clip(1.0 - hnr_db / 20.0, 0.0, 1.0))
    composite = (jitter_norm + shimmer_norm + hnr_stress) / 3.0
    return float(np.clip(composite, 0.0, 1.0))


def _derive_focus_index(biomarkers: Dict[str, Any]) -> float:
    """Derive 0-1 focus proxy from silence ratio and speech rate."""
    silence_ratio = float(biomarkers.get("silence_ratio", 0.0) or 0.0)
    speech_rate = float(biomarkers.get("speech_rate", 0.0) or 0.0)
    # speech_rate ~0-8 syl/s; silence_ratio 0-1
    speech_score = float(np.clip(speech_rate / 4.0, 0.0, 1.0))
    focus = float(np.clip((1.0 - silence_ratio) * speech_score, 0.0, 1.0))
    return focus


def _auto_log_brain_state(
    user_id: str,
    timestamp: float,
    valence: float,
    arousal: float,
    stress_index: float,
    focus_index: float,
    biomarkers: Optional[Dict[str, Any]],
) -> None:
    """Forward check-in measurements to the supplement tracker."""
    tracker = _get_tracker()
    if tracker is None:
        return
    try:
        emotion_data: Dict[str, Any] = {
            "valence": valence,
            "arousal": arousal,
            "stress_index": stress_index,
            "focus_index": focus_index,
            "source": "voice",
        }
        if biomarkers:
            emotion_data["speech_rate"] = float(biomarkers.get("speech_rate", 0.0) or 0.0)
        tracker.log_brain_state(
            user_id=user_id,
            timestamp=timestamp,
            emotion_data=emotion_data,
        )
    except Exception as exc:
        log.warning("Voice check-in auto-log to supplement tracker failed: %s", exc)


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class CheckinRequest(BaseModel):
    audio_b64: str = Field(..., description="Base64-encoded WAV audio (5-30s)")
    user_id: str = Field(..., description="User identifier")
    checkin_type: str = Field(
        "morning",
        description="Time of day: morning | noon | evening",
    )
    text: Optional[str] = Field(None, description="Optional text note for the check-in")
    sample_rate: int = Field(22050, description="Audio sample rate in Hz")
    # Context signals for Bayesian prior adjustment (all optional)
    sleep_quality: Optional[float] = Field(
        None,
        description="Sleep quality score 0-10 (from health tracker or self-report)",
    )
    steps_today: Optional[int] = Field(
        None,
        description="Step count accumulated today (from phone/watch)",
    )
    caffeine_logged: bool = Field(
        False,
        description="True if caffeine was ingested within the last 4 hours",
    )


class CheckinResponse(BaseModel):
    user_id: str
    checkin_type: str
    timestamp: float
    emotion: str
    valence: float
    arousal: float
    confidence: float
    stress_index: float
    focus_index: float
    biomarkers: Optional[Dict[str, Any]] = None
    text: Optional[str] = None
    model_type: str


# ── Submit endpoint ────────────────────────────────────────────────────────────

@router.post("/submit", response_model=CheckinResponse)
def submit_checkin(req: CheckinRequest) -> Dict[str, Any]:
    """Analyse a short voice clip and log the emotional state as a micro check-in."""
    if req.checkin_type not in VALID_CHECKIN_TYPES:
        raise HTTPException(
            422,
            f"Invalid checkin_type '{req.checkin_type}'. Must be one of: "
            f"{sorted(VALID_CHECKIN_TYPES)}",
        )

    # Decode base64 audio
    try:
        wav_bytes = base64.b64decode(req.audio_b64)
    except Exception as exc:
        raise HTTPException(422, f"Invalid base64 audio: {exc}")

    result: Optional[Dict[str, Any]] = None

    # Primary: soundfile → VoiceEmotionModel
    try:
        import soundfile as sf  # type: ignore

        audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        vm = _get_voice_model()
        if vm is not None:
            result = vm.predict_with_biomarkers(audio, sample_rate=int(sr))
    except Exception as exc:
        log.warning("soundfile path failed in voice check-in: %s", exc)

    # Fallback: librosa → VoiceEmotionModel
    if result is None:
        try:
            import librosa  # type: ignore

            audio, _ = librosa.load(io.BytesIO(wav_bytes), sr=req.sample_rate, mono=True)
            if len(audio) < req.sample_rate // 4:
                raise HTTPException(422, "Audio too short — need at least 0.25s")
            vm = _get_voice_model()
            if vm is not None:
                result = vm.predict_with_biomarkers(audio, sample_rate=req.sample_rate)
        except HTTPException:
            raise
        except ImportError:
            log.warning("librosa not available for voice check-in fallback")
        except Exception as exc:
            log.warning("librosa fallback failed in voice check-in: %s", exc)

    if result is None:
        raise HTTPException(
            503,
            "Voice check-in failed — audio processing unavailable or audio too short",
        )

    # Extract fields
    emotion = str(result.get("emotion", "neutral"))
    valence = float(result.get("valence", 0.0))
    arousal = float(result.get("arousal", 0.0))
    confidence = float(result.get("confidence", 0.0))
    model_type = str(result.get("model_type", "unknown"))
    biomarkers: Optional[Dict[str, Any]] = result.get("biomarkers") or None

    # Apply context-aware prior — blend Bayesian prior with ML prediction
    try:
        history_so_far = _CHECKIN_HISTORY.get(req.user_id, [])
        previous_valence: Optional[float] = (
            float(history_so_far[-1]["valence"]) if history_so_far else None
        )
        prior = _context_prior.get_prior(
            hour=datetime.now(tz=timezone.utc).hour,
            sleep_quality=req.sleep_quality,
            steps_today=req.steps_today,
            caffeine_logged=req.caffeine_logged,
            previous_checkin_valence=previous_valence,
        )
        adjusted = blend_with_prior(result, prior, prior_weight=0.20)
        valence = float(adjusted.get("valence", valence))
        arousal = float(adjusted.get("arousal", arousal))
        context_adjustments: List[str] = adjusted.get("adjustments", [])
        if context_adjustments:
            log.debug(
                "Context prior applied for user %s: adjustments=%s",
                req.user_id,
                context_adjustments,
            )
    except Exception as exc:
        log.warning("Context prior failed — using raw prediction: %s", exc)

    # Derive stress / focus from biomarkers when available
    if biomarkers:
        stress_index = _derive_stress_index(biomarkers)
        focus_index = _derive_focus_index(biomarkers)
    else:
        stress_index = 0.0
        focus_index = 0.0

    ts = time.time()

    # Auto-log to supplement tracker
    _auto_log_brain_state(
        user_id=req.user_id,
        timestamp=ts,
        valence=valence,
        arousal=arousal,
        stress_index=stress_index,
        focus_index=focus_index,
        biomarkers=biomarkers,
    )

    # Store in-memory history
    record: Dict[str, Any] = {
        "timestamp": ts,
        "checkin_type": req.checkin_type,
        "emotion": emotion,
        "valence": valence,
        "arousal": arousal,
        "confidence": confidence,
        "stress_index": stress_index,
        "focus_index": focus_index,
        "biomarkers": biomarkers,
        "text": req.text,
        "model_type": model_type,
    }
    _CHECKIN_HISTORY[req.user_id].append(record)

    # Persist as a session JSON so it appears in /sessions list (Fix #8)
    try:
        session_id = str(uuid.uuid4())[:8]
        session_meta: Dict[str, Any] = {
            "session_id": session_id,
            "user_id": req.user_id,
            "session_type": "voice_checkin",
            "start_time": ts,
            "end_time": ts + 10,  # ~10s recording
            "status": "completed",
            "metadata": {"checkin_type": req.checkin_type, "model_type": model_type},
            "summary": {
                "duration_sec": 10,
                "n_frames": 1,
                "n_channels": 0,
                "n_samples": 0,
                "avg_stress": stress_index,
                "avg_focus": focus_index,
                "avg_relaxation": max(0.0, 1.0 - stress_index),
                "avg_valence": valence,
                "avg_arousal": arousal,
                "dominant_emotion": emotion,
                "avg_flow": 0.0,
            },
            "analysis_timeline": [],
        }
        meta_path = _SESSIONS_DIR / f"{session_id}.json"
        with open(meta_path, "w") as f:
            json.dump(session_meta, f, indent=2, default=str)
        log.info("Voice check-in session saved: %s", session_id)
    except Exception as exc:
        log.warning("Failed to persist voice check-in session: %s", exc)

    return {
        "user_id": req.user_id,
        "checkin_type": req.checkin_type,
        "timestamp": ts,
        "emotion": emotion,
        "valence": valence,
        "arousal": arousal,
        "confidence": confidence,
        "stress_index": stress_index,
        "focus_index": focus_index,
        "biomarkers": biomarkers,
        "text": req.text,
        "model_type": model_type,
    }


# ── History endpoint ───────────────────────────────────────────────────────────

@router.get("/history/{user_id}")
def get_checkin_history(
    user_id: str, last_n: int = 30
) -> Dict[str, Any]:
    """Return the last N voice check-in records for a user."""
    history = _CHECKIN_HISTORY.get(user_id, [])
    trimmed = history[-last_n:] if len(history) > last_n else history
    return {
        "user_id": user_id,
        "count": len(trimmed),
        "checkins": trimmed,
    }


# ── Daily summary endpoint ─────────────────────────────────────────────────────

@router.get("/daily-summary/{user_id}")
def get_daily_summary(user_id: str) -> Dict[str, Any]:
    """Return morning/noon/evening mood trajectory for today.

    Groups today's check-ins by checkin_type and returns mean
    valence, arousal, stress_index, focus_index, and dominant emotion
    for each slot that has data.
    """
    # Start of today in UTC (midnight)
    now = time.time()
    today_start = datetime(
        *datetime.now(tz=timezone.utc).timetuple()[:3],
        tzinfo=timezone.utc,
    ).timestamp()

    history = _CHECKIN_HISTORY.get(user_id, [])
    today_records = [r for r in history if r["timestamp"] >= today_start]

    # Group by checkin_type
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in today_records:
        groups[record["checkin_type"]].append(record)

    trajectory: Dict[str, Any] = {}
    for slot in VALID_CHECKIN_TYPES:
        slot_records = groups.get(slot, [])
        if not slot_records:
            trajectory[slot] = None
            continue

        valences = [r["valence"] for r in slot_records]
        arousals = [r["arousal"] for r in slot_records]
        stresses = [r["stress_index"] for r in slot_records]
        focuses = [r["focus_index"] for r in slot_records]

        # Dominant emotion: most frequent
        emotion_counts: Dict[str, int] = defaultdict(int)
        for r in slot_records:
            emotion_counts[r["emotion"]] += 1
        dominant_emotion = max(emotion_counts, key=lambda e: emotion_counts[e])

        trajectory[slot] = {
            "count": len(slot_records),
            "dominant_emotion": dominant_emotion,
            "avg_valence": round(float(np.mean(valences)), 4),
            "avg_arousal": round(float(np.mean(arousals)), 4),
            "avg_stress_index": round(float(np.mean(stresses)), 4),
            "avg_focus_index": round(float(np.mean(focuses)), 4),
            "latest_timestamp": max(r["timestamp"] for r in slot_records),
        }

    return {
        "user_id": user_id,
        "date": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
        "total_checkins_today": len(today_records),
        "trajectory": trajectory,
    }

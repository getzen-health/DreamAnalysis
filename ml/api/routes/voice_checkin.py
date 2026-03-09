"""Voice micro check-in endpoints with supplement-tracker auto-logging.

POST /voice-checkin/submit              -- classify a 10-second voice clip
GET  /voice-checkin/history/{user_id}  -- last N check-ins for a user
GET  /voice-checkin/daily-summary/{user_id} -- today's dominant mood + trend

Every submitted check-in is automatically logged to the SupplementTracker
with source="voice", enabling supplement ↔ mood correlations without EEG.
"""
from __future__ import annotations

import base64
import io
import logging
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
router = APIRouter(prefix="/voice-checkin", tags=["voice-checkin"])

_DEFAULT_SR = 16000

# ── In-memory store: 90 check-ins per user ──────────────────────────────────
_store: Dict[str, Deque[Dict[str, Any]]] = defaultdict(lambda: deque(maxlen=90))

# ── Time slots ───────────────────────────────────────────────────────────────
CHECKIN_SLOTS = [(8, "morning"), (13, "afternoon"), (20, "evening")]
_SLOT_WINDOW_HOURS = 2


def _current_slot() -> str:
    hour = datetime.now().hour
    for slot_hour, name in CHECKIN_SLOTS:
        if abs(hour - slot_hour) <= _SLOT_WINDOW_HOURS:
            return name
    return "manual"


# ── Audio decode ─────────────────────────────────────────────────────────────

def _decode_audio(b64: str, sample_rate: int) -> np.ndarray:
    raw = base64.b64decode(b64)
    try:
        import soundfile as sf
        arr, _ = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        return arr
    except Exception:
        pass
    try:
        import librosa
        arr, _ = librosa.load(io.BytesIO(raw), sr=sample_rate, mono=True)
        return arr
    except Exception as exc:
        raise ValueError(f"Cannot decode audio: {exc}") from exc


# ── Request / response schemas ────────────────────────────────────────────────

class SubmitCheckinRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    audio_b64: str = Field(..., description="Base64-encoded WAV audio (mono, ≥3 s)")
    sample_rate: int = Field(_DEFAULT_SR, description="Sample rate in Hz")
    note: Optional[str] = Field(None, description="Optional free-text note")


class CheckinResult(BaseModel):
    checkin_id: str
    user_id: str
    timestamp: float
    slot: str
    emotion: str
    probabilities: Dict[str, float]
    valence: float
    arousal: float
    stress_index: float
    note: Optional[str] = None


class DailySummary(BaseModel):
    user_id: str
    date: str
    checkin_count: int
    dominant_emotion: Optional[str]
    avg_valence: Optional[float]
    avg_arousal: Optional[float]
    avg_stress: Optional[float]
    trend: str  # "improving" | "declining" | "stable" | "insufficient_data"


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/submit", response_model=CheckinResult)
def submit_checkin(req: SubmitCheckinRequest) -> CheckinResult:
    """Classify a voice clip and log the result to the supplement tracker."""

    # 1. Decode audio
    try:
        audio = _decode_audio(req.audio_b64, req.sample_rate)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # 2. Classify via VoiceEmotionModel (lazy import)
    result: Dict[str, Any] = {}
    try:
        from models.voice_emotion_model import get_voice_model
        result = get_voice_model().predict(audio, req.sample_rate) or {}
    except Exception as exc:
        log.warning("VoiceEmotionModel failed in voice_checkin: %s", exc)

    emotion = str(result.get("emotion", "neutral"))
    probs: Dict[str, float] = {
        k: float(v)
        for k, v in result.get("probabilities", {}).items()
    }
    valence = float(result.get("valence", 0.0))
    arousal = float(result.get("arousal", 0.5))
    stress = float(result.get("stress_index", 0.0))
    focus = float(result.get("focus_index", 0.0))

    now = time.time()
    checkin_id = str(uuid.uuid4())

    entry = {
        "checkin_id": checkin_id,
        "user_id": req.user_id,
        "timestamp": now,
        "slot": _current_slot(),
        "emotion": emotion,
        "probabilities": probs,
        "valence": valence,
        "arousal": arousal,
        "stress_index": stress,
        "note": req.note,
    }
    _store[req.user_id].append(entry)

    # 3. Auto-log to SupplementTracker so supplement correlations use voice data
    try:
        from .supplement_tracker import get_tracker
        get_tracker().log_brain_state(
            user_id=req.user_id,
            timestamp=now,
            emotion_data={
                "valence": valence,
                "arousal": arousal,
                "stress_index": stress,
                "focus_index": focus,
                "source": "voice",
                "alpha_beta_ratio": 0.0,
                "theta_power": 0.0,
                "faa": 0.0,
                "speech_rate": float(result.get("speech_rate", 0.0)),
            },
        )
    except Exception as exc:
        log.warning("supplement_tracker.log_brain_state failed: %s", exc)

    return CheckinResult(**entry)


@router.get("/history/{user_id}", response_model=List[CheckinResult])
def get_history(user_id: str, last_n: int = 30) -> List[CheckinResult]:
    """Return the last N check-ins for a user (newest first)."""
    entries = list(_store[user_id])
    entries.sort(key=lambda e: e["timestamp"], reverse=True)
    return [CheckinResult(**e) for e in entries[:last_n]]


@router.get("/daily-summary/{user_id}", response_model=DailySummary)
def get_daily_summary(user_id: str) -> DailySummary:
    """Summarise today's check-ins: dominant emotion, averages, trend."""
    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    ).timestamp()

    today = [
        e for e in _store[user_id] if e["timestamp"] >= today_start
    ]

    if not today:
        return DailySummary(
            user_id=user_id,
            date=datetime.now().strftime("%Y-%m-%d"),
            checkin_count=0,
            dominant_emotion=None,
            avg_valence=None,
            avg_arousal=None,
            avg_stress=None,
            trend="insufficient_data",
        )

    from collections import Counter
    dominant_emotion = Counter(e["emotion"] for e in today).most_common(1)[0][0]
    avg_valence = float(np.mean([e["valence"] for e in today]))
    avg_arousal = float(np.mean([e["arousal"] for e in today]))
    avg_stress = float(np.mean([e["stress_index"] for e in today]))

    # Trend: compare first half vs second half valence
    trend = "stable"
    if len(today) >= 2:
        mid = len(today) // 2
        early_v = float(np.mean([e["valence"] for e in today[:mid]]))
        late_v = float(np.mean([e["valence"] for e in today[mid:]]))
        if late_v - early_v > 0.1:
            trend = "improving"
        elif early_v - late_v > 0.1:
            trend = "declining"

    return DailySummary(
        user_id=user_id,
        date=datetime.now().strftime("%Y-%m-%d"),
        checkin_count=len(today),
        dominant_emotion=dominant_emotion,
        avg_valence=round(avg_valence, 3),
        avg_arousal=round(avg_arousal, 3),
        avg_stress=round(avg_stress, 3),
        trend=trend,
    )

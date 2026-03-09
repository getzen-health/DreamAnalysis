"""Voice micro-check-in system.

Users record a short 10-second audio clip up to 3 times per day (08:00, 13:00, 20:00).
The audio is classified via VoiceEmotionModel and the result is stored in-memory and
also forwarded to SupplementTracker so brain-state correlations stay up to date.

Endpoints
---------
POST /voice-checkin/submit
    Accept base64 WAV, run VoiceEmotionModel, store, log brain state.

GET  /voice-checkin/history/{user_id}
    Return all check-ins for a user (most recent first, capped at 90).

GET  /voice-checkin/daily-summary/{user_id}
    Aggregate today's check-ins into a single valence/arousal/emotion summary.
"""
from __future__ import annotations

import base64
import io
import logging
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
router = APIRouter(prefix="/voice-checkin", tags=["Voice Check-In"])

# ── In-memory store ───────────────────────────────────────────────────────────
# Per-user deque of check-in dicts, capped at 90 entries (~30 days × 3/day).
_MAX_PER_USER = 90
_store: Dict[str, Deque[Dict[str, Any]]] = defaultdict(lambda: deque(maxlen=_MAX_PER_USER))

# Scheduled check-in windows: (hour, label)
CHECKIN_SLOTS = [(8, "morning"), (13, "afternoon"), (20, "evening")]


# ── Request / Response models ─────────────────────────────────────────────────

class CheckinSubmitRequest(BaseModel):
    user_id: str = Field("default", description="User identifier")
    audio_b64: str = Field(..., description="Base64-encoded WAV (~10 seconds, 16 kHz mono)")
    sample_rate: int = Field(16000, description="Audio sample rate in Hz")
    note: Optional[str] = Field(None, max_length=500, description="Optional user note")


class CheckinResult(BaseModel):
    checkin_id: str
    user_id: str
    timestamp: float
    slot: str                        # "morning" | "afternoon" | "evening" | "manual"
    emotion: str
    probabilities: Dict[str, float]
    valence: float
    arousal: float
    stress_index: float
    note: Optional[str]


class DailySummary(BaseModel):
    user_id: str
    date: str                        # ISO date string YYYY-MM-DD
    checkin_count: int
    dominant_emotion: Optional[str]
    avg_valence: Optional[float]
    avg_arousal: Optional[float]
    avg_stress: Optional[float]
    trend: str                       # "improving" | "declining" | "stable" | "insufficient_data"
    slots_completed: List[str]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _slot_for_hour(hour: int) -> str:
    """Map UTC hour to check-in slot label."""
    for h, label in CHECKIN_SLOTS:
        if abs(hour - h) <= 2:
            return label
    return "manual"


def _decode_audio(audio_b64: str, sample_rate: int) -> tuple[np.ndarray, int]:
    """Decode base64 WAV bytes → (float32 mono array, actual sample rate)."""
    try:
        wav_bytes = base64.b64decode(audio_b64)
    except Exception as exc:
        raise HTTPException(422, f"Invalid base64 audio: {exc}") from exc

    # Try soundfile first (handles WAV/FLAC/OGG)
    try:
        import soundfile as sf
        audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, int(sr)
    except Exception:
        pass

    # Fallback: librosa
    try:
        import librosa
        y, sr = librosa.load(io.BytesIO(wav_bytes), sr=sample_rate, mono=True)
        return y.astype(np.float32), int(sr)
    except Exception as exc:
        raise HTTPException(
            503, f"No audio decoding library available: {exc}"
        ) from exc


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/voice-checkin/submit", response_model=CheckinResult)
async def submit_checkin(req: CheckinSubmitRequest) -> CheckinResult:
    """Classify a 10-second audio clip and store the check-in result."""
    audio, sr = _decode_audio(req.audio_b64, req.sample_rate)

    # Run VoiceEmotionModel
    result: Optional[Dict[str, Any]] = None
    try:
        from models.voice_emotion_model import get_voice_model  # lazy import
        result = get_voice_model().predict(audio, sample_rate=sr, real_time=False)
    except Exception as exc:
        log.warning("VoiceEmotionModel failed in check-in: %s", exc)

    if not result or "emotion" not in result:
        raise HTTPException(503, "Voice emotion model could not process audio")

    now = time.time()
    import datetime
    hour = datetime.datetime.utcfromtimestamp(now).hour
    slot = _slot_for_hour(hour)

    checkin_id = f"{req.user_id}:{int(now * 1000)}"

    entry: Dict[str, Any] = {
        "checkin_id": checkin_id,
        "user_id": req.user_id,
        "timestamp": now,
        "slot": slot,
        "emotion": result.get("emotion", "neutral"),
        "probabilities": result.get("probabilities", {}),
        "valence": float(result.get("valence", 0.0)),
        "arousal": float(result.get("arousal", 0.0)),
        "stress_index": float(result.get("stress_index", 0.0)),
        "note": req.note,
    }
    _store[req.user_id].appendleft(entry)

    # Forward to SupplementTracker for correlation tracking
    try:
        from .supplement_tracker import get_tracker
        get_tracker().log_brain_state(
            user_id=req.user_id,
            timestamp=now,
            emotion_data={
                "valence": entry["valence"],
                "arousal": entry["arousal"],
                "stress_index": entry["stress_index"],
                "focus_index": float(result.get("focus_index", 0.0)),
                "alpha_beta_ratio": 0.0,  # not available from voice
                "theta_power": 0.0,
                "faa": 0.0,
            },
        )
    except Exception as exc:
        log.debug("Supplement tracker log skipped: %s", exc)

    return CheckinResult(**entry)


@router.get("/voice-checkin/history/{user_id}", response_model=List[CheckinResult])
async def get_history(user_id: str, limit: int = 30) -> List[CheckinResult]:
    """Return most-recent check-ins for a user."""
    entries = list(_store.get(user_id, []))
    return [CheckinResult(**e) for e in entries[:min(limit, _MAX_PER_USER)]]


@router.get("/voice-checkin/daily-summary/{user_id}", response_model=DailySummary)
async def get_daily_summary(user_id: str) -> DailySummary:
    """Aggregate today's check-ins into a mood summary."""
    import datetime

    now = time.time()
    today = datetime.datetime.utcfromtimestamp(now).date()
    today_str = today.isoformat()
    day_start = datetime.datetime(today.year, today.month, today.day).timestamp()

    entries = [
        e for e in _store.get(user_id, [])
        if e["timestamp"] >= day_start
    ]

    if not entries:
        return DailySummary(
            user_id=user_id,
            date=today_str,
            checkin_count=0,
            dominant_emotion=None,
            avg_valence=None,
            avg_arousal=None,
            avg_stress=None,
            trend="insufficient_data",
            slots_completed=[],
        )

    valences = [e["valence"] for e in entries]
    arousals = [e["arousal"] for e in entries]
    stresses = [e["stress_index"] for e in entries]
    emotions = [e["emotion"] for e in entries]

    # Dominant emotion by frequency
    from collections import Counter
    dominant = Counter(emotions).most_common(1)[0][0]

    # Trend: compare first-half vs second-half valence
    if len(valences) >= 3:
        mid = len(valences) // 2
        early = float(np.mean(valences[mid:]))   # entries are newest-first
        late = float(np.mean(valences[:mid]))
        delta = late - early
        if delta > 0.1:
            trend = "improving"
        elif delta < -0.1:
            trend = "declining"
        else:
            trend = "stable"
    else:
        trend = "stable"

    slots_completed = list({e["slot"] for e in entries})

    return DailySummary(
        user_id=user_id,
        date=today_str,
        checkin_count=len(entries),
        dominant_emotion=dominant,
        avg_valence=round(float(np.mean(valences)), 3),
        avg_arousal=round(float(np.mean(arousals)), 3),
        avg_stress=round(float(np.mean(stresses)), 3),
        trend=trend,
        slots_completed=slots_completed,
    )

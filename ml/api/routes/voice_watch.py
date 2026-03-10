"""Voice + Apple Watch emotion analysis — upgraded to 6-class + result cache.

POST /voice-watch/analyze    — 6-class emotion from mic audio + watch biometrics
POST /voice-watch/cache      — cache latest voice result for WebSocket fusion
GET  /voice-watch/latest/{user_id} — retrieve cached result (< 5 min TTL)
GET  /voice-watch/status     — model availability
"""
from __future__ import annotations

import base64
import io
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
router = APIRouter(prefix="/voice-watch", tags=["voice-watch"])

_ML_ROOT = Path(__file__).resolve().parent.parent.parent
_LGBM_PATH = _ML_ROOT / "models" / "saved" / "audio_emotion_lgbm.pkl"

# ── In-memory voice result cache (5-minute TTL) ───────────────────────────────
_VOICE_CACHE: Dict[str, Dict] = {}
_VOICE_CACHE_TTL = 300  # seconds

# ── In-memory voice history (persistent within session, keyed by user_id) ─────
# Each entry: {timestamp, emotion, valence, arousal, stress_index, focus_index}
# Used by brain_report.py as the canonical voice data source.
from collections import defaultdict as _defaultdict
_VOICE_HISTORY: Dict[str, List[Dict[str, Any]]] = _defaultdict(list)

# ── Librosa state (lazy) ──────────────────────────────────────────────────────
_librosa_ok = False
_SR = 22050
_N_MFCC = 40
N_FEATS = 92

EMOTIONS_6 = ["happy", "sad", "angry", "fear", "surprise", "neutral"]


def _ensure_librosa() -> bool:
    global _librosa_ok
    if _librosa_ok:
        return True
    try:
        import librosa  # noqa: F401
        _librosa_ok = True
        return True
    except ImportError:
        return False


def _extract_features(y: np.ndarray, sr: int = _SR) -> np.ndarray:
    """92-dim MFCC feature vector — identical to original voice_watch pipeline."""
    import librosa

    if len(y) < sr // 4:
        return np.zeros(N_FEATS, dtype=np.float32)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=_N_MFCC)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    sc  = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    sb  = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    sr2 = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    sf  = librosa.feature.spectral_flatness(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]

    def ms(x: np.ndarray) -> List[float]:
        return [float(x.mean()), float(x.std())]

    feat = np.concatenate([
        mfcc_mean, mfcc_std, ms(sc), ms(sb), ms(sr2), ms(zcr), ms(rms), ms(sf),
    ]).astype(np.float32)
    return np.where(np.isfinite(feat), feat, 0.0)


def _watch_to_stress(
    hr: Optional[float],
    hrv: Optional[float],
    spo2: Optional[float],
) -> float:
    """Return 0-1 stress estimate from Apple Watch biometrics."""
    stress = 0.0
    if hr is not None and hr > 0:
        if hr > 100:
            stress += 0.4
        elif hr < 60:
            stress -= 0.1
    if hrv is not None and hrv > 0:
        if hrv < 20:
            stress += 0.5
        elif hrv < 30:
            stress += 0.3
        elif hrv > 60:
            stress -= 0.2
        elif hrv > 40:
            stress -= 0.1
    if spo2 is not None and spo2 > 0:
        if spo2 < 95:
            stress += 0.4
        elif spo2 < 97:
            stress += 0.1
    return float(np.clip(stress, 0.0, 1.0))


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class VoiceWatchRequest(BaseModel):
    audio_b64: str = Field(..., description="Base64-encoded WAV (5-10s)")
    sample_rate: int = Field(22050, description="Audio sample rate in Hz")
    user_id: str = Field("default", description="User identifier")
    hr: Optional[float] = Field(None, description="Heart rate bpm")
    hrv: Optional[float] = Field(None, description="HRV SDNN ms")
    spo2: Optional[float] = Field(None, description="SpO2 percentage")
    real_time: bool = Field(
        False,
        description="Prefer SenseVoice fast path (<100ms) for WebSocket streaming",
    )


class CacheRequest(BaseModel):
    user_id: str = Field("default", description="User identifier")
    emotion_result: Dict[str, Any] = Field(..., description="Voice emotion result dict")


class EmotionResult(BaseModel):
    emotion: str
    probabilities: Dict[str, float] = {}
    valence: float
    arousal: float
    confidence: float
    model_type: str
    stress_from_watch: Optional[float] = None


def _voice_stress_index(result: Dict[str, Any]) -> float:
    """Composite 0-1 stress score from voice biomarkers + watch signal."""
    biomarkers = result.get("biomarkers", {}) or {}
    mental = result.get("mental_health", {}) or {}

    hnr_db = float(biomarkers.get("hnr_db", 20.0) or 20.0)
    jitter = float(biomarkers.get("jitter_local", 0.01) or 0.01)
    model_stress = float(mental.get("stress", 0.0) or 0.0)
    watch_stress = float(result.get("stress_from_watch", 0.0) or 0.0)

    low_hnr = float(np.clip((18.0 - hnr_db) / 18.0, 0.0, 1.0))
    jitter_tension = float(np.clip((0.01 - min(jitter, 0.01)) / 0.01, 0.0, 1.0))
    composite = 0.45 * model_stress + 0.30 * low_hnr + 0.15 * jitter_tension + 0.10 * watch_stress
    return float(np.clip(composite, 0.0, 1.0))


def _voice_focus_index(result: Dict[str, Any]) -> float:
    """Approximate 0-1 focus proxy from voice prosody stability."""
    biomarkers = result.get("biomarkers", {}) or {}
    speech_rate = float(biomarkers.get("speech_rate", 0.0) or 0.0)
    confidence = float(result.get("confidence", 0.0) or 0.0)
    arousal = float(result.get("arousal", 0.0) or 0.0)

    if speech_rate <= 0:
      speech_score = 0.45
    else:
      speech_score = float(np.clip(1.0 - abs(speech_rate - 4.5) / 4.5, 0.0, 1.0))
    arousal_balance = float(np.clip(1.0 - abs(arousal - 0.55) / 0.55, 0.0, 1.0))
    return float(np.clip(0.45 * confidence + 0.30 * speech_score + 0.25 * arousal_balance, 0.0, 1.0))


def _auto_log_voice_brain_state(user_id: str, timestamp: float, result: Dict[str, Any]) -> None:
    """Store voice-derived brain-state proxies in supplement tracker."""
    try:
        from api.routes.supplement_tracker import get_tracker as get_supplement_tracker

        biomarkers = result.get("biomarkers", {}) or {}
        tracker = get_supplement_tracker()
        tracker.log_brain_state(
            user_id=user_id,
            timestamp=timestamp,
            emotion_data={
                "valence": float(result.get("valence", 0.0)),
                "arousal": float(result.get("arousal", 0.0)),
                "stress_index": _voice_stress_index(result),
                "focus_index": _voice_focus_index(result),
                "source": "voice",
                "speech_rate": float(biomarkers.get("speech_rate", 0.0) or 0.0),
            },
        )
    except Exception as exc:
        log.warning("Voice supplement auto-log failed: %s", exc)


# ── Analyze endpoint ──────────────────────────────────────────────────────────

@router.post("/analyze", response_model=EmotionResult)
def voice_watch_analyze(req: VoiceWatchRequest) -> Dict[str, Any]:
    """6-class emotion from microphone audio + optional Apple Watch biometrics."""
    # Decode base64 audio
    try:
        wav_bytes = base64.b64decode(req.audio_b64)
    except Exception as exc:
        raise HTTPException(422, f"Invalid base64 audio: {exc}")

    result: Optional[Dict] = None

    # ── Primary: VoiceEmotionModel (emotion2vec+ or LightGBM fallback) ────────
    try:
        import soundfile as sf  # type: ignore
        audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        from models.voice_emotion_model import get_voice_model
        result = get_voice_model().predict_with_biomarkers(
            audio, sample_rate=int(sr), real_time=req.real_time
        )
        # Validate result has all required keys
        _required = {"emotion", "probabilities", "valence", "arousal", "confidence", "model_type"}
        if result is not None and not _required.issubset(result.keys()):
            log.warning("VoiceEmotionModel returned incomplete result: %s", list(result.keys()))
            result = None
    except Exception as exc:
        log.warning("VoiceEmotionModel path failed: %s", exc)

    # ── Fallback: re-load audio with librosa and retry VoiceEmotionModel ────────
    if result is None:
        if not _ensure_librosa():
            raise HTTPException(503, "No audio processing library available")
        try:
            import librosa
            y, _ = librosa.load(io.BytesIO(wav_bytes), sr=_SR, mono=True)
        except Exception as exc:
            raise HTTPException(422, f"Could not decode WAV: {exc}")
        if len(y) < _SR // 4:
            raise HTTPException(422, "Audio too short — need at least 0.25s")
        try:
            from models.voice_emotion_model import get_voice_model
            result = get_voice_model().predict_with_biomarkers(
                y, sample_rate=_SR, real_time=req.real_time
            )
        except Exception as exc:
            log.warning("VoiceEmotionModel librosa fallback failed: %s", exc)

    # ── Fail honestly if all paths failed ───────────────────────────────────────
    if result is None:
        raise HTTPException(503, "Voice emotion analysis failed — model could not process audio")

    # ── Blend watch biometric stress signal ───────────────────────────────────
    has_watch = any(v is not None for v in [req.hr, req.hrv, req.spo2])
    if has_watch:
        stress_w = _watch_to_stress(req.hr, req.hrv, req.spo2)
        result["valence"] = float(np.clip(result["valence"] - stress_w * 0.3, -1.0, 1.0))
        result["arousal"] = float(np.clip(result["arousal"] + stress_w * 0.2, 0.0, 1.0))
        result["stress_from_watch"] = round(stress_w, 4)

    ts = time.time()
    _auto_log_voice_brain_state(req.user_id, ts, result)

    # Persist to history for brain_report and other cross-route consumers
    _VOICE_HISTORY[req.user_id].append({
        "timestamp": ts,
        "emotion": result.get("emotion", "neutral"),
        "valence": float(result.get("valence", 0.0)),
        "arousal": float(result.get("arousal", 0.5)),
        "stress_index": _voice_stress_index(result),
        "focus_index": _voice_focus_index(result),
    })

    return result


# ── Cache endpoints ────────────────────────────────────────────────────────────

@router.post("/cache")
def cache_voice_result(req: CacheRequest) -> Dict[str, str]:
    """Store a voice emotion result for WebSocket fusion (5-min TTL)."""
    _VOICE_CACHE[req.user_id] = {"result": req.emotion_result, "ts": time.time()}
    return {"status": "cached", "user_id": req.user_id}


@router.get("/latest/{user_id}")
def get_latest_voice(user_id: str) -> Optional[Dict]:
    """Return cached voice result if < 5 minutes old, else None."""
    entry = _VOICE_CACHE.get(user_id)
    if not entry:
        return None
    if time.time() - entry["ts"] > _VOICE_CACHE_TTL:
        _VOICE_CACHE.pop(user_id, None)
        return None
    return entry["result"]


# ── Status endpoint ───────────────────────────────────────────────────────────

@router.get("/status")
def voice_watch_status() -> Dict[str, Any]:
    """Return voice model availability."""
    e2v_large_ok = False
    e2v_ok = False
    sensevoice_ok = False
    preferred_tier = "heuristic"
    try:
        from models.voice_emotion_model import get_voice_model
        vm = get_voice_model()
        e2v_large_ok = vm._load_e2v_large()
        e2v_ok = vm._load_e2v()
        sensevoice_ok = vm._sensevoice.available
        if e2v_large_ok:
            preferred_tier = "emotion2vec_large"
        elif e2v_ok:
            preferred_tier = "emotion2vec_base"
        elif sensevoice_ok:
            preferred_tier = "sensevoice"
    except Exception:
        pass
    librosa_ok = _ensure_librosa()
    lgbm_ok = _LGBM_PATH.exists()
    if preferred_tier == "heuristic":
        if lgbm_ok:
            preferred_tier = "lgbm_fallback"
        elif librosa_ok:
            preferred_tier = "feature_heuristic"
    return {
        "emotion2vec_large_available": e2v_large_ok,
        "emotion2vec_available": e2v_ok,
        "sensevoice_available": sensevoice_ok,
        "lgbm_fallback_available": lgbm_ok,
        "librosa_available": librosa_ok,
        "preferred_model_tier": preferred_tier,
        "ready": e2v_large_ok or e2v_ok or sensevoice_ok or lgbm_ok or librosa_ok,
    }


# ── History & daily-summary endpoints (#302) ─────────────────────────────────
# _VOICE_HISTORY is the canonical per-user ring buffer populated by /analyze.
# These routes expose it so client helpers have real endpoints to call.

@router.get("/history/{user_id}")
def get_voice_history(user_id: str, last_n: int = 50) -> Dict[str, Any]:
    """Return the last N voice-watch results for a user.

    Each record: {timestamp, emotion, valence, arousal, stress_index, focus_index}
    """
    history = _VOICE_HISTORY.get(user_id, [])
    trimmed = history[-last_n:] if len(history) > last_n else history
    return {"user_id": user_id, "count": len(trimmed), "history": trimmed}


@router.get("/daily-summary/{user_id}")
def get_daily_summary(user_id: str) -> Dict[str, Any]:
    """Return morning/noon/evening aggregates for the current calendar day (UTC).

    Groups today's _VOICE_HISTORY entries by approximate time slot:
      morning  = 04:00-11:59 UTC
      noon     = 12:00-16:59 UTC
      evening  = 17:00-03:59 UTC

    Returns mean valence, arousal, stress_index, focus_index and dominant emotion
    for each slot that has at least one entry.
    """
    import datetime as _dt
    from collections import defaultdict as _dd

    now = time.time()
    # UTC midnight for today
    today_dt = _dt.datetime.utcfromtimestamp(now).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    today_start = today_dt.timestamp()

    history = _VOICE_HISTORY.get(user_id, [])
    today = [r for r in history if r.get("timestamp", 0) >= today_start]

    def _slot(ts: float) -> str:
        h = _dt.datetime.utcfromtimestamp(ts).hour
        if 4 <= h < 12:
            return "morning"
        if 12 <= h < 17:
            return "noon"
        return "evening"

    groups: Dict[str, List[Dict]] = _dd(list)
    for r in today:
        groups[_slot(r["timestamp"])].append(r)

    summary: Dict[str, Any] = {}
    for slot, records in groups.items():
        vals = [r.get("valence", 0.0) for r in records]
        aros = [r.get("arousal", 0.5) for r in records]
        strs = [r.get("stress_index", 0.0) for r in records]
        foci = [r.get("focus_index", 0.0) for r in records]
        emo_counts: Dict[str, int] = _dd(int)
        for r in records:
            emo_counts[r.get("emotion", "neutral")] += 1
        dominant = max(emo_counts, key=lambda e: emo_counts[e]) if emo_counts else "neutral"
        summary[slot] = {
            "count": len(records),
            "dominant_emotion": dominant,
            "avg_valence": round(float(np.mean(vals)), 4),
            "avg_arousal": round(float(np.mean(aros)), 4),
            "avg_stress_index": round(float(np.mean(strs)), 4),
            "avg_focus_index": round(float(np.mean(foci)), 4),
            "latest_timestamp": max(r["timestamp"] for r in records),
        }

    return {
        "user_id": user_id,
        "date": today_dt.strftime("%Y-%m-%d"),
        "total_today": len(today),
        "trajectory": summary,
    }

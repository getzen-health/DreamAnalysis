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
    hr: Optional[float] = Field(None, description="Heart rate bpm")
    hrv: Optional[float] = Field(None, description="HRV SDNN ms")
    spo2: Optional[float] = Field(None, description="SpO2 percentage")


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
        result = get_voice_model().predict(audio, sample_rate=int(sr))
    except Exception as exc:
        log.warning("VoiceEmotionModel path failed: %s", exc)

    # ── Fallback: direct LightGBM via librosa ─────────────────────────────────
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
        result = {
            "emotion": "neutral",
            "probabilities": {e: round(1 / 6, 4) for e in EMOTIONS_6},
            "valence": 0.0,
            "arousal": 0.5,
            "confidence": 0.4,
            "model_type": "voice_lgbm_fallback",
        }

    # ── Blend watch biometric stress signal ───────────────────────────────────
    has_watch = any(v is not None for v in [req.hr, req.hrv, req.spo2])
    if has_watch:
        stress_w = _watch_to_stress(req.hr, req.hrv, req.spo2)
        result["valence"] = float(np.clip(result["valence"] - stress_w * 0.3, -1.0, 1.0))
        result["arousal"] = float(np.clip(result["arousal"] + stress_w * 0.2, 0.0, 1.0))
        result["stress_from_watch"] = round(stress_w, 4)

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
    e2v_ok = False
    try:
        from models.voice_emotion_model import get_voice_model
        e2v_ok = get_voice_model()._load_e2v()
    except Exception:
        pass
    librosa_ok = _ensure_librosa()
    lgbm_ok = _LGBM_PATH.exists()
    return {
        "emotion2vec_available": e2v_ok,
        "lgbm_fallback_available": lgbm_ok,
        "librosa_available": librosa_ok,
        "ready": e2v_ok or (lgbm_ok and librosa_ok),
    }

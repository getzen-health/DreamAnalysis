"""Voice + Apple Watch emotion analysis — upgraded to 6-class + result cache.

POST /voice-watch/analyze    — 6-class emotion from mic audio + watch biometrics
POST /voice-watch/cache      — cache latest voice result for WebSocket fusion
GET  /voice-watch/latest/{user_id} — retrieve cached result (< 5 min TTL)
GET  /voice-watch/status     — model availability
"""
from __future__ import annotations

import base64
import io
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
router = APIRouter(prefix="/voice-watch", tags=["voice-watch"])

_ML_ROOT = Path(__file__).resolve().parent.parent.parent
_LGBM_PATH = _ML_ROOT / "models" / "saved" / "audio_emotion_lgbm.pkl"

# Session storage directory (shared with SessionRecorder)
_SESSIONS_DIR = _ML_ROOT / "sessions"
_SESSIONS_DIR.mkdir(exist_ok=True)

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
    user_id: str = Field(..., description="User identifier")
    hr: Optional[float] = Field(None, description="Heart rate bpm")
    hrv: Optional[float] = Field(None, description="HRV SDNN ms")
    spo2: Optional[float] = Field(None, description="SpO2 percentage")
    real_time: bool = Field(
        False,
        description="Prefer SenseVoice fast path (<100ms) for WebSocket streaming",
    )


class CalibrationFrameRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    audio_features: Dict[str, float] = Field(
        ...,
        description=(
            "Acoustic feature dict (from extract_acoustic_features). "
            "Keys: pitch_mean, pitch_std, energy_mean, energy_std, "
            "speaking_rate_proxy, spectral_centroid_mean."
        ),
    )


class CacheRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    emotion_result: Dict[str, Any] = Field(..., description="Voice emotion result dict")


class EmotionResult(BaseModel):
    emotion: str
    probabilities: Dict[str, float] = {}
    valence: float
    arousal: float
    confidence: float
    model_type: str
    stress_from_watch: Optional[float] = None
    ensemble_active: Optional[bool] = None
    smoothed: Optional[bool] = None


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Extract a float from a value that may be a dict with 'risk_score'."""
    if isinstance(val, dict):
        return float(val.get("risk_score", default))
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def _voice_stress_index(result: Dict[str, Any]) -> float:
    """Composite 0-1 stress score from voice biomarkers + watch signal."""
    biomarkers = result.get("biomarkers", {}) or {}
    mental = result.get("mental_health", {}) or {}

    hnr_db = _safe_float(biomarkers.get("hnr_db"), 20.0)
    jitter = _safe_float(biomarkers.get("jitter_local"), 0.01)
    model_stress = _safe_float(mental.get("stress"), 0.0)
    watch_stress = _safe_float(result.get("stress_from_watch"), 0.0)

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
                "valence": _safe_float(result.get("valence"), 0.0),
                "arousal": _safe_float(result.get("arousal"), 0.0),
                "stress_index": _voice_stress_index(result),
                "focus_index": _voice_focus_index(result),
                "source": "voice",
                "speech_rate": _safe_float(biomarkers.get("speech_rate"), 0.0),
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

    # ── Primary: VoiceEnsemble (emotion2vec+ + acoustic features + smoothing) ──
    audio_arr: Optional[np.ndarray] = None
    audio_sr: int = _SR

    try:
        import soundfile as sf  # type: ignore
        audio_arr, audio_sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if audio_arr.ndim > 1:
            audio_arr = audio_arr.mean(axis=1)
    except Exception as exc:
        log.debug("soundfile decode failed — will retry with librosa: %s", exc)

    if audio_arr is None:
        if not _ensure_librosa():
            raise HTTPException(503, "No audio processing library available")
        try:
            import librosa
            audio_arr, audio_sr = librosa.load(io.BytesIO(wav_bytes), sr=_SR, mono=True)
        except Exception as exc:
            raise HTTPException(422, f"Could not decode WAV: {exc}")

    if audio_arr is None or len(audio_arr) < audio_sr // 4:
        raise HTTPException(422, "Audio too short — need at least 0.25s")

    # ── Per-user baseline calibration — normalize acoustic features if ready ──
    try:
        from models.voice_ensemble import (  # type: ignore
            extract_acoustic_features,
            get_voice_calibrator,
            get_voice_ensemble,
        )
        calibrator = get_voice_calibrator(req.user_id)
        if calibrator.is_ready:
            raw_acoustic = extract_acoustic_features(audio_arr, sr=int(audio_sr))
            normalized_acoustic = calibrator.normalize(raw_acoustic)
            log.debug(
                "Voice calibration active for user %s — normalizing acoustic features",
                req.user_id,
            )
        else:
            normalized_acoustic = None
    except Exception as exc:
        log.debug("Voice calibrator lookup failed: %s", exc)
        normalized_acoustic = None
        from models.voice_ensemble import get_voice_ensemble  # type: ignore  # noqa: F811

    try:
        ensemble = get_voice_ensemble()
        result = ensemble.predict(
            audio_arr,
            sample_rate=int(audio_sr),
            real_time=req.real_time,
        )
        # If calibration is active, overwrite the acoustic_features in the
        # result with the normalized version so the ensemble blending uses
        # per-user baseline-corrected features in subsequent frames.
        if result is not None and normalized_acoustic is not None:
            result["acoustic_features"] = normalized_acoustic
            result["calibration_applied"] = True
        _required = {"emotion", "probabilities", "valence", "arousal", "confidence", "model_type"}
        if result is not None and not _required.issubset(result.keys()):
            log.warning("VoiceEnsemble returned incomplete result: %s", list(result.keys()))
            result = None
    except Exception as exc:
        log.warning("VoiceEnsemble path failed: %s", exc)
        result = None

    # ── Fallback: VoiceEmotionModel without ensemble ─────────────────────────
    if result is None:
        try:
            from models.voice_emotion_model import get_voice_model
            result = get_voice_model().predict_with_biomarkers(
                audio_arr, sample_rate=int(audio_sr), real_time=req.real_time
            )
            _required = {"emotion", "probabilities", "valence", "arousal", "confidence", "model_type"}
            if result is not None and not _required.issubset(result.keys()):
                log.warning("VoiceEmotionModel returned incomplete result: %s", list(result.keys()))
                result = None
        except Exception as exc:
            log.warning("VoiceEmotionModel fallback failed: %s", exc)

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

    # Enrich with voice cognitive load (lightweight, no model file needed)
    try:
        from models.voice_cognitive_load import VoiceCognitiveLoadEstimator
        cog_estimator = VoiceCognitiveLoadEstimator()
        cog_result = cog_estimator.predict(audio_arr, sample_rate=int(audio_sr))
        if cog_result:
            result["voice_cognitive_load"] = cog_result.get("voice_load_index", None)
    except Exception as exc:
        log.debug("Voice cognitive load enrichment failed: %s", exc)

    # Enrich with fatigue biomarkers for better stress estimation
    try:
        from models.voice_fatigue_model import get_voice_fatigue_scanner
        scanner = get_voice_fatigue_scanner()
        fatigue = scanner.scan(audio_arr, sr=int(audio_sr))
        result["fatigue_index"] = fatigue.fatigue_index
        result["fatigue_hnr_db"] = fatigue.hnr_db
    except Exception as exc:
        log.debug("Voice fatigue enrichment failed: %s", exc)

    ts = time.time()
    _auto_log_voice_brain_state(req.user_id, ts, result)

    # Persist to history for brain_report and other cross-route consumers
    try:
        _VOICE_HISTORY[req.user_id].append({
            "timestamp": ts,
            "emotion": result.get("emotion", "neutral"),
            "valence": _safe_float(result.get("valence"), 0.0),
            "arousal": _safe_float(result.get("arousal"), 0.5),
            "stress_index": _voice_stress_index(result),
            "focus_index": _voice_focus_index(result),
        })
    except Exception as exc:
        log.warning("Voice history append failed: %s", exc)

    # Auto-cache so /voice-watch/latest/{user_id} returns this result
    # (used by Daily Brain Report)
    _VOICE_CACHE[req.user_id] = {"result": result, "ts": ts}

    # Persist as a session JSON so it appears in /sessions list
    try:
        session_id = str(uuid.uuid4())[:8]
        stress_idx = _voice_stress_index(result)
        focus_idx = _voice_focus_index(result)
        session_meta: Dict[str, Any] = {
            "session_id": session_id,
            "user_id": req.user_id,
            "session_type": "voice_checkin",
            "start_time": ts,
            "end_time": ts + 10,
            "status": "completed",
            "metadata": {"source": "voice-watch", "model_type": result.get("model_type", "unknown")},
            "summary": {
                "duration_sec": 10,
                "n_frames": 1,
                "n_channels": 0,
                "n_samples": 0,
                "avg_stress": stress_idx,
                "avg_focus": focus_idx,
                "avg_relaxation": max(0.0, 1.0 - stress_idx),
                "avg_valence": float(result.get("valence", 0.0)),
                "avg_arousal": float(result.get("arousal", 0.5)),
                "dominant_emotion": str(result.get("emotion", "neutral")),
                "avg_flow": 0.0,
            },
            "analysis_timeline": [],
        }
        meta_path = _SESSIONS_DIR / f"{session_id}.json"
        with open(meta_path, "w") as f:
            json.dump(session_meta, f, indent=2, default=str)
    except Exception as exc:
        log.warning("Failed to persist voice-watch session: %s", exc)

    return result


# ── Calibration endpoints (#352) ──────────────────────────────────────────────

@router.post("/calibrate/add-frame")
def calibrate_add_frame(req: CalibrationFrameRequest) -> Dict[str, Any]:
    """Add one neutral-speech acoustic feature frame to the user's calibrator.

    Call this endpoint repeatedly (10+ times, ~1s audio per frame) while the
    user speaks neutrally.  Once 10 frames are collected, ``is_ready`` becomes
    True and the ``/analyze`` endpoint will automatically normalize features
    before emotion classification.

    Request body:
        user_id:        User identifier.
        audio_features: Acoustic feature dict with keys pitch_mean, pitch_std,
                        energy_mean, energy_std, speaking_rate_proxy,
                        spectral_centroid_mean.

    Returns:
        Calibration status including ``is_ready``, ``n_frames``, and
        ``progress_pct``.
    """
    try:
        from models.voice_ensemble import get_voice_calibrator  # type: ignore
        calibrator = get_voice_calibrator(req.user_id)
    except Exception as exc:
        raise HTTPException(503, f"Calibration unavailable: {exc}")

    just_ready = calibrator.add_frame(req.audio_features)
    status = calibrator.get_status()
    status["just_became_ready"] = just_ready
    return status


@router.get("/calibrate/status")
def calibrate_status(user_id: str) -> Dict[str, Any]:
    """Return calibration progress for a user.

    Query params:
        user_id: User identifier.

    Returns:
        dict with is_ready, n_frames, frames_needed, progress_pct, baseline_mean.
    """
    try:
        from models.voice_ensemble import get_voice_calibrator  # type: ignore
        calibrator = get_voice_calibrator(user_id)
    except Exception as exc:
        raise HTTPException(503, f"Calibration unavailable: {exc}")

    return calibrator.get_status()


@router.post("/calibrate/reset")
def calibrate_reset(req: Dict[str, str]) -> Dict[str, str]:
    """Clear calibration data for a user and restart the calibration phase.

    Request body:
        {"user_id": "<id>"}

    Returns:
        {"status": "reset", "user_id": "<id>"}
    """
    user_id = req.get("user_id", "")
    if not user_id:
        raise HTTPException(422, "user_id is required")

    try:
        from models.voice_ensemble import get_voice_calibrator  # type: ignore
        calibrator = get_voice_calibrator(user_id)
    except Exception as exc:
        raise HTTPException(503, f"Calibration unavailable: {exc}")

    calibrator.reset()
    return {"status": "reset", "user_id": user_id}


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
    ensemble_ok = False
    try:
        from models.voice_ensemble import get_voice_ensemble  # type: ignore
        get_voice_ensemble()  # instantiate singleton
        ensemble_ok = True
    except Exception:
        pass

    return {
        "emotion2vec_large_available": e2v_large_ok,
        "emotion2vec_available": e2v_ok,
        "sensevoice_available": sensevoice_ok,
        "lgbm_fallback_available": lgbm_ok,
        "librosa_available": librosa_ok,
        "ensemble_available": ensemble_ok,
        "preferred_model_tier": ("ensemble_" + preferred_tier) if ensemble_ok else preferred_tier,
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


# ── Issue #366: Multi-speaker analysis endpoint ───────────────────────────────

class MultiSpeakerRequest(BaseModel):
    audio_b64: str = Field(..., description="Base64-encoded WAV")
    sample_rate: int = Field(22050, description="Audio sample rate in Hz")
    user_id: str = Field(..., description="User identifier")


@router.post("/analyze-multi-speaker")
def voice_analyze_multi_speaker(req: MultiSpeakerRequest) -> Dict[str, Any]:
    """Diarize audio into per-speaker segments and return emotion timeline.

    For each detected speaker segment the endpoint runs the VoiceEnsemble
    pipeline and returns per-speaker emotion results in chronological order.

    Request body:
        audio_b64:   Base64-encoded WAV audio.
        sample_rate: Sample rate in Hz (default 22050).
        user_id:     User identifier.

    Returns:
        {
          "user_id": str,
          "n_speakers": int,
          "segments": [
            {
              "speaker_id": str,
              "start_time": float,
              "end_time": float,
              "duration": float,
              "emotion": str,
              "probabilities": {str: float},
              "valence": float,
              "arousal": float,
              "confidence": float,
              "model_type": str
            },
            ...
          ]
        }
    """
    try:
        wav_bytes = base64.b64decode(req.audio_b64)
    except Exception as exc:
        raise HTTPException(422, f"Invalid base64 audio: {exc}")

    # Decode audio
    audio_arr: Optional[np.ndarray] = None
    audio_sr: int = req.sample_rate

    try:
        import soundfile as sf  # type: ignore
        audio_arr, audio_sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if audio_arr.ndim > 1:
            audio_arr = audio_arr.mean(axis=1)
    except Exception as exc:
        log.debug("soundfile decode failed: %s", exc)

    if audio_arr is None:
        if not _ensure_librosa():
            raise HTTPException(503, "No audio processing library available")
        try:
            import librosa
            audio_arr, audio_sr = librosa.load(io.BytesIO(wav_bytes), sr=_SR, mono=True)
        except Exception as exc:
            raise HTTPException(422, f"Could not decode WAV: {exc}")

    if audio_arr is None or len(audio_arr) < audio_sr // 4:
        raise HTTPException(422, "Audio too short — need at least 0.25s")

    # Diarize
    try:
        from models.voice_ensemble import SimpleSpeakerDiarizer  # type: ignore
        diarizer = SimpleSpeakerDiarizer()
        segments = diarizer.diarize(audio_arr, sr=int(audio_sr))
    except Exception as exc:
        log.warning("Speaker diarization failed: %s", exc)
        raise HTTPException(503, f"Speaker diarization failed: {exc}")

    # Run ensemble on each segment
    speaker_ids = sorted({s.speaker_id for s in segments})
    results: List[Dict[str, Any]] = []

    try:
        from models.voice_ensemble import get_voice_ensemble  # type: ignore
        ensemble = get_voice_ensemble()
    except Exception as exc:
        log.warning("VoiceEnsemble unavailable for multi-speaker: %s", exc)
        ensemble = None

    for seg in segments:
        seg_result: Dict[str, Any] = {
            "speaker_id": seg.speaker_id,
            "start_time": round(seg.start_time, 3),
            "end_time": round(seg.end_time, 3),
            "duration": round(seg.end_time - seg.start_time, 3),
            "emotion": "neutral",
            "probabilities": {},
            "valence": 0.0,
            "arousal": 0.5,
            "confidence": 0.0,
            "model_type": "unavailable",
        }
        if ensemble is not None and len(seg.audio_segment) >= audio_sr // 4:
            try:
                pred = ensemble.predict(
                    seg.audio_segment,
                    sample_rate=int(audio_sr),
                    apply_temporal_smoothing=False,
                )
                if pred is not None:
                    seg_result.update({
                        "emotion": pred.get("emotion", "neutral"),
                        "probabilities": pred.get("probabilities", {}),
                        "valence": pred.get("valence", 0.0),
                        "arousal": pred.get("arousal", 0.5),
                        "confidence": pred.get("confidence", 0.0),
                        "model_type": pred.get("model_type", "unknown"),
                    })
            except Exception as exc:
                log.debug("Ensemble failed on segment %s: %s", seg.speaker_id, exc)

        results.append(seg_result)

    return {
        "user_id": req.user_id,
        "n_speakers": len(speaker_ids),
        "speaker_ids": speaker_ids,
        "segments": results,
    }


# ── Issue #377: Voice fatigue scan endpoint ───────────────────────────────────

class FatigueScanRequest(BaseModel):
    audio_b64: str = Field(..., description="Base64-encoded WAV")
    sample_rate: int = Field(22050, description="Audio sample rate in Hz")
    user_id: str = Field(..., description="User identifier")
    baseline: Optional[Dict[str, float]] = Field(
        None,
        description=(
            "Personal rested-voice baseline. Keys: hnr_db, jitter_pct, "
            "shimmer_db, speaking_rate_proxy. When omitted, population norms are used."
        ),
    )


@router.post("/fatigue-scan")
def voice_fatigue_scan(req: FatigueScanRequest) -> Dict[str, Any]:
    """Analyse vocal fatigue from acoustic biomarkers (issue #377).

    Extracts HNR, jitter, shimmer, and speaking rate from the provided audio
    and computes a composite fatigue index (0-100).  Optionally compares
    against a personal rested-voice baseline for more accurate scoring.

    Request body:
        audio_b64:   Base64-encoded WAV (recommend 3-10 s of continuous speech).
        sample_rate: Sample rate in Hz (default 22050).
        user_id:     User identifier.
        baseline:    Optional personal baseline dict.

    Returns:
        {
          "user_id": str,
          "fatigue_index": float,   // 0-100, higher = more fatigued
          "hnr_db": float,
          "hnr_delta": float,
          "jitter_pct": float,
          "jitter_ratio": float,
          "shimmer_db": float,
          "shimmer_ratio": float,
          "speaking_rate_proxy": float,
          "speaking_rate_ratio": float,
          "confidence": float,
          "recommendations": [str, ...],
          "baseline_used": bool
        }
    """
    try:
        wav_bytes = base64.b64decode(req.audio_b64)
    except Exception as exc:
        raise HTTPException(422, f"Invalid base64 audio: {exc}")

    # Decode audio
    audio_arr: Optional[np.ndarray] = None
    audio_sr: int = req.sample_rate

    try:
        import soundfile as sf  # type: ignore
        audio_arr, audio_sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if audio_arr.ndim > 1:
            audio_arr = audio_arr.mean(axis=1)
    except Exception as exc:
        log.debug("soundfile decode failed: %s", exc)

    if audio_arr is None:
        if not _ensure_librosa():
            raise HTTPException(503, "No audio processing library available")
        try:
            import librosa
            audio_arr, audio_sr = librosa.load(io.BytesIO(wav_bytes), sr=_SR, mono=True)
        except Exception as exc:
            raise HTTPException(422, f"Could not decode WAV: {exc}")

    if audio_arr is None or len(audio_arr) < audio_sr // 4:
        raise HTTPException(422, "Audio too short — need at least 0.25s")

    # Run fatigue scan
    try:
        from models.voice_fatigue_model import get_voice_fatigue_scanner  # type: ignore
        scanner = get_voice_fatigue_scanner()
        result = scanner.scan(audio_arr, sr=int(audio_sr), baseline=req.baseline)
    except Exception as exc:
        log.warning("VoiceFatigueScanner failed: %s", exc)
        raise HTTPException(503, f"Fatigue scan failed: {exc}")

    return {
        "user_id": req.user_id,
        "fatigue_index": result.fatigue_index,
        "hnr_db": result.hnr_db,
        "hnr_delta": result.hnr_delta,
        "jitter_pct": result.jitter_pct,
        "jitter_ratio": result.jitter_ratio,
        "shimmer_db": result.shimmer_db,
        "shimmer_ratio": result.shimmer_ratio,
        "speaking_rate_proxy": result.speaking_rate_proxy,
        "speaking_rate_ratio": result.speaking_rate_ratio,
        "confidence": result.confidence,
        "recommendations": result.recommendations,
        "baseline_used": result.baseline_used,
    }

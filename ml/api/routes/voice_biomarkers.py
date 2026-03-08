"""Voice biomarker and mental health screening endpoints.

POST /voice/biomarkers           -- raw biomarker extraction from audio
POST /voice/mental-health-screen -- depression + anxiety + stress risk scores
GET  /voice/biomarkers/status    -- availability check
"""
from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
router = APIRouter(prefix="/voice", tags=["voice-biomarkers"])

_DEFAULT_SR = 16000


# -- request / response schemas ------------------------------------------------

class AudioRequest(BaseModel):
    audio_b64: str = Field(
        ...,
        description="Base64-encoded WAV or raw PCM audio (mono, 2+ seconds)",
    )
    sample_rate: int = Field(
        _DEFAULT_SR,
        description="Audio sample rate in Hz (default 16000)",
    )


class BiomarkerResponse(BaseModel):
    f0_mean: float = 0.0
    f0_std: float = 0.0
    f0_range: float = 0.0
    jitter_local: float = 0.0
    jitter_rap: float = 0.0
    jitter_ppq5: float = 0.0
    shimmer_local: float = 0.0
    shimmer_apq3: float = 0.0
    hnr: float = 0.0
    silence_ratio: float = 0.0
    pause_count: int = 0
    mean_pause_duration: float = 0.0
    max_pause_duration: float = 0.0
    total_pause_duration: float = 0.0
    speech_rate: float = 0.0
    articulation_rate: float = 0.0
    energy_mean: float = 0.0
    energy_std: float = 0.0
    gfcc_mean: List[float] = []
    gfcc_std: List[float] = []


class ScreeningResult(BaseModel):
    risk_score: float = 0.0
    severity: str = "unknown"
    indicators: List[str] = []


class MentalHealthScreenResponse(BaseModel):
    biomarkers: Dict[str, Any] = {}
    depression: ScreeningResult = ScreeningResult()
    anxiety: ScreeningResult = ScreeningResult()
    stress: ScreeningResult = ScreeningResult()
    disclaimer: str = (
        "These scores are research-grade estimates, not clinical diagnoses. "
        "Consult a qualified mental health professional for any concerns."
    )


# -- helpers -------------------------------------------------------------------

def _decode_audio(b64_str: str, target_sr: int) -> np.ndarray:
    """Decode base64 audio to a mono float32 numpy array."""
    try:
        wav_bytes = base64.b64decode(b64_str)
    except Exception as exc:
        raise HTTPException(422, f"Invalid base64 audio: {exc}")

    # Try soundfile first (handles WAV, FLAC, OGG)
    try:
        import soundfile as sf  # type: ignore

        audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != target_sr:
            import librosa  # type: ignore

            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        return audio
    except Exception:
        pass

    # Fallback to librosa
    try:
        import librosa  # type: ignore

        audio, _ = librosa.load(
            io.BytesIO(wav_bytes), sr=target_sr, mono=True
        )
        return audio
    except Exception as exc:
        raise HTTPException(422, f"Could not decode audio: {exc}")


# -- endpoints -----------------------------------------------------------------

@router.post("/biomarkers", response_model=BiomarkerResponse)
def extract_biomarkers(req: AudioRequest) -> Dict[str, Any]:
    """Extract raw voice biomarkers from audio.

    Returns jitter, shimmer, HNR, F0 stats, pause metrics, speech rate,
    energy features, and GFCC coefficients.  Minimum 2 seconds of audio.
    """
    audio = _decode_audio(req.audio_b64, req.sample_rate)

    if len(audio) < req.sample_rate * 2:
        raise HTTPException(
            422, "Audio too short -- need at least 2 seconds"
        )

    from models.voice_biomarkers import get_biomarker_extractor

    extractor = get_biomarker_extractor()
    result = extractor.extract(audio, sr=req.sample_rate)

    if result.get("error"):
        raise HTTPException(503, f"Biomarker extraction failed: {result['error']}")

    return result


@router.post("/mental-health-screen", response_model=MentalHealthScreenResponse)
def mental_health_screen(req: AudioRequest) -> Dict[str, Any]:
    """Screen for depression, anxiety, and stress from voice.

    Extracts biomarkers and runs three screening models.  Returns
    risk scores (0-1), severity labels, and contributing indicators.

    Minimum 2 seconds of audio; 10+ seconds recommended for reliability.
    """
    audio = _decode_audio(req.audio_b64, req.sample_rate)

    if len(audio) < req.sample_rate * 2:
        raise HTTPException(
            422, "Audio too short -- need at least 2 seconds"
        )

    from models.voice_biomarkers import get_biomarker_extractor

    extractor = get_biomarker_extractor()
    biomarkers = extractor.extract(audio, sr=req.sample_rate)

    if biomarkers.get("error"):
        raise HTTPException(
            503, f"Biomarker extraction failed: {biomarkers['error']}"
        )

    depression = extractor.screen_depression(biomarkers)
    anxiety = extractor.screen_anxiety(biomarkers)
    stress = extractor.screen_stress(biomarkers)

    return {
        "biomarkers": biomarkers,
        "depression": depression,
        "anxiety": anxiety,
        "stress": stress,
        "disclaimer": (
            "These scores are research-grade estimates, not clinical diagnoses. "
            "Consult a qualified mental health professional for any concerns."
        ),
    }


@router.get("/biomarkers/status")
def biomarkers_status() -> Dict[str, Any]:
    """Check availability of voice biomarker extraction."""
    librosa_ok = False
    try:
        import librosa  # noqa: F401

        librosa_ok = True
    except ImportError:
        pass

    scipy_ok = False
    try:
        from scipy.fft import dct  # noqa: F401

        scipy_ok = True
    except ImportError:
        pass

    return {
        "ready": librosa_ok,
        "librosa_available": librosa_ok,
        "scipy_available": scipy_ok,
        "gfcc_available": librosa_ok and scipy_ok,
        "features": [
            "f0_mean", "f0_std", "f0_range",
            "jitter_local", "jitter_rap", "jitter_ppq5",
            "shimmer_local", "shimmer_apq3",
            "hnr",
            "silence_ratio", "pause_count", "mean_pause_duration",
            "max_pause_duration", "total_pause_duration",
            "speech_rate", "articulation_rate",
            "energy_mean", "energy_std",
            "gfcc_mean", "gfcc_std",
        ],
        "screening_models": ["depression", "anxiety", "stress"],
    }

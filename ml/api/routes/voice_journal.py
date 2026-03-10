"""Voice journal analysis endpoints.

POST /voice-journal/analyze  -- full journal analysis pipeline
GET  /voice-journal/status   -- capability availability check
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

router = APIRouter(prefix="/voice-journal", tags=["Voice Journal"])

_DEFAULT_SR = 16000
_MIN_DURATION_SEC = 5.0  # 422 if shorter


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class JournalAnalyzeRequest(BaseModel):
    audio_b64: str = Field(
        ...,
        description=(
            "Base64-encoded audio (WAV preferred, mono, 16 kHz recommended). "
            "Minimum 5 seconds required."
        ),
    )
    sample_rate: int = Field(
        _DEFAULT_SR,
        description="Audio sample rate in Hz (default 16000)",
    )
    include_transcript: bool = Field(
        True,
        description="Whether to attempt Whisper transcription (requires whisper package)",
    )


class TrajectoryPoint(BaseModel):
    time_sec: float
    emotion: str
    valence: float
    arousal: float
    confidence: float


class BiomarkerPoint(BaseModel):
    time_sec: float
    rms: float
    jitter: float
    shimmer: float
    hnr: float


class TopicEntry(BaseModel):
    word: str
    count: int


class JournalSummary(BaseModel):
    dominant_emotion: Optional[str] = None
    mean_valence: Optional[float] = None
    mean_arousal: Optional[float] = None
    peak_stress_time_sec: Optional[float] = None
    valence_trend: Optional[float] = None


class JournalAnalyzeResponse(BaseModel):
    duration_sec: float
    segment_count: int
    trajectory: List[TrajectoryPoint] = []
    biomarker_timeline: List[BiomarkerPoint] = []
    transcript: str = ""
    topics: List[TopicEntry] = []
    summary: JournalSummary = JournalSummary()
    insights: List[str] = []


# ---------------------------------------------------------------------------
# Audio decoding helper (try soundfile → scipy → raw float32)
# ---------------------------------------------------------------------------

def _decode_audio(b64_str: str, target_sr: int) -> np.ndarray:
    """Decode base64 audio to a mono float32 numpy array at target_sr."""
    try:
        wav_bytes = base64.b64decode(b64_str)
    except Exception as exc:
        raise HTTPException(422, f"Invalid base64 audio: {exc}")

    # 1. Try soundfile (handles WAV, FLAC, OGG)
    try:
        import soundfile as sf  # type: ignore

        audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != target_sr:
            try:
                import librosa  # type: ignore
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            except Exception:
                # Simple linear resampling fallback
                n_out = int(len(audio) * target_sr / sr)
                audio = np.interp(
                    np.linspace(0, len(audio) - 1, n_out),
                    np.arange(len(audio)),
                    audio,
                ).astype(np.float32)
        return audio
    except Exception:
        pass

    # 2. Try scipy.io.wavfile
    try:
        from scipy.io import wavfile  # type: ignore

        sr, audio = wavfile.read(io.BytesIO(wav_bytes))
        audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        # Normalize int ranges to [-1, 1]
        if np.abs(audio).max() > 1.0:
            audio = audio / 32768.0
        if sr != target_sr:
            n_out = int(len(audio) * target_sr / sr)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, n_out),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)
        return audio
    except Exception:
        pass

    # 3. Treat bytes as raw float32 PCM
    try:
        audio = np.frombuffer(wav_bytes, dtype=np.float32).copy()
        if len(audio) < 100:
            raise ValueError("Audio array too short after raw decode")
        return audio
    except Exception as exc:
        raise HTTPException(422, f"Could not decode audio data: {exc}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/analyze", response_model=JournalAnalyzeResponse)
def analyze_journal(req: JournalAnalyzeRequest) -> Dict[str, Any]:
    """Analyze a free-form voice journal recording.

    Segments audio into 2-second chunks, extracts per-segment emotion
    (via VoiceEmotionModel) and biomarkers (jitter/shimmer/HNR/RMS),
    optionally transcribes with Whisper tiny, extracts topics, and
    generates rule-based insights.

    Minimum audio length: 5 seconds.
    """
    audio = _decode_audio(req.audio_b64, req.sample_rate)

    duration_sec = len(audio) / req.sample_rate
    if duration_sec < _MIN_DURATION_SEC:
        raise HTTPException(
            422,
            f"Audio too short ({duration_sec:.1f}s) — minimum {_MIN_DURATION_SEC}s required",
        )

    from models.voice_journal_analyzer import get_voice_journal_analyzer  # type: ignore

    analyzer = get_voice_journal_analyzer()
    result = analyzer.analyze(
        audio,
        sr=req.sample_rate,
        include_transcript=req.include_transcript,
    )

    return result


@router.get("/status")
def journal_status() -> Dict[str, Any]:
    """Return capability status for the voice journal analyzer."""
    whisper_available = False
    try:
        import whisper  # type: ignore  # noqa: F401
        whisper_available = True
    except ImportError:
        pass

    librosa_available = False
    try:
        import librosa  # type: ignore  # noqa: F401
        librosa_available = True
    except ImportError:
        pass

    return {
        "whisper_available": whisper_available,
        "librosa_available": librosa_available,
        "ready": True,  # Always ready — graceful fallbacks for all features
        "features": {
            "emotion_trajectory": True,
            "biomarkers": librosa_available,
            "transcription": whisper_available,
            "topics": True,  # Only requires transcript text, no extra deps
            "insights": True,
        },
        "min_duration_sec": _MIN_DURATION_SEC,
        "segment_length_sec": 2.0,
    }

"""Multimodal emotion analysis endpoint.

POST /multimodal/analyze
  Fuses EEG + audio + video into one emotion prediction.
  Each modality is optional except EEG.

POST /multimodal/analyze-frame
  Lightweight: analyze a single video frame alongside live EEG.
  Faster for real-time webcam streaming.

GET  /multimodal/status
  Returns which modality models are loaded.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
router = APIRouter(prefix="/multimodal", tags=["multimodal"])

# ── Lazy singleton ─────────────────────────────────────────────────────────────
_fusion_model = None

def _get_fusion():
    global _fusion_model
    if _fusion_model is None:
        try:
            from models.multimodal_fusion import MultimodalEmotionFusion
            _fusion_model = MultimodalEmotionFusion()
        except Exception as exc:
            log.error("MultimodalFusion init failed: %s", exc)
            raise HTTPException(500, f"Multimodal model not available: {exc}")
    return _fusion_model


# ── Request / Response schemas ─────────────────────────────────────────────────

class MultimodalRequest(BaseModel):
    # EEG: (4 channels × n_samples) as nested list — required
    eeg: List[List[float]] = Field(
        ..., description="EEG array: 4 channels × n_samples (Muse 2 layout)"
    )
    fs_eeg: float = Field(256.0, description="EEG sampling rate in Hz")

    # Audio: mono samples as flat list — optional
    audio: Optional[List[float]] = Field(
        None, description="Mono audio waveform samples"
    )
    sr_audio: int = Field(22050, description="Audio sample rate in Hz")

    # Video: single BGR frame as base64-encoded JPEG/PNG — optional
    video_frame_b64: Optional[str] = Field(
        None, description="Base64-encoded JPEG/PNG video frame from webcam"
    )


class MultimodalFrameRequest(BaseModel):
    """Lightweight request: live EEG + single webcam frame (no audio buffer needed)."""
    eeg: List[List[float]]
    fs_eeg: float = 256.0
    video_frame_b64: Optional[str] = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _decode_frame(b64_str: str) -> Optional[np.ndarray]:
    """Decode base64 JPEG/PNG → BGR numpy array."""
    try:
        import cv2
        img_bytes = base64.b64decode(b64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    except Exception as exc:
        log.warning("Frame decode failed: %s", exc)
        return None


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/status")
def multimodal_status():
    """Return which modality models are loaded."""
    from pathlib import Path
    _ML_ROOT   = Path(__file__).resolve().parent.parent.parent
    _MODEL_DIR = _ML_ROOT / "models" / "saved"

    eeg_ok   = (_MODEL_DIR / "emotion_mega_lgbm.pkl").exists()
    audio_ok = (_MODEL_DIR / "audio_emotion_lgbm.pkl").exists()
    video_ok = (_MODEL_DIR / "video_emotion_lgbm.pkl").exists()

    n_modalities = sum([eeg_ok, audio_ok, video_ok])

    return {
        "eeg_model_loaded":   eeg_ok,
        "audio_model_loaded": audio_ok,
        "video_model_loaded": video_ok,
        "n_modalities":       n_modalities,
        "fusion_weights": {
            "eeg":   0.50,
            "audio": 0.25 if audio_ok else 0.0,
            "video": 0.25 if video_ok else 0.0,
        },
        "ready": eeg_ok,
    }


@router.post("/analyze")
def multimodal_analyze(req: MultimodalRequest) -> Dict[str, Any]:
    """Full multimodal emotion analysis: EEG + (optional) audio + video.

    Returns fused emotion prediction plus per-modality breakdowns.
    """
    fusion = _get_fusion()

    # EEG
    try:
        eeg_array = np.array(req.eeg, dtype=np.float32)
        if eeg_array.ndim != 2 or eeg_array.shape[0] != 4:
            raise HTTPException(422, "eeg must be shape [4, n_samples]")
        if eeg_array.shape[1] < 64:
            raise HTTPException(422, "EEG needs at least 64 samples per channel")
    except ValueError as exc:
        raise HTTPException(422, f"EEG parse error: {exc}")

    # Audio
    audio_samples = None
    if req.audio is not None and len(req.audio) > 0:
        audio_samples = np.array(req.audio, dtype=np.float32)

    # Video frame
    video_frame = None
    if req.video_frame_b64:
        video_frame = _decode_frame(req.video_frame_b64)

    result = fusion.predict(
        eeg_array=eeg_array,
        fs_eeg=req.fs_eeg,
        audio_samples=audio_samples,
        sr_audio=req.sr_audio,
        video_frame=video_frame,
    )
    return result


@router.post("/analyze-frame")
def multimodal_analyze_frame(req: MultimodalFrameRequest) -> Dict[str, Any]:
    """Fast multimodal analysis: EEG + single webcam frame.

    Ideal for real-time streaming where audio buffering is not available.
    """
    fusion = _get_fusion()

    try:
        eeg_array = np.array(req.eeg, dtype=np.float32)
        if eeg_array.ndim != 2 or eeg_array.shape[0] != 4:
            raise HTTPException(422, "eeg must be shape [4, n_samples]")
    except ValueError as exc:
        raise HTTPException(422, f"EEG parse error: {exc}")

    video_frame = None
    if req.video_frame_b64:
        video_frame = _decode_frame(req.video_frame_b64)

    result = fusion.predict(
        eeg_array=eeg_array,
        fs_eeg=req.fs_eeg,
        video_frame=video_frame,
    )
    return result

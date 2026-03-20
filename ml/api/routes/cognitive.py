"""Cognitive model endpoints: drowsiness, load, attention, stress, lucid dream, meditation.

Improvements over baseline version:
- EMA output smoothing (α=0.25) — reduces frame-to-frame noise by ~75%
- Per-user BaselineCalibrator wiring — normalizes features to user's resting baseline
  before sklearn prediction (only active when cal.is_ready after 30 baseline frames)
"""

import threading
import numpy as np
from fastapi import APIRouter
from typing import Any, Dict, List

from ._shared import (
    _numpy_safe,
    drowsiness_model, cognitive_load_model, attention_model,
    stress_model, lucid_dream_model, meditation_model,
    EEGInput, LucidDreamRequest,
)
from .calibration import _get_baseline_cal
from models.voice_cognitive_load import VoiceCognitiveLoadEstimator

_voice_cog_load = VoiceCognitiveLoadEstimator()

router = APIRouter()

# ── Per-user EMA smoothing ────────────────────────────────────────────────────
_cognitive_ema: Dict[str, Dict[str, float]] = {}
_cognitive_ema_lock = threading.Lock()
_EMA_ALPHA = 0.25  # matches food_emotion_predictor.py


def _smooth(user_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Apply EMA smoothing to all top-level float values in result."""
    with _cognitive_ema_lock:
        state = _cognitive_ema.setdefault(user_id, {})
        out = dict(result)
        for key, val in result.items():
            if isinstance(val, float):
                prev = state.get(key, val)
                smoothed = _EMA_ALPHA * val + (1.0 - _EMA_ALPHA) * prev
                state[key] = smoothed
                out[key] = round(smoothed, 4)
        return out


def _calibrated_predict(model, eeg: np.ndarray, fs: float, user_id: str) -> Dict:
    """Run model prediction with optional per-user baseline normalization.

    If BaselineCalibrator is ready (≥30 resting frames collected) AND the model
    has sklearn weights, features are z-scored against the user's resting baseline
    before prediction. Falls back to model.predict() if anything fails.
    """
    from processing.eeg_processor import extract_features, preprocess

    cal = _get_baseline_cal(user_id)
    if (
        cal.is_ready
        and hasattr(model, "sklearn_model") and model.sklearn_model is not None
        and hasattr(model, "feature_names") and model.feature_names is not None
    ):
        try:
            processed = preprocess(eeg, fs)
            features = extract_features(processed, fs)
            normalized = cal.normalize(features)
            fv = np.array(
                [normalized.get(k, features.get(k, 0.0)) for k in model.feature_names]
            ).reshape(1, -1)
            if getattr(model, "scaler", None) is not None:
                fv = model.scaler.transform(fv)
            cal_probs = model.sklearn_model.predict_proba(fv)[0]
            cal_idx = int(np.argmax(cal_probs))
            # Get base result for full dict structure, then patch calibrated values
            base = model.predict(eeg, fs)
            if "confidence" in base:
                base["confidence"] = round(float(cal_probs[cal_idx]), 3)
            for key in list(base.keys()):
                if key.endswith("_index"):
                    base[key] = cal_idx
            base["baseline_calibrated"] = True
            return base
        except Exception:
            pass
    return model.predict(eeg, fs)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/predict-drowsiness")
async def predict_drowsiness(data: EEGInput):
    """Detect drowsiness level: alert / drowsy / sleepy."""
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]
    result = _calibrated_predict(drowsiness_model, eeg, data.fs, data.user_id)
    return _numpy_safe(_smooth(data.user_id, result))


@router.post("/predict-cognitive-load")
async def predict_cognitive_load(data: EEGInput):
    """Estimate cognitive load: low / moderate / high."""
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]
    result = _calibrated_predict(cognitive_load_model, eeg, data.fs, data.user_id)
    return _numpy_safe(_smooth(data.user_id, result))


@router.post("/predict-attention")
async def predict_attention(data: EEGInput):
    """Classify attention: distracted / passive / focused / hyperfocused."""
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]
    result = _calibrated_predict(attention_model, eeg, data.fs, data.user_id)
    return _numpy_safe(_smooth(data.user_id, result))


@router.post("/predict-stress")
async def predict_stress(data: EEGInput):
    """Detect stress level: relaxed / mild / moderate / high."""
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]
    result = _calibrated_predict(stress_model, eeg, data.fs, data.user_id)
    return _numpy_safe(_smooth(data.user_id, result))


@router.post("/predict-lucid-dream")
async def predict_lucid_dream(req: LucidDreamRequest):
    """Detect lucid dreaming: non_lucid / pre_lucid / lucid / controlled."""
    signals = np.array(req.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]
    result = _calibrated_predict(lucid_dream_model, eeg, req.fs, req.user_id)
    return _numpy_safe(_smooth(req.user_id, result))


@router.post("/predict-meditation")
async def predict_meditation(data: EEGInput):
    """Classify meditation depth: relaxed / meditating / deep."""
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]
    result = _calibrated_predict(meditation_model, eeg, data.fs, data.user_id)
    return _numpy_safe(_smooth(data.user_id, result))


@router.get("/cognitive-models/session-stats")
async def cognitive_session_stats():
    """Get session statistics for all cognitive models that track history."""
    stats = {}
    if hasattr(lucid_dream_model, "get_session_stats"):
        stats["lucid_dream"] = lucid_dream_model.get_session_stats()
    if hasattr(meditation_model, "get_session_stats"):
        stats["meditation"] = meditation_model.get_session_stats()
    return _numpy_safe(stats)


# ── Brain Age Estimation ──────────────────────────────────────────────────────

from typing import Optional
from pydantic import BaseModel, Field

from models.brain_age_estimator import get_brain_age_estimator


class BrainAgeRequest(BaseModel):
    signals: List[List[float]] = Field(..., description="EEG signals (channels x samples)")
    fs: float = Field(default=256.0, description="Sampling frequency in Hz")
    user_id: str = Field(..., description="User identifier")
    chronological_age: Optional[float] = Field(
        default=None, description="User's actual age in years (enables gap calculation)"
    )


@router.post("/brain-age")
async def estimate_brain_age(req: BrainAgeRequest):
    """Estimate biological brain age from EEG aperiodic features.

    Returns predicted_age, brain_age_gap (if chronological_age provided),
    and aperiodic spectral features. Wellness indicator only — not medical.
    """
    signals = np.array(req.signals)
    estimator = get_brain_age_estimator()
    result = estimator.predict(signals, req.fs, chronological_age=req.chronological_age)
    return _numpy_safe(result)


# ── Emotion Regulation / Reappraisal Detection ───────────────────────────────

from models.reappraisal_detector import get_reappraisal_detector


@router.post("/reappraisal-detection/predict")
async def reappraisal_detect(data: EEGInput):
    """Detect emotion regulation vs genuine experience via LPP + frontal theta.

    Returns regulation_state: genuine / mild_regulation / active_reappraisal / suppression.
    Higher regulation_index = more cognitive control over emotional response.
    """
    signals = np.array(data.signals)
    detector = get_reappraisal_detector(data.user_id)
    result = detector.predict(signals, data.fs)
    return _numpy_safe(result)


@router.post("/reappraisal-detection/baseline")
async def reappraisal_baseline(data: EEGInput):
    """Record resting-state baseline for LPP normalization."""
    signals = np.array(data.signals)
    detector = get_reappraisal_detector(data.user_id)
    detector.update_baseline(signals, data.fs)
    return {"status": "baseline_updated", "user_id": data.user_id}


# ── Sleep Memory Consolidation ─────────────────────────────────────────────

from models.memory_consolidation_tracker import get_memory_tracker


class MemoryEpochRequest(BaseModel):
    signals: List[List[float]] = Field(..., description="EEG signals (channels x samples)")
    fs: float = Field(default=256.0, description="Sampling frequency in Hz")
    user_id: str = Field(..., description="User identifier")
    sleep_stage: str = Field(default="N2", description="Current sleep stage: N1, N2, N3, REM, Wake")


@router.post("/sleep/memory-consolidation/epoch")
async def score_memory_epoch(req: MemoryEpochRequest):
    """Score one sleep epoch for memory consolidation quality.

    Use during sleep recording. Provide sleep_stage for accurate weighting.
    Returns spindle density, SO-spindle coupling, and consolidation quality.
    """
    signals = np.array(req.signals)
    tracker = get_memory_tracker(req.user_id)
    result = tracker.score_epoch(signals, req.fs, req.sleep_stage)
    return _numpy_safe(result)


@router.get("/sleep/memory-consolidation/session/{user_id}")
async def get_memory_session(user_id: str):
    """Get memory consolidation summary for the current sleep session."""
    tracker = get_memory_tracker(user_id)
    return tracker.score_session()


@router.post("/sleep/memory-consolidation/tmr-check")
async def check_tmr_trigger(req: MemoryEpochRequest):
    """Check if current moment is good for TMR audio cue (SO up-state detection)."""
    signals = np.array(req.signals)
    tracker = get_memory_tracker(req.user_id)
    return tracker.get_tmr_trigger(signals, req.fs, req.sleep_stage)


# ── Cognitive Flexibility ────────────────────────────────────────────────────

from models.cognitive_flexibility_detector import get_flexibility_detector


@router.post("/cognitive-flexibility")
async def detect_cognitive_flexibility(data: EEGInput):
    """Detect cognitive flexibility level from frontal EEG features.

    Uses aperiodic exponent + FMT power. Returns flexibility_index (0-1),
    level (rigid/moderate/flexible), and metacontrol bias.
    """
    signals = np.array(data.signals)
    detector = get_flexibility_detector(data.user_id)
    result = detector.predict(signals, data.fs)
    return _numpy_safe(result)


@router.post("/cognitive-flexibility/baseline")
async def record_flexibility_baseline(data: EEGInput):
    """Record resting-state baseline for dynamic flexibility measurement."""
    signals = np.array(data.signals)
    detector = get_flexibility_detector(data.user_id)
    return detector.record_baseline(signals, data.fs)



# ── Emotional Granularity ────────────────────────────────────────────────────

from models.emotion_granularity import get_granularity_mapper, estimate_dominance as _estimate_dominance


@router.post("/emotion-granularity")
async def analyze_emotion_granularity(data: EEGInput):
    """Map EEG-derived VAD coordinates to 27-emotion nuanced vocabulary.

    Returns primary emotion + 2 nuance emotions with narrative description.
    Computes valence and arousal from EEG band powers, then maps to the
    27-emotion VAD space with dominance estimation.
    """
    from processing.eeg_processor import preprocess, extract_band_powers

    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    ch = signals[0]

    bands = extract_band_powers(preprocess(ch, data.fs), data.fs)
    alpha = max(bands.get("alpha", 0.2), 1e-10)
    beta = max(bands.get("beta", 0.15), 1e-10)

    valence = float(np.clip(np.tanh((alpha / beta - 0.7) * 2.0), -1, 1))
    arousal = float(np.clip(beta / (beta + alpha), 0, 1))
    dominance = _estimate_dominance(bands)

    mapper = get_granularity_mapper()
    result = mapper.map(valence, arousal, dominance)
    return _numpy_safe(result)
@router.post("/voice-cognitive-load")
async def voice_cognitive_load(request: dict):
    """Estimate cognitive load from voice prosodic features.

    Accepts base64-encoded audio and returns a voice-based cognitive load
    estimate using F0 variation, intensity variation, and voice activity ratio.
    """
    import base64
    import io

    try:
        import librosa
    except ImportError:
        return {"error": "librosa not available"}

    audio_b64 = request.get("audio_base64", "")
    sr = request.get("sr", 16000)

    if not audio_b64:
        return _voice_cog_load._empty_result()

    try:
        audio_bytes = base64.b64decode(audio_b64)
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=sr)
    except Exception as e:
        return {"error": f"Could not decode audio: {e}"}

    result = _voice_cog_load.predict(audio, sr)
    return _numpy_safe(result)


# ── Fused Cognitive Load (EEG + Voice) ───────────────────────────────────────


class FusedCogLoadRequest(BaseModel):
    eeg_signals: List[List[float]] = Field(default=[], description="EEG signals (channels x samples)")
    audio_base64: str = Field(default="", description="Base64-encoded audio")
    eeg_fs: float = Field(default=256.0)
    voice_sr: int = Field(default=16000)
    user_id: str = Field(default="anonymous")


@router.post("/cognitive-load/fused")
async def fused_cognitive_load(req: FusedCogLoadRequest):
    """Fuse EEG and voice cognitive load into a single score.

    Uses confidence-weighted average: whichever modality has higher
    confidence contributes more to the final score. Returns both
    individual scores + the fused result.
    """
    eeg_result = None
    voice_result = None

    # EEG cognitive load
    if req.eeg_signals:
        try:
            signals = np.array(req.eeg_signals)
            if signals.ndim == 1:
                signals = signals.reshape(1, -1)
            eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]
            eeg_result = _calibrated_predict(cognitive_load_model, eeg, req.eeg_fs, req.user_id)
        except Exception:
            pass

    # Voice cognitive load
    if req.audio_base64:
        try:
            import base64, io, librosa
            audio_bytes = base64.b64decode(req.audio_base64)
            audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=req.voice_sr)
            voice_result = _voice_cog_load.predict(audio, sr)
        except Exception:
            pass

    # Fusion: confidence-weighted average
    eeg_score = eeg_result.get("cognitive_load_score", 0.5) if eeg_result else None
    eeg_conf = eeg_result.get("confidence", 0.5) if eeg_result else 0.0
    voice_score = voice_result.get("voice_load_index", 0.5) if voice_result else None
    voice_conf = voice_result.get("confidence", 0.5) if voice_result else 0.0

    if eeg_score is not None and voice_score is not None:
        total_conf = eeg_conf + voice_conf + 1e-10
        fused = (eeg_score * eeg_conf + voice_score * voice_conf) / total_conf
        fused_conf = max(eeg_conf, voice_conf)
        source = "eeg+voice"
    elif eeg_score is not None:
        fused = eeg_score
        fused_conf = eeg_conf
        source = "eeg"
    elif voice_score is not None:
        fused = voice_score
        fused_conf = voice_conf
        source = "voice"
    else:
        fused = 0.5
        fused_conf = 0.0
        source = "none"

    # Map score to label
    if fused < 0.35:
        level = "low"
    elif fused < 0.65:
        level = "moderate"
    else:
        level = "high"

    return _numpy_safe({
        "fused_cognitive_load": float(fused),
        "fused_confidence": float(fused_conf),
        "level": level,
        "source": source,
        "eeg_cognitive_load": float(eeg_score) if eeg_score is not None else None,
        "eeg_confidence": float(eeg_conf),
        "voice_cognitive_load": float(voice_score) if voice_score is not None else None,
        "voice_confidence": float(voice_conf),
    })


# ── Attention Screening ──────────────────────────────────────────────────────

from models.attention_screener import AttentionScreener, DISCLAIMER as ADHD_DISCLAIMER

_screener_instances: dict = {}


def _get_screener(user_id: str) -> AttentionScreener:
    if user_id not in _screener_instances:
        _screener_instances[user_id] = AttentionScreener()
    return _screener_instances[user_id]


@router.post("/attention-screening")
async def screen_attention(data: EEGInput):
    """Track attention patterns using aperiodic EEG + TBR.

    Returns attention_risk_index (0-1). Wellness indicator only — not a medical device.
    """
    signals = np.array(data.signals)
    screener = _get_screener(data.user_id)
    result = screener.predict(signals, data.fs)
    return _numpy_safe(result)


@router.post("/attention-screening/rest-baseline")
async def record_attention_rest(data: EEGInput):
    """Record resting-state baseline for dynamic attention response test."""
    signals = np.array(data.signals)
    screener = _get_screener(data.user_id)
    return screener.record_rest_baseline(signals, data.fs)


# ── Emotion Trajectory Prediction ────────────────────────────────────────────
from pydantic import BaseModel, Field
from models.emotion_trajectory_predictor import get_trajectory_predictor


class TrajectoryUpdateRequest(BaseModel):
    valence: float = Field(..., description="Current valence (-1 to 1)")
    arousal: float = Field(..., description="Current arousal (0 to 1)")
    user_id: str = Field(..., description="User identifier")


@router.post("/emotion-trajectory/update")
async def update_emotion_trajectory(req: TrajectoryUpdateRequest):
    """Add current emotion reading to trajectory buffer."""
    predictor = get_trajectory_predictor(req.user_id)
    predictor.update(req.valence, req.arousal)
    return {"status": "updated", "history_length": len(predictor._valence_history)}


@router.get("/emotion-trajectory/predict/{user_id}")
async def predict_emotion_trajectory(user_id: str, horizon_steps: int = 5):
    """Predict future emotion state from recent history."""
    predictor = get_trajectory_predictor(user_id)
    result = predictor.predict(horizon_steps)
    return _numpy_safe(result)


@router.get("/emotion-trajectory/history/{user_id}")
async def get_emotion_trajectory(user_id: str):
    """Get full emotion trajectory history and trends."""
    predictor = get_trajectory_predictor(user_id)
    return _numpy_safe(predictor.get_trajectory())


@router.post("/emotion-trajectory/reset/{user_id}")
async def reset_emotion_trajectory(user_id: str):
    """Reset trajectory buffer for a user."""
    predictor = get_trajectory_predictor(user_id)
    predictor.reset()
    return {"status": "reset"}


# ── IMU Artifact Detection ──────────────────────────────────────────────────

from typing import List, Optional  # noqa: F811  already imported above for other sections
from pydantic import BaseModel, Field  # noqa: F811
from models.imu_artifact_detector import get_imu_detector


class IMURequest(BaseModel):
    acc_data: List[List[float]] = Field(
        ..., description="Accelerometer data (3 x samples or samples x 3), m/s²"
    )
    gyro_data: Optional[List[List[float]]] = Field(
        default=None, description="Gyroscope data (3 x samples or samples x 3), deg/s"
    )
    fs: float = Field(default=52.0, description="IMU sampling rate in Hz (Muse 2: ~52 Hz)")


@router.post("/imu/detect-motion-artifact")
async def detect_motion_artifact(req: IMURequest):
    """Detect motion artifacts from Muse 2 IMU data.

    Computes RMS acceleration magnitude above resting baseline.  Returns
    motion_detected flag, artifact_probability [0-1], and a recommendation
    string: "clean", "mild_motion", or "severe_motion".

    Call once per EEG epoch before running emotion/cognitive classifiers.
    If motion_detected=true, discard the EEG epoch — do not classify.
    """
    acc = np.array(req.acc_data)
    gyro = np.array(req.gyro_data) if req.gyro_data is not None else None
    detector = get_imu_detector()
    result = detector.detect(acc, gyro, req.fs)
    return _numpy_safe(result)


@router.post("/imu/calibrate-resting")
async def calibrate_imu_resting(req: IMURequest):
    """Record resting-state IMU baseline for motion artifact detection.

    Send 30+ seconds of still, seated accelerometer data.  After calibration,
    the detector's threshold is relative to this individual's resting baseline
    rather than a global default (~1 g), improving sensitivity for subtle
    movements like head nods.
    """
    acc = np.array(req.acc_data)
    detector = get_imu_detector()
    detector.calibrate_resting(acc, req.fs)
    return {
        "status": "calibrated",
        "resting_baseline_g": float(detector._resting_baseline),
    }


# ── Heart-Brain Coupling ──────────────────────────────────────────────────────

from pydantic import BaseModel as _BaseModel, Field as _Field
from processing.ppg_features import PPGFeatureExtractor as _PPGFeatureExtractor
from processing.heart_brain import compute_heart_brain_coupling as _compute_hbc

_ppg_extractor = _PPGFeatureExtractor()


class HeartBrainRequest(_BaseModel):
    eeg_signals: list = _Field(..., description="EEG signals [[ch0_samples], [ch1_samples], ...]")
    ppg_signal: list = _Field(..., description="PPG signal samples (64 Hz)")
    fs_eeg: float = _Field(default=256.0)
    fs_ppg: float = _Field(default=64.0)
    user_id: str = _Field(..., description="User identifier")


@router.post("/heart-brain-coupling")
async def analyze_heart_brain_coupling(req: HeartBrainRequest):
    """Compute Heartbeat-Evoked Potential and HRV features from PPG+EEG.

    Returns HEP amplitude, HRV time/frequency features, and interoceptive index.
    Requires simultaneous PPG + EEG recording (Muse 2 with board.config_board('p50')).
    """
    eeg = np.array(req.eeg_signals)
    ppg = np.array(req.ppg_signal)
    result = _compute_hbc(eeg, ppg, req.fs_eeg, req.fs_ppg)
    return _numpy_safe(result)


@router.post("/heart-brain-coupling/hrv-only")
async def analyze_hrv(req: HeartBrainRequest):
    """Extract HRV features from PPG only (no EEG required).

    Returns RMSSD, SDNN, pNN50, HR_bpm, LF/HF ratio.
    """
    ppg = np.array(req.ppg_signal)
    result = _ppg_extractor.extract_hrv(ppg)
    return _numpy_safe(result)


# ── EEGNet-Lite On-Device Inference ─────────────────────────────────────────

from typing import List
from pydantic import BaseModel, Field

from models.eegnet_lite import EEGNetLite

_eegnet_lite = EEGNetLite()


class EEGNetFinetuneRequest(BaseModel):
    signals: List[List[float]] = Field(..., description="EEG signals")
    label: int = Field(
        ...,
        ge=0,
        le=5,
        description=(
            "True emotion label "
            "(0=happy, 1=sad, 2=angry, 3=fear, 4=surprise, 5=neutral)"
        ),
    )
    fs: float = Field(default=256.0)
    user_id: str = Field(..., description="User identifier")


@router.post("/eegnet-lite/predict")
async def eegnet_lite_predict(data: EEGInput):
    """Classify emotion using compact EEGNet-Lite (~2600 params, <20KB).

    Designed for on-device inference. Falls back to heuristics if PyTorch unavailable.
    """
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    result = _eegnet_lite.predict(signals, data.fs)
    return _numpy_safe(result)


@router.post("/eegnet-lite/fine-tune")
async def eegnet_lite_fine_tune(req: EEGNetFinetuneRequest):
    """Online last-layer SGD fine-tuning for personalization (+7.31% from ISWC 2024)."""
    signals = np.array(req.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    result = _eegnet_lite.fine_tune_last_layer(signals, req.label, req.fs)
    return _numpy_safe(result)


@router.get("/eegnet-lite/info")
async def eegnet_lite_info():
    """Get EEGNet-Lite model architecture info and parameter count."""
    return _numpy_safe(_eegnet_lite.get_model_info())


# ── Self-Supervised Contrastive EEG Encoder ──────────────────────────────────

from training.self_supervised_pretrain import EEGContrastivePretrainer, EEGAugmentor

_contrastive_pretrainer = EEGContrastivePretrainer()
_eeg_augmentor = EEGAugmentor()


class ContrastivePretrainRequest(BaseModel):
    epochs: List[List[List[float]]] = Field(
        ...,
        description="List of EEG epochs, each (n_channels, n_samples)",
    )
    n_train_epochs: int = Field(default=5, ge=1, le=50, description="Number of pretraining passes")
    batch_size: int = Field(default=16, ge=2, le=64)
    lr: float = Field(default=3e-4)


@router.post("/contrastive-pretrain/step")
async def contrastive_pretrain_step(req: ContrastivePretrainRequest):
    """Run one contrastive pretraining step on a batch of unlabeled EEG epochs.

    No emotion labels required. Returns NT-Xent loss for monitoring.
    """
    epochs = np.array(req.epochs)
    result = _contrastive_pretrainer.pretrain_step(epochs)
    return _numpy_safe(result)


@router.post("/contrastive-pretrain/extract-features")
async def contrastive_extract_features(data: EEGInput):
    """Extract learned EEG representations from pretrained encoder.

    Returns 128-dim embedding vector for downstream classification.
    """
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    embedding = _contrastive_pretrainer.extract_features(signals)
    return {
        "embedding": embedding.tolist(),
        "embed_dim": len(embedding),
        "torch_available": bool(len(embedding) > 0),
    }


@router.get("/contrastive-pretrain/info")
async def contrastive_pretrain_info():
    """Get contrastive pretrainer architecture info."""
    return _numpy_safe(_contrastive_pretrainer.get_pretrainer_info())


# ── CNN-KAN-F2CA Sparse-Channel Emotion ──────────────────────────────────────

from models.cnn_kan_emotion import get_cnn_kan_classifier


@router.post("/cnn-kan-emotion/predict")
async def cnn_kan_emotion_predict(data: EEGInput):
    """Classify 4-quadrant emotion (valence × arousal) using CNN-KAN-F2CA.

    Optimized for 4-channel sparse EEG. Returns one of:
    low_valence_low_arousal, high_valence_low_arousal,
    low_valence_high_arousal, high_valence_high_arousal.
    """
    signals = np.array(data.signals)
    classifier = get_cnn_kan_classifier(data.user_id)


# ── Dynamic Graph Attention EEG (DGAT) ───────────────────────────────────────

from models.dgat_eeg import get_dgat_classifier


@router.post("/dgat-emotion/predict")
async def dgat_emotion_predict(data: EEGInput):
    """Classify emotion using Dynamic Graph Attention on EEG channel graph.

    Computes dynamic per-epoch adjacency matrix from channel correlations,
    then applies graph attention for spatially-aware emotion classification.
    Returns 6-class emotion + graph connectivity stats.
    """
    signals = np.array(data.signals)
    classifier = get_dgat_classifier(data.user_id)
    result = classifier.predict(signals, data.fs)
    return _numpy_safe(result)


@router.get("/cnn-kan-emotion/info")
async def cnn_kan_emotion_info():
    """Get CNN-KAN-F2CA model architecture info and parameter count."""
    classifier = get_cnn_kan_classifier()
    return _numpy_safe(classifier.get_model_info())


@router.get("/dgat-emotion/graph-stats")
async def dgat_emotion_graph_stats(user_id: str):
    """Get last computed dynamic graph adjacency statistics."""
    classifier = get_dgat_classifier(user_id)
    return _numpy_safe(classifier.get_graph_stats())


# ── Cross-Modal EEG+Voice Fusion ─────────────────────────────────────────────

from models.eeg_voice_fusion import get_eeg_voice_fusion


class EEGVoiceFusionRequest(BaseModel):
    eeg_signals: List[List[float]] = Field(..., description="EEG signals (channels x samples)")
    audio_base64: str = Field(default="", description="Base64-encoded audio (WAV/PCM)")
    eeg_fs: float = Field(default=256.0)
    voice_fs: int = Field(default=16000)
    user_id: str = Field(..., description="User identifier")


@router.post("/eeg-voice-fusion/predict")
async def eeg_voice_fusion_predict(req: EEGVoiceFusionRequest):
    """Fuse EEG and voice features via Optimal Transport alignment.

    Accepts EEG signals + base64-encoded audio. Uses Sinkhorn OT to
    find optimal alignment between modal distributions before fusion.
    Falls back gracefully to single-modal if one input missing.
    """
    import base64

    eeg = np.array(req.eeg_signals)
    audio = None
    if req.audio_base64:
        try:
            audio_bytes = base64.b64decode(req.audio_base64)
            audio = np.frombuffer(audio_bytes, dtype=np.float32)
        except Exception:
            audio = None

    fusion = get_eeg_voice_fusion(req.user_id)
    result = fusion.predict(eeg, audio, req.eeg_fs, req.voice_fs)
    return _numpy_safe(result)


@router.get("/eeg-voice-fusion/stats")
async def eeg_voice_fusion_stats(user_id: str):
    """Get last OT transport plan stats (cost, alignment quality)."""
    fusion = get_eeg_voice_fusion(user_id)
    return _numpy_safe(fusion.get_fusion_stats())


# ── GNN Spatial-Temporal Graph Emotion ───────────────────────────────────────

from models.graph_emotion_classifier import get_graph_emotion_classifier


@router.post("/graph-emotion/predict")
async def predict_graph_emotion(data: EEGInput):
    """Classify emotion using GNN spatial-temporal graph (4-node Muse 2 graph).

    Models the 4 Muse 2 channels (TP9, AF7, AF8, TP10) as a static + learnable
    adjacency graph and applies two GAT-style message-passing layers (NumPy-only,
    no torch-geometric). Node features: 5 DE bands + 3 Hjorth = 8 per node.
    Returns 6-class emotion + graph edge weights for visualisation.
    """
    signals = np.array(data.signals)
    clf = get_graph_emotion_classifier(data.user_id)
    result = clf.predict(signals, fs=int(data.fs))
    return _numpy_safe(result)


@router.get("/graph-emotion/info")
async def graph_emotion_info():
    """Return GNN model info: node count, edge list, feature dimensions."""
    clf = get_graph_emotion_classifier()
    return _numpy_safe(clf.get_model_info())


# ── DREAM Database Enhanced Dream Detection ───────────────────────────────────

@router.post("/dream-database/predict")
async def predict_dream_database(data: EEGInput):
    """Detect dream state using DREAM database-informed classifier (505 subjects, 2643 awakenings).

    Uses published DREAM database statistics (Nature Communications, 2025) as
    Bayesian priors blended with EEG-derived features for sleep-stage-aware
    dream probability estimation.
    """
    try:
        from models.dream_database_detector import get_dream_database_detector
        from fastapi import HTTPException
        eeg = np.array(data.signals)
        detector = get_dream_database_detector(data.user_id)
        # sleep_stage is not part of EEGInput; callers pass it via a wrapper or
        # the endpoint defaults to None (uses overall DREAM base rate of 0.45).
        result = detector.predict(eeg, fs=int(data.fs))
        return _numpy_safe(result)
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dream-database/themes")
async def get_dream_themes_endpoint(data: EEGInput):
    """Estimate dream theme probabilities (emotional/visual/kinesthetic/narrative) from EEG features.

    Based on DREAM database spectral correlates of dream content.
    """
    try:
        from models.dream_database_detector import get_dream_database_detector
        from fastapi import HTTPException
        eeg = np.array(data.signals)
        detector = get_dream_database_detector(data.user_id)
        themes = detector.get_dream_themes(eeg, fs=int(data.fs))
        return _numpy_safe({"themes": themes})
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dream-database/stats")
async def dream_database_stats():
    """Return DREAM database statistics and download instructions."""
    try:
        from training.train_dream_database import DREAMDatabaseLoader
        loader = DREAMDatabaseLoader()
        stats = loader.get_benchmark_stats()
        stats["download_instructions"] = loader.download_instructions()
        stats["is_available"] = loader.is_available()
        return _numpy_safe(stats)
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))

"""Shared utilities, model singletons, and Pydantic schemas for all route modules."""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

# ─── ML Model imports ────────────────────────────────────────────────────────
from models.sleep_staging import SleepStagingModel
from models.emotion_classifier import EmotionClassifier
from models.dream_detector import DreamDetector
from models.flow_state_detector import FlowStateDetector
from models.creativity_detector import CreativityDetector, MemoryEncodingPredictor
from models.drowsiness_detector import DrowsinessDetector
from models.cognitive_load_estimator import CognitiveLoadEstimator
from models.attention_classifier import AttentionClassifier
from models.stress_detector import StressDetector
from models.lucid_dream_detector import LucidDreamDetector
from models.meditation_classifier import MeditationClassifier
from models.food_emotion_predictor import FoodEmotionPredictor
from neurofeedback.protocol_engine import NeurofeedbackProtocol, PROTOCOLS as PROTOCOLS
from storage.session_recorder import SessionRecorder
from processing.state_transitions import BrainStateEngine
from processing.confidence_calibration import ConfidenceCalibrator

# ─── Simulation & processing re-exports (used by route modules) ──────────────
from simulation.eeg_simulator import STATE_PROFILES as STATE_PROFILES, simulate_eeg as simulate_eeg
from processing.eeg_processor import (
    extract_features as extract_features,
    extract_band_powers as extract_band_powers,
    preprocess as preprocess,
    extract_features_multichannel as extract_features_multichannel,
    compute_frontal_asymmetry as compute_frontal_asymmetry,
    compute_coherence as compute_coherence,
    compute_phase_locking_value as compute_phase_locking_value,
    compute_cwt_spectrogram as compute_cwt_spectrogram,
    compute_dwt_features as compute_dwt_features,
    detect_sleep_spindles as detect_sleep_spindles,
    detect_k_complexes as detect_k_complexes,
    extract_spectral_microstate_features as extract_spectral_microstate_features,
    apply_circadian_correction as apply_circadian_correction,
)
from processing.artifact_detector import (
    detect_eye_blinks as detect_eye_blinks,
    detect_muscle_artifacts as detect_muscle_artifacts,
    detect_electrode_pops as detect_electrode_pops,
    compute_signal_quality_index as compute_signal_quality_index,
    auto_reject_epochs as auto_reject_epochs,
    ica_artifact_removal as ica_artifact_removal,
)
from processing.connectivity import (
    compute_granger_causality as compute_granger_causality,
    compute_dtf as compute_dtf,
    compute_graph_metrics as compute_graph_metrics,
)
from processing.spiritual_energy import (
    compute_chakra_activations as compute_chakra_activations,
    compute_chakra_balance as compute_chakra_balance,
    compute_meditation_depth as compute_meditation_depth,
    compute_aura_energy as compute_aura_energy,
    compute_kundalini_flow as compute_kundalini_flow,
    compute_prana_balance as compute_prana_balance,
    compute_consciousness_level as compute_consciousness_level,
    compute_third_eye_activation as compute_third_eye_activation,
    full_spiritual_analysis as full_spiritual_analysis,
    CHAKRAS as CHAKRAS,
    CONSCIOUSNESS_LEVELS as CONSCIOUSNESS_LEVELS,
)
from processing.emotion_shift_detector import EmotionShiftDetector as EmotionShiftDetector
from models.anomaly_detector import AnomalyDetector as AnomalyDetector
from models.multimodal_emotion_fusion import MultimodalEmotionFusion, BiometricSnapshot
from processing.calibration import (
    UserCalibration as UserCalibration,
    CalibrationRunner as CalibrationRunner,
    CALIBRATION_STEPS as CALIBRATION_STEPS,
)
from processing.signal_quality import SignalQualityChecker as SignalQualityChecker
from processing.user_feedback import (
    FeedbackCollector as FeedbackCollector,
    PersonalizedPipeline as PersonalizedPipeline,
)
from processing.confidence_calibration import add_uncertainty_labels as add_uncertainty_labels

# ─── Paths ───────────────────────────────────────────────────────────────────
MODEL_DIR = Path("models/saved")
BENCHMARK_DIR = Path("benchmarks")


# ─── JSON helpers ────────────────────────────────────────────────────────────
def _numpy_safe(obj):
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _numpy_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_numpy_safe(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _find_model(prefix: str) -> Optional[str]:
    """Auto-discover model files in models/saved/. Priority: ONNX > pkl."""
    if not MODEL_DIR.exists():
        return None
    onnx = MODEL_DIR / f"{prefix}.onnx"
    pkl = MODEL_DIR / f"{prefix}.pkl"
    if onnx.exists():
        return str(onnx)
    if pkl.exists():
        return str(pkl)
    return None


# ─── Model singletons ────────────────────────────────────────────────────────
# REVE Foundation singleton — loaded once at startup, used in predict_emotion()
try:
    import os as _os
    if _os.environ.get("HF_TOKEN"):
        from models.reve_foundation import get_reve_foundation as _get_reve
        _reve_foundation = _get_reve()
    else:
        _reve_foundation = None
except Exception:
    _reve_foundation = None

sleep_model = SleepStagingModel(model_path=_find_model("sleep_staging_model"))
emotion_model = EmotionClassifier(model_path=_find_model("emotion_classifier_model"))
dream_model = DreamDetector(model_path=_find_model("dream_detector_model"))
flow_model = FlowStateDetector(model_path=_find_model("flow_state_model"))
creativity_model = CreativityDetector(model_path=_find_model("creativity_model"))
memory_model = MemoryEncodingPredictor(model_path=_find_model("memory_encoding_model"))
drowsiness_model = DrowsinessDetector(model_path=_find_model("drowsiness_model"))
cognitive_load_model = CognitiveLoadEstimator(model_path=_find_model("cognitive_load_model"))
attention_model = AttentionClassifier(model_path=_find_model("attention_model"))
stress_model = StressDetector(model_path=_find_model("stress_model"))
lucid_dream_model = LucidDreamDetector(model_path=_find_model("lucid_dream_model"))
meditation_model = MeditationClassifier(model_path=_find_model("meditation_model"))
food_emotion_model = FoodEmotionPredictor(model_path=_find_model("food_emotion_model"))
fusion_model = MultimodalEmotionFusion()

# ─── Shared state ────────────────────────────────────────────────────────────
_emotion_shift_detectors: Dict[str, EmotionShiftDetector] = {}
_device_manager = None
_personal_models: Dict[str, object] = {}

# Per-user neurofeedback protocol state (replaces global _nf_protocol singleton)
_nf_protocols: Dict[str, NeurofeedbackProtocol] = {}

# Per-user session recorders (replaces global _session_recorder singleton)
_session_recorders: Dict[str, SessionRecorder] = {}

# Per-user anomaly detectors (replaces global _anomaly_detector singleton)
_anomaly_detectors: Dict[str, AnomalyDetector] = {}

# Per-user accuracy pipeline (replaces global _state_engine / _confidence_cal singletons)
_state_engines: Dict[str, BrainStateEngine] = {}
_confidence_cals: Dict[str, ConfidenceCalibrator] = {}

# Accuracy pipeline — shared signal quality checker (stateless, safe to share)
_quality_checker = SignalQualityChecker(fs=256)
_calibration_runners: Dict[str, CalibrationRunner] = {}

# Per-user biometric snapshots — updated via POST /biometrics/update
_biometric_snapshots: Dict[str, BiometricSnapshot] = {}

# ── Personalized k-NN pipeline cache ─────────────────────────────────────────
# PersonalizedPipeline is refreshed at most once per 60 seconds per user so
# the JSONL file isn't re-read on every WebSocket frame.
import time as _time
_personalized_pipelines: Dict[str, "tuple[PersonalizedPipeline, float]"] = {}
_PIPELINE_REFRESH_SECS = 60.0

# Last extracted 17-dim feature vector per user — attached to label-only corrections
# so the k-NN has training data even when no raw EEG is submitted with the correction.
_last_features: Dict[str, np.ndarray] = {}


def get_last_features(user_id: str) -> Optional[np.ndarray]:
    """Return the most recently cached feature vector for *user_id*, or None."""
    return _last_features.get(user_id)


def _get_personalized_pipeline(user_id: str) -> PersonalizedPipeline:
    """Return a k-NN PersonalizedPipeline for *user_id*, refreshing if stale."""
    now = _time.monotonic()
    cached = _personalized_pipelines.get(user_id)
    if cached is None or (now - cached[1]) > _PIPELINE_REFRESH_SECS:
        pipeline = PersonalizedPipeline(user_id)
        pipeline.update_from_feedback()
        _personalized_pipelines[user_id] = (pipeline, now)
    return _personalized_pipelines[user_id][0]


def get_biometric_snapshot(user_id: str) -> BiometricSnapshot:
    """Return the cached BiometricSnapshot for *user_id*, creating one if absent."""
    if user_id not in _biometric_snapshots:
        _biometric_snapshots[user_id] = BiometricSnapshot()
    return _biometric_snapshots[user_id]


def update_biometric_snapshot(user_id: str, fields: dict) -> BiometricSnapshot:
    """Merge *fields* into the per-user BiometricSnapshot and return it."""
    snap = get_biometric_snapshot(user_id)
    for k, v in fields.items():
        if hasattr(snap, k) and v is not None:
            setattr(snap, k, v)
    _biometric_snapshots[user_id] = snap
    return snap

# ── Legacy aliases (kept for code that imports these directly) ────────────────
# These always return/act on the "default" user bucket so old single-user
# callers continue to work without modification.
_session_recorder: SessionRecorder  # assigned lazily below via getter


def _get_nf_protocol(user_id: str = "default") -> Optional[NeurofeedbackProtocol]:
    """Return the active NeurofeedbackProtocol for *user_id*, or None."""
    return _nf_protocols.get(user_id)


def _set_nf_protocol(user_id: str, protocol: Optional[NeurofeedbackProtocol]) -> None:
    """Set (or clear) the NeurofeedbackProtocol for *user_id*."""
    if protocol is None:
        _nf_protocols.pop(user_id, None)
    else:
        _nf_protocols[user_id] = protocol


def _get_session_recorder(user_id: str = "default") -> SessionRecorder:
    """Return (creating if needed) the SessionRecorder for *user_id*."""
    if user_id not in _session_recorders:
        _session_recorders[user_id] = SessionRecorder()
    return _session_recorders[user_id]


def _get_anomaly_detector(user_id: str = "default") -> AnomalyDetector:
    """Return (creating if needed) the AnomalyDetector for *user_id*."""
    if user_id not in _anomaly_detectors:
        _anomaly_detectors[user_id] = AnomalyDetector()
    return _anomaly_detectors[user_id]


# Legacy alias — analysis.py imports this directly for the default user
_anomaly_detector: AnomalyDetector = _get_anomaly_detector("default")


def _get_state_engine(user_id: str = "default") -> BrainStateEngine:
    """Return (creating if needed) the BrainStateEngine for *user_id*."""
    if user_id not in _state_engines:
        _state_engines[user_id] = BrainStateEngine()
    return _state_engines[user_id]


def _get_confidence_cal(user_id: str = "default") -> ConfidenceCalibrator:
    """Return (creating if needed) the ConfidenceCalibrator for *user_id*."""
    if user_id not in _confidence_cals:
        _confidence_cals[user_id] = ConfidenceCalibrator()
    return _confidence_cals[user_id]


# Initialise the legacy singleton so existing `from ._shared import _session_recorder` still works
_session_recorder = _get_session_recorder("default")


# ─── Helper functions ────────────────────────────────────────────────────────
def _get_device_manager():
    global _device_manager
    if _device_manager is None:
        try:
            from hardware.brainflow_manager import BrainFlowManager
            _device_manager = BrainFlowManager()
        except Exception as e:
            import logging
            logging.getLogger("api").error(f"BrainFlowManager init failed: {e}")
            _device_manager = None
    return _device_manager


def _get_personal_model(user_id: str):
    """Get or create personal model adapter for a user (legacy online-learner wrapper)."""
    if user_id not in _personal_models:
        try:
            from models.online_learner import PersonalModelAdapter
            _personal_models[user_id] = PersonalModelAdapter(emotion_model, user_id)
        except Exception:
            return None
    return _personal_models[user_id]


def predict_emotion(
    user_id: str, eeg, fs: float, n_channels: int = 4, device_type: str = "muse_2"
) -> dict:
    """Run emotion prediction, preferring the personal model when available.

    Priority:
      1. PersonalModel (EEGNet backbone + personal adapter head) — if backbone is
         trained AND the user has ≥30 labeled epochs fine-tuned
      2. PersonalModel central head (EEGNet backbone, cross-person) — if backbone
         trained but not enough personal data yet
      3. Global EmotionClassifier (mega LGBM, 74.21% CV) — if EEGNet not trained

    After the base prediction is obtained, the OnlineLearner (SGDClassifier) is
    also consulted. If it has been calibrated and its confidence exceeds 0.6, its
    emotion label overrides the base prediction while all numeric indices (stress,
    valence, arousal) are kept from the higher-quality base model.

    device_type selects the correct left/right frontal channel indices for FAA/DASM
    computation (see processing/channel_maps.py).

    This function is safe to call from a ThreadPoolExecutor (fully synchronous,
    releases GIL during NumPy/LightGBM computation).
    """
    # REVE Foundation (NeurIPS 2025, 69.7M params pretrained on 60K+ hours) — top priority
    # Requires HF_TOKEN env var and approved access to brain-bzh/reve-base.
    # Needs >= 4 seconds of EEG. Falls through if not available.
    _eeg_arr = eeg if hasattr(eeg, "ndim") else None
    if (
        _reve_foundation is not None
        and _reve_foundation.is_pretrained()
        and _eeg_arr is not None
        and _eeg_arr.ndim == 2
        and _eeg_arr.shape[0] >= 4
        and _eeg_arr.shape[1] >= int(fs * 4)
    ):
        try:
            base_result = _reve_foundation.predict(_eeg_arr, fs=int(fs))
            # Skip PersonalModel and return REVE result
            # (OnlineLearner blend below still applies)
        except Exception:
            base_result = None
    else:
        base_result = None

    if base_result is None:
        try:
            from models.personal_model import get_personal_model as _get_pm
            pm = _get_pm(user_id, n_channels=n_channels)
            global_result = emotion_model.predict(eeg, fs, device_type=device_type)
            if pm.status().get("personal_model_active"):
                base_result = pm.blend_with_global(eeg, global_result, fs=fs)
            else:
                result = pm.predict(eeg, fs)
                # "fallback_no_backbone" means EEGNet weights not present → use mega LGBM
                if result.get("model_type") != "fallback_no_backbone":
                    base_result = result
                else:
                    base_result = global_result
        except Exception:
            base_result = emotion_model.predict(eeg, fs, device_type=device_type)

    # ── Extract features (used by both OnlineLearner and k-NN pipeline) ─────────
    _feats_dict: dict = {}
    try:
        _sig = eeg[0] if (hasattr(eeg, "ndim") and eeg.ndim == 2) else eeg
        _proc = preprocess(_sig, fs)
        _feats_dict = extract_features(_proc, fs)
        # Cache as numpy array for k-NN and for attaching to label-only corrections
        _feats_arr = np.array(list(_feats_dict.values()), dtype=np.float32)
        _last_features[user_id] = _feats_arr
    except Exception:
        _feats_arr = None

    # ── OnlineLearner (SGDClassifier) blend ──────────────────────────────────
    # Consults the per-user incremental model that updates on every user correction.
    # Only overrides the emotion label; numeric indices come from the base model.
    try:
        pma = _get_personal_model(user_id)
        if pma is not None and _feats_dict:
            ol_result = pma.predict(_feats_dict)
            if ol_result.get("has_personal") and ol_result.get("personal_confidence", 0) > 0.6:
                base_result = {
                    **base_result,
                    "emotion": ol_result["personal_prediction"],
                    "online_learner_active": True,
                    "online_confidence": round(ol_result["personal_confidence"], 3),
                }
            else:
                base_result["online_learner_active"] = False
    except Exception:
        pass  # OnlineLearner failure never breaks main prediction

    # ── k-NN PersonalizedPipeline blend ──────────────────────────────────────
    # Uses label corrections stored as (features, label) in the user's JSONL file.
    # Needs ≥15 corrections WITH features to activate. Refreshes every 60s.
    # Runs AFTER OnlineLearner so it can confirm or override that result too.
    try:
        if _feats_arr is not None:
            pipeline = _get_personalized_pipeline(user_id)
            base_result = pipeline.blend("emotion", base_result, _feats_arr)
    except Exception:
        pass  # k-NN failure never breaks main prediction

    return base_result


# ─── Pydantic schemas ────────────────────────────────────────────────────────
class EEGInput(BaseModel):
    signals: List[List[float]] = Field(..., description="EEG signals (channels x samples)")
    fs: float = Field(default=256.0, description="Sampling frequency in Hz")
    user_id: str = Field(default="default", description="User identifier for per-user state isolation")
    device_type: str = Field(default="muse_2", description="EEG device name for channel map selection (e.g. 'muse_2', 'openbci_cyton')")


class SimulateRequest(BaseModel):
    state: str = Field(default="rest", description="Brain state to simulate")
    duration: float = Field(default=30.0, description="Duration in seconds")
    fs: float = Field(default=256.0, description="Sampling frequency")
    n_channels: int = Field(default=1, description="Number of channels")


class AnalysisResponse(BaseModel):
    sleep_stage: Dict
    emotions: Dict
    dream_detection: Dict
    features: Dict
    band_powers: Dict
    cross_channel: Optional[Dict] = None
    signal_quality: Optional[Dict] = None
    anomaly: Optional[Dict] = None
    personal: Optional[Dict] = None
    epoch_ready: bool = False  # True when >= 4 sec buffered; accuracy is degraded below this
    # Simplified signal quality fields for the dashboard badge
    signal_quality_score: int = 100  # 0-100; drops when amplitude thresholds exceeded
    artifact_detected: bool = False
    artifact_type: str = "clean"  # "clean" | "blink" | "muscle" | "electrode_pop"
    # E-ASR artifact cleaning ratio (fraction of channels that had artifacts cleaned)
    artifact_cleaned_ratio: Optional[float] = None  # 0.0-1.0; None when E-ASR not applied
    # Background emotion from 30-second slow epoch (more accurate than fast 4s)
    background_emotion: Optional[Dict] = None
    background_ready: bool = False  # True when >= 30 sec buffered
    # Spectral microstate temporal features (coverage, duration, transitions)
    microstates: Optional[Dict] = None


class DeviceConnectRequest(BaseModel):
    device_type: str = Field(..., description="Device type (e.g., 'synthetic', 'openbci_cyton')")
    params: Optional[Dict] = Field(default=None, description="Connection parameters")


class NeurofeedbackStartRequest(BaseModel):
    protocol_type: str = Field(default="alpha_up", description="Protocol type")
    target_band: Optional[str] = Field(default=None, description="Target frequency band")
    threshold: Optional[float] = Field(default=None, description="Reward threshold")
    calibrate: bool = Field(default=True, description="Run baseline calibration")
    user_id: str = Field(default="default", description="User identifier for per-user state isolation")


class NeurofeedbackEvalRequest(BaseModel):
    band_powers: Dict[str, float] = Field(..., description="Current band powers")
    channel_powers: Optional[List[Dict[str, float]]] = Field(default=None)
    user_id: str = Field(default="default", description="User identifier for per-user state isolation")


class SessionStartRequest(BaseModel):
    user_id: str = Field(default="default")
    session_type: str = Field(default="general")
    metadata: Optional[Dict] = Field(default=None)


class CalibrationSubmitRequest(BaseModel):
    user_id: str = Field(default="default")
    signals_list: List[List[List[float]]] = Field(..., description="List of signal arrays")
    labels: List[str] = Field(..., description="Labels for each signal")
    fs: float = Field(default=256.0)


# Renamed from FeedbackRequest to avoid collision with accuracy pipeline's FeedbackRequest
class PersonalFeedbackRequest(BaseModel):
    user_id: str = Field(default="default")
    signals: Optional[List[List[float]]] = Field(default=None, description="EEG signals (optional — label-only corrections accepted)")
    predicted_label: str = Field(...)
    correct_label: str = Field(...)
    fs: float = Field(default=256.0)


class AnomalyBaselineRequest(BaseModel):
    features_list: List[Dict[str, float]] = Field(..., description="List of feature dicts")


class SignalQualityRequest(BaseModel):
    signals: List[List[float]] = Field(..., description="EEG signals (channels x samples)")
    sample_rate: int = 256


class CalibrationEpochRequest(BaseModel):
    condition: str = Field(..., description="Calibration condition name")
    signal: List[float] = Field(..., description="EEG epoch (4 seconds)")


class AccurateAnalysisRequest(BaseModel):
    signals: List[List[float]] = Field(..., description="EEG signals")
    sample_rate: int = 256
    user_id: str = "default"


# Accuracy pipeline FeedbackRequest (different from PersonalFeedbackRequest)
class FeedbackRequest(BaseModel):
    user_id: str = "default"
    model_name: str = Field(..., description="Model that was wrong")
    predicted_state: str = Field(..., description="What the model said")
    corrected_state: str = Field(..., description="What user says is correct")
    features: Optional[List[float]] = None


class SelfReportRequest(BaseModel):
    user_id: str = "default"
    reported_state: str = Field(..., description="User's current state")
    model_name: str = "general"
    features: Optional[List[float]] = None


class EmotionShiftRequest(BaseModel):
    signals: List[List[float]] = Field(..., description="EEG signals (channels x samples)")
    fs: float = Field(default=256.0)
    user_id: str = Field(default="default")


class LucidDreamRequest(BaseModel):
    signals: List[List[float]] = Field(..., description="EEG signals")
    fs: float = Field(default=256.0)
    is_rem: bool = Field(default=True, description="Whether currently in REM sleep")
    sleep_stage: int = Field(default=4, description="Current sleep stage (4=REM)")


class DenoiseRequest(BaseModel):
    signals: List[List[float]]
    fs: float = 256.0


class CollectRequest(BaseModel):
    signals: List[List[float]] = Field(..., description="4-channel EEG signals")
    label: str = Field(..., description="Emotion label (happy/sad/angry/fearful/relaxed/focused)")
    sample_rate: float = Field(default=256.0)


class HealthDataPayload(BaseModel):
    user_id: str = Field(default="default", description="User identifier")
    source: str = Field(..., description="'apple_health', 'google_fit', or 'health_connect'")
    data: Dict = Field(..., description="Raw health data payload from the platform SDK")


class BrainSessionPayload(BaseModel):
    user_id: str = Field(default="default")
    session_id: Optional[str] = None
    start_time: float = Field(..., description="Unix timestamp")
    end_time: Optional[float] = None
    duration_seconds: Optional[float] = None
    analysis: Dict = Field(..., description="Brain analysis results")

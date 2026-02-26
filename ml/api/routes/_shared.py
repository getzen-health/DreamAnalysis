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
from models.anomaly_detector import AnomalyDetector
from processing.emotion_shift_detector import EmotionShiftDetector
from neurofeedback.protocol_engine import NeurofeedbackProtocol, PROTOCOLS as PROTOCOLS
from storage.session_recorder import SessionRecorder
from processing.signal_quality import SignalQualityChecker
from processing.calibration import CalibrationRunner
from processing.state_transitions import BrainStateEngine
from processing.confidence_calibration import ConfidenceCalibrator

# ─── Simulation & processing re-exports (used by route modules) ──────────────
from simulation.eeg_simulator import STATE_PROFILES as STATE_PROFILES, simulate_eeg as simulate_eeg
from processing.eeg_processor import (
    extract_features as extract_features,
    extract_band_powers as extract_band_powers,
    preprocess as preprocess,
    extract_features_multichannel as extract_features_multichannel,
    compute_coherence as compute_coherence,
    compute_phase_locking_value as compute_phase_locking_value,
    compute_cwt_spectrogram as compute_cwt_spectrogram,
    compute_dwt_features as compute_dwt_features,
    detect_sleep_spindles as detect_sleep_spindles,
    detect_k_complexes as detect_k_complexes,
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
        except Exception:
            _device_manager = None
    return _device_manager


def _get_personal_model(user_id: str):
    """Get or create personal model adapter for a user."""
    if user_id not in _personal_models:
        try:
            from models.online_learner import PersonalModelAdapter
            _personal_models[user_id] = PersonalModelAdapter(emotion_model, user_id)
        except Exception:
            return None
    return _personal_models[user_id]


# ─── Pydantic schemas ────────────────────────────────────────────────────────
class EEGInput(BaseModel):
    signals: List[List[float]] = Field(..., description="EEG signals (channels x samples)")
    fs: float = Field(default=256.0, description="Sampling frequency in Hz")
    user_id: str = Field(default="default", description="User identifier for per-user state isolation")


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
    signals: List[List[float]] = Field(..., description="EEG signals")
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

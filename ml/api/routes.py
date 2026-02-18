"""FastAPI routes for the ML service."""

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import numpy as np

from models.sleep_staging import SleepStagingModel
from models.emotion_classifier import EmotionClassifier
from models.dream_detector import DreamDetector
from models.flow_state_detector import FlowStateDetector
from models.creativity_detector import CreativityDetector, MemoryEncodingPredictor
from processing.signal_quality import SignalQualityChecker
from processing.calibration import UserCalibration, CalibrationRunner, CALIBRATION_STEPS
from processing.state_transitions import BrainStateEngine
from processing.confidence_calibration import ConfidenceCalibrator, add_uncertainty_labels
from processing.user_feedback import FeedbackCollector, PersonalizedPipeline
from simulation.eeg_simulator import simulate_eeg, STATE_PROFILES
from processing.eeg_processor import (
    extract_features,
    extract_band_powers,
    preprocess,
    extract_features_multichannel,
    compute_coherence,
    compute_phase_locking_value,
    compute_cwt_spectrogram,
    compute_dwt_features,
    detect_sleep_spindles,
    detect_k_complexes,
)
from processing.artifact_detector import (
    detect_eye_blinks,
    detect_muscle_artifacts,
    detect_electrode_pops,
    compute_signal_quality_index,
    auto_reject_epochs,
    ica_artifact_removal,
)
from neurofeedback.protocol_engine import NeurofeedbackProtocol, PROTOCOLS
from storage.session_recorder import SessionRecorder
from models.anomaly_detector import AnomalyDetector
from processing.connectivity import (
    compute_granger_causality,
    compute_dtf,
    compute_graph_metrics,
)
from processing.emotion_shift_detector import EmotionShiftDetector
from models.drowsiness_detector import DrowsinessDetector
from models.cognitive_load_estimator import CognitiveLoadEstimator
from models.attention_classifier import AttentionClassifier
from models.stress_detector import StressDetector
from models.lucid_dream_detector import LucidDreamDetector
from models.meditation_classifier import MeditationClassifier
from processing.spiritual_energy import (
    compute_chakra_activations,
    compute_chakra_balance,
    compute_meditation_depth,
    compute_aura_energy,
    compute_kundalini_flow,
    compute_prana_balance,
    compute_consciousness_level,
    compute_third_eye_activation,
    full_spiritual_analysis,
    CHAKRAS,
    CONSCIOUSNESS_LEVELS,
)

router = APIRouter()

MODEL_DIR = Path("models/saved")
BENCHMARK_DIR = Path("benchmarks")


def _numpy_safe(obj):
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _numpy_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_numpy_safe(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
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


# Initialize models with auto-discovery
sleep_model = SleepStagingModel(model_path=_find_model("sleep_staging_model"))
emotion_model = EmotionClassifier(model_path=_find_model("emotion_classifier_model"))
dream_model = DreamDetector(model_path=_find_model("dream_detector_model"))
flow_model = FlowStateDetector(model_path=_find_model("flow_state_model"))
creativity_model = CreativityDetector(model_path=_find_model("creativity_model"))
memory_model = MemoryEncodingPredictor(model_path=_find_model("memory_encoding_model"))

# New cognitive models (Phase 15)
drowsiness_model = DrowsinessDetector(model_path=_find_model("drowsiness_model"))
cognitive_load_model = CognitiveLoadEstimator(model_path=_find_model("cognitive_load_model"))
attention_model = AttentionClassifier(model_path=_find_model("attention_model"))
stress_model = StressDetector(model_path=_find_model("stress_model"))
lucid_dream_model = LucidDreamDetector(model_path=_find_model("lucid_dream_model"))
meditation_model = MeditationClassifier(model_path=_find_model("meditation_model"))

# Emotion shift detectors (per-user)
_emotion_shift_detectors: Dict[str, EmotionShiftDetector] = {}

# Hardware manager (lazy init)
_device_manager = None

# Neurofeedback session (per-server singleton for now)
_nf_protocol: Optional[NeurofeedbackProtocol] = None

# Session recorder
_session_recorder = SessionRecorder()

# Anomaly detector
_anomaly_detector = AnomalyDetector()

# Personal model adapters cache
_personal_models: Dict[str, object] = {}


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
            _personal_models[user_id] = PersonalModelAdapter(sleep_model, user_id)
        except Exception:
            return None
    return _personal_models[user_id]


class EEGInput(BaseModel):
    signals: List[List[float]] = Field(..., description="EEG signals (channels x samples)")
    fs: float = Field(default=256.0, description="Sampling frequency in Hz")


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


class DeviceConnectRequest(BaseModel):
    device_type: str = Field(..., description="Device type (e.g., 'synthetic', 'openbci_cyton')")
    params: Optional[Dict] = Field(default=None, description="Connection parameters")


class NeurofeedbackStartRequest(BaseModel):
    protocol_type: str = Field(default="alpha_up", description="Protocol type")
    target_band: Optional[str] = Field(default=None, description="Target frequency band")
    threshold: Optional[float] = Field(default=None, description="Reward threshold")
    calibrate: bool = Field(default=True, description="Run baseline calibration")


class NeurofeedbackEvalRequest(BaseModel):
    band_powers: Dict[str, float] = Field(..., description="Current band powers")
    channel_powers: Optional[List[Dict[str, float]]] = Field(default=None)


class SessionStartRequest(BaseModel):
    user_id: str = Field(default="default")
    session_type: str = Field(default="general")
    metadata: Optional[Dict] = Field(default=None)


class CalibrationSubmitRequest(BaseModel):
    user_id: str = Field(default="default")
    signals_list: List[List[List[float]]] = Field(..., description="List of signal arrays")
    labels: List[str] = Field(..., description="Labels for each signal")
    fs: float = Field(default=256.0)


class FeedbackRequest(BaseModel):
    user_id: str = Field(default="default")
    signals: List[List[float]] = Field(..., description="EEG signals")
    predicted_label: str = Field(...)
    correct_label: str = Field(...)
    fs: float = Field(default=256.0)


class AnomalyBaselineRequest(BaseModel):
    features_list: List[Dict[str, float]] = Field(..., description="List of feature dicts")


# ─── Core Analysis Endpoints ─────────────────────────────────────────────


@router.post("/analyze-eeg", response_model=AnalysisResponse)
async def analyze_eeg(input_data: EEGInput):
    """Run all 3 models on EEG input with multi-channel support."""
    try:
        signals = np.array(input_data.signals)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        fs = input_data.fs
        n_channels = signals.shape[0]

        if n_channels > 1:
            avg_features = extract_features_multichannel(signals, fs, method="average")
            avg_signal = np.mean(signals, axis=0)
            eeg = avg_signal
        else:
            eeg = signals[0]

        sleep_result = sleep_model.predict(eeg, fs)
        # Pass multichannel data to emotion model for DEAP-trained model
        emotion_result = emotion_model.predict(signals if n_channels >= 2 else eeg, fs)
        dream_result = dream_model.predict(eeg, fs)

        processed = preprocess(eeg, fs)
        features = extract_features(processed, fs)
        bands = extract_band_powers(processed, fs)

        # Cross-channel metrics
        cross_channel = None
        if n_channels > 1:
            try:
                coherence_alpha = compute_coherence(signals, fs, "alpha")
                plv_alpha = compute_phase_locking_value(signals, fs, "alpha")
                cross_channel = {
                    "n_channels": n_channels,
                    "coherence_alpha": coherence_alpha,
                    "plv_alpha": plv_alpha,
                }
            except Exception:
                cross_channel = {"n_channels": n_channels}

        # Signal quality (Phase 6)
        signal_quality = None
        try:
            channel_sqis = [compute_signal_quality_index(signals[ch], fs) for ch in range(n_channels)]
            avg_sqi = float(np.mean(channel_sqis))

            artifacts = []
            blinks = detect_eye_blinks(eeg, fs)
            if blinks:
                artifacts.append("eye_blink")
            muscle = detect_muscle_artifacts(eeg, fs)
            if muscle:
                artifacts.append("muscle")
            pops = detect_electrode_pops(eeg, fs)
            if pops:
                artifacts.append("electrode_pop")

            _, rejected = auto_reject_epochs(signals, fs)
            total_epochs = max(1, int(signals.shape[1] / (5.0 * fs)))
            clean_ratio = 1.0 - len(rejected) / total_epochs

            signal_quality = {
                "sqi": avg_sqi,
                "artifacts_detected": artifacts,
                "clean_ratio": float(clean_ratio),
                "rejected_epochs": rejected,
                "channel_quality": channel_sqis,
            }
        except Exception:
            pass

        # Anomaly detection (Phase 11)
        anomaly = None
        try:
            anomaly_result = _anomaly_detector.detect_anomaly(features)
            spikes = AnomalyDetector.detect_spike(eeg, fs)
            seizure = AnomalyDetector.detect_seizure_pattern(eeg, fs)
            alert_level = AnomalyDetector.get_alert_level(
                anomaly_result["anomaly_score"], seizure["seizure_probability"]
            )

            anomaly = {
                "is_anomaly": anomaly_result["is_anomaly"],
                "anomaly_score": anomaly_result["anomaly_score"],
                "spikes_detected": len(spikes),
                "seizure_probability": seizure["seizure_probability"],
                "alert_level": alert_level,
            }
        except Exception:
            pass

        # Personal model blending (Phase 9)
        personal = None
        try:
            pm = _get_personal_model("default")
            if pm:
                personal = pm.predict(features)
        except Exception:
            pass

        return AnalysisResponse(
            sleep_stage=sleep_result,
            emotions=emotion_result,
            dream_detection=dream_result,
            features=features,
            band_powers=bands,
            cross_channel=cross_channel,
            signal_quality=signal_quality,
            anomaly=anomaly,
            personal=personal,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simulate-eeg")
async def simulate_eeg_endpoint(request: SimulateRequest):
    """Generate realistic simulated EEG for a given brain state."""
    if request.state not in STATE_PROFILES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown state '{request.state}'. Available: {list(STATE_PROFILES.keys())}",
        )

    result = simulate_eeg(
        state=request.state,
        duration=request.duration,
        fs=request.fs,
        n_channels=request.n_channels,
    )

    eeg = np.array(result["signals"][0])
    sleep_result = sleep_model.predict(eeg, request.fs)
    emotion_result = emotion_model.predict(eeg, request.fs)
    dream_result = dream_model.predict(eeg, request.fs)
    flow_result = flow_model.predict(eeg, request.fs)
    creativity_result = creativity_model.predict(eeg, request.fs)
    memory_result = memory_model.predict(eeg, request.fs)

    drowsiness_result = drowsiness_model.predict(eeg, request.fs)
    cognitive_load_result = cognitive_load_model.predict(eeg, request.fs)
    attention_result = attention_model.predict(eeg, request.fs)
    stress_result = stress_model.predict(eeg, request.fs)
    lucid_result = lucid_dream_model.predict(
        eeg, request.fs,
        is_rem=(sleep_result.get("stage") == "REM"),
        sleep_stage=sleep_result.get("stage_index", 0),
    )
    meditation_result = meditation_model.predict(eeg, request.fs)

    return {
        **result,
        "analysis": {
            "sleep_stage": sleep_result,
            "emotions": emotion_result,
            "dream_detection": dream_result,
            "flow_state": flow_result,
            "creativity": creativity_result,
            "memory_encoding": memory_result,
            "drowsiness": drowsiness_result,
            "cognitive_load": cognitive_load_result,
            "attention": attention_result,
            "stress": stress_result,
            "lucid_dream": lucid_result,
            "meditation": meditation_result,
        },
    }


@router.get("/models/status")
async def models_status():
    """Model health check with type information."""
    return {
        "sleep_staging": {
            "loaded": True,
            "type": sleep_model.model_type,
            "classes": ["Wake", "N1", "N2", "N3", "REM"],
        },
        "emotion_classifier": {
            "loaded": True,
            "type": emotion_model.model_type,
            "classes": ["happy", "sad", "angry", "fearful", "relaxed", "focused"],
        },
        "dream_detector": {
            "loaded": True,
            "type": dream_model.model_type,
            "classes": ["dreaming", "not_dreaming"],
        },
        "flow_state": {
            "loaded": True,
            "type": flow_model.model_type,
            "classes": ["no_flow", "micro_flow", "flow", "deep_flow"],
        },
        "creativity": {
            "loaded": True,
            "type": creativity_model.model_type,
            "classes": ["analytical", "transitional", "creative", "insight"],
        },
        "memory_encoding": {
            "loaded": True,
            "type": memory_model.model_type,
            "classes": ["poor_encoding", "weak_encoding", "active_encoding", "deep_encoding"],
        },
        "drowsiness": {
            "loaded": True,
            "type": drowsiness_model.model_type,
            "classes": ["alert", "drowsy", "sleepy"],
        },
        "cognitive_load": {
            "loaded": True,
            "type": cognitive_load_model.model_type,
            "classes": ["low", "moderate", "high"],
        },
        "attention": {
            "loaded": True,
            "type": attention_model.model_type,
            "classes": ["distracted", "passive", "focused", "hyperfocused"],
        },
        "stress": {
            "loaded": True,
            "type": stress_model.model_type,
            "classes": ["relaxed", "mild", "moderate", "high"],
        },
        "lucid_dream": {
            "loaded": True,
            "type": lucid_dream_model.model_type,
            "classes": ["non_lucid", "pre_lucid", "lucid", "controlled"],
        },
        "meditation": {
            "loaded": True,
            "type": meditation_model.model_type,
            "classes": ["surface", "light", "moderate", "deep", "transcendent"],
        },
        "available_states": list(STATE_PROFILES.keys()),
    }


@router.get("/models/benchmarks")
async def models_benchmarks():
    """Serve benchmark results for all models."""
    benchmarks = {}
    if BENCHMARK_DIR.exists():
        for json_file in BENCHMARK_DIR.glob("*_benchmark.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    benchmarks[data.get("model_name", json_file.stem)] = data
            except Exception:
                continue
    return benchmarks


# ─── Wavelet Analysis (Phase 5) ──────────────────────────────────────────


@router.post("/analyze-wavelet")
async def analyze_wavelet(data: EEGInput):
    """Wavelet time-frequency analysis: CWT spectrogram, DWT energies, event detection."""
    try:
        signals = np.array(data.signals)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        eeg = signals[0]
        fs = data.fs

        # Need at least 34 samples for bandpass filter (padlen=33)
        if len(eeg) < 34:
            raise HTTPException(status_code=422, detail=f"Signal too short ({len(eeg)} samples). Need at least 34.")

        processed = preprocess(eeg, fs)

        spectrogram = compute_cwt_spectrogram(processed, fs)
        dwt_energies = compute_dwt_features(processed, fs)
        spindles = detect_sleep_spindles(processed, fs)
        k_complexes = detect_k_complexes(processed, fs)

        return {
            "spectrogram": spectrogram,
            "dwt_energies": dwt_energies,
            "events": {
                "sleep_spindles": spindles,
                "k_complexes": k_complexes,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Signal Quality & Cleaning (Phase 6) ─────────────────────────────────


@router.post("/clean-signal")
async def clean_signal(data: EEGInput):
    """ICA-based artifact removal returning cleaned signals + report."""
    try:
        signals = np.array(data.signals)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        fs = data.fs
        result = ica_artifact_removal(signals, fs)

        # Compute before/after SQI
        before_sqi = [compute_signal_quality_index(signals[ch], fs) for ch in range(signals.shape[0])]
        cleaned = result["cleaned_signals"]
        after_sqi = [compute_signal_quality_index(cleaned[ch], fs) for ch in range(cleaned.shape[0])]

        return {
            "cleaned_signals": cleaned.tolist(),
            "removed_components": result["removed_components"],
            "n_components": result["n_components"],
            "before_sqi": before_sqi,
            "after_sqi": after_sqi,
            "improvement": float(np.mean(after_sqi) - np.mean(before_sqi)),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Neurofeedback (Phase 7) ─────────────────────────────────────────────


@router.get("/neurofeedback/protocols")
async def list_protocols():
    """List available neurofeedback protocols."""
    return {
        key: {"name": p["name"], "description": p["description"]}
        for key, p in PROTOCOLS.items()
    }


@router.post("/neurofeedback/start")
async def start_neurofeedback(request: NeurofeedbackStartRequest):
    """Start a neurofeedback session."""
    global _nf_protocol
    _nf_protocol = NeurofeedbackProtocol(
        protocol_type=request.protocol_type,
        target_band=request.target_band,
        threshold=request.threshold,
    )

    if request.calibrate:
        _nf_protocol.start_calibration()
        return {"status": "calibrating", "protocol": request.protocol_type}

    _nf_protocol.start()
    return {"status": "active", "protocol": request.protocol_type}


@router.post("/neurofeedback/evaluate")
async def evaluate_neurofeedback(request: NeurofeedbackEvalRequest):
    """Evaluate current EEG against the active neurofeedback protocol."""
    global _nf_protocol
    if _nf_protocol is None:
        raise HTTPException(status_code=400, detail="No active neurofeedback session")

    if _nf_protocol.is_calibrating:
        done = _nf_protocol.add_calibration_sample(request.band_powers)
        progress = len(_nf_protocol.baseline_samples) / 30.0
        if done:
            return {"status": "calibration_complete", "baseline": _nf_protocol.baseline, "progress": 1.0}
        return {"status": "calibrating", "progress": float(progress)}

    result = _nf_protocol.evaluate(request.band_powers, request.channel_powers)
    return {"status": "active", **result}


@router.post("/neurofeedback/stop")
async def stop_neurofeedback():
    """Stop the current neurofeedback session and return stats."""
    global _nf_protocol
    if _nf_protocol is None:
        raise HTTPException(status_code=400, detail="No active neurofeedback session")

    stats = _nf_protocol.stop()
    _nf_protocol = None
    return {"status": "stopped", "stats": stats}


# ─── Session Recording (Phase 8) ─────────────────────────────────────────


@router.post("/sessions/start")
async def start_session(request: SessionStartRequest):
    """Start recording an EEG session."""
    session_id = _session_recorder.start_recording(
        user_id=request.user_id,
        session_type=request.session_type,
        metadata=request.metadata,
    )
    return {"status": "recording", "session_id": session_id}


@router.post("/sessions/stop")
async def stop_session():
    """Stop the current recording and return summary."""
    if not _session_recorder.is_recording:
        raise HTTPException(status_code=400, detail="No active recording")

    summary = _session_recorder.stop_recording()
    return summary


@router.get("/sessions")
async def list_sessions(user_id: Optional[str] = None, session_type: Optional[str] = None):
    """List saved sessions."""
    return SessionRecorder.list_sessions(user_id, session_type)


# Session analytics (MUST be above /sessions/{session_id} to avoid catch-all)
from storage.session_analytics import compare_sessions, get_session_trends, get_weekly_report


@router.get("/sessions/trends")
async def session_trends(user_id: Optional[str] = None, last_n: int = 20):
    """Get trends across recent sessions."""
    return _numpy_safe(get_session_trends(user_id, last_n))


@router.get("/sessions/weekly-report")
async def weekly_report(user_id: Optional[str] = None):
    """Generate a weekly progress report comparing this week to last week."""
    return _numpy_safe(get_weekly_report(user_id))


@router.get("/sessions/compare/{session_a}/{session_b}")
async def compare_two_sessions(session_a: str, session_b: str):
    """Compare two sessions side-by-side with per-metric deltas."""
    result = compare_sessions(session_a, session_b)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return _numpy_safe(result)


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get full session data."""
    data = SessionRecorder.load_session(session_id)
    if "error" in data:
        raise HTTPException(status_code=404, detail=data["error"])
    return data


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    deleted = SessionRecorder.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


@router.get("/sessions/{session_id}/export")
async def export_session(session_id: str, format: str = "csv"):
    """Export session data as CSV."""
    data = SessionRecorder.export_session(session_id, format)
    if data is None:
        raise HTTPException(status_code=404, detail="Session not found or export failed")

    media_type = "text/csv" if format == "csv" else "application/octet-stream"
    filename = f"session_{session_id}.{format}"
    return Response(
        content=data,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ─── Muse 2 Data Collection (for training) ───────────────────────────────

_COLLECT_DIR = Path("data/muse2_collected")


class CollectRequest(BaseModel):
    signals: List[List[float]] = Field(..., description="4-channel EEG signals")
    label: str = Field(..., description="Emotion label (happy/sad/angry/fearful/relaxed/focused)")
    sample_rate: float = Field(default=256.0)


@router.post("/collect-training-data")
async def collect_training_data(data: CollectRequest):
    """Save labeled EEG data from live Muse 2 sessions for future training.

    Call this when the user confirms their current emotional state,
    so we can build a personal training dataset over time.
    """
    valid_labels = ["happy", "sad", "angry", "fearful", "relaxed", "focused"]
    if data.label not in valid_labels:
        raise HTTPException(status_code=422, detail=f"Invalid label. Must be one of: {valid_labels}")

    signals = np.array(data.signals)
    if signals.shape[0] < 1 or signals.shape[1] < 34:
        raise HTTPException(status_code=422, detail="Signals too short. Need at least 34 samples per channel.")

    _COLLECT_DIR.mkdir(parents=True, exist_ok=True)

    import time
    filename = f"{data.label}_{int(time.time()*1000)}.json"
    filepath = _COLLECT_DIR / filename

    record = {
        "signals": [ch.tolist() if isinstance(ch, np.ndarray) else ch for ch in data.signals],
        "sample_rate": data.sample_rate,
        "label": data.label,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_channels": signals.shape[0],
        "n_samples": signals.shape[1],
    }
    filepath.write_text(json.dumps(record))

    total_files = len(list(_COLLECT_DIR.glob("*.json")))
    return {
        "saved": filename,
        "total_collected": total_files,
        "label": data.label,
    }


@router.get("/collected-data/stats")
async def collected_data_stats():
    """Get statistics on collected Muse 2 training data."""
    if not _COLLECT_DIR.exists():
        return {"total": 0, "per_label": {}, "ready_to_train": False}

    files = list(_COLLECT_DIR.glob("*.json"))
    per_label = {}
    for f in files:
        label = f.stem.rsplit("_", 1)[0]
        per_label[label] = per_label.get(label, 0) + 1

    return {
        "total": len(files),
        "per_label": per_label,
        "ready_to_train": len(files) >= 30,
    }


# ─── Calibration & Personal Models (Phase 9) ─────────────────────────────


@router.post("/calibration/start")
async def start_calibration():
    """Start a calibration session (returns prompts for the user)."""
    return {
        "status": "started",
        "steps": [
            {"step": 1, "instruction": "Close your eyes and relax for 30 seconds", "label": "relaxed", "duration_sec": 30},
            {"step": 2, "instruction": "Focus on counting backwards from 100", "label": "focused", "duration_sec": 30},
            {"step": 3, "instruction": "Think about something stressful", "label": "stressed", "duration_sec": 30},
        ],
    }


@router.post("/calibration/submit")
async def submit_calibration(request: CalibrationSubmitRequest):
    """Submit labeled calibration data to train personal model."""
    pm = _get_personal_model(request.user_id)
    if pm is None:
        raise HTTPException(status_code=500, detail="Failed to initialize personal model")

    signals_list = [np.array(s) for s in request.signals_list]
    result = pm.calibrate(signals_list, request.labels, request.fs)
    return result


@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """User corrects a prediction — triggers incremental model update."""
    pm = _get_personal_model(request.user_id)
    if pm is None:
        raise HTTPException(status_code=500, detail="Failed to initialize personal model")

    signal = np.array(request.signals)
    result = pm.adapt(signal, request.predicted_label, request.correct_label, request.fs)
    return result


@router.get("/calibration/status")
async def calibration_status(user_id: str = "default"):
    """Check personal model calibration status."""
    pm = _get_personal_model(user_id)
    if pm is None:
        return {"calibrated": False, "n_samples": 0, "personal_accuracy": 0.0, "classes": []}
    return pm.get_calibration_status()


# ─── Brain Connectivity (Phase 10) ───────────────────────────────────────


@router.post("/analyze-connectivity")
async def analyze_connectivity(data: EEGInput):
    """Compute brain network connectivity and graph metrics."""
    try:
        signals = np.array(data.signals)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        fs = data.fs
        n_channels = signals.shape[0]

        if n_channels < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 channels for connectivity analysis")

        # Pairwise correlation matrix
        corr = np.corrcoef(signals)
        np.fill_diagonal(corr, 0)
        connectivity_matrix = np.abs(corr)

        # Graph metrics
        graph_metrics = compute_graph_metrics(connectivity_matrix)

        # Directed flow (Granger causality)
        gc = compute_granger_causality(signals, fs)

        # DTF
        dtf = compute_dtf(signals, fs)

        return {
            "connectivity_matrix": connectivity_matrix.tolist(),
            "graph_metrics": graph_metrics,
            "directed_flow": {
                "granger": gc,
                "dtf_matrix": dtf["dtf_matrix"],
                "dominant_direction": dtf["dominant_direction"],
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Anomaly Detection (Phase 11) ────────────────────────────────────────


@router.post("/anomaly/set-baseline")
async def set_anomaly_baseline(request: AnomalyBaselineRequest):
    """Train anomaly detector on user's normal EEG features."""
    result = _anomaly_detector.fit_baseline(request.features_list)
    return result


# ─── Device Management Endpoints ─────────────────────────────────────────


@router.get("/devices")
async def list_devices():
    """List available EEG devices."""
    manager = _get_device_manager()
    if manager is None:
        return {
            "brainflow_available": False,
            "devices": [],
            "message": "BrainFlow not installed. Install with: pip install brainflow",
        }

    devices = manager.discover_devices()
    return {
        "brainflow_available": True,
        "devices": devices,
        "connected": manager.is_connected,
    }


@router.post("/devices/connect")
async def connect_device(request: DeviceConnectRequest):
    """Connect to an EEG device."""
    manager = _get_device_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="BrainFlow not available")

    try:
        result = manager.connect(request.device_type, request.params or {})
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/devices/disconnect")
async def disconnect_device():
    """Disconnect from the current EEG device."""
    manager = _get_device_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="BrainFlow not available")

    try:
        manager.disconnect()
        return {"status": "disconnected"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/devices/status")
async def device_status():
    """Get current device status."""
    manager = _get_device_manager()
    if manager is None:
        return {"connected": False, "brainflow_available": False}

    return {
        "connected": manager.is_connected,
        "streaming": manager.is_streaming,
        "device_type": manager.current_device_type,
        "n_channels": manager.n_channels,
        "sample_rate": manager.sample_rate,
        "brainflow_available": True,
    }


@router.post("/devices/start-stream")
async def start_stream():
    """Start data streaming from the connected device."""
    manager = _get_device_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="BrainFlow not available")
    if not manager.is_connected:
        raise HTTPException(status_code=400, detail="No device connected")

    try:
        manager.start_streaming()
        return {"status": "streaming", "sample_rate": manager.sample_rate}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/devices/stop-stream")
async def stop_stream():
    """Stop data streaming."""
    manager = _get_device_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="BrainFlow not available")

    try:
        manager.stop_streaming()
        return {"status": "stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Dataset management endpoints
# ---------------------------------------------------------------------------


@router.post("/datasets/download-deap")
async def download_deap():
    """Download DEAP dataset from Kaggle (requires ~/.kaggle/kaggle.json)."""
    from training.data_loaders import download_deap_kaggle
    try:
        path = download_deap_kaggle()
        dat_count = len(list(path.glob("s*.dat")))
        return {"status": "success", "path": str(path), "files": dat_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/datasets/download-dens")
async def download_dens():
    """Download/sync DENS dataset from OpenNeuro."""
    from training.data_loaders import download_dens_openneuro
    try:
        path = download_dens_openneuro()
        subjects = len(list(path.glob("sub-*")))
        return {"status": "success", "path": str(path), "subjects": subjects}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Health Integration (Apple HealthKit / Google Fit / Health Connect)
# ---------------------------------------------------------------------------

import sqlite3
from health.correlation_engine import HealthBrainDB
from health.apple_health import parse_healthkit_payload, format_brain_data_for_healthkit
from health.google_fit import parse_google_fit_payload, parse_health_connect_payload

_health_db = HealthBrainDB()


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


@router.post("/health/ingest")
async def ingest_health_data(payload: HealthDataPayload):
    """Ingest health data from Apple Health, Google Fit, or Health Connect.

    The mobile app reads health data using platform SDKs and sends it here.
    We parse, normalize, and store it for brain-health correlation.
    """
    try:
        if payload.source == "apple_health":
            parsed = parse_healthkit_payload(payload.data)
        elif payload.source == "google_fit":
            parsed = parse_google_fit_payload(payload.data)
        elif payload.source == "health_connect":
            parsed = parse_health_connect_payload(payload.data)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown source '{payload.source}'. Use 'apple_health', 'google_fit', or 'health_connect'."
            )

        _health_db.store_health_samples(
            user_id=payload.user_id,
            metric=parsed["metric"],
            samples=parsed["samples"],
        )

        return {
            "status": "stored",
            "metric": parsed["metric"],
            "samples_stored": parsed["count"],
            "source": payload.source,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/health/brain-session")
async def store_brain_session(payload: BrainSessionPayload):
    """Store a brain state session for correlation with health data.

    Call this after each EEG analysis session to build the
    brain-health correlation database over time.
    """
    try:
        session_data = {
            "session_id": payload.session_id,
            "start_time": payload.start_time,
            "end_time": payload.end_time,
            "duration_seconds": payload.duration_seconds,
            **payload.analysis,
        }
        _health_db.store_brain_session(payload.user_id, session_data)
        return {"status": "stored", "session_id": payload.session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/daily-summary/{user_id}")
async def daily_summary(user_id: str, date: Optional[str] = None):
    """Get combined brain + health summary for a day.

    Combines all brain state sessions and health metrics into one overview.
    """
    try:
        return _health_db.get_daily_summary(user_id, date)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/insights/{user_id}")
async def get_insights(user_id: str, days: int = 30):
    """Generate personalized brain-health correlation insights.

    Examples of insights generated:
    - "You enter flow 2x more after morning exercise"
    - "Your creativity peaks at 2pm"
    - "Better sleep leads to stronger memory encoding next day"
    - "High HRV days correlate with more creative thinking"
    """
    try:
        insights = _health_db.generate_insights(user_id, days)
        return {"user_id": user_id, "period_days": days, "insights": insights}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/trends/{user_id}")
async def brain_trends(user_id: str, days: int = 30):
    """Get brain state trends over time.

    Daily averages for flow, creativity, memory encoding, valence, arousal.
    """
    try:
        return _health_db.get_brain_trends(user_id, days)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/health/export-to-healthkit/{user_id}")
async def export_to_healthkit(user_id: str):
    """Export brain data formatted for Apple HealthKit.

    Returns HealthKit-compatible samples for the iOS app to write
    back to Apple Health (mindful minutes from flow, EEG sleep stages).
    """
    try:
        with sqlite3.connect(_health_db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            sessions = conn.execute(
                """SELECT * FROM brain_sessions
                   WHERE user_id = ? ORDER BY start_time DESC LIMIT 100""",
                (user_id,)
            ).fetchall()

        exports = []
        for session in sessions:
            session_dict = dict(session)
            brain_session = {
                "start_time": session_dict.get("start_time"),
                "end_time": session_dict.get("end_time"),
                "flow_state": {
                    "state": session_dict.get("flow_state"),
                    "flow_score": session_dict.get("flow_score"),
                },
                "sleep_stage": {
                    "stage": session_dict.get("sleep_stage"),
                    "confidence": session_dict.get("sleep_confidence"),
                },
            }
            exports.extend(format_brain_data_for_healthkit(brain_session))

        return {
            "user_id": user_id,
            "healthkit_samples": exports,
            "count": len(exports),
            "instructions": "Pass these samples to HKHealthStore.save() on iOS",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/supported-metrics")
async def supported_metrics():
    """List all supported health metrics from Apple Health and Google Fit."""
    from health.apple_health import HEALTHKIT_TYPE_MAP, EXPORT_TYPES
    from health.google_fit import GOOGLE_FIT_TYPE_MAP, HEALTH_CONNECT_TYPE_MAP

    return {
        "apple_health": {
            "import": HEALTHKIT_TYPE_MAP,
            "export": EXPORT_TYPES,
        },
        "google_fit": GOOGLE_FIT_TYPE_MAP,
        "health_connect": HEALTH_CONNECT_TYPE_MAP,
        "brain_metrics": {
            "flow_state": ["no_flow", "micro_flow", "flow", "deep_flow"],
            "creativity": ["analytical", "transitional", "creative", "insight"],
            "memory_encoding": ["poor_encoding", "weak_encoding", "active_encoding", "deep_encoding"],
            "sleep_staging": ["Wake", "N1", "N2", "N3", "REM"],
            "emotions": ["happy", "sad", "angry", "fearful", "relaxed", "focused"],
            "dream_detection": ["dreaming", "not_dreaming"],
            "drowsiness": ["alert", "drowsy", "sleepy"],
            "cognitive_load": ["low", "moderate", "high"],
            "attention": ["distracted", "passive", "focused", "hyperfocused"],
            "stress": ["relaxed", "mild", "moderate", "high"],
            "lucid_dream": ["non_lucid", "pre_lucid", "lucid", "controlled"],
            "meditation": ["surface", "light", "moderate", "deep", "transcendent"],
        },
    }


# ═══════════════════════════════════════════════════════════════
#  ACCURACY PIPELINE ENDPOINTS
# ═══════════════════════════════════════════════════════════════

# Singletons for accuracy modules
_quality_checker = SignalQualityChecker(fs=256)
_state_engine = BrainStateEngine()
_confidence_cal = ConfidenceCalibrator()
_calibration_runners: Dict[str, CalibrationRunner] = {}


# ── Signal Quality ──

class SignalQualityRequest(BaseModel):
    signals: List[List[float]] = Field(..., description="EEG signals (channels x samples)")
    sample_rate: int = 256


@router.post("/signal-quality")
async def check_signal_quality(req: SignalQualityRequest):
    """Check EEG signal quality before analysis.

    Returns quality score (0-1), usability flag, and per-channel details.
    Noisy signals should be rejected before running models.
    """
    checker = SignalQualityChecker(fs=req.sample_rate)
    signals = np.array(req.signals)

    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    if signals.shape[0] == 1:
        result = checker.check_quality(signals[0])
    else:
        result = checker.check_multichannel(signals)

    return _numpy_safe(result)


# ── Confidence & Reliability ──

@router.get("/confidence/reliability")
async def get_model_reliability():
    """Get reliability assessment for all 6 models.

    Shows expected accuracy, calibration method, and reliability tier
    (high/medium/low) for each model.
    """
    return _confidence_cal.get_all_reliability()


@router.post("/confidence/calibrate")
async def calibrate_confidence(model_name: str, raw_confidence: float):
    """Calibrate a single confidence score.

    Transforms raw model confidence into an honest, calibrated probability.
    """
    result = _confidence_cal.calibrate(model_name, raw_confidence)
    return result


# ── Calibration Protocol ──

@router.get("/calibration/steps")
async def get_calibration_steps():
    """Get the 4-step calibration protocol instructions.

    Each step has a name, instruction, duration (seconds), and purpose.
    """
    return {"steps": CALIBRATION_STEPS, "total_duration_seconds": 120}


@router.post("/calibration/start/{user_id}")
async def start_user_calibration(user_id: str, sample_rate: int = 256):
    """Start a new calibration session for a user."""
    _calibration_runners[user_id] = CalibrationRunner(fs=sample_rate)
    return {
        "status": "started",
        "user_id": user_id,
        "steps": CALIBRATION_STEPS,
    }


class CalibrationEpochRequest(BaseModel):
    condition: str = Field(..., description="Calibration condition name")
    signal: List[float] = Field(..., description="EEG epoch (4 seconds)")


@router.post("/calibration/add-epoch/{user_id}")
async def add_calibration_epoch(user_id: str, req: CalibrationEpochRequest):
    """Add an EEG epoch to the calibration for a specific condition."""
    runner = _calibration_runners.get(user_id)
    if runner is None:
        raise HTTPException(404, f"No calibration session for user {user_id}")

    signal = np.array(req.signal)

    # Check quality first
    checker = SignalQualityChecker(fs=runner.fs)
    quality = checker.check_quality(signal)

    if not quality["is_usable"]:
        return {
            "status": "rejected",
            "reason": "Signal quality too low",
            "quality": quality,
        }

    runner.add_epoch(req.condition, signal)
    progress = runner.get_progress()

    return {
        "status": "accepted",
        "quality": quality,
        "progress": progress,
    }


@router.post("/calibration/complete/{user_id}")
async def complete_calibration(user_id: str):
    """Complete calibration and compute per-user baselines."""
    runner = _calibration_runners.get(user_id)
    if runner is None:
        raise HTTPException(404, f"No calibration session for user {user_id}")

    progress = runner.get_progress()
    if not progress["is_complete"]:
        return {
            "status": "incomplete",
            "progress": progress,
            "message": "Need at least 3 epochs per condition",
        }

    cal = runner.compute_calibration(user_id)
    del _calibration_runners[user_id]

    return {
        "status": "complete",
        "alpha_reactivity": cal.alpha_reactivity,
        "beta_reactivity": cal.beta_reactivity,
        "theta_alpha_ratio_rest": cal.theta_alpha_ratio_rest,
        "global_band_means": cal.global_band_means,
    }


@router.get("/calibration/status/{user_id}")
async def get_calibration_status(user_id: str):
    """Check if a user has a saved calibration profile."""
    cal = UserCalibration.load(user_id)

    # Check for in-progress calibration
    runner = _calibration_runners.get(user_id)
    in_progress = None
    if runner:
        in_progress = runner.get_progress()

    return {
        "is_calibrated": cal.is_calibrated,
        "calibrated_at": cal.calibrated_at,
        "alpha_reactivity": cal.alpha_reactivity,
        "in_progress": in_progress,
    }


# ── State Transitions ──

@router.get("/state-engine/summary")
async def get_state_engine_summary():
    """Get current state of all temporal smoothing trackers."""
    return _state_engine.get_summary()


@router.get("/state-engine/coherence")
async def get_state_coherence():
    """Check if current brain states are physiologically coherent."""
    return _state_engine.get_cross_state_coherence()


# ── User Feedback & Personalization ──

class FeedbackRequest(BaseModel):
    user_id: str = "default"
    model_name: str = Field(..., description="Model that was wrong")
    predicted_state: str = Field(..., description="What the model said")
    corrected_state: str = Field(..., description="What user says is correct")
    features: Optional[List[float]] = None


@router.post("/feedback/correction")
async def submit_correction(req: FeedbackRequest):
    """Submit a state correction (model said X, but I was actually Y)."""
    fc = FeedbackCollector(req.user_id)
    features = np.array(req.features) if req.features else None
    fc.record_state_correction(
        req.model_name, req.predicted_state, req.corrected_state, features
    )
    stats = fc.get_feedback_stats()
    return {
        "status": "recorded",
        "total_feedback": stats["total_entries"],
        "can_personalize": stats.get("can_personalize", {}),
    }


class SelfReportRequest(BaseModel):
    user_id: str = "default"
    reported_state: str = Field(..., description="User's current state")
    model_name: str = "general"
    features: Optional[List[float]] = None


@router.post("/feedback/self-report")
async def submit_self_report(req: SelfReportRequest):
    """Submit a self-report of current state (no model prediction needed)."""
    fc = FeedbackCollector(req.user_id)
    features = np.array(req.features) if req.features else None
    fc.record_self_report(req.reported_state, req.model_name, features)
    return {"status": "recorded"}


@router.get("/personalization/status/{user_id}")
async def get_personalization_status(user_id: str):
    """Get personalization status for a user."""
    pipeline = PersonalizedPipeline(user_id)
    pipeline.update_from_feedback()
    return pipeline.get_personalization_status()


# ── Enhanced Analysis (with full accuracy pipeline) ──

class AccurateAnalysisRequest(BaseModel):
    signals: List[List[float]] = Field(..., description="EEG signals")
    sample_rate: int = 256
    user_id: str = "default"


@router.post("/analyze-eeg-accurate")
async def analyze_eeg_accurate(req: AccurateAnalysisRequest):
    """Run EEG analysis with full accuracy pipeline.

    Unlike /analyze-eeg, this endpoint runs:
    1. Signal quality gate
    2. Per-user calibration (if available)
    3. All 6 models
    4. Confidence calibration
    5. State transition smoothing
    6. Personalization (if feedback exists)
    """
    signals = np.array(req.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    channel_data = signals[0]

    # Step 1: Quality gate
    checker = SignalQualityChecker(fs=req.sample_rate)
    quality = checker.check_quality(channel_data)

    if not quality["is_usable"]:
        return {
            "status": "rejected",
            "reason": "Signal quality too low for analysis",
            "quality": quality,
        }

    # Step 2: Features + calibration
    processed = preprocess(channel_data, req.sample_rate)
    features_dict = extract_features(processed, req.sample_rate)
    features_array = np.array([v for _, v in sorted(features_dict.items())])

    calibration = UserCalibration.load(req.user_id)
    if calibration.is_calibrated:
        features_array = calibration.normalize_features(features_array)

    # Step 3: Run all 12 models
    sleep_pred = sleep_model.predict(channel_data, req.sample_rate)
    analysis = {
        "sleep_staging": sleep_pred,
        "emotions": emotion_model.predict(channel_data, req.sample_rate),
        "dream_detection": dream_model.predict(channel_data, req.sample_rate),
        "flow_state": flow_model.predict(channel_data, req.sample_rate),
        "creativity": creativity_model.predict(channel_data, req.sample_rate),
        "memory_encoding": memory_model.predict(channel_data, req.sample_rate),
        "drowsiness": drowsiness_model.predict(channel_data, req.sample_rate),
        "cognitive_load": cognitive_load_model.predict(channel_data, req.sample_rate),
        "attention": attention_model.predict(channel_data, req.sample_rate),
        "stress": stress_model.predict(channel_data, req.sample_rate),
        "lucid_dream": lucid_dream_model.predict(
            channel_data, req.sample_rate,
            is_rem=(sleep_pred.get("stage") == "REM"),
            sleep_stage=sleep_pred.get("stage_index", 0),
        ),
        "meditation": meditation_model.predict(channel_data, req.sample_rate),
    }

    # Step 4: Confidence calibration
    add_uncertainty_labels(analysis, _confidence_cal)
    conf_summary = analysis.pop("_confidence_summary", {})

    # Step 5: State smoothing
    smoothed = _state_engine.update({
        "sleep": analysis.get("sleep_staging", {}),
        "flow": analysis.get("flow_state", {}),
        "emotion": analysis.get("emotions", {}),
        "creativity": analysis.get("creativity", {}),
        "memory": analysis.get("memory_encoding", {}),
        "dream": analysis.get("dream_detection", {}),
    })

    # Step 6: Personalization
    pipeline = PersonalizedPipeline(req.user_id)
    pipeline.update_from_feedback()
    p_status = pipeline.get_personalization_status()

    coherence = _state_engine.get_cross_state_coherence()

    return _numpy_safe({
        "status": "ok",
        "quality": quality,
        "analysis": analysis,
        "smoothed_states": smoothed,
        "confidence_summary": conf_summary,
        "coherence": coherence,
        "is_calibrated": calibration.is_calibrated,
        "personalization": p_status,
    })


# ═══════════════════════════════════════════════════════════════
#  SPIRITUAL ENERGY / SELF-AWARENESS ENDPOINTS
# ═══════════════════════════════════════════════════════════════


@router.get("/spiritual/chakras/info")
async def chakra_info():
    """Get information about all 7 chakras and their EEG frequency mappings."""
    return {
        "chakras": {
            name: {
                "sanskrit": info["sanskrit"],
                "frequency_band_hz": info["frequency_band"],
                "color": info["color"],
                "element": info["element"],
                "qualities": info["qualities"],
                "location": info["location"],
                "mantra": info["mantra"],
            }
            for name, info in CHAKRAS.items()
        },
        "consciousness_levels": CONSCIOUSNESS_LEVELS,
    }


@router.post("/spiritual/chakras")
async def analyze_chakras(data: EEGInput):
    """Analyze chakra activation levels from EEG brainwaves.

    Maps delta through gamma frequency bands to the 7 energy centers,
    returning activation levels, balance, and guidance.
    """
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]
    processed = preprocess(eeg, data.fs)

    activations = compute_chakra_activations(processed, data.fs)
    balance = compute_chakra_balance(activations)

    return _numpy_safe({
        "chakras": activations,
        "balance": balance,
    })


@router.post("/spiritual/meditation-depth")
async def analyze_meditation(data: EEGInput):
    """Measure meditation depth from EEG patterns.

    Uses alpha/theta ratios, spectral entropy, and gamma detection
    to classify meditation depth on a 0-10 scale with stage labels.
    """
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]
    processed = preprocess(eeg, data.fs)

    return _numpy_safe(compute_meditation_depth(processed, data.fs))


@router.post("/spiritual/aura")
async def analyze_aura(data: EEGInput):
    """Generate aura color and energy visualization from EEG.

    Maps brain frequency content to traditional aura colors with
    layered inner/middle/outer visualization data.
    """
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]
    processed = preprocess(eeg, data.fs)

    return _numpy_safe(compute_aura_energy(processed, data.fs))


@router.post("/spiritual/kundalini")
async def analyze_kundalini(data: EEGInput):
    """Track kundalini energy flow through the chakra system.

    Measures progressive activation from root to crown, flow
    continuity, and awakening status.
    """
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]
    processed = preprocess(eeg, data.fs)

    return _numpy_safe(compute_kundalini_flow(processed, data.fs))


@router.post("/spiritual/prana-balance")
async def analyze_prana(data: EEGInput):
    """Analyze prana/chi energy balance from bilateral EEG.

    Uses hemispheric asymmetry to determine balance between
    Ida (yin/lunar) and Pingala (yang/solar) energy channels.
    Requires at least 2 channels (left and right hemisphere).
    """
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    if signals.shape[0] < 2:
        # With single channel, simulate bilateral by splitting
        mid = len(signals[0]) // 2
        eeg_left = preprocess(signals[0][:mid], data.fs)
        eeg_right = preprocess(signals[0][mid:], data.fs)
    else:
        eeg_left = preprocess(signals[0], data.fs)
        eeg_right = preprocess(signals[1], data.fs)

    return _numpy_safe(compute_prana_balance(eeg_left, eeg_right, data.fs))


@router.post("/spiritual/consciousness")
async def analyze_consciousness(data: EEGInput):
    """Estimate consciousness level from EEG patterns.

    Maps brain activity to a 0-1000 consciousness scale with
    level descriptions from Deep Sleep to Cosmic Consciousness.
    """
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]
    processed = preprocess(eeg, data.fs)

    return _numpy_safe(compute_consciousness_level(processed, data.fs))


@router.post("/spiritual/third-eye")
async def analyze_third_eye(data: EEGInput):
    """Measure third eye (Ajna) activation through gamma analysis.

    Tracks high beta and gamma activity correlated with intuitive
    experiences and heightened perception.
    """
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]
    processed = preprocess(eeg, data.fs)

    return _numpy_safe(compute_third_eye_activation(processed, data.fs))


@router.post("/spiritual/full-analysis")
async def full_spiritual_analysis_endpoint(data: EEGInput):
    """Complete spiritual energy analysis — all metrics in one call.

    Returns chakras, meditation depth, aura, kundalini flow,
    consciousness level, third eye, and personalized insights.
    Includes prana balance if 2+ channels provided.
    """
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]
    processed = preprocess(eeg, data.fs)

    eeg_left = None
    eeg_right = None
    if signals.shape[0] >= 2:
        eeg_left = preprocess(signals[0], data.fs)
        eeg_right = preprocess(signals[1], data.fs)

    result = full_spiritual_analysis(processed, data.fs, eeg_left, eeg_right)
    return _numpy_safe(result)


# ═══════════════════════════════════════════════════════════════
#  EMOTIONAL SHIFT DETECTION — Pre-conscious awareness
# ═══════════════════════════════════════════════════════════════


class EmotionShiftRequest(BaseModel):
    signals: List[List[float]] = Field(..., description="EEG signals (channels x samples)")
    fs: float = Field(default=256.0)
    user_id: str = Field(default="default")


@router.post("/emotion-shift/detect")
async def detect_emotion_shift(req: EmotionShiftRequest):
    """Feed EEG data and detect pre-conscious emotional shifts.

    Call this continuously (~4Hz) during a session. The detector
    watches for EEG patterns that precede conscious emotion changes
    — the same signals animals read in us before we notice ourselves.

    Returns shift alerts with type, guidance, and body awareness cues.
    """
    if req.user_id not in _emotion_shift_detectors:
        _emotion_shift_detectors[req.user_id] = EmotionShiftDetector(fs=req.fs)

    detector = _emotion_shift_detectors[req.user_id]
    signals = np.array(req.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]

    # Also run emotion model for richer context
    emotion_pred = emotion_model.predict(eeg, req.fs)
    result = detector.update(eeg, emotion_pred)

    return _numpy_safe(result)


@router.get("/emotion-shift/summary/{user_id}")
async def emotion_shift_summary(user_id: str):
    """Get session summary of all emotional shifts detected.

    Shows shift timeline, dominant patterns, stability assessment,
    and personalized insights about emotional patterns.
    """
    detector = _emotion_shift_detectors.get(user_id)
    if detector is None:
        return {"total_shifts": 0, "message": "No active session for this user"}

    return _numpy_safe(detector.get_session_summary())


@router.get("/emotion-shift/awareness-score/{user_id}")
async def emotion_awareness_score(user_id: str):
    """Get emotional awareness score for the session.

    Tracks how many shifts were observed — building this muscle
    over time gives humans animal-like emotional perception.
    """
    detector = _emotion_shift_detectors.get(user_id)
    if detector is None:
        return {"awareness_score": 0, "level": "Beginning", "message": "Start a session first."}

    return _numpy_safe(detector.get_emotional_awareness_score())


@router.post("/emotion-shift/reset/{user_id}")
async def reset_emotion_shift(user_id: str):
    """Reset the emotion shift detector for a new session."""
    if user_id in _emotion_shift_detectors:
        del _emotion_shift_detectors[user_id]
    return {"status": "reset", "user_id": user_id}


# ═══════════════════════════════════════════════════════════════
#  COGNITIVE MODELS — Drowsiness, Cognitive Load, Attention,
#  Stress, Lucid Dream, Meditation
# ═══════════════════════════════════════════════════════════════


@router.post("/predict-drowsiness")
async def predict_drowsiness(data: EEGInput):
    """Detect drowsiness level from EEG: alert / drowsy / sleepy.

    Uses theta/beta ratio, alpha attenuation, and slow-wave increase
    to classify alertness state.
    """
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]
    return _numpy_safe(drowsiness_model.predict(eeg, data.fs))


@router.post("/predict-cognitive-load")
async def predict_cognitive_load(data: EEGInput):
    """Estimate cognitive load from EEG: low / moderate / high.

    Uses frontal theta, working memory load markers, and task
    engagement metrics to assess cognitive demand.
    """
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]
    return _numpy_safe(cognitive_load_model.predict(eeg, data.fs))


@router.post("/predict-attention")
async def predict_attention(data: EEGInput):
    """Classify attention state: distracted / passive / focused / hyperfocused.

    Uses theta/beta ratio (ADHD gold standard), beta engagement,
    and spectral concentration to assess attention.
    """
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]
    return _numpy_safe(attention_model.predict(eeg, data.fs))


@router.post("/predict-stress")
async def predict_stress(data: EEGInput):
    """Detect stress level: relaxed / mild / moderate / high.

    Multi-dimensional stress assessment using high-beta activation,
    alpha suppression, cortisol proxy, and autonomic index.
    """
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]
    return _numpy_safe(stress_model.predict(eeg, data.fs))


class LucidDreamRequest(BaseModel):
    signals: List[List[float]] = Field(..., description="EEG signals")
    fs: float = Field(default=256.0)
    is_rem: bool = Field(default=True, description="Whether currently in REM sleep")
    sleep_stage: int = Field(default=4, description="Current sleep stage (4=REM)")


@router.post("/predict-lucid-dream")
async def predict_lucid_dream(req: LucidDreamRequest):
    """Detect lucid dreaming during REM sleep.

    States: non_lucid / pre_lucid / lucid / controlled.
    Uses 40Hz gamma surge (Voss et al.), metacognition index,
    and alpha-gamma coupling. Only meaningful during REM sleep.
    """
    signals = np.array(req.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]
    return _numpy_safe(lucid_dream_model.predict(
        eeg, req.fs, is_rem=req.is_rem, sleep_stage=req.sleep_stage
    ))


@router.post("/predict-meditation")
async def predict_meditation(data: EEGInput):
    """Classify meditation depth: surface / light / moderate / deep / transcendent.

    Uses alpha stability, theta depth, gamma transcendence,
    theta-gamma coupling, and matches to meditation traditions.
    """
    signals = np.array(data.signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    eeg = np.mean(signals, axis=0) if signals.shape[0] > 1 else signals[0]
    return _numpy_safe(meditation_model.predict(eeg, data.fs))


@router.get("/cognitive-models/session-stats")
async def cognitive_session_stats():
    """Get session statistics for all cognitive models that track history."""
    stats = {}
    if hasattr(lucid_dream_model, 'get_session_stats'):
        stats["lucid_dream"] = lucid_dream_model.get_session_stats()
    if hasattr(meditation_model, 'get_session_stats'):
        stats["meditation"] = meditation_model.get_session_stats()
    return _numpy_safe(stats)


# ─── Denoising & Artifact Classification ─────────────────────────────────


class DenoiseRequest(BaseModel):
    signals: List[List[float]]
    fs: float = 256.0


@router.post("/denoise")
async def denoise_signal(req: DenoiseRequest):
    """Denoise EEG signals using the trained autoencoder.

    Falls back to classical bandpass + notch filtering if the
    trained model is not available.

    Returns cleaned signals and SNR improvement estimate.
    """
    from processing.eeg_processor import preprocess_robust

    results = []
    for ch_data in req.signals:
        signal = np.array(ch_data, dtype=np.float64)
        cleaned = preprocess_robust(signal, req.fs, use_denoiser=True)
        results.append(cleaned.tolist())

    return _numpy_safe({
        "cleaned_signals": results,
        "n_channels": len(results),
        "method": "ml_denoiser" if _get_denoiser_status() else "classical_filter",
    })


@router.post("/classify-artifacts")
async def classify_artifacts(req: DenoiseRequest):
    """Classify artifact types in EEG signal segments.

    Returns per-window artifact type classification with confidence scores.
    """
    try:
        from models.artifact_classifier import ArtifactClassifier
        classifier = ArtifactClassifier(model_path="models/saved/artifact_classifier_model.pkl")
    except Exception:
        classifier = None

    if classifier is None or classifier.model is None:
        # Fallback to heuristic detection
        results = []
        for ch_data in req.signals:
            signal = np.array(ch_data, dtype=np.float64)
            blinks = detect_eye_blinks(signal, req.fs)
            muscle = detect_muscle_artifacts(signal, req.fs)
            pops = detect_electrode_pops(signal, req.fs)
            sqi = compute_signal_quality_index(signal, req.fs)
            results.append({
                "sqi": sqi,
                "eye_blinks": len(blinks),
                "muscle_artifacts": len(muscle),
                "electrode_pops": len(pops),
                "method": "heuristic",
            })
        return _numpy_safe({"channels": results})

    all_results = []
    for ch_data in req.signals:
        signal = np.array(ch_data, dtype=np.float64)
        classifications = classifier.classify_signal(signal, req.fs, window_sec=1.0)
        sqi = compute_signal_quality_index(signal, req.fs)
        all_results.append({
            "sqi": sqi,
            "windows": classifications,
            "artifact_summary": _summarize_artifacts(classifications),
            "method": "ml_classifier",
        })

    return _numpy_safe({"channels": all_results})


@router.get("/denoise/status")
async def denoise_status():
    """Check availability of ML denoiser and artifact classifier models."""
    return {
        "denoiser_available": _get_denoiser_status(),
        "artifact_classifier_available": _get_artifact_classifier_status(),
        "denoiser_model_path": "models/saved/denoiser_model.pt",
        "artifact_model_path": "models/saved/artifact_classifier_model.pkl",
    }


@router.get("/datasets")
async def list_datasets():
    """List all available EEG datasets and their download status."""
    try:
        from training.data_loaders import list_available_datasets
        return list_available_datasets()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _get_denoiser_status() -> bool:
    return Path("models/saved/denoiser_model.pt").exists()


def _get_artifact_classifier_status() -> bool:
    return Path("models/saved/artifact_classifier_model.pkl").exists()


def _summarize_artifacts(classifications: list) -> dict:
    """Summarize artifact types from window classifications."""
    from collections import Counter
    types = [c["artifact_type"] for c in classifications]
    counts = Counter(types)
    total = len(types)
    return {
        "total_windows": total,
        "clean_windows": counts.get("clean", 0),
        "clean_ratio": counts.get("clean", 0) / max(total, 1),
        "artifact_counts": dict(counts),
    }

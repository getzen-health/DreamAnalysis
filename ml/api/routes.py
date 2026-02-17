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

router = APIRouter()

MODEL_DIR = Path("models/saved")
BENCHMARK_DIR = Path("benchmarks")


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
flow_model = FlowStateDetector()
creativity_model = CreativityDetector()
memory_model = MemoryEncodingPredictor()

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
        emotion_result = emotion_model.predict(eeg, fs)
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

    return {
        **result,
        "analysis": {
            "sleep_stage": sleep_result,
            "emotions": emotion_result,
            "dream_detection": dream_result,
            "flow_state": flow_result,
            "creativity": creativity_result,
            "memory_encoding": memory_result,
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

@router.get("/datasets")
async def list_datasets():
    """List all available EEG datasets and their download status."""
    from training.data_loaders import list_available_datasets
    return list_available_datasets()


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

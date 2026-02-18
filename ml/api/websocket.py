"""WebSocket endpoint for real-time EEG streaming with full accuracy pipeline."""

import asyncio
import json
import logging
import time
import uuid
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from processing.eeg_processor import extract_features, extract_band_powers, preprocess

logger = logging.getLogger(__name__)

# Per-connection state storage (keyed by connection ID)
_connection_state: dict[str, dict] = {}
MAX_CONNECTIONS = 50


def _numpy_safe(obj):
    """Recursively convert numpy types to native Python for JSON."""
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


async def eeg_stream_endpoint(websocket: WebSocket):
    """WebSocket at /ws/eeg-stream: real-time EEG with full accuracy pipeline.

    Sends JSON frames at ~4Hz containing:
    {
        "signals": [[...], ...],
        "analysis": {
            "band_powers": {...},
            "features": {...},
            "sleep_staging": {...},
            "emotions": {...},
            "dream_detection": {...},
            "flow_state": {...},
            "creativity": {...},
            "memory_encoding": {...}
        },
        "quality": {...},
        "smoothed_states": {...},
        "confidence_summary": {...},
        "coherence": {...},
        "timestamp": 1234567890.123
    }

    Client can send JSON commands:
    - {"command": "set_user", "user_id": "alice"}
    - {"command": "configure", "run_models": true, "run_quality": true}

    Reads from BrainFlow device if connected. Client should handle reconnection.
    """
    # Enforce connection limit
    if len(_connection_state) >= MAX_CONNECTIONS:
        logger.warning("WebSocket connection rejected: max connections (%d) reached", MAX_CONNECTIONS)
        await websocket.close(code=1013, reason="Max connections reached")
        return

    conn_id = str(uuid.uuid4())
    logger.info("WebSocket connected: %s", conn_id)
    await websocket.accept()

    # Per-connection state (stored in module-level dict, cleaned up on disconnect)
    _connection_state[conn_id] = {
        "user_id": "default",
        "shift_detector": None,
    }
    user_id = "default"
    run_models = True
    run_quality = True
    run_smoothing = True
    run_spiritual = False  # opt-in for spiritual energy analysis
    run_emotion_shift = True  # on by default — core feature
    last_pong = time.time()

    # Lazy-init pipeline singletons
    models = None
    quality_checker = None
    state_engine = None
    confidence_cal = None

    def _init_models():
        nonlocal models
        if models is not None:
            return models
        try:
            from api.routes import (
                sleep_model, emotion_model, dream_model,
                flow_model, creativity_model, memory_model,
                drowsiness_model, cognitive_load_model, attention_model,
                stress_model, lucid_dream_model, meditation_model,
            )
            models = {
                "sleep": sleep_model,
                "emotion": emotion_model,
                "dream": dream_model,
                "flow": flow_model,
                "creativity": creativity_model,
                "memory": memory_model,
                "drowsiness": drowsiness_model,
                "cognitive_load": cognitive_load_model,
                "attention": attention_model,
                "stress": stress_model,
                "lucid_dream": lucid_dream_model,
                "meditation": meditation_model,
            }
        except Exception as e:
            logger.error("Failed to load ML models: %s", e)
            models = {}
        return models

    def _init_accuracy_pipeline():
        nonlocal quality_checker, state_engine, confidence_cal
        if quality_checker is not None:
            return
        try:
            from processing.signal_quality import SignalQualityChecker
            from processing.state_transitions import BrainStateEngine
            from processing.confidence_calibration import (
                ConfidenceCalibrator,
            )
            quality_checker = SignalQualityChecker(fs=256)
            state_engine = BrainStateEngine()
            confidence_cal = ConfidenceCalibrator()
        except Exception as e:
            logger.error("Failed to init accuracy pipeline: %s", e)

    try:
        # Get device manager
        device_manager = None
        try:
            from hardware.brainflow_manager import BRAINFLOW_AVAILABLE
            if BRAINFLOW_AVAILABLE:
                from api.routes import _get_device_manager
                device_manager = _get_device_manager()
        except Exception as e:
            logger.warning("BrainFlow not available: %s", e)

        # Get session recorder
        session_recorder = None
        try:
            from api.routes import _session_recorder
            session_recorder = _session_recorder
        except Exception as e:
            logger.warning("Session recorder not available: %s", e)

        frame_interval = 0.25  # 4 Hz

        while True:
            start_time = time.time()

            # Check for incoming commands (non-blocking)
            try:
                msg = await asyncio.wait_for(
                    websocket.receive_text(), timeout=0.01
                )
                try:
                    cmd = json.loads(msg)
                    if cmd.get("command") == "set_user":
                        user_id = cmd.get("user_id", "default")
                    elif cmd.get("command") == "configure":
                        run_models = cmd.get("run_models", run_models)
                        run_quality = cmd.get("run_quality", run_quality)
                        run_smoothing = cmd.get("run_smoothing", run_smoothing)
                        run_spiritual = cmd.get("run_spiritual", run_spiritual)
                        run_emotion_shift = cmd.get("run_emotion_shift", run_emotion_shift)
                except (json.JSONDecodeError, AttributeError):
                    pass
            except asyncio.TimeoutError:
                pass

            frame = None

            if device_manager and device_manager.is_streaming:
                data = device_manager.get_current_data(n_samples=64)
                if data and data["signals"] and len(data["signals"][0]) > 0:
                    signals = np.array(data["signals"])
                    eeg = signals[0] if signals.shape[0] > 0 else np.zeros(64)
                    fs = data.get("sample_rate", 256)

                    # Basic processing
                    try:
                        processed = preprocess(eeg, fs)
                        bands = extract_band_powers(processed, fs)
                        features = extract_features(processed, fs)
                    except Exception as e:
                        logger.warning("EEG processing error: %s", e)
                        bands = {}
                        features = {}

                    analysis = {
                        "band_powers": bands,
                        "features": features,
                    }

                    # Signal quality check
                    quality_result = None
                    if run_quality:
                        _init_accuracy_pipeline()
                        if quality_checker:
                            try:
                                from processing.signal_quality import SignalQualityChecker
                                qc = SignalQualityChecker(fs=fs) if fs != 256 else quality_checker
                                quality_result = qc.check_quality(eeg)
                            except Exception as e:
                                logger.warning("Signal quality check failed: %s", e)

                    # Run all 12 models
                    if run_models:
                        m = _init_models()
                        if m:
                            try:
                                sleep_pred = m["sleep"].predict(eeg, fs) if "sleep" in m else {}
                                analysis["sleep_staging"] = sleep_pred
                                analysis["emotions"] = m["emotion"].predict(eeg, fs) if "emotion" in m else {}
                                analysis["dream_detection"] = m["dream"].predict(eeg, fs) if "dream" in m else {}
                                analysis["flow_state"] = m["flow"].predict(eeg, fs) if "flow" in m else {}
                                analysis["creativity"] = m["creativity"].predict(eeg, fs) if "creativity" in m else {}
                                analysis["memory_encoding"] = m["memory"].predict(eeg, fs) if "memory" in m else {}
                                analysis["drowsiness"] = m["drowsiness"].predict(eeg, fs) if "drowsiness" in m else {}
                                analysis["cognitive_load"] = m["cognitive_load"].predict(eeg, fs) if "cognitive_load" in m else {}
                                analysis["attention"] = m["attention"].predict(eeg, fs) if "attention" in m else {}
                                analysis["stress"] = m["stress"].predict(eeg, fs) if "stress" in m else {}
                                if "lucid_dream" in m:
                                    analysis["lucid_dream"] = m["lucid_dream"].predict(
                                        eeg, fs,
                                        is_rem=(sleep_pred.get("stage") == "REM"),
                                        sleep_stage=sleep_pred.get("stage_index", 0),
                                    )
                                analysis["meditation"] = m["meditation"].predict(eeg, fs) if "meditation" in m else {}
                            except Exception as e:
                                logger.warning("Model inference error: %s", e)

                    # Confidence calibration
                    conf_summary = None
                    if run_models and confidence_cal:
                        try:
                            from processing.confidence_calibration import add_uncertainty_labels
                            add_uncertainty_labels(analysis, confidence_cal)
                            conf_summary = analysis.pop("_confidence_summary", None)
                        except Exception as e:
                            logger.warning("Confidence calibration error: %s", e)

                    # State transition smoothing
                    smoothed = None
                    coherence = None
                    if run_smoothing and state_engine and run_models:
                        try:
                            smoothed = state_engine.update({
                                "sleep": analysis.get("sleep_staging", {}),
                                "flow": analysis.get("flow_state", {}),
                                "emotion": analysis.get("emotions", {}),
                                "creativity": analysis.get("creativity", {}),
                                "memory": analysis.get("memory_encoding", {}),
                                "dream": analysis.get("dream_detection", {}),
                            })
                            coherence = state_engine.get_cross_state_coherence()
                        except Exception as e:
                            logger.warning("State smoothing error: %s", e)

                    frame = {
                        "signals": data["signals"],
                        "analysis": analysis,
                        "timestamp": time.time(),
                        "n_channels": data.get("n_channels", 1),
                        "sample_rate": fs,
                    }

                    if quality_result is not None:
                        frame["quality"] = quality_result
                    if smoothed is not None:
                        frame["smoothed_states"] = smoothed
                    if conf_summary is not None:
                        frame["confidence_summary"] = conf_summary
                    if coherence is not None:
                        frame["coherence"] = coherence

                    # Emotion shift detection (per-connection detector)
                    if run_emotion_shift:
                        try:
                            from processing.emotion_shift_detector import EmotionShiftDetector
                            conn = _connection_state.get(conn_id, {})
                            if conn.get("shift_detector") is None:
                                conn["shift_detector"] = EmotionShiftDetector(fs=fs)
                                _connection_state[conn_id] = conn
                            emotion_pred = analysis.get("emotions")
                            shift = conn["shift_detector"].update(eeg, emotion_pred)
                            if shift.get("shift_detected"):
                                frame["emotion_shift"] = shift
                        except Exception as e:
                            logger.warning("Emotion shift detection error: %s", e)

                    # Spiritual energy analysis (opt-in)
                    if run_spiritual:
                        try:
                            from processing.spiritual_energy import (
                                compute_chakra_activations,
                                compute_chakra_balance,
                                compute_meditation_depth,
                                compute_aura_energy,
                                compute_consciousness_level,
                            )
                            chakras = compute_chakra_activations(processed, fs)
                            frame["spiritual"] = {
                                "chakras": chakras,
                                "chakra_balance": compute_chakra_balance(chakras),
                                "meditation_depth": compute_meditation_depth(processed, fs),
                                "aura": compute_aura_energy(processed, fs),
                                "consciousness": compute_consciousness_level(processed, fs),
                            }
                        except Exception as e:
                            logger.warning("Spiritual energy analysis error: %s", e)

                    # Pipe to session recorder
                    if session_recorder and session_recorder.is_recording:
                        try:
                            session_recorder.add_frame(signals, frame["analysis"])
                        except Exception as e:
                            logger.warning("Session recording error: %s", e)

            if frame:
                await websocket.send_json(_numpy_safe(frame))

            # Maintain frame rate
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_interval - elapsed)
            await asyncio.sleep(sleep_time)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: %s", conn_id)
    except Exception as e:
        logger.error("WebSocket error for %s: %s", conn_id, e)
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        _connection_state.pop(conn_id, None)

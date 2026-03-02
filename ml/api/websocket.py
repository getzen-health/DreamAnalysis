"""WebSocket endpoint for real-time EEG streaming with full accuracy pipeline."""

import asyncio
import json
import logging
import time
import uuid
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from processing.eeg_processor import extract_features, extract_band_powers, preprocess
from storage.timescale_writer import TimescaleWriter
from storage.parquet_writer import ParquetWriter
from api.routes import fusion_model, get_biometric_snapshot, predict_emotion

logger = logging.getLogger(__name__)

# Per-connection state storage (keyed by connection ID)
_connection_state: dict[str, dict] = {}
MAX_CONNECTIONS = 200

# Emotion classification window: 30 seconds at 256 Hz.
# 30s is required for the REVE DETransformer (temporal DE sequence model).
# EEGNet and mega-LGBM also benefit from longer epochs: 2024 paper found
# 30-second segments most effective for consumer EEG emotion classification.
# First result arrives at t=30s; after that refreshes every 30s.
_EMOTION_WINDOW_SEC = 30
_EMOTION_WINDOW_SAMPLES = 256 * _EMOTION_WINDOW_SEC  # 7 680 samples


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
    try:
        from monitoring.datadog_reporter import report_metric
        asyncio.create_task(asyncio.to_thread(
            report_metric, "neural_dream.ws.connections", 1.0, None, "count"
        ))
    except Exception:
        pass

    # Per-connection state (stored in module-level dict, cleaned up on disconnect)
    _connection_state[conn_id] = {
        "user_id": "default",
        "shift_detector": None,
        # Rolling 30s EEG buffer for emotion classification
        "eeg_buffer": None,          # np.ndarray (n_channels, n_samples) or None
        "emotion_result": None,      # last 30s emotion output (shown until next update)
        "emotion_updated_at": 0.0,   # timestamp of last emotion computation
        "emotion_samples_seen": 0,   # total samples accumulated so far
    }
    user_id = "default"
    ts_writer = await TimescaleWriter.create(user_id)  # always-on TimescaleDB recording
    parquet_writer = ParquetWriter(user_id=user_id)     # always-on Parquet recording
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

        # Get per-user session recorder
        session_recorder = None
        try:
            from api.routes._shared import _get_session_recorder
            session_recorder = _get_session_recorder(user_id)
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
                        new_uid = cmd.get("user_id", "default")
                        if new_uid != user_id:
                            await ts_writer.close()
                            parquet_writer.flush()
                            user_id = new_uid
                            ts_writer = await TimescaleWriter.create(user_id)
                            parquet_writer = ParquetWriter(user_id=user_id)
                            # Switch to the new user's session recorder
                            try:
                                from api.routes._shared import _get_session_recorder
                                session_recorder = _get_session_recorder(user_id)
                            except Exception:
                                pass
                        user_id = new_uid
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
                    # Keep full multichannel data for models that support it
                    eeg_multi = signals if signals.ndim == 2 and signals.shape[0] >= 4 else None
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
                                # Datadog — SQI gauge every frame (4 Hz) — sampled 1-in-30 to avoid spam
                                _sqi = (quality_result or {}).get("quality_score", 0)
                                if _sqi and int(time.time()) % 30 == 0:
                                    try:
                                        from monitoring.datadog_reporter import report_metric
                                        asyncio.create_task(asyncio.to_thread(
                                            report_metric, "neural_dream.eeg.sqi", _sqi
                                        ))
                                    except Exception:
                                        pass
                            except Exception as e:
                                logger.warning("Signal quality check failed: %s", e)

                    # ── Accumulate rolling 30s EEG buffer for emotion ──────────────
                    conn_state = _connection_state.get(conn_id, {})
                    chunk_samples = signals.shape[-1]  # typically 64
                    buf = conn_state.get("eeg_buffer")
                    if buf is None:
                        buf = signals.copy()
                    else:
                        buf = np.concatenate([buf, signals], axis=-1)
                        if buf.shape[-1] > _EMOTION_WINDOW_SAMPLES:
                            buf = buf[:, -_EMOTION_WINDOW_SAMPLES:]
                    conn_state["eeg_buffer"] = buf
                    conn_state["emotion_samples_seen"] = conn_state.get("emotion_samples_seen", 0) + chunk_samples
                    _connection_state[conn_id] = conn_state

                    # Run all 12 models
                    if run_models:
                        m = _init_models()
                        if m:
                            try:
                                sleep_pred = m["sleep"].predict(eeg, fs) if "sleep" in m else {}
                                analysis["sleep_staging"] = sleep_pred

                                # ── Emotion: 30s window only ──────────────────────────────
                                # Accumulate 30s before first result; then re-run every 30s.
                                # In the meantime the last result is reused so the UI stays
                                # populated rather than flickering on every 64-sample chunk.
                                buf_full = buf.shape[-1] >= _EMOTION_WINDOW_SAMPLES
                                now = time.time()
                                due = now - conn_state.get("emotion_updated_at", 0) >= _EMOTION_WINDOW_SEC
                                if buf_full and (due or conn_state.get("emotion_result") is None):
                                    try:
                                        ws_user_id = conn_state.get("user_id", "default")
                                        eeg_30s = buf if buf.shape[0] >= 4 else buf[0]
                                        n_ch = eeg_30s.shape[0] if eeg_30s.ndim == 2 else 1
                                        # Personal model → EEGNet central → mega LGBM fallback
                                        emotion_result = predict_emotion(ws_user_id, eeg_30s, fs, n_ch)
                                        # Multimodal fusion: blend with any cached biometrics
                                        try:
                                            bio = get_biometric_snapshot(ws_user_id)
                                            emotion_result = fusion_model.fuse(emotion_result, bio)
                                        except Exception:
                                            pass  # keep raw result if fusion fails
                                        conn_state["emotion_result"] = emotion_result
                                        conn_state["emotion_updated_at"] = now
                                        _connection_state[conn_id] = conn_state
                                        _mtype = emotion_result.get("model_type", "?")
                                        _emo   = emotion_result.get("emotion", "?")
                                        _conf  = emotion_result.get("confidence", 0)
                                        _val   = emotion_result.get("valence", 0)
                                        _aro   = emotion_result.get("arousal", 0)
                                        logger.info(
                                            "[emotion] user=%s model=%s emotion=%s conf=%.2f val=%.3f aro=%.3f",
                                            ws_user_id, _mtype, _emo, _conf, _val, _aro,
                                        )
                                        # Datadog — fire-and-forget metrics every 30s
                                        try:
                                            from monitoring.datadog_reporter import report_metric
                                            asyncio.create_task(asyncio.to_thread(
                                                lambda: (
                                                    report_metric("neural_dream.emotion.confidence", _conf,
                                                                  tags=[f"model:{_mtype}", f"emotion:{_emo}"]),
                                                    report_metric("neural_dream.emotion.valence", _val,
                                                                  tags=[f"model:{_mtype}"]),
                                                    report_metric("neural_dream.emotion.arousal", _aro,
                                                                  tags=[f"model:{_mtype}"]),
                                                )
                                            ))
                                        except Exception:
                                            pass
                                    except Exception as e:
                                        logger.warning("30s emotion error: %s", e)

                                # Attach last known emotion result (or none if still buffering)
                                last_emotion = conn_state.get("emotion_result")
                                if last_emotion:
                                    samples_seen = conn_state.get("emotion_samples_seen", 0)
                                    analysis["emotions"] = {
                                        **last_emotion,
                                        "window_sec": _EMOTION_WINDOW_SEC,
                                        "buffered_sec": round(min(samples_seen, _EMOTION_WINDOW_SAMPLES) / fs, 1),
                                        "ready": buf_full,
                                    }
                                else:
                                    # Still buffering — tell the UI how far along we are
                                    buffered_sec = round(buf.shape[-1] / fs, 1)
                                    analysis["emotions"] = {
                                        "emotion": None,
                                        "confidence": 0,
                                        "ready": False,
                                        "buffered_sec": buffered_sec,
                                        "window_sec": _EMOTION_WINDOW_SEC,
                                    }

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

                    # Always-on: push to TimescaleDB + Parquet
                    ts_writer.push_frame(frame, frame.get("signals", []))
                    parquet_writer.push_frame(frame.get("analysis", {}), frame.get("timestamp"))

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
        await ts_writer.close()
        await parquet_writer.close()
        _connection_state.pop(conn_id, None)

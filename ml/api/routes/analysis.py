"""Core EEG analysis endpoints: /analyze-eeg, /simulate-eeg."""

import threading
from typing import Dict
import numpy as np
from fastapi import APIRouter, HTTPException

from ._shared import (
    _numpy_safe,
    _get_personal_model,
    sleep_model, emotion_model, dream_model, flow_model,
    creativity_model, memory_model,
    drowsiness_model, cognitive_load_model, attention_model,
    stress_model, lucid_dream_model, meditation_model,
    _anomaly_detector,
    EEGInput, SimulateRequest, AnalysisResponse,
    STATE_PROFILES, simulate_eeg,
    extract_features, extract_band_powers, preprocess, extract_features_multichannel,
    compute_coherence, compute_phase_locking_value,
    detect_eye_blinks, detect_muscle_artifacts, detect_electrode_pops,
    compute_signal_quality_index, auto_reject_epochs,
    AnomalyDetector,
)

router = APIRouter()


# ─── 4-second epoch buffer (sliding window, 50% overlap) ─────────────────────
# Research consensus: 4-8 sec epochs are needed for stable Welch PSD estimates.
# Below 4 seconds, theta/alpha power estimates have high variance.
# Buffer accumulates frames and exposes the last 4 seconds on each call.
_EPOCH_SECONDS = 4          # seconds of EEG to accumulate before classifying
_EPOCH_HOP_SECONDS = 2      # slide by 2 seconds (50% overlap)
_DEFAULT_FS = 256

class _EpochBuffer:
    """Thread-safe ring buffer that accumulates EEG frames.

    When fewer than _EPOCH_SECONDS of data has been collected, the buffer
    returns whatever is available so the API still responds (degraded accuracy,
    flagged with epoch_ready=False).  Once full, it slides by _EPOCH_HOP_SECONDS.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._buf: np.ndarray | None = None   # (n_channels, n_accumulated)
        self._n_channels: int = 0

    def push_and_get(self, signals: np.ndarray, fs: float) -> tuple[np.ndarray, bool]:
        """Add new samples and return the best epoch available.

        Returns:
            (epoch, epoch_ready) where epoch_ready=True means >= 4 seconds.
        """
        epoch_samples = int(_EPOCH_SECONDS * fs)
        hop_samples = int(_EPOCH_HOP_SECONDS * fs)
        n_channels = signals.shape[0]

        with self._lock:
            # Reset buffer if channel count changed
            if self._buf is None or self._n_channels != n_channels:
                self._buf = signals.copy()
                self._n_channels = n_channels
            else:
                self._buf = np.concatenate([self._buf, signals], axis=1)

            # Trim buffer to maximum needed size (4 seconds)
            if self._buf.shape[1] > epoch_samples:
                self._buf = self._buf[:, -epoch_samples:]

            buf_copy = self._buf.copy()

        epoch_ready = buf_copy.shape[1] >= epoch_samples
        return buf_copy, epoch_ready


_epoch_buffers: Dict[str, _EpochBuffer] = {}
_epoch_buffers_lock = threading.Lock()

def _get_epoch_buffer(user_id: str) -> _EpochBuffer:
    """Return the per-user epoch buffer, creating it on first use."""
    with _epoch_buffers_lock:
        if user_id not in _epoch_buffers:
            _epoch_buffers[user_id] = _EpochBuffer()
        return _epoch_buffers[user_id]
# ─────────────────────────────────────────────────────────────────────────────


@router.post("/analyze-eeg", response_model=AnalysisResponse)
async def analyze_eeg(input_data: EEGInput):
    """Run all 3 models on EEG input with multi-channel support."""
    try:
        signals = np.array(input_data.signals)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        fs = input_data.fs
        user_id = input_data.user_id
        n_channels = signals.shape[0]

        # Accumulate into per-user epoch buffer; use 4-second window when available
        signals, epoch_ready = _get_epoch_buffer(user_id).push_and_get(signals, fs)
        n_channels = signals.shape[0]   # re-read after buffer update

        if n_channels > 1:
            extract_features_multichannel(signals, fs, method="average")
            avg_signal = np.mean(signals, axis=0)
            eeg = avg_signal
        else:
            eeg = signals[0]

        sleep_result = sleep_model.predict(eeg, fs)
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

        # Signal quality
        signal_quality = None
        try:
            channel_sqis = [compute_signal_quality_index(signals[ch], fs) for ch in range(n_channels)]
            avg_sqi = float(np.mean(channel_sqis))

            artifacts = []
            if detect_eye_blinks(eeg, fs):
                artifacts.append("eye_blink")
            if detect_muscle_artifacts(eeg, fs):
                artifacts.append("muscle")
            if detect_electrode_pops(eeg, fs):
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

        # Anomaly detection
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

        # Personal model blending
        personal = None
        try:
            pm = _get_personal_model(user_id)
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
            epoch_ready=epoch_ready,
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

    return {
        **result,
        "analysis": {
            "sleep_stage": sleep_result,
            "emotions": emotion_model.predict(eeg, request.fs),
            "dream_detection": dream_model.predict(eeg, request.fs),
            "flow_state": flow_model.predict(eeg, request.fs),
            "creativity": creativity_model.predict(eeg, request.fs),
            "memory_encoding": memory_model.predict(eeg, request.fs),
            "drowsiness": drowsiness_model.predict(eeg, request.fs),
            "cognitive_load": cognitive_load_model.predict(eeg, request.fs),
            "attention": attention_model.predict(eeg, request.fs),
            "stress": stress_model.predict(eeg, request.fs),
            "lucid_dream": lucid_dream_model.predict(
                eeg, request.fs,
                is_rem=(sleep_result.get("stage") == "REM"),
                sleep_stage=sleep_result.get("stage_index", 0),
            ),
            "meditation": meditation_model.predict(eeg, request.fs),
        },
    }

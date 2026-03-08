"""Core EEG analysis endpoints: /analyze-eeg, /simulate-eeg."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict
import numpy as np
from fastapi import APIRouter, HTTPException

from ._shared import (
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
    fusion_model, get_biometric_snapshot, predict_emotion,
)
from processing.e_asr import EmbeddedASR

router = APIRouter()

# ─── Per-user E-ASR instances ────────────────────────────────────────────────
# Each user gets an independent EmbeddedASR so baseline calibration is per-session.
# Instances are created lazily on first use. Baseline is fitted via the
# /calibration/baseline/add-frame endpoint (BaselineCalibrator) — E-ASR does
# not require separate calibration; it uses the same resting-state data.
_easr_instances: Dict[str, EmbeddedASR] = {}
_easr_instances_lock = threading.Lock()


def _get_easr(user_id: str, fs: int = 256) -> EmbeddedASR:
    """Return (creating if needed) the per-user EmbeddedASR instance."""
    with _easr_instances_lock:
        if user_id not in _easr_instances:
            _easr_instances[user_id] = EmbeddedASR(fs=fs)
        return _easr_instances[user_id]

# Shared thread-pool for parallel ML inference.
# LightGBM, NumPy and SciPy release the GIL during native computation,
# so multiple models genuinely run concurrently on multi-core hosts.
_MODEL_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="ml_inference")


# ─── Dual epoch buffer (4s fast + 30s slow, sliding windows) ─────────────────
# Research consensus: 4-8 sec epochs are needed for stable Welch PSD estimates.
# Below 4 seconds, theta/alpha power estimates have high variance.
# 30-second epochs produce the most accurate emotion classification (2024 paper).
#
# Fast buffer (4s, 2s hop): real-time display, updates every 2 seconds.
# Slow buffer (30s, 15s hop): background emotion state, higher accuracy.
_EPOCH_SECONDS = 4          # fast epoch: seconds of EEG to classify in real-time
_EPOCH_HOP_SECONDS = 2      # fast epoch: slide by 2 seconds (50% overlap)
_SLOW_EPOCH_SECONDS = 30    # slow epoch: seconds of EEG for accurate background state
_SLOW_EPOCH_HOP_SECONDS = 15  # slow epoch: slide by 15 seconds (50% overlap)
_DEFAULT_FS = 256

class _EpochBuffer:
    """Thread-safe dual ring buffer that accumulates EEG frames.

    Maintains two parallel windows:
      - Fast buffer (4s): for real-time emotion display. Returns whatever is
        available, flagged with epoch_ready=False until 4 seconds are buffered.
      - Slow buffer (30s): for accurate background emotion state. Only emits
        when 30 seconds are accumulated, producing more stable classification.

    The slow buffer is independent -- it accumulates all incoming data up to 30s
    and trims to that cap.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._fast_buf: np.ndarray | None = None   # (n_channels, n_accumulated) -- max 4s
        self._slow_buf: np.ndarray | None = None   # (n_channels, n_accumulated) -- max 30s
        self._n_channels: int = 0

    def push_and_get(
        self, signals: np.ndarray, fs: float
    ) -> tuple[np.ndarray, bool, np.ndarray | None, bool]:
        """Add new samples and return both fast and slow epochs.

        Returns:
            (fast_epoch, fast_ready, slow_epoch, slow_ready)
            - fast_epoch: up to 4 seconds of buffered data (always returned).
            - fast_ready: True when fast_epoch contains >= 4 seconds.
            - slow_epoch: 30 seconds of buffered data, or None if not ready.
            - slow_ready: True when slow buffer contains >= 30 seconds.
        """
        fast_samples = int(_EPOCH_SECONDS * fs)
        slow_samples = int(_SLOW_EPOCH_SECONDS * fs)
        n_channels = signals.shape[0]

        with self._lock:
            # Reset both buffers if channel count changed
            if self._fast_buf is None or self._n_channels != n_channels:
                self._fast_buf = signals.copy()
                self._slow_buf = signals.copy()
                self._n_channels = n_channels
            else:
                self._fast_buf = np.concatenate([self._fast_buf, signals], axis=1)
                self._slow_buf = np.concatenate([self._slow_buf, signals], axis=1)

            # Trim fast buffer to 4 seconds
            if self._fast_buf.shape[1] > fast_samples:
                self._fast_buf = self._fast_buf[:, -fast_samples:]

            # Trim slow buffer to 30 seconds
            if self._slow_buf.shape[1] > slow_samples:
                self._slow_buf = self._slow_buf[:, -slow_samples:]

            fast_copy = self._fast_buf.copy()
            slow_copy = self._slow_buf.copy()

        fast_ready = fast_copy.shape[1] >= fast_samples
        slow_ready = slow_copy.shape[1] >= slow_samples
        slow_epoch = slow_copy if slow_ready else None

        return fast_copy, fast_ready, slow_epoch, slow_ready


_epoch_buffers: Dict[str, _EpochBuffer] = {}
_epoch_buffers_lock = threading.Lock()

# ─── EMA smoothing state (per user/session) ───────────────────────────────────
# Stores the previous smoothed probability dict for each user_id.
# alpha=0.3 means: smoothed = 0.3 * new + 0.7 * previous
# This reduces label flips from momentary signal noise without adding latency.
_EMA_ALPHA = 0.3
_ema_state: Dict[str, Dict[str, float]] = {}
_ema_state_lock = threading.Lock()


def _apply_ema(user_id: str, raw_probs: Dict[str, float]) -> Dict[str, float]:
    """Apply exponential moving average to emotion probabilities.

    On first call for a user, returns raw_probs unchanged (no prior state).
    On subsequent calls, blends: smoothed = EMA_ALPHA * raw + (1 - EMA_ALPHA) * prev.
    After blending, normalizes so probabilities sum to 1.0.

    Args:
        user_id: Key used to isolate per-session EMA state.
        raw_probs: Dict mapping emotion labels to probability values [0.0, 1.0].

    Returns:
        Smoothed, normalized probability dict.
    """
    with _ema_state_lock:
        prev = _ema_state.get(user_id)
        if prev is None:
            # First frame — no smoothing, store as initial state
            smoothed = {k: float(v) for k, v in raw_probs.items()}
        else:
            smoothed = {
                k: _EMA_ALPHA * float(raw_probs.get(k, 0.0)) + (1 - _EMA_ALPHA) * float(prev.get(k, 0.0))
                for k in raw_probs
            }
        # Normalize to ensure probabilities sum to 1.0
        total = sum(smoothed.values())
        if total > 0:
            smoothed = {k: v / total for k, v in smoothed.items()}
        _ema_state[user_id] = smoothed
    return smoothed
# ─────────────────────────────────────────────────────────────────────────────


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
        device_type = input_data.device_type
        n_channels = signals.shape[0]

        # Accumulate into per-user epoch buffer; use 4-second window when available
        signals, epoch_ready, slow_epoch, slow_ready = _get_epoch_buffer(user_id).push_and_get(signals, fs)
        n_channels = signals.shape[0]   # re-read after buffer update

        # ── E-ASR artifact cleaning (preserves data instead of discarding) ─────
        # Applied per-channel before model inference. If the per-user E-ASR
        # instance has a fitted baseline, full subspace reconstruction is used;
        # otherwise falls back to simple interpolation of extreme amplitudes.
        easr = _get_easr(user_id, fs=int(fs))
        signals, artifact_cleaned_ratio = easr.clean_multichannel(signals)

        if n_channels > 1:
            avg_signal = np.mean(signals, axis=0)
            eeg = avg_signal
        else:
            eeg = signals[0]

        # ── Parallel inference: 3 models + preprocessing run concurrently ──
        # LightGBM/NumPy/SciPy release the GIL, so this is genuine parallelism.
        loop = asyncio.get_event_loop()
        emotion_input = signals if n_channels >= 2 else eeg
        (
            sleep_result,
            emotion_result,
            dream_result,
            processed,
        ) = await asyncio.gather(
            loop.run_in_executor(_MODEL_EXECUTOR, sleep_model.predict, eeg, fs),
            loop.run_in_executor(_MODEL_EXECUTOR, predict_emotion, user_id, emotion_input, fs, n_channels, device_type),
            loop.run_in_executor(_MODEL_EXECUTOR, dream_model.predict, eeg, fs),
            loop.run_in_executor(_MODEL_EXECUTOR, preprocess, eeg, fs),
        )

        # ── Multimodal fusion: enrich emotion_result with any available biometrics ──
        try:
            bio = get_biometric_snapshot(user_id)
            emotion_result = fusion_model.fuse(emotion_result, bio)
        except Exception:
            pass  # fusion failure must never break the main inference path

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
                # Blend personal prediction into emotion result when confident
                if personal.get("has_personal") and personal.get("personal_confidence", 0.0) >= 0.5:
                    emotion_result = dict(emotion_result)
                    emotion_result["emotion"] = personal["personal_prediction"]
                    emotion_result["personal_override"] = True
                elif personal is not None:
                    emotion_result = dict(emotion_result)
                    emotion_result["personal_override"] = False
        except Exception:
            pass

        # ── EMA smoothing on emotion probabilities ────────────────────────────
        # Reduces frame-to-frame label jitter without altering model weights.
        # Applied after all blending (personal model, fusion) so the smoothing
        # operates on the final output values seen by the client.
        raw_probs = emotion_result.get("probabilities")
        if isinstance(raw_probs, dict) and raw_probs:
            smoothed_probs = _apply_ema(user_id, raw_probs)
            emotion_result = dict(emotion_result)
            emotion_result["probabilities"] = smoothed_probs
            emotion_result["ema_smoothing"] = True
            emotion_result["ema_alpha"] = _EMA_ALPHA
        # ─────────────────────────────────────────────────────────────────────

        # ── Background emotion from 30s slow epoch ────────────────────────────
        # When the slow buffer has accumulated 30 seconds, run a second emotion
        # prediction on the longer window for more accurate background state.
        background_emotion = None
        if slow_ready and slow_epoch is not None:
            try:
                slow_n_channels = slow_epoch.shape[0]
                slow_emotion_input = slow_epoch if slow_n_channels >= 2 else slow_epoch[0]
                background_emotion = await loop.run_in_executor(
                    _MODEL_EXECUTOR,
                    predict_emotion,
                    user_id,
                    slow_emotion_input,
                    fs,
                    slow_n_channels,
                    device_type,
                )
            except Exception:
                pass  # slow-path failure must never break the fast path

        # ── Simple amplitude-threshold signal quality for dashboard badge ────────
        # Computes a 0-100 score and detects artifact type from raw amplitude alone.
        # This is intentionally simple — the detailed signal_quality dict above is
        # for the advanced signal quality panel; this is for the recording area badge.
        _max_amp = float(np.max(np.abs(signals))) if signals.size > 0 else 0.0
        if _max_amp > 100.0:
            _sq_score = max(0, 100 - int((_max_amp - 100.0) / 5.0))
            _artifact_detected = True
            # Determine artifact type from heuristics on the averaged signal
            _diff_max = float(np.max(np.abs(np.diff(eeg)))) if len(eeg) > 1 else 0.0
            _high_freq = float(np.mean(np.abs(eeg - np.mean(eeg))))
            if _diff_max > 50.0:
                _artifact_type = "electrode_pop"
            elif _high_freq > 30.0:
                _artifact_type = "muscle"
            else:
                _artifact_type = "blink"
        elif _max_amp > 75.0:
            # Amplitude in warning zone — quality degrades linearly 100→70
            _sq_score = 100 - int((_max_amp - 75.0) / 25.0 * 30.0)
            _artifact_detected = False
            _artifact_type = "clean"
        else:
            _sq_score = 100
            _artifact_detected = False
            _artifact_type = "clean"

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
            signal_quality_score=_sq_score,
            artifact_detected=_artifact_detected,
            artifact_type=_artifact_type,
            artifact_cleaned_ratio=round(artifact_cleaned_ratio, 4),
            background_emotion=background_emotion,
            background_ready=slow_ready,
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
    fs = request.fs
    loop = asyncio.get_event_loop()

    # Sleep runs first — lucid_dream prediction depends on its output.
    sleep_result = await loop.run_in_executor(_MODEL_EXECUTOR, sleep_model.predict, eeg, fs)
    is_rem = sleep_result.get("stage") == "REM"
    sleep_stage_idx = sleep_result.get("stage_index", 0)

    def _lucid():
        return lucid_dream_model.predict(eeg, fs, is_rem=is_rem, sleep_stage=sleep_stage_idx)

    # Remaining 11 models run in parallel.
    (
        emotions,
        dream_detection,
        flow_state,
        creativity,
        memory_encoding,
        drowsiness,
        cognitive_load,
        attention,
        stress,
        lucid_dream,
        meditation,
    ) = await asyncio.gather(
        loop.run_in_executor(_MODEL_EXECUTOR, emotion_model.predict, eeg, fs),
        loop.run_in_executor(_MODEL_EXECUTOR, dream_model.predict, eeg, fs),
        loop.run_in_executor(_MODEL_EXECUTOR, flow_model.predict, eeg, fs),
        loop.run_in_executor(_MODEL_EXECUTOR, creativity_model.predict, eeg, fs),
        loop.run_in_executor(_MODEL_EXECUTOR, memory_model.predict, eeg, fs),
        loop.run_in_executor(_MODEL_EXECUTOR, drowsiness_model.predict, eeg, fs),
        loop.run_in_executor(_MODEL_EXECUTOR, cognitive_load_model.predict, eeg, fs),
        loop.run_in_executor(_MODEL_EXECUTOR, attention_model.predict, eeg, fs),
        loop.run_in_executor(_MODEL_EXECUTOR, stress_model.predict, eeg, fs),
        loop.run_in_executor(_MODEL_EXECUTOR, _lucid),
        loop.run_in_executor(_MODEL_EXECUTOR, meditation_model.predict, eeg, fs),
    )

    return {
        **result,
        "analysis": {
            "sleep_stage": sleep_result,
            "emotions": emotions,
            "dream_detection": dream_detection,
            "flow_state": flow_state,
            "creativity": creativity,
            "memory_encoding": memory_encoding,
            "drowsiness": drowsiness,
            "cognitive_load": cognitive_load,
            "attention": attention,
            "stress": stress,
            "lucid_dream": lucid_dream,
            "meditation": meditation,
        },
    }

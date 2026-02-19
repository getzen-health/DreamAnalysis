"""Core EEG analysis endpoints: /analyze-eeg, /simulate-eeg."""

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

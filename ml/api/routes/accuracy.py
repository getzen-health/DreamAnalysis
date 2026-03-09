"""Accuracy pipeline: signal quality, confidence, calibration protocol,
state engine, user feedback, personalization, and /analyze-eeg-accurate."""


import numpy as np
from fastapi import APIRouter, HTTPException

from ._shared import (
    _numpy_safe,
    sleep_model, emotion_model, dream_model, flow_model,
    creativity_model, memory_model,
    drowsiness_model, cognitive_load_model, attention_model,
    stress_model, lucid_dream_model, meditation_model,
    _get_state_engine, _get_confidence_cal, _calibration_runners,
    SignalQualityChecker, UserCalibration, CalibrationRunner, CALIBRATION_STEPS,
    FeedbackCollector, PersonalizedPipeline,
    add_uncertainty_labels,
    preprocess, extract_features,
    SignalQualityRequest, CalibrationEpochRequest,
    AccurateAnalysisRequest, FeedbackRequest, SelfReportRequest,
)

router = APIRouter()


# ── Signal Quality ──────────────────────────────────────────────────────────

@router.post("/signal-quality")
async def check_signal_quality(req: SignalQualityRequest):
    """Check EEG signal quality before analysis."""
    checker = SignalQualityChecker(fs=req.sample_rate)
    signals = np.array(req.signals)

    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    if signals.shape[0] == 1:
        result = checker.check_quality(signals[0])
    else:
        result = checker.check_multichannel(signals)

    return _numpy_safe(result)


# ── Confidence & Reliability ────────────────────────────────────────────────

@router.get("/confidence/reliability")
async def get_model_reliability(user_id: str = "default"):
    """Get reliability assessment for all models."""
    return _get_confidence_cal(user_id).get_all_reliability()


@router.post("/confidence/calibrate")
async def calibrate_confidence(model_name: str, raw_confidence: float, user_id: str = "default"):
    """Calibrate a single confidence score."""
    return _get_confidence_cal(user_id).calibrate(model_name, raw_confidence)


# ── Calibration Protocol ────────────────────────────────────────────────────

@router.get("/calibration/steps")
async def get_calibration_steps():
    """Get the 4-step calibration protocol instructions."""
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


@router.post("/calibration/add-epoch/{user_id}")
async def add_calibration_epoch(user_id: str, req: CalibrationEpochRequest):
    """Add an EEG epoch to the calibration for a specific condition."""
    runner = _calibration_runners.get(user_id)
    if runner is None:
        raise HTTPException(404, f"No calibration session for user {user_id}")

    signal = np.array(req.signal)
    checker = SignalQualityChecker(fs=runner.fs)
    quality = checker.check_quality(signal)

    if not quality["is_usable"]:
        return {
            "status": "rejected",
            "reason": "Signal quality too low",
            "quality": quality,
        }

    runner.add_epoch(req.condition, signal)
    return {
        "status": "accepted",
        "quality": quality,
        "progress": runner.get_progress(),
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
    runner = _calibration_runners.get(user_id)
    return {
        "is_calibrated": cal.is_calibrated,
        "calibrated_at": cal.calibrated_at,
        "alpha_reactivity": cal.alpha_reactivity,
        "in_progress": runner.get_progress() if runner else None,
    }


# ── State Engine ────────────────────────────────────────────────────────────

@router.get("/state-engine/summary")
async def get_state_engine_summary():
    """Get current state of all temporal smoothing trackers."""
    return _get_state_engine().get_summary()


@router.get("/state-engine/coherence")
async def get_state_coherence():
    """Check if current brain states are physiologically coherent."""
    return _get_state_engine().get_cross_state_coherence()


# ── User Feedback & Personalization ────────────────────────────────────────

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


# ── Enhanced Analysis ───────────────────────────────────────────────────────

@router.post("/analyze-eeg-accurate")
async def analyze_eeg_accurate(req: AccurateAnalysisRequest):
    """Run EEG analysis with the full accuracy pipeline.

    Steps: signal quality gate → per-user calibration → all 12 models →
    confidence calibration → state transition smoothing → personalization.
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

    # Step 4: Confidence calibration (per-user)
    user_confidence_cal = _get_confidence_cal(req.user_id)
    add_uncertainty_labels(analysis, user_confidence_cal)
    conf_summary = analysis.pop("_confidence_summary", {})

    # Step 5: State smoothing (per-user)
    user_state_engine = _get_state_engine(req.user_id)
    smoothed = user_state_engine.update({
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

    coherence = user_state_engine.get_cross_state_coherence()

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

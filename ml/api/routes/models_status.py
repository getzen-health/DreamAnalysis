"""Model status and benchmark endpoints."""

import json
from fastapi import APIRouter

from ._shared import (
    BENCHMARK_DIR, STATE_PROFILES,
    sleep_model, emotion_model, dream_model, flow_model,
    creativity_model, memory_model,
    drowsiness_model, cognitive_load_model, attention_model,
    stress_model, lucid_dream_model, meditation_model,
)

router = APIRouter()


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
            "classes": ["no_flow", "shallow", "moderate", "deep"],
            "binary_classes": ["no_flow", "flow"],
            "accuracy": "62.86% CV",
            "accuracy_note": "Marginal accuracy; binary mode recommended. Calibration required.",
            "calibrated": flow_model.is_calibrated,
        },
        "creativity": {
            "loaded": True,
            "type": creativity_model.model_type,
            "classes": ["analytical", "transitional", "creative", "insight"],
            "experimental": creativity_model.EXPERIMENTAL,
            "confidence_note": creativity_model.CONFIDENCE_NOTE,
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

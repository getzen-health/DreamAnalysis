"""On-device emotion inference capabilities — reports what can run locally.

Describes which inference components can run client-side (zero server dependency)
vs which require server processing, helping clients decide routing strategy.

GET /on-device/capabilities  — full capability report
GET /on-device/models        — ONNX models available for client-side use
GET /on-device/status        — health check

Privacy model:
- EEG emotion (LGBM ONNX) — already runs on device (model in client/public/models/)
- Voice biomarkers (jitter/shimmer/HNR/pauses) — pure DSP, no ML, client-capable
- Health analysis (HRV/sleep/activity) — statistical math, client-capable
- EI composite scoring — weighted averages, client-capable
- Voice emotion (emotion2vec+) — requires server (PyTorch, 300MB model)
- LLM coaching — requires server (Cerebras API)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter

router = APIRouter(tags=["On-Device Inference"])

# ── Capability definitions ─────────────────────────────────────────────────────

_CAPABILITIES: List[Dict] = [
    {
        "component": "eeg_emotion",
        "display_name": "EEG Emotion Classification",
        "can_run_on_device": True,
        "method": "ONNX (LightGBM, 2.2 MB)",
        "model_file": "client/public/models/emotion_mega_lgbm.onnx",
        "accuracy": "74.21% cross-subject CV",
        "latency_ms": "<10",
        "notes": "Already runs client-side via onnxruntime-web. No server needed.",
        "data_sent_to_server": "None",
    },
    {
        "component": "voice_biomarkers",
        "display_name": "Voice Biomarkers (jitter/shimmer/HNR/pauses)",
        "can_run_on_device": True,
        "method": "Pure DSP (Web Audio API)",
        "model_file": None,
        "accuracy": "Deterministic (no ML)",
        "latency_ms": "<50",
        "notes": "jitter, shimmer, HNR, pause ratio, speech rate — computable in JS from WebRTC audio.",
        "data_sent_to_server": "None",
    },
    {
        "component": "health_analysis",
        "display_name": "Health Data Analysis (HRV/sleep/activity)",
        "can_run_on_device": True,
        "method": "Statistical math (JS)",
        "model_file": None,
        "accuracy": "Deterministic",
        "latency_ms": "<5",
        "notes": "RMSSD, SDNN, sleep stage totals, step counts — pure arithmetic on Apple Health / Fitbit data.",
        "data_sent_to_server": "None",
    },
    {
        "component": "ei_composite",
        "display_name": "EI Composite Scoring",
        "can_run_on_device": True,
        "method": "Weighted average (JS)",
        "model_file": None,
        "accuracy": "N/A (derived metric)",
        "latency_ms": "<1",
        "notes": "Weighted combination of voice + health + EEG scores — trivial JS computation.",
        "data_sent_to_server": "None",
    },
    {
        "component": "supplement_correlation",
        "display_name": "Supplement-Mood Correlation",
        "can_run_on_device": True,
        "method": "Statistical correlation (JS)",
        "model_file": None,
        "accuracy": "N/A (statistical)",
        "latency_ms": "<10",
        "notes": "Pearson/Spearman correlation between supplement log and mood scores. Runs locally.",
        "data_sent_to_server": "None",
    },
    {
        "component": "sleep_mood_prediction",
        "display_name": "Sleep-to-Mood Prediction",
        "can_run_on_device": True,
        "method": "Small ONNX (exportable)",
        "model_file": "ml/models/saved/sleep_staging_model.pkl",
        "accuracy": "92.98% sleep staging",
        "latency_ms": "<20",
        "notes": "Sleep staging model can be exported to ONNX (~500KB). Mood prediction is linear regression.",
        "data_sent_to_server": "None (after export)",
    },
    {
        "component": "voice_emotion",
        "display_name": "Voice Emotion (emotion2vec+)",
        "can_run_on_device": False,
        "method": "PyTorch transformer (server)",
        "model_file": "ml/models/voice_emotion_model.py",
        "accuracy": "emotion2vec_plus_large (state-of-art)",
        "latency_ms": "200-800 (network + inference)",
        "notes": (
            "emotion2vec+ requires PyTorch and >300MB model — not feasible in browser. "
            "Fallback: DistilHuBERT ONNX export (~23MB) can run on-device with reduced accuracy. "
            "Alternative: send MFCC features only (not raw audio) to minimize data transfer."
        ),
        "data_sent_to_server": "Raw audio (PCM) or MFCC features (privacy-preserving option)",
    },
    {
        "component": "llm_coaching",
        "display_name": "AI Emotion Coach (LLM)",
        "can_run_on_device": False,
        "method": "Cerebras API (llama3.1-8b)",
        "model_file": None,
        "accuracy": "N/A (generative)",
        "latency_ms": "500-2000",
        "notes": (
            "LLM coaching requires cloud API. Rule-based intervention fallback available offline. "
            "No personal data required for API call — only anonymized emotion state is sent."
        ),
        "data_sent_to_server": "Anonymized emotion state + user-selected context (no biometrics)",
    },
]

# ── ONNX models available for client download ──────────────────────────────────

def _get_available_onnx_models() -> List[Dict]:
    """Scan for ONNX models that exist on disk and can be served to clients."""
    results = []
    model_dir = Path("models/saved")
    client_model_dir = Path("../client/public/models")

    candidates = [
        {
            "name": "emotion_mega_lgbm",
            "description": "EEG emotion classifier (6 classes + valence/arousal)",
            "size_note": "~2.2 MB",
            "accuracy": "74.21% cross-subject CV",
            "input": "41 EEG features (DASM/RASM/FAA/band powers)",
        },
        {
            "name": "sleep_staging_model",
            "description": "Sleep stage classifier (Wake/N1/N2/N3/REM)",
            "size_note": "~500 KB (estimated after export)",
            "accuracy": "92.98%",
            "input": "17 EEG features from single channel",
        },
    ]

    for c in candidates:
        onnx_server = model_dir / f"{c['name']}.onnx"
        onnx_client = client_model_dir / f"{c['name']}.onnx"
        available = onnx_server.exists() or onnx_client.exists()
        results.append({
            **c,
            "available": available,
            "path": str(onnx_client) if onnx_client.exists() else str(onnx_server),
        })

    return results


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/on-device/capabilities")
def get_capabilities() -> dict:
    """Return full on-device vs server-side capability report.

    Use this to decide whether to route inference locally or to the ML backend.
    Components where can_run_on_device=True never need to send data to the server.
    """
    on_device = [c for c in _CAPABILITIES if c["can_run_on_device"]]
    server_required = [c for c in _CAPABILITIES if not c["can_run_on_device"]]

    return {
        "summary": {
            "total_components": len(_CAPABILITIES),
            "on_device": len(on_device),
            "server_required": len(server_required),
            "privacy_note": (
                f"{len(on_device)}/{len(_CAPABILITIES)} components send zero data to server. "
                "Voice emotion is the primary server-dependent component."
            ),
        },
        "on_device_components": on_device,
        "server_required_components": server_required,
        "recommended_architecture": {
            "eeg": "Client-side ONNX (already implemented)",
            "voice_biomarkers": "Client-side Web Audio API DSP",
            "health": "Client-side statistical computation",
            "voice_emotion": "Server (or DistilHuBERT ONNX fallback for privacy mode)",
            "llm_coach": "Server (Cerebras API, anonymized state only)",
        },
        "privacy_mode_instructions": (
            "Enable privacy mode in settings to use on-device fallbacks: "
            "voice features replace voice emotion transformer, "
            "rule-based coach replaces LLM coach."
        ),
    }


@router.get("/on-device/models")
def get_onnx_models() -> dict:
    """List ONNX models available for client-side inference."""
    models = _get_available_onnx_models()
    available = [m for m in models if m["available"]]

    return {
        "models": models,
        "available_count": len(available),
        "usage": (
            "Load these models with onnxruntime-web in the browser. "
            "Input/output shapes match the server-side inference pipeline."
        ),
        "runtime": "onnxruntime-web v1.17+ recommended",
    }


@router.get("/on-device/status")
def status() -> dict:
    return {
        "status": "ready",
        "capabilities": [c["component"] for c in _CAPABILITIES if c["can_run_on_device"]],
        "server_dependent": [c["component"] for c in _CAPABILITIES if not c["can_run_on_device"]],
        "privacy_first": True,
        "note": (
            "All EEG, health, biomarker, and EI scoring runs on-device by default. "
            "Voice emotion and LLM coaching require server connectivity."
        ),
    }

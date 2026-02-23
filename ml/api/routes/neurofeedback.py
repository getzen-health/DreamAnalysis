"""Neurofeedback session endpoints."""

from fastapi import APIRouter, HTTPException

from ._shared import (
    NeurofeedbackProtocol, PROTOCOLS,
    NeurofeedbackStartRequest, NeurofeedbackEvalRequest,
)
import api.routes._shared as _state

router = APIRouter()


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
    _state._nf_protocol = NeurofeedbackProtocol(
        protocol_type=request.protocol_type,
        target_band=request.target_band,
        threshold=request.threshold,
    )

    if request.calibrate:
        _state._nf_protocol.start_calibration()
        return {"status": "calibrating", "protocol": request.protocol_type}

    _state._nf_protocol.start()
    return {"status": "active", "protocol": request.protocol_type}


@router.post("/neurofeedback/evaluate")
async def evaluate_neurofeedback(request: NeurofeedbackEvalRequest):
    """Evaluate current EEG against the active neurofeedback protocol."""
    if _state._nf_protocol is None:
        raise HTTPException(status_code=400, detail="No active neurofeedback session")

    if _state._nf_protocol.is_calibrating:
        done = _state._nf_protocol.add_calibration_sample(request.band_powers)
        progress = len(_state._nf_protocol.baseline_samples) / 30.0
        if done:
            return {
                "status": "calibration_complete",
                "baseline": _state._nf_protocol.baseline,
                "progress": 1.0,
            }
        return {"status": "calibrating", "progress": float(progress)}

    result = _state._nf_protocol.evaluate(request.band_powers, request.channel_powers)
    return {"status": "active", **result}


@router.post("/neurofeedback/stop")
async def stop_neurofeedback():
    """Stop the current neurofeedback session and return stats."""
    if _state._nf_protocol is None:
        raise HTTPException(status_code=400, detail="No active neurofeedback session")

    stats = _state._nf_protocol.stop()
    _state._nf_protocol = None
    return {"status": "stopped", "stats": stats}

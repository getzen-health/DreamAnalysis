"""EEG-guided binaural beat neurofeedback endpoints."""

import logging

from fastapi import APIRouter, Body

log = logging.getLogger(__name__)
router = APIRouter()

# ── Binaural Beat Neurofeedback ──────────────────────────────────────────
_binaural_controller = None


def _get_binaural_controller():
    global _binaural_controller
    if _binaural_controller is None:
        from models.binaural_feedback import BinauralFeedbackController
        _binaural_controller = BinauralFeedbackController()
    return _binaural_controller


@router.post("/binaural/start")
async def binaural_start(payload: dict = Body(default={})):
    """Start a binaural beat neurofeedback session.

    target_state: "focus" | "relax" | "meditation" | "sleep" | "flow" | "calm"
    volume: 0.0-1.0
    """
    ctrl = _get_binaural_controller()
    result = ctrl.start_session(
        target_state=payload.get("target_state", "relax"),
        volume=float(payload.get("volume", 0.3)),
    )
    return result


@router.post("/binaural/stop")
async def binaural_stop():
    """Stop the binaural beat session."""
    ctrl = _get_binaural_controller()
    return ctrl.stop_session()


@router.post("/binaural/update")
async def binaural_update(payload: dict = Body(...)):
    """Feed EEG band powers to adapt beat frequency.

    eeg_features: {alpha, beta, theta, delta} band powers
    """
    ctrl = _get_binaural_controller()
    features = payload.get("eeg_features", {})
    return ctrl.update_from_eeg(features)


@router.get("/binaural/status")
async def binaural_status():
    """Get current binaural beat parameters."""
    ctrl = _get_binaural_controller()
    return ctrl.get_status()


@router.get("/binaural/presets")
async def binaural_presets():
    """List available entrainment presets."""
    from models.binaural_feedback import ENTRAINMENT_TARGETS
    return {"presets": ENTRAINMENT_TARGETS}

"""EEG device management endpoints (BrainFlow + Emotiv adapters)."""

from fastapi import APIRouter, HTTPException

from ._shared import _get_device_manager, DeviceConnectRequest, emotion_model

router = APIRouter()

# ── Emotiv adapter singleton ──────────────────────────────────────────────────
# Lazy-initialised on first Emotiv connect; None when no Emotiv device is active.
_emotiv: "EmotivAdapter | None" = None  # type: ignore[name-defined]


def _is_emotiv(device_type: str) -> bool:
    return device_type.startswith("emotiv_")


def _get_emotiv():
    """Return (or create) the shared EmotivAdapter instance."""
    global _emotiv
    if _emotiv is None:
        try:
            from hardware.emotiv_adapter import EmotivAdapter
            _emotiv = EmotivAdapter()
        except Exception:
            pass
    return _emotiv


def _active_manager(device_type: str | None = None):
    """Return the correct adapter for a device_type.

    If device_type is given and is an Emotiv device, returns EmotivAdapter.
    Otherwise returns the BrainFlowManager (or None if BrainFlow unavailable).
    Falls back to whichever adapter is currently connected.
    """
    if device_type and _is_emotiv(device_type):
        return _get_emotiv()

    # Check if the Emotiv adapter is currently active
    if _emotiv is not None and _emotiv.is_connected:
        return _emotiv

    return _get_device_manager()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/devices")
async def list_devices():
    """List all available EEG devices (BrainFlow + Emotiv)."""
    bf_manager = _get_device_manager()
    emotiv_adapter = _get_emotiv()

    brainflow_devices = []
    brainflow_available = False
    if bf_manager is not None:
        brainflow_available = True
        brainflow_devices = bf_manager.discover_devices()

    emotiv_devices = []
    if emotiv_adapter is not None:
        emotiv_devices = emotiv_adapter.discover_devices()

    return {
        "brainflow_available": brainflow_available,
        "devices": brainflow_devices + emotiv_devices,
        "connected": (
            (bf_manager is not None and bf_manager.is_connected)
            or (_emotiv is not None and _emotiv.is_connected)
        ),
    }


@router.post("/devices/connect")
async def connect_device(request: DeviceConnectRequest):
    """Connect to an EEG device (BrainFlow or Emotiv)."""
    if _is_emotiv(request.device_type):
        adapter = _get_emotiv()
        if adapter is None:
            raise HTTPException(
                status_code=503,
                detail="Emotiv adapter unavailable — check hardware/emotiv_adapter.py",
            )
        try:
            result = adapter.connect(request.device_type, request.params or {})
            emotion_model.set_device_type(request.device_type)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # BrainFlow path
    is_ble_device = request.device_type.startswith("muse_") or request.device_type in (
        "openbci_ganglion",
    )
    manager = _get_device_manager()
    if manager is None:
        if is_ble_device:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{request.device_type} requires Bluetooth — not available on this "
                    "remote server. Select 'Synthetic' board to demo the app, or run "
                    "the ML backend locally for real Muse data."
                ),
            )
        raise HTTPException(status_code=400, detail="BrainFlow not available on this server.")
    try:
        result = manager.connect(request.device_type, request.params or {})
        emotion_model.set_device_type(request.device_type)
        return result
    except RuntimeError as e:
        msg = str(e)
        if is_ble_device:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{request.device_type} requires Bluetooth — not available on this "
                    "remote server. Select 'Synthetic' board to demo the app, or run "
                    f"the ML backend locally for real Muse data. (Detail: {msg})"
                ),
            )
        raise HTTPException(status_code=500, detail=msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/devices/disconnect")
async def disconnect_device():
    """Disconnect from the current EEG device."""
    # Disconnect whichever adapter is connected
    if _emotiv is not None and _emotiv.is_connected:
        try:
            _emotiv.disconnect()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        emotion_model.set_device_type(None)
        return {"status": "disconnected"}

    manager = _get_device_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="BrainFlow not available")
    try:
        manager.disconnect()
        emotion_model.set_device_type(None)
        return {"status": "disconnected"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/devices/status")
async def device_status():
    """Get current device status."""
    # Check Emotiv first
    if _emotiv is not None and _emotiv.is_connected:
        return {
            "connected": True,
            "streaming": _emotiv.is_streaming,
            "device_type": _emotiv.current_device_type,
            "n_channels": _emotiv.n_channels,
            "sample_rate": _emotiv.sample_rate,
            "adapter": "emotiv",
            "brainflow_available": _get_device_manager() is not None,
        }

    manager = _get_device_manager()
    if manager is None:
        return {"connected": False, "brainflow_available": False}
    return {
        "connected": manager.is_connected,
        "streaming": manager.is_streaming,
        "device_type": manager.current_device_type,
        "n_channels": manager.n_channels,
        "sample_rate": manager.sample_rate,
        "adapter": "brainflow",
        "brainflow_available": True,
    }


@router.post("/devices/start-stream")
async def start_stream():
    """Start data streaming from the connected device."""
    adapter = _emotiv if (_emotiv is not None and _emotiv.is_connected) else _get_device_manager()
    if adapter is None:
        raise HTTPException(status_code=503, detail="No adapter available")
    if not adapter.is_connected:
        raise HTTPException(status_code=400, detail="No device connected")
    try:
        adapter.start_streaming()
        return {"status": "streaming", "sample_rate": adapter.sample_rate}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/devices/stop-stream")
async def stop_stream():
    """Stop data streaming."""
    adapter = _emotiv if (_emotiv is not None and _emotiv.is_connected) else _get_device_manager()
    if adapter is None:
        raise HTTPException(status_code=503, detail="No adapter available")
    try:
        adapter.stop_streaming()
        return {"status": "stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

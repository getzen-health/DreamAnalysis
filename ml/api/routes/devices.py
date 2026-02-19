"""EEG device management endpoints (BrainFlow / Muse 2)."""

from fastapi import APIRouter, HTTPException

from ._shared import _get_device_manager, DeviceConnectRequest

router = APIRouter()


@router.get("/devices")
async def list_devices():
    """List available EEG devices."""
    manager = _get_device_manager()
    if manager is None:
        return {
            "brainflow_available": False,
            "devices": [],
            "message": "BrainFlow not installed. Install with: pip install brainflow",
        }
    devices = manager.discover_devices()
    return {
        "brainflow_available": True,
        "devices": devices,
        "connected": manager.is_connected,
    }


@router.post("/devices/connect")
async def connect_device(request: DeviceConnectRequest):
    """Connect to an EEG device."""
    manager = _get_device_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="BrainFlow not available")
    try:
        return manager.connect(request.device_type, request.params or {})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/devices/disconnect")
async def disconnect_device():
    """Disconnect from the current EEG device."""
    manager = _get_device_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="BrainFlow not available")
    try:
        manager.disconnect()
        return {"status": "disconnected"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/devices/status")
async def device_status():
    """Get current device status."""
    manager = _get_device_manager()
    if manager is None:
        return {"connected": False, "brainflow_available": False}
    return {
        "connected": manager.is_connected,
        "streaming": manager.is_streaming,
        "device_type": manager.current_device_type,
        "n_channels": manager.n_channels,
        "sample_rate": manager.sample_rate,
        "brainflow_available": True,
    }


@router.post("/devices/start-stream")
async def start_stream():
    """Start data streaming from the connected device."""
    manager = _get_device_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="BrainFlow not available")
    if not manager.is_connected:
        raise HTTPException(status_code=400, detail="No device connected")
    try:
        manager.start_streaming()
        return {"status": "streaming", "sample_rate": manager.sample_rate}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/devices/stop-stream")
async def stop_stream():
    """Stop data streaming."""
    manager = _get_device_manager()
    if manager is None:
        raise HTTPException(status_code=503, detail="BrainFlow not available")
    try:
        manager.stop_streaming()
        return {"status": "stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

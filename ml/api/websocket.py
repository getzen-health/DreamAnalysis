"""WebSocket endpoint for real-time EEG streaming."""

import asyncio
import time
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from processing.eeg_processor import extract_features, extract_band_powers, preprocess


async def eeg_stream_endpoint(websocket: WebSocket):
    """WebSocket at /ws/eeg-stream: sends JSON frames at ~4Hz.

    Each frame contains:
    {
        "signals": [[...], ...],  // channels x samples
        "analysis": {
            "band_powers": {...},
            "features": {...}
        },
        "timestamp": 1234567890.123
    }

    Reads from BrainFlow device if connected, otherwise sends nothing.
    When a session is being recorded, automatically pipes frames to the recorder.
    Client should handle reconnection.
    """
    await websocket.accept()

    try:
        # Try to get device manager
        device_manager = None
        try:
            from hardware.brainflow_manager import BRAINFLOW_AVAILABLE
            if BRAINFLOW_AVAILABLE:
                from api.routes import _get_device_manager
                device_manager = _get_device_manager()
        except Exception:
            pass

        # Get session recorder reference
        session_recorder = None
        try:
            from api.routes import _session_recorder
            session_recorder = _session_recorder
        except Exception:
            pass

        frame_interval = 0.25  # 4 Hz

        while True:
            start_time = time.time()

            frame = None

            if device_manager and device_manager.is_streaming:
                data = device_manager.get_current_data(n_samples=64)
                if data and data["signals"] and len(data["signals"][0]) > 0:
                    signals = np.array(data["signals"])
                    # Analyze first channel
                    eeg = signals[0] if signals.shape[0] > 0 else np.zeros(64)
                    fs = data.get("sample_rate", 256)

                    try:
                        processed = preprocess(eeg, fs)
                        bands = extract_band_powers(processed, fs)
                        features = extract_features(processed, fs)
                    except Exception:
                        bands = {}
                        features = {}

                    frame = {
                        "signals": data["signals"],
                        "analysis": {
                            "band_powers": bands,
                            "features": features,
                        },
                        "timestamp": time.time(),
                        "n_channels": data.get("n_channels", 1),
                        "sample_rate": fs,
                    }

                    # Pipe to session recorder if recording is active
                    if session_recorder and session_recorder.is_recording:
                        try:
                            session_recorder.add_frame(signals, frame["analysis"])
                        except Exception:
                            pass

            if frame:
                await websocket.send_json(frame)

            # Maintain frame rate
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_interval - elapsed)
            await asyncio.sleep(sleep_time)

    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass

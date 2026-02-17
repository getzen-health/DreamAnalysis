"""BrainFlow hardware manager for EEG device integration.

Supports: OpenBCI Cyton/Ganglion, Muse 2/S, Emotiv EPOC, NeuroSky, Synthetic.
Gracefully degrades if brainflow is not installed.
"""

import threading
from typing import Dict, List, Optional, Callable

try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds  # noqa: F401
    from brainflow.data_filter import DataFilter  # noqa: F401

    BRAINFLOW_AVAILABLE = True
except ImportError:
    BRAINFLOW_AVAILABLE = False

# Device type to BrainFlow board ID mapping
DEVICE_MAP = {
    "synthetic": {"board_id": -1, "name": "Synthetic Board", "channels": 16, "sample_rate": 256},
    "openbci_cyton": {"board_id": 0, "name": "OpenBCI Cyton", "channels": 8, "sample_rate": 250},
    "openbci_ganglion": {"board_id": 1, "name": "OpenBCI Ganglion", "channels": 4, "sample_rate": 200},
    "openbci_cyton_daisy": {"board_id": 2, "name": "OpenBCI Cyton+Daisy", "channels": 16, "sample_rate": 125},
    "muse_2": {"board_id": 22, "name": "Muse 2", "channels": 4, "sample_rate": 256},
    "muse_s": {"board_id": 21, "name": "Muse S", "channels": 4, "sample_rate": 256},
    "emotiv_epoc": {"board_id": 45, "name": "Emotiv EPOC", "channels": 14, "sample_rate": 256},
    "neurosky": {"board_id": 13, "name": "NeuroSky MindWave", "channels": 1, "sample_rate": 512},
}


class BrainFlowManager:
    """Manages BrainFlow board connections and data streaming."""

    def __init__(self):
        self.board = None
        self.is_connected = False
        self.is_streaming = False
        self.current_device_type = None
        self.n_channels = 0
        self.sample_rate = 0
        self._stream_thread = None
        self._stream_callback = None
        self._stop_event = threading.Event()

    def discover_devices(self) -> List[Dict]:
        """List available devices. Always includes synthetic board."""
        devices = []
        for device_type, info in DEVICE_MAP.items():
            devices.append({
                "type": device_type,
                "name": info["name"],
                "channels": info["channels"],
                "sample_rate": info["sample_rate"],
                "available": BRAINFLOW_AVAILABLE,
            })
        return devices

    def connect(self, device_type: str, params: Dict = None) -> Dict:
        """Connect to an EEG device.

        Args:
            device_type: Key from DEVICE_MAP (e.g., 'synthetic', 'openbci_cyton')
            params: Optional parameters (serial_port, mac_address, etc.)

        Returns:
            Connection status dict.
        """
        if not BRAINFLOW_AVAILABLE:
            raise RuntimeError("BrainFlow is not installed")

        if self.is_connected:
            self.disconnect()

        if device_type not in DEVICE_MAP:
            raise ValueError(f"Unknown device: {device_type}. Available: {list(DEVICE_MAP.keys())}")

        device_info = DEVICE_MAP[device_type]
        board_id = device_info["board_id"]

        bf_params = BrainFlowInputParams()
        if params:
            if "serial_port" in params:
                bf_params.serial_port = params["serial_port"]
            if "mac_address" in params:
                bf_params.mac_address = params["mac_address"]
            if "ip_address" in params:
                bf_params.ip_address = params["ip_address"]
            if "ip_port" in params:
                bf_params.ip_port = params["ip_port"]

        self.board = BoardShim(board_id, bf_params)
        self.board.prepare_session()

        self.is_connected = True
        self.current_device_type = device_type
        self.n_channels = len(BoardShim.get_eeg_channels(board_id))
        self.sample_rate = BoardShim.get_sampling_rate(board_id)

        return {
            "status": "connected",
            "device": device_info["name"],
            "channels": self.n_channels,
            "sample_rate": self.sample_rate,
        }

    def disconnect(self):
        """Disconnect from the current device."""
        if self.is_streaming:
            self.stop_streaming()

        if self.board is not None:
            try:
                self.board.release_session()
            except Exception:
                pass

        self.board = None
        self.is_connected = False
        self.current_device_type = None
        self.n_channels = 0
        self.sample_rate = 0

    def start_streaming(self, callback: Optional[Callable] = None):
        """Start data streaming from the connected board.

        Args:
            callback: Optional function called with (signals, timestamp) at each read.
        """
        if not self.is_connected or self.board is None:
            raise RuntimeError("No device connected")

        self.board.start_stream()
        self.is_streaming = True
        self._stream_callback = callback
        self._stop_event.clear()

    def stop_streaming(self):
        """Stop data streaming."""
        self._stop_event.set()

        if self.board is not None:
            try:
                self.board.stop_stream()
            except Exception:
                pass

        self.is_streaming = False
        self._stream_callback = None

    def get_current_data(self, n_samples: int = 256) -> Optional[Dict]:
        """Get the latest n_samples from the board's ring buffer.

        Args:
            n_samples: Number of samples to retrieve.

        Returns:
            Dict with 'signals' (channels x samples), 'timestamp', 'sample_rate'.
        """
        if not self.is_connected or self.board is None:
            return None

        if not BRAINFLOW_AVAILABLE:
            return None

        board_id = DEVICE_MAP.get(self.current_device_type, {}).get("board_id", -1)

        try:
            data = self.board.get_current_board_data(n_samples)
            eeg_channels = BoardShim.get_eeg_channels(board_id)
            timestamp_channel = BoardShim.get_timestamp_channel(board_id)

            signals = data[eeg_channels].tolist()
            timestamps = data[timestamp_channel].tolist() if timestamp_channel < data.shape[0] else []

            return {
                "signals": signals,
                "timestamps": timestamps,
                "sample_rate": self.sample_rate,
                "n_channels": len(eeg_channels),
            }
        except Exception:
            return None

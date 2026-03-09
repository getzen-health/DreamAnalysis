"""BrainFlow hardware manager for EEG device integration.

Supports: OpenBCI Cyton/Ganglion, Muse 2/S, Emotiv EPOC, NeuroSky, Synthetic.
Gracefully degrades if brainflow is not installed.
"""

import logging
import threading
from typing import Dict, List, Optional, Callable

import numpy as np

log = logging.getLogger(__name__)

try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds  # noqa: F401
    from brainflow.data_filter import DataFilter  # noqa: F401

    BRAINFLOW_AVAILABLE = True
except Exception:
    # Catches ImportError, ModuleNotFoundError, pkg_resources deprecation errors,
    # and any other failure during brainflow import.
    BRAINFLOW_AVAILABLE = False

# Device type to BrainFlow board ID mapping
# Board IDs verified against BrainFlow v5.20.1
DEVICE_MAP = {
    "synthetic": {"board_id": -1, "name": "Synthetic Board", "channels": 16, "sample_rate": 256},
    "openbci_cyton": {"board_id": 0, "name": "OpenBCI Cyton", "channels": 8, "sample_rate": 250},
    "openbci_ganglion": {"board_id": 1, "name": "OpenBCI Ganglion", "channels": 4, "sample_rate": 200},
    "openbci_cyton_daisy": {"board_id": 2, "name": "OpenBCI Cyton+Daisy", "channels": 16, "sample_rate": 125},
    "muse_2": {
        "board_id": 38, "name": "Muse 2", "channels": 4, "sample_rate": 256,
        "eeg_names": ["TP9", "AF7", "AF8", "TP10"],
        "notes": "Native Bluetooth (macOS/Linux). No dongle needed.",
    },
    "muse_2_bled": {
        "board_id": 22, "name": "Muse 2 (BLED Dongle)", "channels": 4, "sample_rate": 256,
        "eeg_names": ["TP9", "AF7", "AF8", "TP10"],
        "notes": "Requires BLED112 USB Bluetooth dongle.",
    },
    "muse_s": {
        "board_id": 39, "name": "Muse S", "channels": 4, "sample_rate": 256,
        "eeg_names": ["TP9", "AF7", "AF8", "TP10"],
        "notes": "Native Bluetooth (macOS/Linux). No dongle needed.",
    },
    "muse_s_bled": {
        "board_id": 21, "name": "Muse S (BLED Dongle)", "channels": 4, "sample_rate": 256,
        "eeg_names": ["TP9", "AF7", "AF8", "TP10"],
        "notes": "Requires BLED112 USB Bluetooth dongle.",
    },
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
        self.eeg_channel_names = []
        self._board_id = None
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
            if "timeout" in params:
                bf_params.timeout = int(params["timeout"])

        # BLE devices (Muse) need longer discovery timeout
        is_ble = device_type.startswith("muse_")
        if is_ble and bf_params.timeout == 0:
            bf_params.timeout = 15  # 15s BLE scan (default 6s is often too short)

        # Enable verbose BrainFlow logging for debugging
        if is_ble:
            BoardShim.enable_dev_board_logger()

        self.board = BoardShim(board_id, bf_params)
        try:
            self.board.prepare_session()
        except Exception as e:
            # Clean up so subsequent connect() calls start fresh
            self.board = None
            if is_ble:
                raise RuntimeError(
                    f"Bluetooth scan failed for {device_type}. "
                    "If you're using the remote (Railway) backend, Bluetooth is unavailable — "
                    "select 'Synthetic' board instead. "
                    f"Original error: {e}"
                ) from e
            raise RuntimeError(f"BrainFlow session error: {e}") from e

        self.is_connected = True
        self.current_device_type = device_type
        self._board_id = board_id
        self.n_channels = len(BoardShim.get_eeg_channels(board_id))
        self.sample_rate = BoardShim.get_sampling_rate(board_id)

        try:
            names = BoardShim.get_eeg_names(board_id)
            self.eeg_channel_names = names.split(",") if isinstance(names, str) else list(names)
        except Exception:
            self.eeg_channel_names = [f"CH{i}" for i in range(self.n_channels)]

        return {
            "status": "connected",
            "device": device_info["name"],
            "channels": self.n_channels,
            "channel_names": self.eeg_channel_names,
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
        self._board_id = None
        self.n_channels = 0
        self.sample_rate = 0
        self.eeg_channel_names = []

    def start_streaming(self, callback: Optional[Callable] = None):
        """Start data streaming from the connected board.  Idempotent — safe to
        call when already streaming.

        Args:
            callback: Optional function called with (signals, timestamp) at each read.
        """
        if not self.is_connected or self.board is None:
            raise RuntimeError("No device connected")
        if self.is_streaming:
            return  # already streaming — no-op

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

        if not BRAINFLOW_AVAILABLE or self._board_id is None:
            return None

        try:
            data = self.board.get_current_board_data(n_samples)
            eeg_channels = BoardShim.get_eeg_channels(self._board_id)
            timestamp_channel = BoardShim.get_timestamp_channel(self._board_id)

            signals_np = data[eeg_channels]  # shape: (n_channels, n_samples)

            # Apply mastoid re-reference for Muse devices (ch0=TP9, ch3=TP10 are mastoids)
            is_muse = self.current_device_type and self.current_device_type.startswith("muse_")
            if is_muse and signals_np.shape[0] >= 4:
                from processing.eeg_processor import rereference_to_mastoid
                signals_np = rereference_to_mastoid(signals_np, left_mastoid_ch=0, right_mastoid_ch=3)

            signals = signals_np.tolist()
            timestamps = data[timestamp_channel].tolist() if timestamp_channel < data.shape[0] else []

            return {
                "signals": signals,
                "timestamps": timestamps,
                "sample_rate": self.sample_rate,
                "n_channels": len(eeg_channels),
                "channel_names": self.eeg_channel_names,
            }
        except Exception:
            return None

    def get_ppg_data(self) -> Optional[np.ndarray]:
        """Get latest PPG data from Muse 2 ANCILLARY preset.

        Returns 1D numpy array of PPG samples, or None if unavailable.
        Note: requires board started with ANCILLARY preset (config 'p50').
        """
        try:
            if not self.is_connected or self.board is None:
                return None
            if not BRAINFLOW_AVAILABLE or self._board_id is None:
                return None
            from brainflow.board_shim import BoardShim, BoardIds
            # PPG channels for Muse 2
            ppg_channels = BoardShim.get_ppg_channels(BoardIds.MUSE_2_BOARD.value)
            if not ppg_channels:
                return None
            data = self.board.get_current_board_data(256)
            if data.shape[1] == 0:
                return None
            # Return first PPG channel
            return data[ppg_channels[0], :]
        except Exception as e:
            log.debug("PPG data unavailable: %s", e)
            return None

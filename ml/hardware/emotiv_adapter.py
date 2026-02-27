"""Emotiv EPOC / EPOC X adapter for NeuralDreamWorkshop.

Provides the same interface as BrainFlowManager so the ML pipeline can consume
Emotiv data without modification — just swap the manager class.

Two backends (tried in order):
    1. Cortex WebSocket API — connects to Emotiv's local WebSocket server that
       ships with EmotivPro. No proprietary binary SDK required; uses the
       documented JSON-RPC protocol on wss://localhost:6789.
    2. EDF / CSV file reader — plays back saved recordings at real-time rate.
       Used for AMIGOS dataset offline analysis and lab replay.

Output format mirrors BrainFlowManager.get_current_data():
    {
        "signals":       [[float, ...], ...],  # (n_channels, n_samples) lists
        "timestamps":    [float, ...],
        "sample_rate":   int,
        "n_channels":    int,
        "channel_names": [str, ...],
    }

Emotiv EPOC X hardware spec:
    Channels (14): AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
    Sampling rate: 128 Hz (standard) / 256 Hz (Researcher plan)
    ADC: 14-bit, ±4 mV range, 0.5–43 Hz hardware bandpass
    Reference: CMS (Common Mode Sense) + DRL (Driven Right Leg)
    Connectivity: Bluetooth (BTLE / Emotiv USB dongle)

Usage:
    from hardware.emotiv_adapter import EmotivAdapter

    adapter = EmotivAdapter()
    adapter.connect("emotiv_epoc_x", params={"client_id": "...", "client_secret": "..."})
    adapter.start_streaming()
    data = adapter.get_current_data(256)  # same as BrainFlowManager
    adapter.disconnect()

    # For AMIGOS dataset replay:
    adapter = EmotivAdapter()
    adapter.connect("emotiv_epoc_x_file", params={"file": "amigos_subject_01.edf"})
    adapter.start_streaming()
    data = adapter.get_current_data(256)

Cortex API reference: https://emotiv.gitbook.io/cortex-api
"""

from __future__ import annotations

import json
import logging
import queue
import ssl
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

# ── Emotiv EPOC / EPOC X hardware constants ──────────────────────────────────

EPOC_CHANNEL_NAMES: List[str] = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]
N_CHANNELS = len(EPOC_CHANNEL_NAMES)  # 14

# Standard Emotiv sampling rates
SAMPLE_RATE_STD = 128    # Hz — free / Personal plan
SAMPLE_RATE_HIGH = 256   # Hz — Researcher / Researcher Pro plan

# Cortex local WebSocket server (started automatically by EmotivPro)
CORTEX_URL = "wss://localhost:6789"

# Ring-buffer size (seconds × sample rate keeps ~10 seconds)
_RING_BUFFER_SECS = 10


# ── Cortex API client ─────────────────────────────────────────────────────────

class CortexClient:
    """Minimal Cortex API v2 JSON-RPC client over a local WebSocket.

    Handles the auth flow and EEG subscription.  Incoming EEG samples are
    pushed into a thread-safe ring buffer that EmotivAdapter reads from.
    """

    def __init__(
        self,
        url: str = CORTEX_URL,
        client_id: str = "",
        client_secret: str = "",
        sample_rate: int = SAMPLE_RATE_STD,
    ) -> None:
        self._url = url
        self._client_id = client_id
        self._client_secret = client_secret
        self._sample_rate = sample_rate

        self._ws = None           # websocket.WebSocketApp instance
        self._ws_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._id_counter = 0
        self._pending: Dict[int, queue.Queue] = {}   # msg_id → response queue

        # Ring buffer: deque of (n_channels,) arrays, max _RING_BUFFER_SECS seconds
        _maxlen = _RING_BUFFER_SECS * sample_rate
        self._buffer: deque = deque(maxlen=_maxlen)
        self._ts_buffer: deque = deque(maxlen=_maxlen)

        self._auth_token: Optional[str] = None
        self._session_id: Optional[str] = None
        self._connected = False
        self._ready = threading.Event()   # set when EEG subscription is active

    # ── Connection lifecycle ───────────────────────────────────────────────

    def connect(self) -> bool:
        """Open WebSocket and complete the Cortex auth + session flow.

        Returns True if EEG data will arrive; False on any failure.
        """
        try:
            import websocket  # websocket-client package
        except ImportError:
            log.error(
                "websocket-client not installed. "
                "Run: pip install websocket-client"
            )
            return False

        ssl_opts = {"cert_reqs": ssl.CERT_NONE}  # self-signed cert from Cortex

        self._ws = websocket.WebSocketApp(
            self._url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._ws_thread = threading.Thread(
            target=self._ws.run_forever,
            kwargs={"sslopt": ssl_opts},
            daemon=True,
            name="emotiv-cortex-ws",
        )
        self._ws_thread.start()

        # Wait up to 10 seconds for the full auth flow to complete
        if not self._ready.wait(timeout=10):
            log.error("Cortex auth/session timeout — is EmotivPro running?")
            return False

        return True

    def disconnect(self) -> None:
        """Close the WebSocket and clean up."""
        if self._ws is not None:
            try:
                # Unsubscribe and close session before disconnecting
                if self._session_id and self._auth_token:
                    self._call("unsubscribeRequest", {
                        "cortexToken": self._auth_token,
                        "session": self._session_id,
                        "streams": ["eeg"],
                    }, timeout=3)
                    self._call("closeSession", {
                        "cortexToken": self._auth_token,
                        "session": self._session_id,
                    }, timeout=3)
            except Exception:
                pass
            self._ws.close()
        self._connected = False
        self._auth_token = None
        self._session_id = None

    def get_samples(self, n: int) -> tuple:
        """Return last n samples as (signals, timestamps).

        signals shape: (n_channels, n)
        timestamps shape: (n,)
        """
        with self._lock:
            buf_list = list(self._buffer)
            ts_list  = list(self._ts_buffer)

        n = min(n, len(buf_list))
        if n == 0:
            return np.zeros((N_CHANNELS, 0)), np.array([])

        signals = np.array(buf_list[-n:], dtype=np.float32).T  # (14, n)
        timestamps = np.array(ts_list[-n:])
        return signals, timestamps

    # ── WebSocket callbacks ────────────────────────────────────────────────

    def _on_open(self, ws) -> None:
        self._connected = True
        log.info("Cortex WebSocket connected — starting auth flow")
        threading.Thread(target=self._auth_flow, daemon=True).start()

    def _on_message(self, ws, raw: str) -> None:
        msg = json.loads(raw)

        # Route response to waiting caller
        msg_id = msg.get("id")
        if msg_id is not None:
            with self._lock:
                q = self._pending.get(msg_id)
            if q is not None:
                q.put(msg)
                return

        # EEG data stream event (no id field)
        if "eeg" in msg:
            self._handle_eeg(msg)

    def _on_error(self, ws, err) -> None:
        log.warning("Cortex WebSocket error: %s", err)

    def _on_close(self, ws, code, reason) -> None:
        self._connected = False
        log.info("Cortex WebSocket closed (%s %s)", code, reason)

    # ── Cortex auth flow ───────────────────────────────────────────────────

    def _next_id(self) -> int:
        with self._lock:
            self._id_counter += 1
            return self._id_counter

    def _call(self, method: str, params: dict, timeout: float = 8.0) -> Optional[dict]:
        """Send a JSON-RPC request and wait for the response."""
        if not self._connected or self._ws is None:
            return None
        msg_id = self._next_id()
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._pending[msg_id] = q
        payload = json.dumps({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": msg_id,
        })
        self._ws.send(payload)
        try:
            resp = q.get(timeout=timeout)
        except queue.Empty:
            log.warning("Cortex %s timed out", method)
            resp = None
        finally:
            with self._lock:
                self._pending.pop(msg_id, None)
        return resp

    def _auth_flow(self) -> None:
        """Execute the Cortex auth sequence in order."""
        # 1. Request access (prompts user in EmotivPro on first run)
        r = self._call("requestAccess", {
            "clientId": self._client_id,
            "clientSecret": self._client_secret,
        })
        if not r or "result" not in r:
            log.error("requestAccess failed: %s", r)
            return

        # 2. Authorize — get cortex token
        r = self._call("authorize", {
            "clientId": self._client_id,
            "clientSecret": self._client_secret,
            "debit": 1,
        })
        if not r or "result" not in r:
            log.error("authorize failed: %s", r)
            return

        self._auth_token = r["result"].get("cortexToken")
        if not self._auth_token:
            log.error("No cortexToken in authorize response")
            return
        log.info("Cortex token obtained")

        # 3. Query headsets
        r = self._call("queryHeadsets", {})
        headsets = (r or {}).get("result", [])
        if not headsets:
            log.error("No Emotiv headsets found — is the device paired?")
            return
        headset_id = headsets[0].get("id", "")
        log.info("Using headset: %s", headset_id)

        # 4. Create session
        r = self._call("createSession", {
            "cortexToken": self._auth_token,
            "headset": headset_id,
            "status": "active",
        })
        if not r or "result" not in r:
            log.error("createSession failed: %s", r)
            return
        self._session_id = r["result"].get("id", "")
        log.info("Session created: %s", self._session_id)

        # 5. Subscribe to EEG stream
        r = self._call("subscribe", {
            "cortexToken": self._auth_token,
            "session": self._session_id,
            "streams": ["eeg"],
        })
        if not r or "result" not in r:
            log.error("subscribe failed: %s", r)
            return

        log.info("EEG stream subscribed — Emotiv data flowing")
        self._ready.set()

    # ── EEG data ingestion ─────────────────────────────────────────────────

    def _handle_eeg(self, msg: dict) -> None:
        """Parse incoming Cortex EEG event and push to ring buffer.

        Cortex EEG format:
            { "eeg": [COUNTER, INTERPOLATED, AF3, F7, F3, FC5, T7, P7, O1,
                       O2, P8, T8, FC6, F4, F8, AF4, MARKER] }
        The first 2 and last 1 values are metadata, not EEG channels.
        """
        try:
            row = msg["eeg"]
            # row[0]=counter, row[1]=interpolated, row[2:16]=EEG channels, row[16]=marker
            if len(row) < 16:
                return
            eeg_sample = np.array(row[2:16], dtype=np.float32)  # (14,)
            timestamp = time.time()
            with self._lock:
                self._buffer.append(eeg_sample)
                self._ts_buffer.append(timestamp)
        except Exception as exc:
            log.debug("EEG parse error: %s", exc)


# ── EDF / CSV file reader (AMIGOS dataset playback) ───────────────────────────

class EDFReader:
    """Plays back an EDF or CSV file at real-time rate into a ring buffer.

    Supports:
      - EDF/BDF files (uses pyEDFlib if available)
      - CSV files with header row (channel names as columns, one row per sample)
      - NPZ files with keys 'X' (n_channels, n_samples) and optionally 'fs'
    """

    def __init__(self, file_path: Path, sample_rate: int = SAMPLE_RATE_STD) -> None:
        self._path = Path(file_path)
        self._sample_rate = sample_rate
        self._data: Optional[np.ndarray] = None   # (n_channels, n_total_samples)
        self._channel_names: List[str] = []
        self._play_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        _maxlen = _RING_BUFFER_SECS * sample_rate
        self._buffer: deque = deque(maxlen=_maxlen)
        self._ts_buffer: deque = deque(maxlen=_maxlen)

    def load(self) -> bool:
        """Load the file. Returns True on success."""
        suffix = self._path.suffix.lower()
        try:
            if suffix in (".edf", ".bdf"):
                return self._load_edf()
            elif suffix == ".csv":
                return self._load_csv()
            elif suffix == ".npz":
                return self._load_npz()
            else:
                log.error("Unsupported file format: %s", suffix)
                return False
        except Exception as exc:
            log.error("Failed to load %s: %s", self._path, exc)
            return False

    def _load_edf(self) -> bool:
        try:
            import pyedflib
        except ImportError:
            log.error("pyedflib not installed. Run: pip install pyEDFlib")
            return False

        with pyedflib.EdfReader(str(self._path)) as f:
            n_signals = f.signals_in_file
            self._channel_names = f.getSignalLabels()
            # Only keep EEG channels matching Emotiv layout (if labeled)
            eeg_mask = [
                i for i, name in enumerate(self._channel_names)
                if any(ch in name.upper() for ch in EPOC_CHANNEL_NAMES)
            ]
            if not eeg_mask:
                # Fallback: use first N_CHANNELS signals
                eeg_mask = list(range(min(N_CHANNELS, n_signals)))

            self._channel_names = [self._channel_names[i] for i in eeg_mask]
            self._data = np.array([
                f.readSignal(i) for i in eeg_mask
            ], dtype=np.float32)  # (n_channels, n_samples)

            # Use file's declared sample rate if available
            try:
                self._sample_rate = int(f.getSampleFrequency(eeg_mask[0]))
            except Exception:
                pass

        log.info(
            "EDF loaded: %d channels × %d samples @ %d Hz",
            self._data.shape[0], self._data.shape[1], self._sample_rate,
        )
        return True

    def _load_csv(self) -> bool:
        import pandas as pd

        df = pd.read_csv(self._path)
        # Keep columns that match Emotiv channel names
        eeg_cols = [c for c in df.columns if c.upper() in EPOC_CHANNEL_NAMES]
        if not eeg_cols:
            # Use all numeric columns except time/timestamp
            eeg_cols = [c for c in df.columns
                        if df[c].dtype in (np.float32, np.float64, np.int64)
                        and "time" not in c.lower()]
        self._channel_names = eeg_cols
        self._data = df[eeg_cols].values.T.astype(np.float32)
        log.info("CSV loaded: %d channels × %d samples", *self._data.shape)
        return True

    def _load_npz(self) -> bool:
        arr = np.load(self._path)
        if "X" not in arr:
            log.error("NPZ missing 'X' key")
            return False
        self._data = arr["X"].astype(np.float32)
        if self._data.ndim == 1:
            self._data = self._data.reshape(1, -1)
        if "fs" in arr:
            self._sample_rate = int(arr["fs"])
        self._channel_names = [f"CH{i}" for i in range(self._data.shape[0])]
        log.info("NPZ loaded: %d channels × %d samples", *self._data.shape)
        return True

    def start(self) -> None:
        """Begin streaming file data into the ring buffer at real-time rate."""
        if self._data is None:
            raise RuntimeError("Call load() before start()")
        self._stop_event.clear()
        self._play_thread = threading.Thread(
            target=self._play_loop, daemon=True, name="emotiv-edf-player"
        )
        self._play_thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def get_samples(self, n: int) -> tuple:
        with self._lock:
            buf_list = list(self._buffer)
            ts_list  = list(self._ts_buffer)
        n = min(n, len(buf_list))
        if n == 0:
            return np.zeros((len(self._channel_names) or N_CHANNELS, 0)), np.array([])
        signals = np.array(buf_list[-n:], dtype=np.float32).T
        timestamps = np.array(ts_list[-n:])
        return signals, timestamps

    def _play_loop(self) -> None:
        """Push samples from the file into the buffer at real-time cadence."""
        n_total = self._data.shape[1]
        interval = 1.0 / self._sample_rate  # seconds per sample
        idx = 0
        while not self._stop_event.is_set():
            sample = self._data[:, idx % n_total]
            ts = time.time()
            with self._lock:
                self._buffer.append(sample)
                self._ts_buffer.append(ts)
            idx += 1
            time.sleep(interval)


# ── Main adapter class ────────────────────────────────────────────────────────

class EmotivAdapter:
    """Drop-in replacement for BrainFlowManager for Emotiv EPOC / EPOC X.

    Supported device_type values:
        "emotiv_epoc"        — EPOC (14 ch, 128 Hz) via Cortex API
        "emotiv_epoc_x"      — EPOC X (14 ch, 128/256 Hz) via Cortex API
        "emotiv_epoc_file"   — Offline EDF/CSV/NPZ replay (any EPOC recording)
        "emotiv_epoc_x_file" — Same for EPOC X files

    Connection params (dict passed to connect()):
        For live Cortex devices:
            client_id     : str  — from Emotiv developer portal
            client_secret : str  — from Emotiv developer portal
            sample_rate   : int  — 128 or 256 (default: 128)
            cortex_url    : str  — override WebSocket URL (default: wss://localhost:6789)

        For file replay:
            file          : str  — path to EDF, CSV, or NPZ file
            sample_rate   : int  — override if not embedded in file
    """

    SUPPORTED_DEVICES = {
        "emotiv_epoc": {
            "name": "Emotiv EPOC",
            "channels": N_CHANNELS,
            "sample_rate": SAMPLE_RATE_STD,
            "backend": "cortex",
        },
        "emotiv_epoc_x": {
            "name": "Emotiv EPOC X",
            "channels": N_CHANNELS,
            "sample_rate": SAMPLE_RATE_STD,
            "backend": "cortex",
        },
        "emotiv_epoc_file": {
            "name": "Emotiv EPOC (File Replay)",
            "channels": N_CHANNELS,
            "sample_rate": SAMPLE_RATE_STD,
            "backend": "file",
        },
        "emotiv_epoc_x_file": {
            "name": "Emotiv EPOC X (File Replay)",
            "channels": N_CHANNELS,
            "sample_rate": SAMPLE_RATE_STD,
            "backend": "file",
        },
    }

    def __init__(self) -> None:
        self.is_connected: bool = False
        self.is_streaming: bool = False
        self.current_device_type: Optional[str] = None
        self.n_channels: int = N_CHANNELS
        self.sample_rate: int = SAMPLE_RATE_STD
        self.eeg_channel_names: List[str] = list(EPOC_CHANNEL_NAMES)

        self._cortex: Optional[CortexClient] = None
        self._edf: Optional[EDFReader] = None
        self._stream_callback: Optional[Callable] = None

    # ── Public interface (mirrors BrainFlowManager) ────────────────────────

    def discover_devices(self) -> List[Dict]:
        """Return list of Emotiv devices this adapter supports."""
        devices = []
        for device_type, info in self.SUPPORTED_DEVICES.items():
            devices.append({
                "type": device_type,
                "name": info["name"],
                "channels": info["channels"],
                "sample_rate": info["sample_rate"],
                "available": True,
                "adapter": "emotiv",
            })
        return devices

    def connect(self, device_type: str, params: Optional[Dict] = None) -> Dict:
        """Connect to an Emotiv device or open a file for replay.

        Args:
            device_type: One of SUPPORTED_DEVICES keys.
            params:      Dict with client_id/client_secret (live) or file (replay).

        Returns:
            Status dict (same shape as BrainFlowManager.connect()).

        Raises:
            ValueError: Unknown device_type.
            RuntimeError: Connection failure.
        """
        if device_type not in self.SUPPORTED_DEVICES:
            raise ValueError(
                f"Unknown Emotiv device '{device_type}'. "
                f"Available: {list(self.SUPPORTED_DEVICES.keys())}"
            )

        if self.is_connected:
            self.disconnect()

        params = params or {}
        info = self.SUPPORTED_DEVICES[device_type]
        self.sample_rate = int(params.get("sample_rate", info["sample_rate"]))
        self.current_device_type = device_type

        if info["backend"] == "file":
            self._connect_file(params)
        else:
            self._connect_cortex(params)

        self.is_connected = True
        return {
            "status": "connected",
            "device": info["name"],
            "channels": self.n_channels,
            "channel_names": self.eeg_channel_names,
            "sample_rate": self.sample_rate,
            "adapter": "emotiv",
        }

    def disconnect(self) -> None:
        """Disconnect and clean up."""
        if self.is_streaming:
            self.stop_streaming()
        if self._cortex is not None:
            self._cortex.disconnect()
            self._cortex = None
        if self._edf is not None:
            self._edf.stop()
            self._edf = None
        self.is_connected = False
        self.current_device_type = None

    def start_streaming(self, callback: Optional[Callable] = None) -> None:
        """Begin streaming EEG data.

        Args:
            callback: Optional function called with (signals, timestamp) dict
                      each time get_current_data() is polled. For compatibility
                      with WebSocket path; polling via get_current_data() works
                      without a callback.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected — call connect() first")
        if self.is_streaming:
            return  # idempotent

        self._stream_callback = callback

        if self._edf is not None:
            self._edf.start()

        # Cortex client starts streaming automatically after subscription
        self.is_streaming = True
        log.info("EmotivAdapter streaming started")

    def stop_streaming(self) -> None:
        """Stop streaming."""
        if self._edf is not None:
            self._edf.stop()
        self.is_streaming = False
        self._stream_callback = None

    def get_current_data(self, n_samples: int = 256) -> Optional[Dict]:
        """Return the latest n_samples from the ring buffer.

        Returns the same dict shape as BrainFlowManager.get_current_data().
        Returns None if not connected or no data available yet.
        """
        if not self.is_connected:
            return None

        if self._cortex is not None:
            signals, timestamps = self._cortex.get_samples(n_samples)
        elif self._edf is not None:
            signals, timestamps = self._edf.get_samples(n_samples)
        else:
            return None

        if signals.shape[1] == 0:
            return None  # no data buffered yet

        # Pad to requested length if buffer hasn't filled yet (repeat first frame)
        if signals.shape[1] < n_samples:
            pad_n = n_samples - signals.shape[1]
            signals = np.concatenate(
                [np.tile(signals[:, :1], (1, pad_n)), signals], axis=1
            )

        result = {
            "signals": signals.tolist(),
            "timestamps": timestamps.tolist(),
            "sample_rate": self.sample_rate,
            "n_channels": self.n_channels,
            "channel_names": self.eeg_channel_names,
        }

        # Fire callback if registered
        if self._stream_callback is not None:
            try:
                self._stream_callback(signals, time.time())
            except Exception:
                pass

        return result

    # ── Private connection helpers ─────────────────────────────────────────

    def _connect_cortex(self, params: Dict) -> None:
        """Connect to the Cortex WebSocket API."""
        client_id     = params.get("client_id", "")
        client_secret = params.get("client_secret", "")
        cortex_url    = params.get("cortex_url", CORTEX_URL)

        self._cortex = CortexClient(
            url=cortex_url,
            client_id=client_id,
            client_secret=client_secret,
            sample_rate=self.sample_rate,
        )
        ok = self._cortex.connect()
        if not ok:
            self._cortex = None
            raise RuntimeError(
                "Cortex API connection failed. "
                "Check that EmotivPro is running and the headset is paired."
            )
        self.eeg_channel_names = list(EPOC_CHANNEL_NAMES)
        log.info("EmotivAdapter: Cortex API connected")

    def _connect_file(self, params: Dict) -> None:
        """Open a recording file for replay."""
        file_path = params.get("file")
        if not file_path:
            raise ValueError("params['file'] is required for file replay mode")

        self._edf = EDFReader(Path(file_path), sample_rate=self.sample_rate)
        ok = self._edf.load()
        if not ok:
            self._edf = None
            raise RuntimeError(f"Failed to load recording file: {file_path}")

        # Update channel names if the file has its own labels
        if self._edf._channel_names:
            self.eeg_channel_names = self._edf._channel_names
            self.n_channels = len(self.eeg_channel_names)
        if self._edf._sample_rate:
            self.sample_rate = self._edf._sample_rate
        log.info("EmotivAdapter: file replay ready — %s", file_path)


# ── Convenience function ──────────────────────────────────────────────────────

def make_emotiv_adapter(
    device_type: str = "emotiv_epoc_x",
    **params,
) -> EmotivAdapter:
    """Create and connect an EmotivAdapter in one call.

    Example:
        # Live device:
        adapter = make_emotiv_adapter(
            "emotiv_epoc_x",
            client_id="my_id", client_secret="my_secret"
        )

        # AMIGOS dataset replay:
        adapter = make_emotiv_adapter(
            "emotiv_epoc_x_file",
            file="data/amigos/s01_eeg.edf"
        )
    """
    adapter = EmotivAdapter()
    adapter.connect(device_type, params=params)
    return adapter

"""Session Recording & Playback Module.

Records EEG sessions (signals + analysis) to disk for later review,
comparison, and export. Uses numpy compressed format for signal data
and JSON for metadata/analysis timelines.
"""

import json
import uuid
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


SESSIONS_DIR = Path(__file__).parent.parent / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)


class SessionRecorder:
    """Records and manages EEG sessions."""

    def __init__(self):
        self.active_session: Optional[str] = None
        self.session_meta: Optional[Dict] = None
        self.signal_buffer: List[np.ndarray] = []
        self.analysis_timeline: List[Dict] = []
        self.start_time: Optional[float] = None

    def start_recording(
        self, user_id: str = "default", session_type: str = "general", metadata: Optional[Dict] = None
    ) -> str:
        """Start a new recording session.

        Returns:
            session_id: Unique identifier for this session.
        """
        session_id = str(uuid.uuid4())[:8]
        self.active_session = session_id
        self.start_time = time.time()
        self.signal_buffer = []
        self.analysis_timeline = []

        self.session_meta = {
            "session_id": session_id,
            "user_id": user_id,
            "session_type": session_type,
            "start_time": self.start_time,
            "metadata": metadata or {},
            "status": "recording",
        }

        return session_id

    def add_frame(self, signals: np.ndarray, analysis: Optional[Dict] = None):
        """Append a frame to the active recording.

        Args:
            signals: 2D array (n_channels, n_samples) or 1D array.
            analysis: Optional analysis dict for this frame.
        """
        if self.active_session is None:
            return

        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        self.signal_buffer.append(signals)

        if analysis:
            self.analysis_timeline.append({
                "time_offset": time.time() - self.start_time,
                **analysis,
            })

    def stop_recording(self) -> Dict:
        """Finalize and save the current recording.

        Returns:
            Session summary dict.
        """
        if self.active_session is None:
            return {"error": "No active recording"}

        session_id = self.active_session
        duration = time.time() - self.start_time

        # Concatenate signal buffer
        if self.signal_buffer:
            all_signals = np.concatenate(self.signal_buffer, axis=1)
        else:
            all_signals = np.array([[]])

        # Save signals as compressed numpy
        signal_path = SESSIONS_DIR / f"{session_id}.npz"
        np.savez_compressed(str(signal_path), signals=all_signals)

        # Compute summary
        summary = {
            "duration_sec": duration,
            "n_frames": len(self.signal_buffer),
            "n_channels": all_signals.shape[0] if all_signals.size > 0 else 0,
            "n_samples": all_signals.shape[1] if all_signals.size > 0 else 0,
        }

        # Save metadata + analysis timeline
        self.session_meta["status"] = "completed"
        self.session_meta["end_time"] = time.time()
        self.session_meta["summary"] = summary
        self.session_meta["analysis_timeline"] = self.analysis_timeline

        meta_path = SESSIONS_DIR / f"{session_id}.json"
        with open(meta_path, "w") as f:
            json.dump(self.session_meta, f, indent=2, default=str)

        # Reset state
        result = {**self.session_meta}
        self.active_session = None
        self.session_meta = None
        self.signal_buffer = []
        self.analysis_timeline = []
        self.start_time = None

        return result

    @property
    def is_recording(self) -> bool:
        return self.active_session is not None

    @staticmethod
    def list_sessions(user_id: Optional[str] = None, session_type: Optional[str] = None) -> List[Dict]:
        """List saved sessions with optional filters."""
        sessions = []
        for meta_file in SESSIONS_DIR.glob("*.json"):
            try:
                with open(meta_file) as f:
                    meta = json.load(f)

                if user_id and meta.get("user_id") != user_id:
                    continue
                if session_type and meta.get("session_type") != session_type:
                    continue

                sessions.append({
                    "session_id": meta["session_id"],
                    "user_id": meta.get("user_id", "unknown"),
                    "session_type": meta.get("session_type", "general"),
                    "start_time": meta.get("start_time"),
                    "status": meta.get("status"),
                    "summary": meta.get("summary", {}),
                })
            except (json.JSONDecodeError, KeyError):
                continue

        # Sort by start time, newest first
        sessions.sort(key=lambda s: s.get("start_time", 0), reverse=True)
        return sessions

    @staticmethod
    def load_session(session_id: str) -> Dict:
        """Load a full session (metadata + signals + analysis timeline)."""
        meta_path = SESSIONS_DIR / f"{session_id}.json"
        signal_path = SESSIONS_DIR / f"{session_id}.npz"

        if not meta_path.exists():
            return {"error": f"Session {session_id} not found"}

        with open(meta_path) as f:
            meta = json.load(f)

        signals = None
        if signal_path.exists():
            data = np.load(str(signal_path))
            signals = data["signals"].tolist()

        return {
            **meta,
            "signals": signals,
        }

    @staticmethod
    def delete_session(session_id: str) -> bool:
        """Delete a session's files."""
        meta_path = SESSIONS_DIR / f"{session_id}.json"
        signal_path = SESSIONS_DIR / f"{session_id}.npz"

        deleted = False
        if meta_path.exists():
            meta_path.unlink()
            deleted = True
        if signal_path.exists():
            signal_path.unlink()
            deleted = True

        return deleted

    @staticmethod
    def export_session(session_id: str, format: str = "csv") -> Optional[bytes]:
        """Export session data as CSV or raw bytes.

        Args:
            session_id: Session identifier.
            format: Export format ('csv' or 'edf').

        Returns:
            Bytes of the exported data, or None on error.
        """
        meta_path = SESSIONS_DIR / f"{session_id}.json"
        signal_path = SESSIONS_DIR / f"{session_id}.npz"

        if not signal_path.exists() or not meta_path.exists():
            return None

        with open(meta_path) as f:
            meta = json.load(f)

        data = np.load(str(signal_path))
        signals = data["signals"]

        if format == "csv":
            import io
            output = io.StringIO()
            # Header
            n_channels = signals.shape[0]
            header = ",".join([f"channel_{i}" for i in range(n_channels)])
            output.write(f"sample,{header}\n")
            # Data rows
            for s in range(signals.shape[1]):
                row = ",".join([str(signals[ch, s]) for ch in range(n_channels)])
                output.write(f"{s},{row}\n")
            return output.getvalue().encode("utf-8")

        return None

"""Session Recording & Playback Module.

Records EEG sessions (signals + analysis) to disk for later review,
comparison, and export. Uses numpy compressed format for signal data
and JSON for metadata/analysis timelines.
"""

import json
import uuid
import time
import logging
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

SESSIONS_DIR = Path(__file__).parent.parent / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)

# Batch flush every N frames during active recording (≈1 min at 1 fps)
_BATCH_FLUSH_INTERVAL = 60

# Express backend URL for persisting emotion readings
_EXPRESS_URL = os.environ.get("EXPRESS_URL", "http://localhost:5000")


def _post_emotion_batch(frames: List[Dict], user_id: str, session_id: str) -> int:
    """POST analysis_timeline frames to the Express batch endpoint.

    Returns number of records inserted, or 0 on failure.
    """
    if not frames:
        return 0
    try:
        import urllib.request
        readings = []
        for frame in frames:
            emotions = frame.get("emotions") or {}
            if not emotions:
                continue
            flow = frame.get("flow_state") or {}
            readings.append({
                "userId": user_id,
                "sessionId": session_id,
                "stress": float(emotions.get("stress_index", 0)),
                "happiness": float((emotions["valence"] + 1) / 2) if emotions.get("valence") is not None else 0.5,
                "focus": float(emotions.get("focus_index", 0)),
                "energy": float(emotions.get("arousal", 0)),
                "dominantEmotion": str(emotions.get("emotion", "unknown")),
                "valence": float(emotions.get("valence", 0)) if emotions.get("valence") is not None else None,
                "arousal": float(emotions.get("arousal", 0)) if emotions.get("arousal") is not None else None,
            })
        if not readings:
            return 0
        body = json.dumps({"readings": readings}).encode("utf-8")
        req = urllib.request.Request(
            f"{_EXPRESS_URL}/api/emotion-readings/batch",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            return result.get("inserted", 0)
    except Exception as exc:
        logger.warning(f"[session_recorder] batch persist failed: {exc}")
        return 0


class SessionRecorder:
    """Records and manages EEG sessions."""

    def __init__(self):
        self.active_session: Optional[str] = None
        self.session_meta: Optional[Dict] = None
        self.signal_buffer: List[np.ndarray] = []
        self.analysis_timeline: List[Dict] = []
        self.start_time: Optional[float] = None
        self._unflushed_frames: List[Dict] = []  # pending frames not yet POSTed

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
        self._unflushed_frames = []

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
            entry = {"time_offset": time.time() - self.start_time, **analysis}
            self.analysis_timeline.append(entry)
            self._unflushed_frames.append(entry)

            # Batch-flush every _BATCH_FLUSH_INTERVAL frames so partial sessions are persisted
            if len(self._unflushed_frames) >= _BATCH_FLUSH_INTERVAL:
                user_id = (self.session_meta or {}).get("user_id", "default")
                session_id = self.active_session or "unknown"
                _post_emotion_batch(self._unflushed_frames, user_id, session_id)
                self._unflushed_frames = []

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

        # Save signals — local file always written as fallback
        signal_path = SESSIONS_DIR / f"{session_id}.npz"
        np.savez_compressed(str(signal_path), signals=all_signals)

        # Upload to R2 (no-op when credentials not set)
        user_id = self.session_meta.get("user_id", "default")
        signal_r2_key: str | None = None
        try:
            from storage.r2_client import r2
            if r2.available:
                r2_key = r2.session_key(user_id, session_id)
                with open(signal_path, "rb") as f:
                    if r2.upload_bytes(f.read(), r2_key):
                        signal_r2_key = r2_key
        except Exception:
            pass

        # Compute summary
        summary = {
            "duration_sec": duration,
            "n_frames": len(self.signal_buffer),
            "n_channels": all_signals.shape[0] if all_signals.size > 0 else 0,
            "n_samples": all_signals.shape[1] if all_signals.size > 0 else 0,
        }

        # Compute per-session metric averages from analysis timeline
        if self.analysis_timeline:
            from collections import Counter
            emotions_data = [t["emotions"] for t in self.analysis_timeline if t.get("emotions")]
            flow_data = [t["flow_state"] for t in self.analysis_timeline if t.get("flow_state")]
            creativity_data = [t["creativity"] for t in self.analysis_timeline if t.get("creativity")]

            if emotions_data:
                summary["avg_stress"] = sum(e.get("stress_index", 0) for e in emotions_data) / len(emotions_data)
                summary["avg_focus"] = sum(e.get("focus_index", 0) for e in emotions_data) / len(emotions_data)
                summary["avg_relaxation"] = sum(e.get("relaxation_index", 0) for e in emotions_data) / len(emotions_data)
                summary["avg_valence"] = sum(e.get("valence", 0) for e in emotions_data) / len(emotions_data)
                summary["avg_arousal"] = sum(e.get("arousal", 0) for e in emotions_data) / len(emotions_data)
                emotion_labels = [e.get("emotion", "") for e in emotions_data if e.get("emotion")]
                if emotion_labels:
                    summary["dominant_emotion"] = Counter(emotion_labels).most_common(1)[0][0]

            if flow_data:
                summary["avg_flow"] = sum(f.get("flow_score", 0) for f in flow_data) / len(flow_data)
            if creativity_data:
                summary["avg_creativity"] = sum(c.get("creativity_score", 0) for c in creativity_data) / len(creativity_data)

        # Save metadata + analysis timeline
        self.session_meta["status"] = "completed"
        self.session_meta["end_time"] = time.time()
        self.session_meta["summary"] = summary
        self.session_meta["analysis_timeline"] = self.analysis_timeline

        meta_path = SESSIONS_DIR / f"{session_id}.json"
        with open(meta_path, "w") as f:
            json.dump(self.session_meta, f, indent=2, default=str)

        # Persist metadata to PostgreSQL (no-op when DATABASE_URL not set)
        try:
            from storage.pg_session_store import upsert_session
            upsert_session(
                session_id=session_id,
                user_id=self.session_meta.get("user_id", "default"),
                session_type=self.session_meta.get("session_type", "general"),
                start_time=self.session_meta.get("start_time"),
                end_time=self.session_meta.get("end_time"),
                status="completed",
                summary=summary,
                signal_r2_key=signal_r2_key,
            )
        except Exception:
            pass

        # POST any remaining unflushed frames + full timeline to Express
        # (unflushed_frames contains only frames not yet sent during recording)
        flush_user = self.session_meta.get("user_id", "default")
        if self._unflushed_frames:
            _post_emotion_batch(self._unflushed_frames, flush_user, session_id)

        # Reset state
        result = {**self.session_meta}
        self.active_session = None
        self.session_meta = None
        self.signal_buffer = []
        self.analysis_timeline = []
        self.start_time = None
        self._unflushed_frames = []

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
        """Delete a session's local files, R2 object, and PostgreSQL row."""
        meta_path = SESSIONS_DIR / f"{session_id}.json"
        signal_path = SESSIONS_DIR / f"{session_id}.npz"

        # Read user_id from local JSON before deleting
        user_id = "default"
        try:
            if meta_path.exists():
                with open(meta_path) as f:
                    user_id = json.load(f).get("user_id", "default")
        except Exception:
            pass

        deleted = False
        if meta_path.exists():
            meta_path.unlink()
            deleted = True
        if signal_path.exists():
            signal_path.unlink()
            deleted = True

        # Remove from R2
        try:
            from storage.r2_client import r2
            if r2.available:
                r2.delete(r2.session_key(user_id, session_id))
        except Exception:
            pass

        # Remove from PostgreSQL
        try:
            from storage.pg_session_store import delete_session as pg_delete
            pg_delete(session_id)
        except Exception:
            pass

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

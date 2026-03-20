"""Apache Parquet EEG storage pipeline.

Writes EEG analysis frames as Parquet files — columnar format gives:
  - 5-10× smaller files vs NPZ (Snappy compression)
  - Column pruning: read only alpha/beta/theta without loading full rows
  - Native pandas/PyArrow integration for trend queries
  - Direct compatibility with Spark / Polars for future batch ML

Storage layout (local or S3/R2-compatible):
  {PARQUET_BASE_DIR}/
    users/
      {user_id}/
        {YYYY-MM-DD}/
          {HH}-{session_id}.parquet

Each Parquet file is a batch of EEG frames for a single session.
Rows = time steps (typically 4 Hz), columns = EEG features + metadata.

Usage:
    writer = ParquetWriter(user_id="alice")
    writer.push_frame(analysis_dict, timestamp)
    writer.flush()          # writes pending rows to Parquet
    await writer.close()    # final flush + optional R2 upload

Environment variables:
    PARQUET_BASE_DIR      Local base directory (default: ml/data/parquet)
    R2_ACCESS_KEY_ID      Cloudflare R2 key (optional — skipped if absent)
    R2_SECRET_ACCESS_KEY  Cloudflare R2 secret
    R2_ENDPOINT_URL       e.g. https://<account>.r2.cloudflarestorage.com
    R2_BUCKET_NAME        Bucket name (default: neural-dream-eeg)
"""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
_BASE_DIR = Path(os.environ.get("PARQUET_BASE_DIR", Path(__file__).parent.parent / "data" / "parquet"))
_FLUSH_INTERVAL = float(os.environ.get("PARQUET_FLUSH_INTERVAL", "30"))   # seconds
_BATCH_MAX = int(os.environ.get("PARQUET_BATCH_MAX", "120"))              # rows (≈ 30 s at 4 Hz)

# R2 config (optional)
_R2_KEY    = os.environ.get("R2_ACCESS_KEY_ID")
_R2_SECRET = os.environ.get("R2_SECRET_ACCESS_KEY")
_R2_ENDPOINT = os.environ.get("R2_ENDPOINT_URL")
_R2_BUCKET = os.environ.get("R2_BUCKET_NAME", "neural-dream-eeg")

# ── Column schema (flattened EEG frame) ───────────────────────────────────────
_EEG_COLUMNS = [
    "timestamp", "user_id", "session_id",
    # Band powers
    "alpha", "beta", "theta", "delta", "gamma",
    "low_beta", "high_beta",
    # Cognitive scores
    "focus_index", "stress_index", "relaxation_index",
    "flow_score", "creativity_score",
    "valence", "arousal",
    # Emotion
    "emotion", "emotion_confidence",
    # Sleep
    "sleep_stage",
    # Signal quality
    "sqi",
]


def _extract_row(
    analysis: Dict[str, Any],
    user_id: str,
    session_id: str,
    timestamp: float,
) -> Dict[str, Any]:
    """Flatten a nested analysis dict into a single Parquet row."""
    bp = analysis.get("band_powers", {})
    emotions = analysis.get("emotions", {})
    sleep = analysis.get("sleep_staging", {})
    quality = analysis.get("signal_quality", {})

    return {
        "timestamp": timestamp,
        "user_id": user_id,
        "session_id": session_id,
        # Band powers
        "alpha":      float(bp.get("alpha", 0.0)),
        "beta":       float(bp.get("beta", 0.0)),
        "theta":      float(bp.get("theta", 0.0)),
        "delta":      float(bp.get("delta", 0.0)),
        "gamma":      float(bp.get("gamma", 0.0)),
        "low_beta":   float(bp.get("low_beta", 0.0)),
        "high_beta":  float(bp.get("high_beta", 0.0)),
        # Cognitive scores
        "focus_index":      float(emotions.get("focus_index", 0.0)),
        "stress_index":     float(emotions.get("stress_index", 0.0)),
        "relaxation_index": float(emotions.get("relaxation_index", 0.0)),
        "flow_score":       float(analysis.get("flow_state", {}).get("flow_score", 0.0)),
        "creativity_score": float(analysis.get("creativity", {}).get("creativity_score", 0.0)),
        "valence":          float(emotions.get("valence", 0.0)),
        "arousal":          float(emotions.get("arousal", 0.0)),
        # Emotion
        "emotion":            str(emotions.get("emotion", "")),
        "emotion_confidence": float(emotions.get("confidence", 0.0)),
        # Sleep
        "sleep_stage": str(sleep.get("stage", "")),
        # Signal quality
        "sqi": float(quality.get("sqi", quality.get("overall_quality", 0.0))),
    }


class ParquetWriter:
    """Per-user writer that batches EEG frames and flushes to Parquet."""

    def __init__(self, user_id: str = "default", session_id: Optional[str] = None):
        if not re.match(r'^[a-zA-Z0-9_-]{1,128}$', user_id):
            raise ValueError(f"Invalid user_id: {user_id!r}")
        if session_id is not None and not re.match(r'^[a-zA-Z0-9_-]{1,128}$', session_id):
            raise ValueError(f"Invalid session_id: {session_id!r}")
        self._user_id = user_id
        self._session_id = session_id or f"s{int(time.time())}"
        self._rows: List[Dict[str, Any]] = []
        self._last_flush = time.time()
        self._enabled = self._check_pyarrow()

    @staticmethod
    def _check_pyarrow() -> bool:
        try:
            import pyarrow  # noqa: F401
            return True
        except ImportError:
            logger.warning(
                "[ParquetWriter] pyarrow not installed — Parquet writing disabled. "
                "Run: pip install pyarrow"
            )
            return False

    def push_frame(self, analysis: Dict[str, Any], timestamp: Optional[float] = None) -> None:
        """Add one EEG frame to the pending batch."""
        if not self._enabled:
            return
        row = _extract_row(analysis, self._user_id, self._session_id, timestamp or time.time())
        self._rows.append(row)

        now = time.time()
        if len(self._rows) >= _BATCH_MAX or (now - self._last_flush) >= _FLUSH_INTERVAL:
            self.flush()

    def flush(self) -> Optional[Path]:
        """Write pending rows to a Parquet file. Returns path or None."""
        if not self._enabled or not self._rows:
            return None

        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            return None

        rows = self._rows
        self._rows = []
        self._last_flush = time.time()

        try:
            now_utc = datetime.now(timezone.utc)
            day_dir = _BASE_DIR / "users" / self._user_id / now_utc.strftime("%Y-%m-%d")
            day_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{now_utc.strftime('%H%M%S')}-{self._session_id}.parquet"
            out_path = day_dir / filename

            table = pa.Table.from_pylist(rows)
            pq.write_table(
                table,
                out_path,
                compression="snappy",     # fast + good ratio
                use_dictionary=["emotion", "sleep_stage", "user_id", "session_id"],
            )
            logger.debug("[Parquet] Wrote %d rows → %s", len(rows), out_path)

            # Optional R2 upload (fire-and-forget)
            self._upload_to_r2(out_path)

            return out_path

        except Exception as e:
            logger.error("[Parquet] flush failed: %s", e)
            # Put rows back so they aren't lost
            self._rows = rows + self._rows
            return None

    def _upload_to_r2(self, local_path: Path) -> None:
        """Upload Parquet file to Cloudflare R2 (no-op if not configured)."""
        if not (_R2_KEY and _R2_SECRET and _R2_ENDPOINT):
            return
        try:
            import boto3
            s3 = boto3.client(
                "s3",
                aws_access_key_id=_R2_KEY,
                aws_secret_access_key=_R2_SECRET,
                endpoint_url=_R2_ENDPOINT,
            )
            key = f"parquet/{local_path.relative_to(_BASE_DIR)}"
            s3.upload_file(str(local_path), _R2_BUCKET, str(key))
            logger.debug("[Parquet] Uploaded → R2: %s", key)
        except Exception as e:
            logger.warning("[Parquet] R2 upload failed: %s", e)

    async def close(self) -> None:
        """Final flush before shutdown."""
        self.flush()


# ── Query helpers ─────────────────────────────────────────────────────────────

def read_user_parquet(
    user_id: str,
    date_str: Optional[str] = None,
    columns: Optional[List[str]] = None,
) -> "Any":
    """
    Read all Parquet files for *user_id* on *date_str* (YYYY-MM-DD).
    Returns a pandas DataFrame, or None if pyarrow/pandas unavailable.

    Example:
        df = read_user_parquet("alice", "2026-02-26", columns=["timestamp", "alpha", "stress_index"])
        df.plot(x="timestamp", y="stress_index")
    """
    try:
        import pyarrow.parquet as pq
        import pandas as pd
    except ImportError:
        logger.warning("[Parquet] pyarrow/pandas not installed.")
        return None

    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    day_dir = _BASE_DIR / "users" / user_id / date_str
    if not day_dir.exists():
        return pd.DataFrame(columns=columns or _EEG_COLUMNS)

    files = sorted(day_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=columns or _EEG_COLUMNS)

    tables = []
    for f in files:
        try:
            tables.append(pq.read_table(f, columns=columns))
        except Exception as e:
            logger.warning("[Parquet] read failed for %s: %s", f, e)

    if not tables:
        return pd.DataFrame(columns=columns or _EEG_COLUMNS)

    import pyarrow as pa
    combined = pa.concat_tables(tables)
    return combined.to_pandas().sort_values("timestamp").reset_index(drop=True)


def get_user_daily_summary(user_id: str, date_str: Optional[str] = None) -> Dict[str, Any]:
    """
    Compute daily EEG summary stats from Parquet data.
    Faster than querying Postgres brain_readings for large date ranges.
    """
    df = read_user_parquet(
        user_id, date_str,
        columns=["timestamp", "alpha", "beta", "theta", "focus_index",
                 "stress_index", "valence", "flow_score", "sqi"],
    )
    if df is None or df.empty:
        return {"error": "no_data"}

    return {
        "date": date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "n_frames": len(df),
        "duration_min": round((df["timestamp"].max() - df["timestamp"].min()) / 60, 1),
        "avg_focus":    round(float(df["focus_index"].mean()), 3),
        "avg_stress":   round(float(df["stress_index"].mean()), 3),
        "avg_valence":  round(float(df["valence"].mean()), 3),
        "avg_flow":     round(float(df["flow_score"].mean()), 3),
        "avg_sqi":      round(float(df["sqi"].mean()), 3),
        "peak_focus":   round(float(df["focus_index"].max()), 3),
        "peak_flow":    round(float(df["flow_score"].max()), 3),
    }

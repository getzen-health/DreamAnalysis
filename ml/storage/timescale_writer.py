"""
TimescaleWriter — always-on, non-blocking asyncpg batch writer.

One instance per WebSocket connection. Receives EEG frames at 4 Hz,
downsamples to 1 Hz by averaging, then batch-writes rows to Neon/TimescaleDB.

If DATABASE_URL is not set the writer silently no-ops so the rest of the
application continues to work without a database.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import statistics
from collections import deque
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional asyncpg import — skip gracefully if not installed
# ---------------------------------------------------------------------------
try:
    import asyncpg  # type: ignore
    _ASYNCPG_AVAILABLE = True
except ImportError:
    asyncpg = None  # type: ignore
    _ASYNCPG_AVAILABLE = False


# Whitelist of numeric keys we average across the 4-frame window
_NUMERIC_KEYS = (
    "alpha", "beta", "theta", "gamma", "delta",
    "valence", "arousal",
    "focus_index", "stress_index", "relaxation_idx",
    "flow_score", "creativity_score", "attention_score",
    "sqi",
)

_WRITE_BATCH_MAX = 60   # rows per executemany call
_FLUSH_INTERVAL  = 1.0  # seconds between 4→1 Hz downsamples


class _NoOpWriter:
    """Returned when DATABASE_URL is absent or asyncpg not installed."""

    def push_frame(self, frame: dict, signals: list) -> None:
        pass

    async def close(self) -> None:
        pass


class TimescaleWriter:
    """
    Async batch writer for brain_readings.

    Usage
    -----
        writer = await TimescaleWriter.create(user_id="default")
        writer.push_frame(frame_dict, raw_signals)   # called at 4 Hz
        await writer.close()
    """

    def __init__(self, user_id: str, pool: "asyncpg.Pool") -> None:
        self._user_id   = user_id
        self._pool      = pool
        self._frame_buf: deque[tuple[dict, list]] = deque()
        self._write_q: asyncio.Queue[dict] = asyncio.Queue()
        self._closed    = False
        self._flush_task = asyncio.ensure_future(self._flush_loop())
        self._write_task = asyncio.ensure_future(self._write_loop())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    async def create(cls, user_id: str = "default") -> "TimescaleWriter | _NoOpWriter":
        """
        Factory.  Returns a no-op writer when the DB is unavailable so callers
        never have to handle None.
        """
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            logger.debug("DATABASE_URL not set — TimescaleWriter disabled")
            return _NoOpWriter()
        if not _ASYNCPG_AVAILABLE:
            logger.warning("asyncpg not installed — TimescaleWriter disabled")
            return _NoOpWriter()
        try:
            pool = await asyncpg.create_pool(
                dsn=db_url,
                min_size=1,
                max_size=3,
                command_timeout=10,
            )
            logger.info("TimescaleWriter connected (user=%s)", user_id)
            return cls(user_id, pool)
        except Exception as exc:
            logger.warning("TimescaleWriter could not connect: %s", exc)
            return _NoOpWriter()

    def push_frame(self, frame: dict, signals: list) -> None:
        """
        Non-blocking.  Call at 4 Hz from the WebSocket handler.
        ``frame`` is the dict already sent to the frontend (band powers, emotion …).
        ``signals`` is the raw channel matrix (list of 4 lists).
        """
        if self._closed:
            return
        self._frame_buf.append((frame, signals))

    async def close(self) -> None:
        """Flush remaining data and shut down."""
        self._closed = True
        # Give flush loop one last cycle
        await asyncio.sleep(_FLUSH_INTERVAL * 1.1)
        self._flush_task.cancel()
        self._write_task.cancel()
        # Drain write queue
        remaining: list[dict] = []
        while not self._write_q.empty():
            try:
                remaining.append(self._write_q.get_nowait())
            except asyncio.QueueEmpty:
                break
        if remaining:
            await self._execute_batch(remaining)
        await self._pool.close()
        logger.info("TimescaleWriter closed (user=%s)", self._user_id)

    # ------------------------------------------------------------------
    # Internal tasks
    # ------------------------------------------------------------------

    async def _flush_loop(self) -> None:
        """Every 1 s: average buffered 4-Hz frames → 1 row → write queue."""
        while not self._closed:
            await asyncio.sleep(_FLUSH_INTERVAL)
            frames = list(self._frame_buf)
            self._frame_buf.clear()
            if not frames:
                continue
            row = self._average_frames(frames)
            await self._write_q.put(row)

    async def _write_loop(self) -> None:
        """Drain write queue in batches of up to _WRITE_BATCH_MAX."""
        batch: list[dict] = []
        while not self._closed:
            try:
                row = await asyncio.wait_for(self._write_q.get(), timeout=2.0)
                batch.append(row)
                # Collect any additional rows already queued (non-blocking)
                while len(batch) < _WRITE_BATCH_MAX:
                    try:
                        batch.append(self._write_q.get_nowait())
                    except asyncio.QueueEmpty:
                        break
                await self._execute_batch(batch)
                batch.clear()
            except asyncio.TimeoutError:
                if batch:
                    await self._execute_batch(batch)
                    batch.clear()
            except asyncio.CancelledError:
                break

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _average_frames(self, frames: list[tuple[dict, list]]) -> dict:
        """Average numeric fields across frames and build a row dict."""
        row: dict[str, Any] = {
            "time":    datetime.now(timezone.utc),
            "user_id": self._user_id,
        }

        all_frames = [f for f, _ in frames]
        all_signals = [s for _, s in frames]

        for key in _NUMERIC_KEYS:
            vals = [f.get(key) for f in all_frames if f.get(key) is not None]
            row[key] = statistics.mean(vals) if vals else None

        # Dominant emotion (mode)
        emotions = [f.get("emotion") for f in all_frames if f.get("emotion")]
        row["emotion"] = max(set(emotions), key=emotions.count) if emotions else None

        sleep_stages = [f.get("sleep_stage") for f in all_frames if f.get("sleep_stage")]
        row["sleep_stage"] = sleep_stages[-1] if sleep_stages else None

        # Raw snapshot: last 4×64 matrix (list of 4 channels)
        if all_signals:
            last_sig = all_signals[-1]
            row["raw_snapshot"] = json.dumps(last_sig) if isinstance(last_sig, list) else None
        else:
            row["raw_snapshot"] = None

        return row

    async def _execute_batch(self, batch: list[dict]) -> None:
        if not batch:
            return
        try:
            async with self._pool.acquire() as conn:
                await conn.executemany(
                    """
                    INSERT INTO brain_readings (
                        time, user_id,
                        alpha, beta, theta, gamma, delta,
                        emotion, valence, arousal,
                        focus_index, stress_index, relaxation_idx,
                        flow_score, creativity_score, attention_score,
                        sleep_stage, sqi, raw_snapshot
                    ) VALUES (
                        $1,  $2,
                        $3,  $4,  $5,  $6,  $7,
                        $8,  $9,  $10,
                        $11, $12, $13,
                        $14, $15, $16,
                        $17, $18, $19
                    )
                    """,
                    [
                        (
                            r["time"],          r["user_id"],
                            r.get("alpha"),     r.get("beta"),      r.get("theta"),
                            r.get("gamma"),     r.get("delta"),
                            r.get("emotion"),   r.get("valence"),   r.get("arousal"),
                            r.get("focus_index"),   r.get("stress_index"), r.get("relaxation_idx"),
                            r.get("flow_score"),    r.get("creativity_score"), r.get("attention_score"),
                            r.get("sleep_stage"),   r.get("sqi"),
                            r.get("raw_snapshot"),
                        )
                        for r in batch
                    ],
                )
            logger.debug("TimescaleWriter wrote %d rows", len(batch))
        except Exception as exc:
            logger.error("TimescaleWriter write failed: %s", exc)

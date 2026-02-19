"""Continuous brain timeline endpoints (TimescaleDB continuous aggregates)."""

import io
import os
import time as _time
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

router = APIRouter()

_VALID_METRICS = {
    "focus_index", "stress_index", "relaxation_idx", "flow_score",
    "creativity_score", "attention_score", "valence", "arousal",
    "alpha", "beta", "theta", "gamma", "delta",
}

_BUCKET_TO_VIEW = {
    "1m": "brain_readings_1min",
    "1h": "brain_readings_1hr",
    "1d": "brain_readings_1day",
}


@router.get("/brain/timeline")
async def brain_timeline(
    user_id: str = "default",
    from_ts: Optional[float] = None,
    to_ts: Optional[float] = None,
    metric: str = "focus_index",
    bucket: str = "1h",
):
    """Apple Health-style continuous timeline from TimescaleDB continuous aggregates."""
    if metric not in _VALID_METRICS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid metric '{metric}'. Choose from: {sorted(_VALID_METRICS)}",
        )
    view = _BUCKET_TO_VIEW.get(bucket)
    if view is None:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid bucket '{bucket}'. Choose from: {list(_BUCKET_TO_VIEW)}",
        )

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return {"buckets": [], "message": "DATABASE_URL not configured"}

    try:
        import psycopg2
        import psycopg2.extras

        now = _time.time()
        ts_from = from_ts if from_ts else now - 86400
        ts_to = to_ts if to_ts else now

        with psycopg2.connect(db_url) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # metric and view are whitelisted above — safe to interpolate
                cur.execute(
                    f"""
                    SELECT
                        bucket                           AS time,
                        {metric}                         AS value,
                        dominant_emotion
                    FROM {view}
                    WHERE user_id = %s
                      AND bucket >= to_timestamp(%s)
                      AND bucket <= to_timestamp(%s)
                    ORDER BY bucket
                    """,
                    (user_id, ts_from, ts_to),
                )
                rows = cur.fetchall() or []

        buckets = [
            {
                "time": row["time"].isoformat(),
                "value": round(float(row["value"]), 4) if row["value"] is not None else None,
                "dominant_emotion": row["dominant_emotion"],
            }
            for row in rows
        ]
        return {"buckets": buckets}

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/brain/export")
async def brain_export(
    user_id: str = "default",
    from_ts: Optional[float] = None,
    to_ts: Optional[float] = None,
    format: str = "csv",
    metrics: str = "focus_index,stress_index,relaxation_idx,flow_score,valence",
):
    """Export raw brain_readings for a date range as CSV or JSON."""
    requested = [m.strip() for m in metrics.split(",") if m.strip()]
    safe_cols = [m for m in requested if m in _VALID_METRICS]
    if not safe_cols:
        safe_cols = ["focus_index", "stress_index", "relaxation_idx"]

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise HTTPException(status_code=503, detail="DATABASE_URL not configured")

    now = _time.time()
    ts_from = from_ts if from_ts else now - 86400
    ts_to = to_ts if to_ts else now

    try:
        import psycopg2
        import psycopg2.extras

        col_list = ", ".join(safe_cols)  # whitelisted
        with psycopg2.connect(db_url) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT time, user_id, emotion, {col_list}
                    FROM brain_readings
                    WHERE user_id = %s
                      AND time >= to_timestamp(%s)
                      AND time <= to_timestamp(%s)
                    ORDER BY time
                    """,
                    (user_id, ts_from, ts_to),
                )
                rows = cur.fetchall() or []

        if format.lower() == "json":
            data = [
                {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in row.items()}
                for row in rows
            ]
            return JSONResponse(content=data)

        output = io.StringIO()
        if rows:
            headers = list(rows[0].keys())
            output.write(",".join(headers) + "\n")
            for row in rows:
                output.write(",".join(
                    str(v.isoformat() if hasattr(v, "isoformat") else v) if v is not None else ""
                    for v in row.values()
                ) + "\n")

        filename = f"brain_readings_{user_id}_{int(ts_from)}_{int(ts_to)}.csv"
        output.seek(0)
        return StreamingResponse(
            iter([output.read()]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

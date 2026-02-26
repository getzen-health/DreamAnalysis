"""Parquet EEG data query endpoints.

Provides fast columnar access to EEG history stored as Parquet files.
Complements the Postgres brain_readings table — Parquet is faster for
large date-range queries (column pruning, no row-scan overhead).
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException

from storage.parquet_writer import get_user_daily_summary, read_user_parquet

router = APIRouter()


@router.get("/brain/parquet/daily-summary")
async def parquet_daily_summary(
    user_id: str = "default",
    date: Optional[str] = None,
):
    """
    Return EEG daily summary stats from Parquet storage.

    Faster than Postgres for long date ranges — uses columnar read.
    date format: YYYY-MM-DD (defaults to today UTC)
    """
    result = get_user_daily_summary(user_id, date)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/brain/parquet/columns")
async def parquet_column_query(
    user_id: str = "default",
    date: Optional[str] = None,
    columns: str = "timestamp,focus_index,stress_index,valence",
):
    """
    Query specific EEG columns from Parquet storage for a given day.

    columns: comma-separated list of EEG column names.
    Supported: timestamp, alpha, beta, theta, delta, gamma, low_beta, high_beta,
               focus_index, stress_index, relaxation_index, flow_score,
               creativity_score, valence, arousal, emotion, emotion_confidence,
               sleep_stage, sqi
    """
    col_list: List[str] = [c.strip() for c in columns.split(",") if c.strip()]
    if not col_list:
        raise HTTPException(status_code=400, detail="No columns specified")

    df = read_user_parquet(user_id, date, columns=col_list)
    if df is None:
        raise HTTPException(status_code=503, detail="pyarrow not installed on this server")
    if df.empty:
        return {"date": date, "user_id": user_id, "rows": [], "columns": col_list}

    # Cap at 10 000 rows to avoid huge JSON responses
    if len(df) > 10_000:
        df = df.iloc[-10_000:]

    return {
        "date": date,
        "user_id": user_id,
        "n_rows": len(df),
        "columns": col_list,
        "rows": df.to_dict(orient="records"),
    }

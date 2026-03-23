"""Training sync — pull corrections from Supabase for ML retraining.

GET  /training/corrections/{user_id} — fetch corrections from Supabase
POST /training/sync/{user_id}        — sync Supabase corrections to local store + trigger retrain
POST /training/retrain/{user_id}     — force retrain from all available data
GET  /training/status/{user_id}      — training data stats
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

log = logging.getLogger(__name__)
router = APIRouter(prefix="/training", tags=["training-sync"])

# Local correction store: JSONL per user in ml/user_data/corrections/
CORRECTIONS_DIR = Path(__file__).parent.parent.parent / "user_data" / "corrections"


class SyncResult(BaseModel):
    synced: int = 0
    total_corrections: int = 0
    new_corrections: int = 0
    retrain_triggered: bool = False
    message: str = ""


class TrainingStatus(BaseModel):
    user_id: str
    total_corrections: int = 0
    local_corrections: int = 0
    supabase_corrections: int = 0
    last_sync: Optional[str] = None
    retrain_available: bool = False


def _corrections_path(user_id: str) -> Path:
    """Return path to a user's local corrections JSONL file."""
    CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    return CORRECTIONS_DIR / f"{user_id}_corrections.jsonl"


def _load_local_corrections(user_id: str) -> List[Dict[str, Any]]:
    """Load all local corrections for a user."""
    path = _corrections_path(user_id)
    if not path.exists():
        return []
    corrections = []
    for line in path.read_text().strip().split("\n"):
        if line.strip():
            try:
                corrections.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return corrections


def _append_corrections(user_id: str, new_records: List[Dict[str, Any]]) -> int:
    """Append corrections to local JSONL, deduplicating by created_at."""
    existing = _load_local_corrections(user_id)
    existing_timestamps = {r.get("created_at") for r in existing}

    path = _corrections_path(user_id)
    added = 0
    with open(path, "a") as f:
        for record in new_records:
            ts = record.get("created_at")
            if ts and ts in existing_timestamps:
                continue  # dedup
            f.write(json.dumps(record) + "\n")
            existing_timestamps.add(ts)
            added += 1
    return added


def _get_supabase_corrections(user_id: str) -> List[Dict[str, Any]]:
    """Fetch corrections from Supabase for a user. Returns [] if Supabase unavailable."""
    try:
        from lib.supabase_client import get_supabase

        client = get_supabase()
        if client is None:
            return []

        result = (
            client.table("user_feedback")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at")
            .execute()
        )
        return result.data or []
    except Exception as exc:
        log.warning("Failed to fetch corrections from Supabase: %s", exc)
        return []


@router.get("/corrections/{user_id}")
async def get_corrections(user_id: str):
    """Fetch corrections from Supabase for a user."""
    corrections = _get_supabase_corrections(user_id)
    return {
        "user_id": user_id,
        "count": len(corrections),
        "corrections": corrections,
    }


@router.post("/sync/{user_id}", response_model=SyncResult)
async def sync_corrections(user_id: str):
    """Sync Supabase corrections to local store and trigger retrain if enough new data."""
    supabase_corrections = _get_supabase_corrections(user_id)
    if not supabase_corrections:
        local_count = len(_load_local_corrections(user_id))
        return SyncResult(
            synced=0,
            total_corrections=local_count,
            new_corrections=0,
            retrain_triggered=False,
            message="No Supabase corrections found (Supabase may not be configured)",
        )

    new_count = _append_corrections(user_id, supabase_corrections)
    total = len(_load_local_corrections(user_id))

    retrain_triggered = False
    if new_count >= 5:
        try:
            from training.auto_retrainer import retrain_personal_model

            import asyncio

            result = await asyncio.to_thread(retrain_personal_model, user_id)
            retrain_triggered = result.get("trained", False)
            log.info(
                "[training-sync] Retrain triggered for %s: %s", user_id, result
            )
        except Exception as exc:
            log.warning("[training-sync] Retrain failed for %s: %s", user_id, exc)

    return SyncResult(
        synced=len(supabase_corrections),
        total_corrections=total,
        new_corrections=new_count,
        retrain_triggered=retrain_triggered,
        message=f"Synced {new_count} new corrections"
        + (", retrain triggered" if retrain_triggered else ""),
    )


@router.post("/retrain/{user_id}")
async def force_retrain(user_id: str):
    """Force retrain from all available local data."""
    try:
        from training.auto_retrainer import retrain_personal_model

        import asyncio

        result = await asyncio.to_thread(retrain_personal_model, user_id)
        return {"user_id": user_id, "result": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrain failed: {exc}")


@router.get("/status/{user_id}", response_model=TrainingStatus)
async def training_status(user_id: str):
    """Show training data stats for a user."""
    local = _load_local_corrections(user_id)

    # Count Supabase corrections (best-effort)
    supabase_count = 0
    try:
        from lib.supabase_client import get_supabase

        client = get_supabase()
        if client is not None:
            result = (
                client.table("user_feedback")
                .select("*", count="exact")
                .eq("user_id", user_id)
                .execute()
            )
            supabase_count = result.count or 0
    except Exception:
        pass

    last_sync = None
    if local:
        last_ts = max(
            (r.get("created_at", "") for r in local), default=None
        )
        if last_ts:
            last_sync = last_ts

    total = max(len(local), supabase_count)

    return TrainingStatus(
        user_id=user_id,
        total_corrections=total,
        local_corrections=len(local),
        supabase_corrections=supabase_count,
        last_sync=last_sync,
        retrain_available=total >= 5,
    )

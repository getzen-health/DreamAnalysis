"""Unified multi-wearable data ingestion and normalization.

Normalizes HRV, stress, sleep, and activity data from:
- Apple Watch / HealthKit
- Fitbit Web API
- Garmin Connect IQ
- Whoop API

Endpoints
---------
POST /wearables/ingest    — accept any platform's data, return normalized snapshot
GET  /wearables/platforms — list supported platforms + fields
GET  /wearables/status    — health check
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/wearables", tags=["Wearables"])

# ── Normalized schema ──────────────────────────────────────────────────────


class WearableSnapshot(BaseModel):
    """Platform-agnostic wearable data snapshot."""

    hrv_rmssd: Optional[float] = Field(None, description="HRV RMSSD in ms")
    hrv_sdnn: Optional[float] = Field(None, description="HRV SDNN in ms")
    heart_rate_bpm: Optional[float] = Field(None, description="Current or resting heart rate (bpm)")
    stress_score: Optional[float] = Field(None, description="Stress score 0-100 (platform-normalized)")
    recovery_score: Optional[float] = Field(None, description="Recovery/readiness score 0-100")
    sleep_quality: Optional[float] = Field(None, description="Sleep quality 0-1")
    deep_sleep_minutes: Optional[int] = Field(None, description="Deep sleep duration (minutes)")
    rem_sleep_minutes: Optional[int] = Field(None, description="REM sleep duration (minutes)")
    spo2: Optional[float] = Field(None, description="Blood oxygen saturation (%)")
    respiratory_rate: Optional[float] = Field(None, description="Respiratory rate (breaths/min)")
    steps_today: Optional[int] = Field(None, description="Step count for the day")
    active_energy_kcal: Optional[float] = Field(None, description="Active energy burned (kcal)")
    body_battery: Optional[float] = Field(None, description="Garmin Body Battery score 0-100")
    platform: str = Field("unknown", description="Source platform identifier")


# ── Request model ──────────────────────────────────────────────────────────


class WearableIngestRequest(BaseModel):
    """Request body for wearable data ingestion."""

    platform: str = Field(
        ...,
        description="Platform identifier: apple_watch, fitbit, garmin, whoop",
    )
    data: Dict[str, Any] = Field(
        ...,
        description="Raw payload in the platform's native field format",
    )


# ── Adapter helpers ────────────────────────────────────────────────────────

# Emotion relevance annotations — shared across all adapters.
_EMOTION_RELEVANCE: Dict[str, str] = {
    "hrv_rmssd": "high — autonomic stress marker, feeds stress_management EI domain",
    "hrv_sdnn": "high — overall autonomic variability, feeds all EI domains",
    "heart_rate_bpm": "medium — sympathetic activation proxy",
    "stress_score": "high — platform-computed stress, feeds stress_management EI domain",
    "recovery_score": "high — readiness baseline, feeds all EI domains",
    "sleep_quality": "high — prior-night sleep, strongest for focus + mood",
    "deep_sleep_minutes": "medium — slow-wave sleep, feeds memory consolidation domain",
    "rem_sleep_minutes": "high — emotional processing sleep, feeds emotion regulation domain",
    "spo2": "medium — oxygenation baseline, impacts cognitive performance",
    "respiratory_rate": "medium — autonomic marker, complements HRV",
    "steps_today": "low — activity proxy, indirect mood correlate",
    "active_energy_kcal": "low — exertion indicator, indirect arousal correlate",
    "body_battery": "high (Garmin) — validated composite energy score",
}


def _safe_float(value: Any) -> Optional[float]:
    """Convert value to float or return None."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    """Convert value to int or return None."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _adapt_apple_watch(data: Dict[str, Any]) -> WearableSnapshot:
    """Normalize Apple Watch / HealthKit payload.

    Expected input fields (all optional):
        hrv_ms              — HRV RMSSD from HealthKit (ms)
        heart_rate          — heart rate (bpm)
        sleep_hours         — total sleep duration (hours)
        deep_sleep_hours    — deep sleep duration (hours)
        rem_sleep_hours     — REM sleep duration (hours)
        steps               — step count
        active_calories     — active energy (kcal)
        spo2                — blood oxygen (%)
        respiratory_rate    — respiratory rate (breaths/min)
    """
    hrv_rmssd = _safe_float(data.get("hrv_ms"))

    # Derive sleep_quality from total sleep hours (assumes 8h = 1.0)
    sleep_hours = _safe_float(data.get("sleep_hours"))
    sleep_quality: Optional[float] = None
    if sleep_hours is not None:
        sleep_quality = min(1.0, sleep_hours / 8.0)

    deep_sleep_hours = _safe_float(data.get("deep_sleep_hours"))
    deep_sleep_minutes = _safe_int(deep_sleep_hours * 60) if deep_sleep_hours is not None else None

    rem_sleep_hours = _safe_float(data.get("rem_sleep_hours"))
    rem_sleep_minutes = _safe_int(rem_sleep_hours * 60) if rem_sleep_hours is not None else None

    return WearableSnapshot(
        hrv_rmssd=hrv_rmssd,
        heart_rate_bpm=_safe_float(data.get("heart_rate")),
        sleep_quality=sleep_quality,
        deep_sleep_minutes=deep_sleep_minutes,
        rem_sleep_minutes=rem_sleep_minutes,
        spo2=_safe_float(data.get("spo2")),
        respiratory_rate=_safe_float(data.get("respiratory_rate")),
        steps_today=_safe_int(data.get("steps")),
        active_energy_kcal=_safe_float(data.get("active_calories")),
        platform="apple_watch",
    )


def _adapt_fitbit(data: Dict[str, Any]) -> WearableSnapshot:
    """Normalize Fitbit Web API payload.

    Expected input fields (all optional):
        hrv             — dict with key "dailyRmssd" (ms)
        heartRate       — resting heart rate (bpm)
        stress_score    — 0-100 where 0=no stress, 100=max stress
        sleep           — dict with "efficiency" (0-100) and "stages"
                          stages: list of {"type": "deep"|"rem", "minutes": int}
        steps           — step count
    """
    # HRV from nested dict
    hrv_block = data.get("hrv")
    hrv_rmssd: Optional[float] = None
    if isinstance(hrv_block, dict):
        hrv_rmssd = _safe_float(hrv_block.get("dailyRmssd"))

    # Stress: Fitbit 0-100, already on a 0-100 scale matching our convention
    stress_score = _safe_float(data.get("stress_score"))

    # Sleep quality from efficiency field (0-100 → 0-1)
    sleep_block = data.get("sleep")
    sleep_quality: Optional[float] = None
    deep_sleep_minutes: Optional[int] = None
    rem_sleep_minutes: Optional[int] = None
    if isinstance(sleep_block, dict):
        efficiency = _safe_float(sleep_block.get("efficiency"))
        if efficiency is not None:
            sleep_quality = min(1.0, efficiency / 100.0)

        stages = sleep_block.get("stages")
        if isinstance(stages, list):
            for stage in stages:
                if not isinstance(stage, dict):
                    continue
                stage_type = str(stage.get("type", "")).lower()
                stage_minutes = _safe_int(stage.get("minutes"))
                if stage_minutes is None:
                    continue
                if stage_type == "deep":
                    deep_sleep_minutes = (deep_sleep_minutes or 0) + stage_minutes
                elif stage_type == "rem":
                    rem_sleep_minutes = (rem_sleep_minutes or 0) + stage_minutes

    return WearableSnapshot(
        hrv_rmssd=hrv_rmssd,
        heart_rate_bpm=_safe_float(data.get("heartRate")),
        stress_score=stress_score,
        sleep_quality=sleep_quality,
        deep_sleep_minutes=deep_sleep_minutes,
        rem_sleep_minutes=rem_sleep_minutes,
        steps_today=_safe_int(data.get("steps")),
        platform="fitbit",
    )


def _adapt_garmin(data: Dict[str, Any]) -> WearableSnapshot:
    """Normalize Garmin Connect IQ payload.

    Expected input fields (all optional):
        hrv_weekly_average  — weekly average HRV RMSSD (ms)
        stress_level_value  — stress 0-100 (0=rest, 100=high stress)
        body_battery_value  — Body Battery 0-100 (already normalized)
        sleep_score         — sleep score 0-100 → converted to 0-1
        total_steps         — step count
        active_energy       — active calories (kcal)
        heart_rate          — heart rate (bpm)
    """
    # Sleep score 0-100 → quality 0-1
    sleep_score = _safe_float(data.get("sleep_score"))
    sleep_quality: Optional[float] = None
    if sleep_score is not None:
        sleep_quality = min(1.0, sleep_score / 100.0)

    return WearableSnapshot(
        hrv_rmssd=_safe_float(data.get("hrv_weekly_average")),
        heart_rate_bpm=_safe_float(data.get("heart_rate")),
        stress_score=_safe_float(data.get("stress_level_value")),
        body_battery=_safe_float(data.get("body_battery_value")),
        sleep_quality=sleep_quality,
        steps_today=_safe_int(data.get("total_steps")),
        active_energy_kcal=_safe_float(data.get("active_energy")),
        platform="garmin",
    )


def _adapt_whoop(data: Dict[str, Any]) -> WearableSnapshot:
    """Normalize Whoop API payload.

    Expected input fields (all optional):
        hrv_rmssd_milli     — HRV RMSSD in ms
        recovery_score      — recovery 0-100 (already normalized)
        strain              — strain score (0-21, Whoop-specific; not mapped)
        sleep_performance   — sleep performance 0-100 → converted to 0-1
        naps                — list of nap dicts (not mapped to standard schema)
        blood_oxygen        — SpO2 (%)
    """
    # Sleep performance 0-100 → quality 0-1
    sleep_perf = _safe_float(data.get("sleep_performance"))
    sleep_quality: Optional[float] = None
    if sleep_perf is not None:
        sleep_quality = min(1.0, sleep_perf / 100.0)

    return WearableSnapshot(
        hrv_rmssd=_safe_float(data.get("hrv_rmssd_milli")),
        recovery_score=_safe_float(data.get("recovery_score")),
        sleep_quality=sleep_quality,
        spo2=_safe_float(data.get("blood_oxygen")),
        platform="whoop",
    )


# ── Adapter dispatch table ─────────────────────────────────────────────────

_ADAPTERS = {
    "apple_watch": _adapt_apple_watch,
    "fitbit": _adapt_fitbit,
    "garmin": _adapt_garmin,
    "whoop": _adapt_whoop,
}

# ── Platform metadata ──────────────────────────────────────────────────────

_PLATFORM_METADATA = [
    {
        "name": "apple_watch",
        "display_name": "Apple Watch / HealthKit",
        "fields": [
            "hrv_ms",
            "heart_rate",
            "sleep_hours",
            "deep_sleep_hours",
            "rem_sleep_hours",
            "steps",
            "active_calories",
            "spo2",
            "respiratory_rate",
        ],
        "notes": (
            "HealthKit exports RMSSD as hrv_ms. "
            "Sleep stages require a Sleep app or third-party sleep tracker. "
            "spo2 requires Apple Watch Series 6 or later."
        ),
    },
    {
        "name": "fitbit",
        "display_name": "Fitbit Web API",
        "fields": [
            "hrv (dict: {dailyRmssd})",
            "heartRate",
            "stress_score",
            "sleep (dict: {efficiency, stages})",
            "steps",
        ],
        "notes": (
            "stress_score range is 0 (no stress) to 100 (max stress). "
            "Sleep stages list uses Fitbit stage type strings: 'deep', 'rem', 'light', 'wake'. "
            "HRV is available via the Fitbit HRV API (requires premium)."
        ),
    },
    {
        "name": "garmin",
        "display_name": "Garmin Connect IQ",
        "fields": [
            "hrv_weekly_average",
            "stress_level_value",
            "body_battery_value",
            "sleep_score",
            "total_steps",
            "active_energy",
            "heart_rate",
        ],
        "notes": (
            "body_battery_value is 0-100, already a composite energy score. "
            "stress_level_value is 0 (at rest) to 100 (high stress). "
            "HRV is a weekly average rather than daily; use with caution for daily readings."
        ),
    },
    {
        "name": "whoop",
        "display_name": "Whoop API",
        "fields": [
            "hrv_rmssd_milli",
            "recovery_score",
            "strain",
            "sleep_performance",
            "naps",
            "blood_oxygen",
        ],
        "notes": (
            "recovery_score is 0-100 (green ≥67, yellow 34-66, red ≤33). "
            "strain is 0-21 (Whoop-specific exertion scale) — not mapped to standard schema. "
            "sleep_performance is 0-100 percentage of needed sleep obtained."
        ),
    },
]


# ── Helper: build emotion relevance for a snapshot ─────────────────────────


def _build_emotion_relevance(snapshot: WearableSnapshot) -> Dict[str, str]:
    """Return emotion relevance annotations for fields present in the snapshot."""
    relevance: Dict[str, str] = {}
    snap_dict = snapshot.model_dump()
    for field, annotation in _EMOTION_RELEVANCE.items():
        if snap_dict.get(field) is not None:
            relevance[field] = annotation
    return relevance


# ── Endpoints ──────────────────────────────────────────────────────────────


@router.post("/ingest")
def ingest_wearable_data(request: WearableIngestRequest) -> dict:
    """Accept wearable data from any supported platform and return a normalized snapshot.

    The response includes:
    - ``snapshot``: normalized ``WearableSnapshot`` with all available fields
    - ``emotion_relevance``: per-field annotation indicating relevance to EI domains
    - ``fields_received``: count of non-null fields in the normalized snapshot
    """
    adapter = _ADAPTERS.get(request.platform)
    if adapter is None:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported platform '{request.platform}'. "
                f"Supported: {sorted(_ADAPTERS.keys())}"
            ),
        )

    try:
        snapshot = adapter(request.data)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Failed to normalize data for platform '{request.platform}': {exc}",
        ) from exc

    snap_dict = snapshot.model_dump()
    fields_received = sum(
        1 for k, v in snap_dict.items() if k != "platform" and v is not None
    )
    emotion_relevance = _build_emotion_relevance(snapshot)

    return {
        "snapshot": snap_dict,
        "emotion_relevance": emotion_relevance,
        "fields_received": fields_received,
        "platform": request.platform,
    }


@router.get("/platforms")
def list_platforms() -> dict:
    """List all supported wearable platforms and their expected input fields."""
    return {
        "platforms": _PLATFORM_METADATA,
        "supported_platform_ids": sorted(_ADAPTERS.keys()),
    }


@router.get("/status")
def status() -> dict:
    """Health check for the wearables ingestion service."""
    return {
        "status": "ready",
        "supported_platforms": sorted(_ADAPTERS.keys()),
    }

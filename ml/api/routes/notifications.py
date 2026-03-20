"""Smart daily notification API routes.

Endpoints
---------
GET  /notifications/morning-report/{user_id}   — generate morning brain report notification
GET  /notifications/evening-winddown/{user_id} — generate evening wind-down suggestion
GET  /notifications/weekly-summary/{user_id}   — generate weekly recap notification
GET  /notifications/preferences/{user_id}      — get user notification preferences
PUT  /notifications/preferences/{user_id}      — update notification preferences
POST /notifications/test/{user_id}             — send a test notification payload
GET  /notifications/history/{user_id}          — past notifications sent
GET  /notifications/schedule/{user_id}         — next scheduled send times

Related issue: #350 — Smart Daily Notifications
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from notifications.smart_notifications import (
    NotificationRecord,
    NotificationPreferences,
    WIND_DOWN_ACTIVITIES,
    get_morning_generator,
    get_evening_generator,
    get_weekly_generator,
    get_scheduler,
    get_history,
    get_preferences,
    update_preferences,
    _record_notification,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/notifications", tags=["Notifications"])


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class PreferencesUpdateRequest(BaseModel):
    enabled: Optional[bool] = None
    morning_report_enabled: Optional[bool] = None
    evening_winddown_enabled: Optional[bool] = None
    post_session_enabled: Optional[bool] = None
    weekly_summary_enabled: Optional[bool] = None
    quiet_hours_start: Optional[int] = Field(None, ge=0, le=23)
    quiet_hours_end: Optional[int] = Field(None, ge=0, le=23)
    morning_hour: Optional[int] = Field(None, ge=0, le=23)
    evening_hour: Optional[int] = Field(None, ge=0, le=23)
    timezone_offset_hours: Optional[int] = Field(None, ge=-12, le=14)
    skip_weekends_morning: Optional[bool] = None
    min_stress_for_evening: Optional[float] = Field(None, ge=0.0, le=1.0)


class TestNotificationRequest(BaseModel):
    notification_type: str = Field(
        "morning_report",
        description="'morning_report' | 'evening_winddown' | 'weekly_summary'",
    )
    override_stress: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Override stress index for evening test"
    )


class NotificationEngageRequest(BaseModel):
    notification_id: str
    action: str = Field(..., description="'opened' | 'dismissed'")


# ── Morning report ────────────────────────────────────────────────────────────

@router.get("/morning-report/{user_id}")
async def get_morning_report(user_id: str) -> Dict[str, Any]:
    """Generate a morning brain-report notification payload for the given user.

    Pulls yesterday's voice check-in + health data, computes focus/sleep/
    readiness scores, and returns a ready-to-send notification.
    """
    generator = get_morning_generator()
    result = generator.generate(user_id)

    notification_id = str(uuid.uuid4())
    record = NotificationRecord(
        notification_id=notification_id,
        user_id=user_id,
        notification_type="morning_report",
        title=result["title"],
        body=result["body"],
        route=result["route"],
        scheduled_at=time.time(),
        sent_at=time.time(),
        metadata={
            "scores": result["scores"],
            "data_sources": result["data_sources"],
            "has_data": result["has_data"],
        },
    )
    _record_notification(record)
    get_scheduler().mark_sent(user_id, "morning_report")

    return {
        "notification_id": notification_id,
        "user_id": user_id,
        "type": "morning_report",
        **result,
    }


# ── Evening wind-down ─────────────────────────────────────────────────────────

@router.get("/evening-winddown/{user_id}")
async def get_evening_winddown(user_id: str) -> Dict[str, Any]:
    """Generate an evening wind-down notification payload for the given user.

    Analyzes today's stress patterns and selects the most appropriate
    breathing or relaxation activity. Returns skip=True when stress is
    below the user's configured minimum threshold.
    """
    generator = get_evening_generator()
    result = generator.generate(user_id)

    notification_id = str(uuid.uuid4())
    if not result["skip"]:
        record = NotificationRecord(
            notification_id=notification_id,
            user_id=user_id,
            notification_type="evening_winddown",
            title=result["title"],
            body=result["body"],
            route=result["route"],
            scheduled_at=time.time(),
            sent_at=time.time(),
            metadata={
                "activity_id": result["activity_id"],
                "stress_index": result["stress_index"],
            },
        )
        _record_notification(record)
        get_scheduler().mark_sent(user_id, "evening_winddown")

    return {
        "notification_id": notification_id,
        "user_id": user_id,
        "type": "evening_winddown",
        **result,
    }


# ── Weekly summary ────────────────────────────────────────────────────────────

@router.get("/weekly-summary/{user_id}")
async def get_weekly_summary(user_id: str) -> Dict[str, Any]:
    """Generate a weekly recap notification payload for the given user.

    Aggregates the past 7 days of voice check-ins, health metrics, and
    session data into a concise summary with key stats and a deep-link
    to the full weekly-brain-summary page.

    Oura sends weekly email summaries as a re-engagement mechanism;
    this endpoint provides the equivalent payload for push notifications.
    """
    prefs = get_preferences(user_id)
    if not prefs.enabled or not prefs.weekly_summary_enabled:
        return {
            "user_id": user_id,
            "type": "weekly_summary",
            "skip": True,
            "skip_reason": "Weekly summary notifications are disabled",
        }

    generator = get_weekly_generator()
    result = generator.generate(user_id)

    notification_id = str(uuid.uuid4())
    record = NotificationRecord(
        notification_id=notification_id,
        user_id=user_id,
        notification_type="weekly_summary",
        title=result["title"],
        body=result["body"],
        route=result["route"],
        scheduled_at=time.time(),
        sent_at=time.time(),
        metadata={
            "stats": result["stats"],
            "has_data": result["has_data"],
        },
    )
    _record_notification(record)
    get_scheduler().mark_sent(user_id, "weekly_summary")

    return {
        "notification_id": notification_id,
        "user_id": user_id,
        "type": "weekly_summary",
        "skip": False,
        **result,
    }


# ── Preferences ───────────────────────────────────────────────────────────────

@router.get("/preferences/{user_id}")
async def get_notification_preferences(user_id: str) -> Dict[str, Any]:
    """Return the current notification preferences for a user."""
    return get_preferences(user_id).to_dict()


@router.put("/preferences/{user_id}")
async def update_notification_preferences(
    user_id: str,
    req: PreferencesUpdateRequest,
) -> Dict[str, Any]:
    """Update notification preferences for a user.

    Only the supplied fields are changed — omit a field to leave it unchanged.
    """
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No fields supplied to update")
    prefs = update_preferences(user_id, updates)
    return {"status": "updated", "preferences": prefs.to_dict()}


# ── Test notification ─────────────────────────────────────────────────────────

@router.post("/test/{user_id}")
async def send_test_notification(
    user_id: str,
    req: TestNotificationRequest,
) -> Dict[str, Any]:
    """Generate and return a test notification payload without side effects.

    Useful for verifying that the notification content looks right before
    enabling automated delivery.  Does NOT update cooldown timers.
    """
    if req.notification_type == "morning_report":
        generator = get_morning_generator()
        result = generator.generate(user_id)
        ntype = "morning_report"
    elif req.notification_type == "evening_winddown":
        generator_e = get_evening_generator()
        voice_override = (
            {"avg_stress": req.override_stress, "avg_valence": 0.0, "avg_arousal": 0.5}
            if req.override_stress is not None
            else None
        )
        result = generator_e.generate(user_id, voice_data=voice_override)
        ntype = "evening_winddown"
    elif req.notification_type == "weekly_summary":
        generator_w = get_weekly_generator()
        result = generator_w.generate(user_id)
        ntype = "weekly_summary"
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown notification_type '{req.notification_type}'. "
                   "Use 'morning_report', 'evening_winddown', or 'weekly_summary'.",
        )

    return {
        "test": True,
        "user_id": user_id,
        "type": ntype,
        "notification_id": str(uuid.uuid4()),
        **result,
    }


# ── History ───────────────────────────────────────────────────────────────────

@router.get("/history/{user_id}")
async def get_notification_history(
    user_id: str,
    limit: int = 50,
    notification_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Return past notifications generated for the user (most recent first).

    Optionally filter by notification_type.  Results are capped at `limit`
    (max 200).
    """
    limit = min(limit, 200)
    history = get_history(user_id)
    if notification_type:
        history = [n for n in history if n["notification_type"] == notification_type]
    history = list(reversed(history))[:limit]
    return {
        "user_id": user_id,
        "count": len(history),
        "notifications": history,
    }


# ── Schedule ──────────────────────────────────────────────────────────────────

@router.get("/schedule/{user_id}")
async def get_notification_schedule(user_id: str) -> Dict[str, Any]:
    """Return the next scheduled send times for morning and evening notifications."""
    scheduler = get_scheduler()
    prefs = get_preferences(user_id)
    now = time.time()

    morning_ts = scheduler.next_morning_ts(user_id, now)
    evening_ts = scheduler.next_evening_ts(user_id, now)

    # Weekly summary: next Sunday at morning_hour
    import datetime as _dt
    local_now = _dt.datetime.fromtimestamp(now, tz=_dt.timezone.utc).replace(tzinfo=None)
    local_now += _dt.timedelta(hours=prefs.timezone_offset_hours)
    next_sunday = local_now + _dt.timedelta(days=(6 - local_now.weekday()) % 7 or 7)
    next_sunday = next_sunday.replace(
        hour=prefs.morning_hour, minute=0, second=0, microsecond=0
    )
    weekly_ts = next_sunday.timestamp() - prefs.timezone_offset_hours * 3600

    return {
        "user_id": user_id,
        "morning_report": {
            "enabled": prefs.enabled and prefs.morning_report_enabled,
            "next_scheduled_utc": morning_ts,
            "hour_local": prefs.morning_hour,
            "skip_weekends": prefs.skip_weekends_morning,
        },
        "evening_winddown": {
            "enabled": prefs.enabled and prefs.evening_winddown_enabled,
            "next_scheduled_utc": evening_ts,
            "hour_local": prefs.evening_hour,
            "min_stress_threshold": prefs.min_stress_for_evening,
        },
        "weekly_summary": {
            "enabled": prefs.enabled and prefs.weekly_summary_enabled,
            "next_scheduled_utc": weekly_ts,
            "day": "Sunday",
            "hour_local": prefs.morning_hour,
        },
        "quiet_hours": {
            "start": prefs.quiet_hours_start,
            "end": prefs.quiet_hours_end,
        },
        "timezone_offset_hours": prefs.timezone_offset_hours,
    }


# ── Engagement tracking ───────────────────────────────────────────────────────

@router.post("/engage/{user_id}")
async def record_engagement(
    user_id: str,
    req: NotificationEngageRequest,
) -> Dict[str, Any]:
    """Record that a notification was opened or dismissed.

    Updates the in-memory history record if the notification_id is found.
    """
    from notifications.smart_notifications import _HISTORY

    history = _HISTORY.get(user_id, [])
    matched = next(
        (r for r in history if r.notification_id == req.notification_id), None
    )
    if not matched:
        raise HTTPException(
            status_code=404,
            detail=f"Notification {req.notification_id} not found for user {user_id}",
        )

    if req.action == "opened":
        matched.opened_at = time.time()
    elif req.action == "dismissed":
        matched.dismissed_at = time.time()
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown action '{req.action}'. Use 'opened' or 'dismissed'.",
        )

    return {"status": "recorded", "notification_id": req.notification_id, "action": req.action}


# ── Activities catalogue ──────────────────────────────────────────────────────

@router.get("/activities")
async def list_wind_down_activities() -> List[Dict[str, Any]]:
    """Return all available wind-down activities with metadata."""
    return [
        {
            "id": a.id,
            "name": a.name,
            "tagline": a.tagline,
            "duration_min": a.duration_min,
            "stress_threshold": a.stress_threshold,
            "route": a.route,
        }
        for a in WIND_DOWN_ACTIVITIES
    ]

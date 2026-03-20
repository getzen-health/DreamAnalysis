"""Smart daily notification engine — core logic.

Provides four classes:
  MorningReportGenerator  — generates a morning brain-report notification text
  EveningWindDownGenerator — picks a wind-down activity based on today's stress
  NotificationScheduler   — determines optimal send times from user wake/sleep patterns
  NotificationPreferences — serialisable per-user settings

No external dependencies beyond stdlib + the existing voice/health caches.
All heavy ML calls are isolated in the Generator classes so tests can patch
them with simple dicts.
"""

from __future__ import annotations

import datetime
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ── Wind-down activity catalogue ──────────────────────────────────────────────

@dataclass
class WindDownActivity:
    id: str
    name: str
    tagline: str
    duration_min: int
    stress_threshold: float   # recommend when stress >= this value (0-1)
    route: str                # frontend route to deep-link into

WIND_DOWN_ACTIVITIES: List[WindDownActivity] = [
    WindDownActivity(
        id="resonance_breathing",
        name="Resonance Breathing",
        tagline="5.5 breaths/min lowers HRV-LF power and cortisol",
        duration_min=5,
        stress_threshold=0.0,   # always appropriate
        route="/biofeedback",
    ),
    WindDownActivity(
        id="box_breathing",
        name="Box Breathing",
        tagline="4-4-4-4 pattern resets sympathetic overdrive",
        duration_min=4,
        stress_threshold=0.4,
        route="/biofeedback",
    ),
    WindDownActivity(
        id="cyclic_sigh",
        name="Cyclic Sighing",
        tagline="Double inhale + long exhale — fastest acute stress relief (Stanford 2023)",
        duration_min=3,
        stress_threshold=0.6,
        route="/biofeedback",
    ),
    WindDownActivity(
        id="478_breathing",
        name="4-7-8 Breathing",
        tagline="Inhale 4s, hold 7s, exhale 8s — pre-sleep optimised",
        duration_min=5,
        stress_threshold=0.3,
        route="/biofeedback",
    ),
    WindDownActivity(
        id="body_scan",
        name="Body Scan Meditation",
        tagline="Progressive muscle relaxation — 10-min parasympathetic shift",
        duration_min=10,
        stress_threshold=0.5,
        route="/neurofeedback",
    ),
    WindDownActivity(
        id="alpha_neurofeedback",
        name="Alpha Relaxation Protocol",
        tagline="Eyes-closed 10 Hz entrainment for pre-sleep alpha dominance",
        duration_min=8,
        stress_threshold=0.35,
        route="/neurofeedback",
    ),
]


# ── NotificationRecord ────────────────────────────────────────────────────────

@dataclass
class NotificationRecord:
    notification_id: str
    user_id: str
    notification_type: str         # "morning_report" | "evening_winddown" | "post_session" | "weekly_summary"
    title: str
    body: str
    route: str                     # deep-link route
    scheduled_at: float            # UNIX timestamp
    sent_at: Optional[float] = None
    opened_at: Optional[float] = None
    dismissed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "notification_id": self.notification_id,
            "user_id": self.user_id,
            "notification_type": self.notification_type,
            "title": self.title,
            "body": self.body,
            "route": self.route,
            "scheduled_at": self.scheduled_at,
            "sent_at": self.sent_at,
            "opened_at": self.opened_at,
            "dismissed_at": self.dismissed_at,
            "metadata": self.metadata,
        }


# ── NotificationPreferences ───────────────────────────────────────────────────

@dataclass
class NotificationPreferences:
    user_id: str
    enabled: bool = True
    morning_report_enabled: bool = True
    evening_winddown_enabled: bool = True
    post_session_enabled: bool = True
    weekly_summary_enabled: bool = True
    quiet_hours_start: int = 22    # 10 PM local — no notifications after this
    quiet_hours_end: int = 7       # 7 AM local — no notifications before this
    morning_hour: int = 8          # preferred morning send hour (local)
    evening_hour: int = 21         # preferred evening send hour (local)
    timezone_offset_hours: int = 0  # UTC offset in whole hours
    skip_weekends_morning: bool = False
    min_stress_for_evening: float = 0.3  # skip evening prompt if stress < this
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "enabled": self.enabled,
            "morning_report_enabled": self.morning_report_enabled,
            "evening_winddown_enabled": self.evening_winddown_enabled,
            "post_session_enabled": self.post_session_enabled,
            "weekly_summary_enabled": self.weekly_summary_enabled,
            "quiet_hours_start": self.quiet_hours_start,
            "quiet_hours_end": self.quiet_hours_end,
            "morning_hour": self.morning_hour,
            "evening_hour": self.evening_hour,
            "timezone_offset_hours": self.timezone_offset_hours,
            "skip_weekends_morning": self.skip_weekends_morning,
            "min_stress_for_evening": self.min_stress_for_evening,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NotificationPreferences":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})  # type: ignore[attr-defined]


# ── Preference store (in-memory, keyed by user_id) ────────────────────────────

_PREFERENCES: Dict[str, NotificationPreferences] = {}
_HISTORY: Dict[str, List[NotificationRecord]] = {}   # user_id → list of records


def _get_prefs(user_id: str) -> NotificationPreferences:
    return _PREFERENCES.get(user_id, NotificationPreferences(user_id=user_id))


def _save_prefs(prefs: NotificationPreferences) -> None:
    _PREFERENCES[prefs.user_id] = prefs


def _record_notification(record: NotificationRecord) -> None:
    _HISTORY.setdefault(record.user_id, []).append(record)
    # Cap history per user at 200 entries
    if len(_HISTORY[record.user_id]) > 200:
        _HISTORY[record.user_id] = _HISTORY[record.user_id][-200:]


# ── Score helpers ─────────────────────────────────────────────────────────────

def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _score_label(score: float) -> str:
    """Convert 0-100 score to a brief readable label."""
    if score >= 85:
        return "excellent"
    if score >= 70:
        return "good"
    if score >= 55:
        return "fair"
    if score >= 40:
        return "low"
    return "poor"


def _pull_yesterday_voice(user_id: str) -> Dict[str, Any]:
    """Pull the most recent voice cache entry for a user (best-effort)."""
    try:
        from api.routes.voice_watch import _VOICE_CACHE, _VOICE_CACHE_TTL  # type: ignore[import]
        entry = _VOICE_CACHE.get(user_id)
        if not entry:
            return {}
        if time.time() - entry.get("ts", 0) > _VOICE_CACHE_TTL:
            return {}
        r = entry.get("result", {})
        stress_raw = r.get("stress_from_watch")
        stress_index = (
            min(1.0, float(stress_raw) / 10.0) if stress_raw is not None
            else float(r.get("stress_index", 0.0))
        )
        return {
            "avg_valence": float(r.get("valence", 0.0)),
            "avg_stress": stress_index,
            "avg_arousal": float(r.get("arousal", 0.5)),
            "dominant_emotion": r.get("emotion", "neutral"),
        }
    except Exception as exc:
        log.debug("Voice cache unavailable: %s", exc)
        return {}


def _pull_health_summary(user_id: str) -> Dict[str, Any]:
    """Pull today's health daily summary (best-effort)."""
    try:
        from api.routes.health import _health_db  # type: ignore[attr-defined,import]
        result = _health_db.get_daily_summary(user_id, None)
        if isinstance(result, dict):
            return result
    except Exception as exc:
        log.debug("Health DB unavailable: %s", exc)
    return {}


# ── MorningReportGenerator ────────────────────────────────────────────────────

class MorningReportGenerator:
    """Generates a morning brain-report notification for a user.

    Pulls yesterday's voice check-in data and health data, computes three
    summary scores (brain/cognitive, sleep quality, readiness), then produces
    a short notification title + body and a suggested deep-link route.

    If any health metric deviates more than 1.5 SD from the 30-day baseline,
    the most extreme anomaly is appended to the notification body (capped at
    one anomaly per notification so it stays concise).
    """

    def generate(
        self,
        user_id: str,
        *,
        voice_data: Optional[Dict[str, Any]] = None,
        health_data: Optional[Dict[str, Any]] = None,
        baseline: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return dict with: title, body, route, scores, has_data, anomalies."""
        voice = voice_data if voice_data is not None else _pull_yesterday_voice(user_id)
        health = health_data if health_data is not None else _pull_health_summary(user_id)

        scores = self._compute_scores(voice, health)
        has_data = bool(voice or health)

        title, body = self._build_message(scores, voice, health, has_data)

        # ── Anomaly detection ──────────────────────────────────────────────
        anomalies: List[Any] = []
        anomaly_description: Optional[str] = None
        try:
            from notifications.anomaly_detector import AnomalyDetector  # type: ignore[import]
            if baseline and (voice or health):
                user_snapshot = self._build_user_snapshot(voice, health)
                detector = AnomalyDetector()
                anomalies = detector.detect_anomalies(user_snapshot, baseline)
                if anomalies:
                    # Only surface the single most extreme anomaly per notification
                    top = anomalies[0]
                    anomaly_description = top.description
                    body = f"{body}  |  {anomaly_description}"
        except Exception as exc:
            log.debug("Anomaly detection skipped: %s", exc)

        return {
            "title": title,
            "body": body,
            "route": "/brain-report",
            "scores": scores,
            "has_data": has_data,
            "data_sources": (
                (["voice"] if voice else []) + (["health"] if health else [])
            ),
            "anomalies": [
                {
                    "metric": a.metric,
                    "value": a.value,
                    "baseline_mean": a.baseline_mean,
                    "z_score": a.z_score,
                    "direction": a.direction,
                    "description": a.description,
                }
                for a in anomalies
            ],
        }

    @staticmethod
    def _build_user_snapshot(
        voice: Dict[str, Any],
        health: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Assemble a flat metric snapshot from voice + health dicts."""
        snapshot: Dict[str, Any] = {}

        # Sleep quality: normalise to 0-1
        sleep_raw = health.get("sleep_efficiency") or health.get("sleep_score")
        if sleep_raw is not None:
            sq = float(sleep_raw)
            snapshot["sleep_quality"] = sq / 100.0 if sq > 1 else sq

        # Voice valence
        if voice.get("avg_valence") is not None:
            snapshot["voice_valence"] = float(voice["avg_valence"])

        # HRV
        hrv = health.get("hrv_sdnn")
        if hrv is not None:
            snapshot["hrv_avg"] = float(hrv)

        # Dream recall rate (dreams per night — caller must pre-compute)
        dream_recall = health.get("dream_recall_rate")
        if dream_recall is not None:
            snapshot["dream_recall_rate"] = float(dream_recall)

        # Stress index
        if voice.get("avg_stress") is not None:
            snapshot["stress_index"] = float(voice["avg_stress"])

        return snapshot

    # ── Internal ──────────────────────────────────────────────────────────────

    def _compute_scores(
        self,
        voice: Dict[str, Any],
        health: Dict[str, Any],
    ) -> Dict[str, float]:
        """Compute focus, sleep, and readiness scores (0-100)."""

        # Sleep quality
        sleep_quality: Optional[float] = None
        if health.get("sleep_efficiency") is not None:
            sleep_quality = _clamp(float(health["sleep_efficiency"]))
        elif health.get("sleep_score") is not None:
            sleep_quality = _clamp(float(health["sleep_score"]))

        # HRV
        hrv_sdnn: Optional[float] = health.get("hrv_sdnn")
        hrv_norm: float = 0.5
        if hrv_sdnn and hrv_sdnn > 0:
            hrv_norm = _clamp((float(hrv_sdnn) - 20) / 50, 0.0, 1.0)

        # Focus forecast
        focus_components: List[float] = []
        focus_weights: List[float] = []
        if sleep_quality is not None:
            focus_components.append(sleep_quality / 100)
            focus_weights.append(0.40)
        focus_components.append(hrv_norm)
        focus_weights.append(0.30)
        if voice.get("avg_valence") is not None:
            focus_components.append((float(voice["avg_valence"]) + 1) / 2)
            focus_weights.append(0.30)

        total_w = sum(focus_weights) or 1.0
        focus = _clamp(
            sum(c * w for c, w in zip(focus_components, focus_weights)) / total_w * 100
        )

        # Readiness (HRV-weighted blend of sleep + stress inverse)
        stress_raw = float(voice.get("avg_stress", 0.5))
        readiness_components = [hrv_norm, 1.0 - stress_raw]
        readiness_weights = [0.6, 0.4]
        if sleep_quality is not None:
            readiness_components.append(sleep_quality / 100)
            readiness_weights.append(0.4)
        total_rw = sum(readiness_weights) or 1.0
        readiness = _clamp(
            sum(c * w for c, w in zip(readiness_components, readiness_weights)) / total_rw * 100
        )

        scores: Dict[str, float] = {
            "focus": round(focus, 1),
            "readiness": round(readiness, 1),
        }
        if sleep_quality is not None:
            scores["sleep"] = round(sleep_quality, 1)
        return scores

    def _build_message(
        self,
        scores: Dict[str, float],
        voice: Dict[str, Any],
        health: Dict[str, Any],
        has_data: bool,
    ) -> Tuple[str, str]:
        """Build (title, body) notification strings."""
        if not has_data:
            return (
                "Good morning — your brain report is ready",
                "Complete a voice check-in or sync health data for a personalised report.",
            )

        focus = scores.get("focus", 50.0)
        sleep = scores.get("sleep")
        readiness = scores.get("readiness", 50.0)

        # Title — lead with the most important score
        title = "Good morning! Your brain report is ready"

        # Body — pack key numbers into one line
        parts: List[str] = []
        parts.append(f"Focus forecast: {focus:.0f}/100 ({_score_label(focus)})")
        if sleep is not None:
            parts.append(f"Sleep: {sleep:.0f}/100")
        parts.append(f"Readiness: {readiness:.0f}/100")

        mood = voice.get("dominant_emotion")
        if mood and mood != "neutral":
            parts.append(f"Mood yesterday: {mood}")

        body = "  |  ".join(parts)
        return title, body


# ── EveningWindDownGenerator ──────────────────────────────────────────────────

class EveningWindDownGenerator:
    """Analyzes today's stress patterns and suggests a wind-down activity.

    If the user's stress is below NotificationPreferences.min_stress_for_evening,
    no notification is generated (returns has_data=True, skip=True).
    """

    def generate(
        self,
        user_id: str,
        *,
        voice_data: Optional[Dict[str, Any]] = None,
        health_data: Optional[Dict[str, Any]] = None,
        prefs: Optional[NotificationPreferences] = None,
    ) -> Dict[str, Any]:
        """Return dict with: title, body, route, activity, stress_index, skip."""
        voice = voice_data if voice_data is not None else _pull_yesterday_voice(user_id)
        health = health_data if health_data is not None else _pull_health_summary(user_id)
        _prefs = prefs or _get_prefs(user_id)

        stress_index = self._estimate_stress(voice, health)
        skip = stress_index < _prefs.min_stress_for_evening

        activity = self._pick_activity(stress_index)
        title, body = self._build_message(stress_index, activity, skip)

        return {
            "title": title,
            "body": body,
            "route": activity.route,
            "activity_id": activity.id,
            "activity_name": activity.name,
            "stress_index": round(stress_index, 3),
            "skip": skip,
            "skip_reason": (
                f"Stress {stress_index:.2f} below threshold {_prefs.min_stress_for_evening:.2f}"
                if skip else None
            ),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _estimate_stress(
        self,
        voice: Dict[str, Any],
        health: Dict[str, Any],
    ) -> float:
        """Compute a 0-1 stress estimate from available signals."""
        components: List[float] = []
        weights: List[float] = []

        # Voice stress index
        if voice.get("avg_stress") is not None:
            components.append(float(voice["avg_stress"]))
            weights.append(0.5)

        # HRV inverse (low HRV = high stress)
        hrv_sdnn: Optional[float] = health.get("hrv_sdnn")
        if hrv_sdnn and hrv_sdnn > 0:
            hrv_norm = _clamp((float(hrv_sdnn) - 20) / 50, 0.0, 1.0)
            components.append(1.0 - hrv_norm)
            weights.append(0.3)

        # Resting HR elevation (if available)
        resting_hr: Optional[float] = health.get("resting_heart_rate")
        if resting_hr and resting_hr > 0:
            # Normalise: 50 bpm = low (0), 90 bpm = high (1)
            hr_stress = _clamp((float(resting_hr) - 50) / 40, 0.0, 1.0)
            components.append(hr_stress)
            weights.append(0.2)

        if not components:
            return 0.4  # neutral default when no data available

        total_w = sum(weights)
        return _clamp(
            sum(c * w for c, w in zip(components, weights)) / total_w, 0.0, 1.0
        )

    def _pick_activity(self, stress_index: float) -> WindDownActivity:
        """Select the most appropriate wind-down activity for the given stress level."""
        # Filter to activities appropriate for this stress level, pick most specific
        candidates = [
            a for a in WIND_DOWN_ACTIVITIES
            if stress_index >= a.stress_threshold
        ]
        if not candidates:
            return WIND_DOWN_ACTIVITIES[0]   # resonance breathing — always safe

        # Among candidates, pick the one with the highest (most specific) threshold
        return max(candidates, key=lambda a: a.stress_threshold)

    def _build_message(
        self,
        stress_index: float,
        activity: WindDownActivity,
        skip: bool,
    ) -> Tuple[str, str]:
        if skip:
            return "", ""

        if stress_index >= 0.7:
            urgency = "Your stress was elevated today."
            cta = "A quick wind-down session can make a real difference."
        elif stress_index >= 0.45:
            urgency = "Moderate stress detected today."
            cta = "A short breathing session helps signal your brain to sleep mode."
        else:
            urgency = "Time to wind down."
            cta = "A gentle practice before bed improves sleep onset."

        title = "Time to wind down"
        body = (
            f"{urgency} Try {activity.duration_min}-min {activity.name} now. {cta}"
        )
        return title, body


# ── WeeklySummaryGenerator ────────────────────────────────────────────────────

class WeeklySummaryGenerator:
    """Generates a weekly recap notification summarising the past 7 days.

    Aggregates voice check-in history, health metrics, and session data
    into a concise notification with week-over-week trend arrows and a
    deep-link to the full weekly-brain-summary page.

    Oura and Whoop both send automated weekly summaries as a key re-engagement
    mechanism.  The notification schema already supports ``weekly_summary``
    as a type and ``weekly_summary_enabled`` exists in preferences, but no
    generator existed until now.
    """

    def generate(
        self,
        user_id: str,
        *,
        voice_history: Optional[List[Dict[str, Any]]] = None,
        health_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return dict with: title, body, route, stats, has_data."""
        voice_week = voice_history if voice_history is not None else self._pull_voice_week(user_id)
        health = health_data if health_data is not None else _pull_health_summary(user_id)

        stats = self._compute_weekly_stats(voice_week, health)
        has_data = stats["total_checkins"] > 0 or stats["has_health"]

        title, body = self._build_message(stats, has_data)

        return {
            "title": title,
            "body": body,
            "route": "/weekly-brain-summary",
            "stats": stats,
            "has_data": has_data,
        }

    @staticmethod
    def _pull_voice_week(user_id: str) -> List[Dict[str, Any]]:
        """Pull the last 7 days of voice check-in records (best-effort)."""
        try:
            from api.routes.voice_watch import _VOICE_HISTORY  # type: ignore[import]
            history = _VOICE_HISTORY.get(user_id, [])
            cutoff = time.time() - 7 * 86_400
            return [
                r for r in history
                if r.get("timestamp", 0) >= cutoff
            ]
        except Exception as exc:
            log.debug("Voice history unavailable for weekly summary: %s", exc)
            return []

    def _compute_weekly_stats(
        self,
        voice_records: List[Dict[str, Any]],
        health: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Aggregate week's voice + health data into summary stats."""
        stats: Dict[str, Any] = {
            "total_checkins": len(voice_records),
            "has_health": bool(health),
        }

        # Voice aggregations
        if voice_records:
            valences = [float(r["valence"]) for r in voice_records if r.get("valence") is not None]
            stresses = [float(r["stress_index"]) for r in voice_records if r.get("stress_index") is not None]
            focuses = [float(r["focus_index"]) for r in voice_records if r.get("focus_index") is not None]

            if valences:
                stats["avg_valence"] = round(sum(valences) / len(valences), 3)
            if stresses:
                stats["avg_stress"] = round(sum(stresses) / len(stresses), 3)
            if focuses:
                stats["avg_focus"] = round(sum(focuses) / len(focuses), 3)

            # Dominant emotion
            emotions = [r.get("emotion") for r in voice_records if r.get("emotion")]
            if emotions:
                counts: Dict[str, int] = {}
                for e in emotions:
                    counts[e] = counts.get(e, 0) + 1
                stats["dominant_emotion"] = max(counts, key=lambda k: counts[k])
                stats["emotion_variety"] = len(counts)

            # Streak: consecutive days with at least one check-in
            if voice_records:
                days_with_checkin = set()
                for r in voice_records:
                    ts = r.get("timestamp", 0)
                    if ts > 0:
                        day = datetime.datetime.fromtimestamp(ts).date()
                        days_with_checkin.add(day)
                stats["active_days"] = len(days_with_checkin)

        # Health aggregations
        sleep_score = health.get("sleep_efficiency") or health.get("sleep_score")
        if sleep_score is not None:
            stats["sleep_score"] = round(float(sleep_score), 1)

        hrv = health.get("hrv_sdnn")
        if hrv is not None and float(hrv) > 0:
            stats["hrv_avg"] = round(float(hrv), 1)

        return stats

    def _build_message(
        self,
        stats: Dict[str, Any],
        has_data: bool,
    ) -> Tuple[str, str]:
        """Build (title, body) notification strings."""
        if not has_data:
            return (
                "Your weekly brain summary is ready",
                "Start tracking with voice check-ins to get a personalised weekly recap.",
            )

        checkins = stats.get("total_checkins", 0)
        active_days = stats.get("active_days", 0)

        title = "Your weekly brain summary is ready"

        parts: List[str] = []

        if checkins > 0:
            parts.append(f"{checkins} check-in{'s' if checkins != 1 else ''} across {active_days} day{'s' if active_days != 1 else ''}")

        avg_stress = stats.get("avg_stress")
        if avg_stress is not None:
            stress_pct = round(avg_stress * 100)
            parts.append(f"Avg stress: {stress_pct}%")

        avg_focus = stats.get("avg_focus")
        if avg_focus is not None:
            focus_pct = round(avg_focus * 100)
            parts.append(f"Focus: {focus_pct}%")

        dominant = stats.get("dominant_emotion")
        if dominant and dominant != "neutral":
            parts.append(f"Top mood: {dominant}")

        sleep = stats.get("sleep_score")
        if sleep is not None:
            parts.append(f"Sleep: {sleep:.0f}/100")

        body = "  |  ".join(parts) if parts else "Tap to see your full weekly report."

        return title, body


# ── NotificationScheduler ────────────────────────────────────────────────────

class NotificationScheduler:
    """Determines optimal send times and whether a notification should fire.

    Uses user preferences (wake/sleep hour) to compute the next scheduled
    timestamp for morning and evening notifications. Also checks quiet hours
    and weekend-skip rules.
    """

    # History of recent send times to avoid duplicates (user_id → last send ts per type)
    _last_sent: Dict[str, Dict[str, float]] = {}

    # Minimum gap between the same notification type (seconds)
    COOLDOWN_MORNING: float = 20 * 3600   # 20 hours
    COOLDOWN_EVENING: float = 20 * 3600

    def should_send_morning(self, user_id: str, now: Optional[float] = None) -> bool:
        """True if a morning notification should be sent right now."""
        prefs = _get_prefs(user_id)
        if not prefs.enabled or not prefs.morning_report_enabled:
            return False
        ts = now or time.time()
        local_dt = self._local_dt(ts, prefs.timezone_offset_hours)
        if self._in_quiet_hours(local_dt.hour, prefs):
            return False
        if prefs.skip_weekends_morning and local_dt.weekday() >= 5:
            return False
        if local_dt.hour != prefs.morning_hour:
            return False
        return not self._on_cooldown(user_id, "morning_report", ts)

    def should_send_evening(self, user_id: str, now: Optional[float] = None) -> bool:
        """True if an evening wind-down notification should be sent right now."""
        prefs = _get_prefs(user_id)
        if not prefs.enabled or not prefs.evening_winddown_enabled:
            return False
        ts = now or time.time()
        local_dt = self._local_dt(ts, prefs.timezone_offset_hours)
        if self._in_quiet_hours(local_dt.hour, prefs):
            return False
        if local_dt.hour != prefs.evening_hour:
            return False
        return not self._on_cooldown(user_id, "evening_winddown", ts)

    def next_morning_ts(self, user_id: str, now: Optional[float] = None) -> float:
        """Return UNIX timestamp of the next scheduled morning notification."""
        prefs = _get_prefs(user_id)
        ts = now or time.time()
        local_dt = self._local_dt(ts, prefs.timezone_offset_hours)
        target = local_dt.replace(
            hour=prefs.morning_hour, minute=0, second=0, microsecond=0
        )
        if target <= local_dt:
            target += datetime.timedelta(days=1)
        # Skip weekend if requested
        if prefs.skip_weekends_morning:
            while target.weekday() >= 5:
                target += datetime.timedelta(days=1)
        # Convert back to UTC UNIX
        return target.timestamp() - prefs.timezone_offset_hours * 3600

    def next_evening_ts(self, user_id: str, now: Optional[float] = None) -> float:
        """Return UNIX timestamp of the next scheduled evening notification."""
        prefs = _get_prefs(user_id)
        ts = now or time.time()
        local_dt = self._local_dt(ts, prefs.timezone_offset_hours)
        target = local_dt.replace(
            hour=prefs.evening_hour, minute=0, second=0, microsecond=0
        )
        if target <= local_dt:
            target += datetime.timedelta(days=1)
        return target.timestamp() - prefs.timezone_offset_hours * 3600

    def mark_sent(self, user_id: str, notification_type: str, ts: Optional[float] = None) -> None:
        """Record that a notification of the given type was sent."""
        self._last_sent.setdefault(user_id, {})[notification_type] = ts or time.time()

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _local_dt(utc_ts: float, tz_offset_hours: int) -> datetime.datetime:
        """Convert UNIX timestamp to a naive local datetime using a fixed UTC offset."""
        utc_dt = datetime.datetime.fromtimestamp(utc_ts, tz=datetime.timezone.utc).replace(tzinfo=None)
        return utc_dt + datetime.timedelta(hours=tz_offset_hours)

    @staticmethod
    def _in_quiet_hours(hour: int, prefs: NotificationPreferences) -> bool:
        """True if the local hour falls in the user's quiet hours."""
        start, end = prefs.quiet_hours_start, prefs.quiet_hours_end
        if start > end:   # wraps midnight (e.g., 22 → 7)
            return hour >= start or hour < end
        return start <= hour < end

    def _on_cooldown(self, user_id: str, notification_type: str, now: float) -> bool:
        last = self._last_sent.get(user_id, {}).get(notification_type)
        if last is None:
            return False
        cooldown = (
            self.COOLDOWN_MORNING if notification_type == "morning_report"
            else self.COOLDOWN_EVENING
        )
        return (now - last) < cooldown


# ── Module-level singletons ───────────────────────────────────────────────────

_morning_generator = MorningReportGenerator()
_evening_generator = EveningWindDownGenerator()
_weekly_generator = WeeklySummaryGenerator()
_scheduler = NotificationScheduler()


def get_morning_generator() -> MorningReportGenerator:
    return _morning_generator


def get_evening_generator() -> EveningWindDownGenerator:
    return _evening_generator


def get_weekly_generator() -> WeeklySummaryGenerator:
    return _weekly_generator


def get_scheduler() -> NotificationScheduler:
    return _scheduler


def get_history(user_id: str) -> List[Dict[str, Any]]:
    return [r.to_dict() for r in _HISTORY.get(user_id, [])]


def get_preferences(user_id: str) -> NotificationPreferences:
    return _get_prefs(user_id)


def update_preferences(user_id: str, updates: Dict[str, Any]) -> NotificationPreferences:
    prefs = _get_prefs(user_id)
    for k, v in updates.items():
        if hasattr(prefs, k) and k not in ("user_id",):
            setattr(prefs, k, v)
    prefs.updated_at = time.time()
    _save_prefs(prefs)
    return prefs

"""Smart daily notification engine for Neural Dream Workshop.

Exports
-------
MorningReportGenerator  — pulls yesterday's session data, computes scores, generates summary text
EveningWindDownGenerator — analyzes today's stress patterns, suggests wind-down activities
NotificationScheduler   — determines optimal send times based on user patterns
NotificationPreferences — user settings (enabled/disabled, quiet hours, notification types)
"""

from .smart_notifications import (
    MorningReportGenerator,
    EveningWindDownGenerator,
    NotificationScheduler,
    NotificationPreferences,
    NotificationRecord,
    WindDownActivity,
    WIND_DOWN_ACTIVITIES,
)

__all__ = [
    "MorningReportGenerator",
    "EveningWindDownGenerator",
    "NotificationScheduler",
    "NotificationPreferences",
    "NotificationRecord",
    "WindDownActivity",
    "WIND_DOWN_ACTIVITIES",
]

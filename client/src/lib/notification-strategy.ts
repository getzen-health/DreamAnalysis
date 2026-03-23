import { sbGetSetting, sbSaveGeneric } from "./supabase-store";
/**
 * Evidence-based notification strategy for Neural Dream Workshop.
 *
 * Research basis:
 * - Sweet spot: 1-3 notifications per day, batched
 * - First 90 days: 1+ push notification -> 3x higher retention
 * - ML-driven send-time optimization beats fixed schedules
 * - Mental health apps = 4% retention at 15 days
 * - Excessive check-in prompts increase rumination and anxiety
 * - Default minimal (1x/day), let users opt up, never guilt-trip
 *
 * Language rules (MANDATORY):
 * - Never say "You missed..." or "You haven't..."
 * - Never use guilt language ("Don't break your streak!")
 * - Always frame as invitations: "When you're ready..." / "Your brain data is here for you"
 * - Keep messages short (under 60 chars for title, under 120 for body)
 */

// ── Types ─────────────────────────────────────────────────────────────────

export type NotificationFrequency = "minimal" | "balanced" | "engaged";

export interface NotificationPreferences {
  frequency: NotificationFrequency;
  quietHoursStart: number;  // 0-23, default 22 (10 PM)
  quietHoursEnd: number;    // 0-23, default 7 (7 AM)
  enabledTypes: {
    morningBrief: boolean;
    sessionReminder: boolean;
    weeklyInsight: boolean;
    streakEncouragement: boolean;
  };
}

export interface NotificationContent {
  title: string;
  body: string;
  type: "morning_brief" | "session_reminder" | "weekly_insight" | "streak";
}

// ── Constants ─────────────────────────────────────────────────────────────

const STORAGE_KEY = "ndw_notification_prefs";

// ── Frequency presets ─────────────────────────────────────────────────────

export function getPresetForFrequency(
  frequency: NotificationFrequency,
): NotificationPreferences["enabledTypes"] {
  switch (frequency) {
    case "minimal":
      return {
        morningBrief: true,
        sessionReminder: false,
        weeklyInsight: false,
        streakEncouragement: false,
      };
    case "balanced":
      return {
        morningBrief: true,
        sessionReminder: true,
        weeklyInsight: true,
        streakEncouragement: false,
      };
    case "engaged":
      return {
        morningBrief: true,
        sessionReminder: true,
        weeklyInsight: true,
        streakEncouragement: true,
      };
  }
}

// ── Defaults ──────────────────────────────────────────────────────────────

export function getDefaultPreferences(): NotificationPreferences {
  return {
    frequency: "minimal",
    quietHoursStart: 22,
    quietHoursEnd: 7,
    enabledTypes: getPresetForFrequency("minimal"),
  };
}

// ── Persistence ───────────────────────────────────────────────────────────

export function saveNotificationPreferences(prefs: NotificationPreferences): void {
  try {
    sbSaveGeneric(STORAGE_KEY, prefs);
  } catch {
    // localStorage full or unavailable
  }
}

export function getNotificationPreferences(): NotificationPreferences {
  try {
    const raw = sbGetSetting(STORAGE_KEY);
    if (!raw) return getDefaultPreferences();
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object" || !parsed.frequency) {
      return getDefaultPreferences();
    }
    return parsed as NotificationPreferences;
  } catch {
    return getDefaultPreferences();
  }
}

// ── Quiet hours check ─────────────────────────────────────────────────────

export function isInQuietHours(
  currentHour: number,
  start: number,
  end: number,
): boolean {
  if (start > end) {
    // Wraps midnight: e.g. 22-7 means 22,23,0,1,2,3,4,5,6
    return currentHour >= start || currentHour < end;
  }
  // Non-wrapping: e.g. 1-6 means 1,2,3,4,5
  return currentHour >= start && currentHour < end;
}

// ── Content generators ────────────────────────────────────────────────────

export function generateMorningBrief(data: {
  lastEmotion?: string;
  streakDays?: number;
  chronotype?: string;
}): NotificationContent {
  const { lastEmotion, streakDays, chronotype } = data;

  // Pick a contextual body based on available data.
  // HIPAA: Never include emotion labels, health metrics, or PHI in notification text.
  // Push notifications are visible on lock screens — use generic prompts instead.
  let body: string;

  if (lastEmotion && streakDays && streakDays > 0) {
    body = `Your ${streakDays}-day streak is going strong. Open the app to see yesterday's insights.`;
  } else if (streakDays && streakDays > 0) {
    body = `Your ${streakDays}-day check-in streak is going strong. How are you feeling today?`;
  } else if (lastEmotion) {
    body = "Your latest insights are ready. Start today with a voice check-in.";
  } else {
    body = "No pressure -- when you're ready, your brain data is here for you.";
  }

  // Truncate to stay under 120 chars
  if (body.length > 120) {
    body = body.slice(0, 117) + "...";
  }

  return {
    title: "Good morning",
    body,
    type: "morning_brief",
  };
}

export function generateSessionReminder(): NotificationContent {
  return {
    title: "Neurofeedback window",
    body: "Your brain is ready for a session. 20 minutes is all you need.",
    type: "session_reminder",
  };
}

export function generateWeeklyInsight(data: {
  voiceCheckins: number;
  neurofeedbackSessions: number;
  stressTrend: "improving" | "stable" | "worsening";
}): NotificationContent {
  const { voiceCheckins, neurofeedbackSessions, stressTrend } = data;

  // HIPAA: Do not include health trend details in push notifications.
  // Stress trend info is only shown inside the app.
  let trendNote = "";
  if (stressTrend === "improving") {
    trendNote = " Your wellness trend looks good.";
  } else if (stressTrend === "worsening") {
    trendNote = " Open the app for your full wellness summary.";
  }

  let body = `This week: ${voiceCheckins} voice check-ins, ${neurofeedbackSessions} neurofeedback sessions.${trendNote}`;

  if (body.length > 120) {
    body = `${voiceCheckins} check-ins, ${neurofeedbackSessions} sessions this week.${trendNote}`;
  }
  if (body.length > 120) {
    body = body.slice(0, 117) + "...";
  }

  return {
    title: "Your weekly summary",
    body,
    type: "weekly_insight",
  };
}

export function generateStreakEncouragement(streakDays: number): NotificationContent {
  let body: string;

  if (streakDays <= 0) {
    body = "When you're ready, a quick voice check-in takes under a minute.";
  } else if (streakDays < 7) {
    body = `${streakDays}-day streak and counting. Keep it going at your own pace.`;
  } else if (streakDays < 30) {
    body = `${streakDays} days in a row. Your consistency is building real insight.`;
  } else {
    body = `${streakDays}-day streak. Your commitment to self-awareness is remarkable.`;
  }

  if (body.length > 120) {
    body = body.slice(0, 117) + "...";
  }

  return {
    title: "Check-in streak",
    body,
    type: "streak",
  };
}

/**
 * Streak protector — contextual nudges to prevent streak loss and encourage consistency.
 * Shows on Today page when streak is at risk or to celebrate weekly progress.
 */
import { useState, useEffect } from "react";
import { Flame, TrendingUp, Clock } from "lucide-react";

interface StreakNudge {
  type: "protection" | "reflection" | "contextual";
  icon: typeof Flame;
  title: string;
  message: string;
  color: string;
  action?: { label: string; route: string };
}

function getStreakNudge(): StreakNudge | null {
  const streak = parseInt(localStorage.getItem("ndw_streak_count") || "0", 10);
  const lastDate = localStorage.getItem("ndw_streak_last_date");
  const today = new Date().toISOString().slice(0, 10);
  const hour = new Date().getHours();
  const checkedInToday = lastDate === today;

  // 1. Streak protection — not checked in today and it's afternoon+
  if (streak >= 2 && !checkedInToday && hour >= 14) {
    const hoursLeft = 24 - hour;
    return {
      type: "protection",
      icon: Flame,
      title: `${streak}-day streak at risk!`,
      message: hoursLeft <= 4
        ? `Only ${hoursLeft}h left — don't lose your ${streak}-day streak`
        : `Check in today to keep your ${streak}-day streak alive`,
      color: hoursLeft <= 4 ? "hsl(0 65% 55%)" : "hsl(38 85% 52%)",
      action: { label: "Check in now", route: "" },
    };
  }

  // 2. Weekly reflection — show on Sundays if checked in today
  const dayOfWeek = new Date().getDay();
  if (dayOfWeek === 0 && checkedInToday) {
    // Count how many days this week had check-ins
    const weekDays: number[] = [];
    for (let i = 0; i < 7; i++) {
      const d = new Date(Date.now() - i * 86400000).toISOString().slice(0, 10);
      const key = `voice-checkin-${d}-morning`;
      const key2 = `voice-checkin-${d}-noon`;
      const key3 = `voice-checkin-${d}-evening`;
      if (localStorage.getItem(key) || localStorage.getItem(key2) || localStorage.getItem(key3)) {
        weekDays.push(i);
      }
    }
    if (weekDays.length >= 3) {
      return {
        type: "reflection",
        icon: TrendingUp,
        title: "Weekly Reflection",
        message: `You checked in ${weekDays.length}/7 days this week${weekDays.length >= 5 ? " — amazing consistency!" : ". Keep building the habit!"}`,
        color: "hsl(165 60% 45%)",
      };
    }
  }

  // 3. Contextual nudge — if not checked in during usual time
  if (!checkedInToday && hour >= 10 && hour <= 12 && streak >= 3) {
    return {
      type: "contextual",
      icon: Clock,
      title: "Your usual check-in time",
      message: "Most of your check-ins happen around this time. How are you feeling?",
      color: "hsl(200 70% 55%)",
      action: { label: "Quick check-in", route: "" },
    };
  }

  // 4. Progress pride — milestone approaching
  const nextMilestone = [7, 14, 30, 60, 100].find(m => m > streak && m - streak <= 2);
  if (nextMilestone && checkedInToday) {
    return {
      type: "contextual",
      icon: Flame,
      title: `${nextMilestone - streak} day${nextMilestone - streak > 1 ? "s" : ""} to go!`,
      message: `You're ${nextMilestone - streak} day${nextMilestone - streak > 1 ? "s" : ""} away from a ${nextMilestone}-day streak`,
      color: "hsl(270 40% 60%)",
    };
  }

  return null;
}

export function StreakProtector() {
  const [nudge, setNudge] = useState<StreakNudge | null>(null);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    setNudge(getStreakNudge());
    const handler = () => setNudge(getStreakNudge());
    window.addEventListener("ndw-voice-updated", handler);
    window.addEventListener("ndw-emotion-update", handler);
    return () => {
      window.removeEventListener("ndw-voice-updated", handler);
      window.removeEventListener("ndw-emotion-update", handler);
    };
  }, []);

  if (!nudge || dismissed) return null;

  const Icon = nudge.icon;
  return (
    <div
      style={{
        background: "var(--card)",
        border: `1px solid ${nudge.color}40`,
        borderRadius: 14,
        padding: "12px 14px",
        marginBottom: 12,
        display: "flex",
        alignItems: "center",
        gap: 10,
      }}
    >
      <div style={{
        width: 36, height: 36, borderRadius: 10,
        background: `${nudge.color}15`,
        display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
      }}>
        <Icon style={{ width: 18, height: 18, color: nudge.color }} />
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 12, fontWeight: 600, color: "var(--foreground)" }}>{nudge.title}</div>
        <div style={{ fontSize: 11, color: "var(--muted-foreground)", marginTop: 1 }}>{nudge.message}</div>
      </div>
      {nudge.action && (
        <button
          onClick={() => {
            window.dispatchEvent(new Event("ndw-open-voice-checkin"));
          }}
          style={{
            fontSize: 11, fontWeight: 600, color: nudge.color,
            background: `${nudge.color}12`, border: `1px solid ${nudge.color}30`,
            borderRadius: 8, padding: "6px 10px", whiteSpace: "nowrap" as const,
            cursor: "pointer",
          }}
        >
          {nudge.action.label}
        </button>
      )}
      <button
        onClick={() => setDismissed(true)}
        aria-label="Dismiss"
        style={{
          fontSize: 14, color: "var(--muted-foreground)", background: "none", border: "none",
          cursor: "pointer", padding: "4px", lineHeight: 1,
        }}
      >
        ×
      </button>
    </div>
  );
}

/**
 * Achievements — simple badge/milestone system for the You page.
 *
 * Badges are computed from localStorage + API data, not stored server-side.
 * This keeps it lightweight and privacy-first.
 */

import { useState, useEffect } from "react";

export interface Badge {
  id: string;
  emoji: string;
  name: string;
  description: string;
  earned: boolean;
}

function checkBadges(): Badge[] {
  const badges: Badge[] = [];

  // First Analysis
  const hasCheckin = !!localStorage.getItem("ndw_last_emotion");
  badges.push({
    id: "first-checkin",
    emoji: "🎙️",
    name: "First Voice",
    description: "Completed your first voice analysis",
    earned: hasCheckin,
  });

  // Streak badges
  const streakStr = localStorage.getItem("ndw_streak_count");
  const streak = streakStr ? parseInt(streakStr, 10) : 0;
  badges.push({
    id: "streak-3",
    emoji: "🔥",
    name: "3-Day Streak",
    description: "Checked in 3 days in a row",
    earned: streak >= 3,
  });
  badges.push({
    id: "streak-7",
    emoji: "⭐",
    name: "Weekly Warrior",
    description: "7-day analysis streak",
    earned: streak >= 7,
  });
  badges.push({
    id: "streak-30",
    emoji: "🏆",
    name: "Monthly Champion",
    description: "30-day analysis streak",
    earned: streak >= 30,
  });

  // Early Bird / Night Owl
  try {
    const raw = localStorage.getItem("ndw_last_emotion");
    if (raw) {
      const ts = JSON.parse(raw)?.timestamp;
      if (ts) {
        const hour = new Date(ts).getHours();
        badges.push({
          id: "early-bird",
          emoji: "🌅",
          name: "Early Bird",
          description: "Checked in before 8 AM",
          earned: hour < 8,
        });
        badges.push({
          id: "night-owl",
          emoji: "🦉",
          name: "Night Owl",
          description: "Checked in after 10 PM",
          earned: hour >= 22,
        });
      }
    }
  } catch { /* ignore */ }

  // Health Connected
  const healthConnected = localStorage.getItem("ndw_health_connect_granted") === "true"
    || localStorage.getItem("ndw_apple_health_granted") === "true";
  badges.push({
    id: "health-sync",
    emoji: "❤️",
    name: "Health Synced",
    description: "Connected a health data source",
    earned: healthConnected,
  });

  // Meal Logged
  const mealLogged = !!localStorage.getItem("ndw_meal_logged");
  badges.push({
    id: "first-meal",
    emoji: "🍽️",
    name: "First Meal",
    description: "Logged your first meal",
    earned: mealLogged,
  });

  // Onboarding Complete
  const onboarded = localStorage.getItem("ndw_onboarding_complete") === "true";
  badges.push({
    id: "onboarded",
    emoji: "🎓",
    name: "All Set Up",
    description: "Completed app onboarding",
    earned: onboarded,
  });

  return badges;
}

export function AchievementBadges() {
  const [badges, setBadges] = useState<Badge[]>([]);

  useEffect(() => {
    setBadges(checkBadges());
    // Re-check when voice data updates
    function handler() { setBadges(checkBadges()); }
    window.addEventListener("ndw-voice-updated", handler);
    window.addEventListener("ndw-emotion-update", handler);
    return () => {
      window.removeEventListener("ndw-voice-updated", handler);
      window.removeEventListener("ndw-emotion-update", handler);
    };
  }, []);

  const earned = badges.filter(b => b.earned);
  const locked = badges.filter(b => !b.earned);

  if (badges.length === 0) return null;

  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{
        fontSize: 11, fontWeight: 600, color: "var(--muted-foreground)",
        textTransform: "uppercase" as const, letterSpacing: "0.5px", marginBottom: 8,
      }}>
        Achievements ({earned.length}/{badges.length})
      </div>

      {/* Earned badges */}
      <div style={{
        display: "flex", flexWrap: "wrap" as const, gap: 8, marginBottom: earned.length > 0 && locked.length > 0 ? 8 : 0,
      }}>
        {earned.map(b => (
          <div key={b.id} style={{
            background: "var(--card)", border: "1px solid var(--border)",
            borderRadius: 12, padding: "8px 12px",
            display: "flex", alignItems: "center", gap: 6,
          }}>
            <span style={{ fontSize: 18 }}>{b.emoji}</span>
            <div>
              <div style={{ fontSize: 11, fontWeight: 600, color: "var(--foreground)" }}>{b.name}</div>
              <div style={{ fontSize: 9, color: "var(--muted-foreground)" }}>{b.description}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Locked badges — dimmed */}
      <div style={{ display: "flex", flexWrap: "wrap" as const, gap: 8 }}>
        {locked.map(b => (
          <div key={b.id} style={{
            background: "var(--card)", border: "1px solid var(--border)",
            borderRadius: 12, padding: "8px 12px", opacity: 0.4,
            display: "flex", alignItems: "center", gap: 6,
          }}>
            <span style={{ fontSize: 18 }}>🔒</span>
            <div>
              <div style={{ fontSize: 11, fontWeight: 600, color: "var(--foreground)" }}>{b.name}</div>
              <div style={{ fontSize: 9, color: "var(--muted-foreground)" }}>{b.description}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

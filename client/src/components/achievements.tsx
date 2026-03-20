/**
 * Achievements — simple badge/milestone system for the You page.
 *
 * Badges are computed from localStorage + API data, not stored server-side.
 * This keeps it lightweight and privacy-first.
 */

import { useState, useEffect } from "react";
import {
  Mic, Flame, Star, Sparkles, Trophy, Sunrise, Moon, Heart,
  UtensilsCrossed, GraduationCap, Palette, Brain, Lock,
  type LucideIcon,
} from "lucide-react";

export interface Badge {
  id: string;
  icon: LucideIcon;
  iconColor: string;
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
    icon: Mic,
    iconColor: "#0891b2",
    name: "First Voice",
    description: "Completed your first voice analysis",
    earned: hasCheckin,
  });

  // Streak badges
  const streakStr = localStorage.getItem("ndw_streak_count");
  const streak = streakStr ? parseInt(streakStr, 10) : 0;
  badges.push({
    id: "streak-3",
    icon: Flame,
    iconColor: "#ea580c",
    name: "3-Day Streak",
    description: "Checked in 3 days in a row",
    earned: streak >= 3,
  });
  badges.push({
    id: "streak-7",
    icon: Star,
    iconColor: "#d4a017",
    name: "Weekly Warrior",
    description: "7-day analysis streak",
    earned: streak >= 7,
  });
  badges.push({
    id: "streak-14",
    icon: Sparkles,
    iconColor: "#a78bfa",
    name: "Two-Week Titan",
    description: "14-day analysis streak",
    earned: streak >= 14,
  });
  badges.push({
    id: "streak-30",
    icon: Trophy,
    iconColor: "#d4a017",
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
          icon: Sunrise,
          iconColor: "#d4a017",
          name: "Early Bird",
          description: "Checked in before 8 AM",
          earned: hour < 8,
        });
        badges.push({
          id: "night-owl",
          icon: Moon,
          iconColor: "#7c3aed",
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
    icon: Heart,
    iconColor: "#e879a8",
    name: "Health Synced",
    description: "Connected a health data source",
    earned: healthConnected,
  });

  // Meal Logged
  const mealLogged = !!localStorage.getItem("ndw_meal_logged");
  badges.push({
    id: "first-meal",
    icon: UtensilsCrossed,
    iconColor: "#ea580c",
    name: "First Meal",
    description: "Logged your first meal",
    earned: mealLogged,
  });

  // Onboarding Complete
  const onboarded = localStorage.getItem("ndw_onboarding_complete") === "true";
  badges.push({
    id: "onboarded",
    icon: GraduationCap,
    iconColor: "#0891b2",
    name: "All Set Up",
    description: "Completed app onboarding",
    earned: onboarded,
  });

  // Emotion Explorer — logged all 6 emotions
  const ALL_EMOTIONS = ["happy", "sad", "angry", "fear", "surprise", "neutral"];
  try {
    const seen = JSON.parse(localStorage.getItem("ndw_emotions_seen") || "[]") as string[];
    const allSeen = ALL_EMOTIONS.every(e => seen.includes(e));
    badges.push({
      id: "emotion-explorer",
      icon: Palette,
      iconColor: "#4ade80",
      name: "Emotion Explorer",
      description: "Logged all 6 emotions",
      earned: allSeen,
    });
  } catch {
    badges.push({
      id: "emotion-explorer",
      icon: Palette,
      iconColor: "#4ade80",
      name: "Emotion Explorer",
      description: "Logged all 6 emotions",
      earned: false,
    });
  }

  // Muse Connected — connected an EEG device
  const museConnected = localStorage.getItem("ndw_muse_connected") === "true";
  badges.push({
    id: "muse-connected",
    icon: Brain,
    iconColor: "#6366f1",
    name: "Brain Reader",
    description: "Connected an EEG headband",
    earned: museConnected,
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
        {earned.map(b => {
          const IconComp = b.icon;
          return (
          <div key={b.id} style={{
            background: "var(--card)", border: "1px solid var(--border)",
            borderRadius: 12, padding: "8px 12px",
            display: "flex", alignItems: "center", gap: 6,
          }}>
            <IconComp style={{ width: 18, height: 18, color: b.iconColor, flexShrink: 0 }} />
            <div>
              <div style={{ fontSize: 11, fontWeight: 600, color: "var(--foreground)" }}>{b.name}</div>
              <div style={{ fontSize: 9, color: "var(--muted-foreground)" }}>{b.description}</div>
            </div>
          </div>
          );
        })}
      </div>

      {/* Locked badges — dimmed */}
      <div style={{ display: "flex", flexWrap: "wrap" as const, gap: 8 }}>
        {locked.map(b => (
          <div key={b.id} style={{
            background: "var(--card)", border: "1px solid var(--border)",
            borderRadius: 12, padding: "8px 12px", opacity: 0.4,
            display: "flex", alignItems: "center", gap: 6,
          }}>
            <Lock style={{ width: 18, height: 18, color: "var(--muted-foreground)", flexShrink: 0 }} />
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

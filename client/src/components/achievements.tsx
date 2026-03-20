/**
 * Achievements — premium badge/milestone system for the You page.
 *
 * Badges are computed from localStorage + API data, not stored server-side.
 * This keeps it lightweight and privacy-first.
 *
 * Visual tiers: bronze (easy), silver (medium), gold (hard).
 * Categories: Sessions, Streaks, Milestones, Wellness, Brain.
 */

import { useState, useEffect } from "react";
import {
  Mic, Flame, Star, Sparkles, Trophy, Sunrise, Moon, Heart,
  UtensilsCrossed, GraduationCap, Palette, Brain, Lock, Zap,
  type LucideIcon,
} from "lucide-react";

// ── Visual tier system ──────────────────────────────────────────────────

export type AchievementTier = "bronze" | "silver" | "gold";
export type AchievementCategory = "Sessions" | "Streaks" | "Milestones" | "Wellness" | "Brain";

const TIER_GRADIENTS: Record<AchievementTier, string> = {
  bronze: "linear-gradient(135deg, #92400e 0%, #b45309 50%, #d97706 100%)",
  silver: "linear-gradient(135deg, #6b7280 0%, #9ca3af 50%, #d1d5db 100%)",
  gold:   "linear-gradient(135deg, #92400e 0%, #d4a017 40%, #fde68a 70%, #d4a017 100%)",
};

const TIER_BORDER: Record<AchievementTier, string> = {
  bronze: "rgba(217, 119, 6, 0.3)",
  silver: "rgba(156, 163, 175, 0.3)",
  gold:   "rgba(212, 160, 23, 0.4)",
};

const TIER_GLOW: Record<AchievementTier, string> = {
  bronze: "0 0 12px rgba(217, 119, 6, 0.15)",
  silver: "0 0 14px rgba(156, 163, 175, 0.15)",
  gold:   "0 0 18px rgba(212, 160, 23, 0.2)",
};

// ── Shimmer keyframes injected once ─────────────────────────────────────

let shimmerInjected = false;
function injectShimmer() {
  if (shimmerInjected || typeof document === "undefined") return;
  shimmerInjected = true;
  const style = document.createElement("style");
  style.textContent = `
    @keyframes ndw-achievement-shimmer {
      0%   { background-position: -200% 0; }
      100% { background-position: 200% 0; }
    }
  `;
  document.head.appendChild(style);
}

// ── Badge interface ─────────────────────────────────────────────────────

export interface Badge {
  id: string;
  icon: LucideIcon;
  iconColor: string;
  name: string;
  description: string;
  earned: boolean;
  tier: AchievementTier;
  category: AchievementCategory;
  /** Progress toward earning (0-1). Shown as bar for locked badges. */
  progress?: number;
}

// ── Badge definitions ───────────────────────────────────────────────────

export function checkBadges(): Badge[] {
  const badges: Badge[] = [];

  // ── Sessions category ──

  const hasCheckin = !!localStorage.getItem("ndw_last_emotion");
  badges.push({
    id: "first-checkin",
    icon: Mic,
    iconColor: "#0891b2",
    name: "First Voice",
    description: "Completed your first voice analysis",
    earned: hasCheckin,
    tier: "bronze",
    category: "Sessions",
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
          tier: "silver",
          category: "Sessions",
        });
        badges.push({
          id: "night-owl",
          icon: Moon,
          iconColor: "#7c3aed",
          name: "Night Owl",
          description: "Checked in after 10 PM",
          earned: hour >= 22,
          tier: "silver",
          category: "Sessions",
        });
      }
    }
  } catch { /* ignore */ }

  // ── Streaks category ──

  const streakStr = localStorage.getItem("ndw_streak_count");
  const streak = streakStr ? parseInt(streakStr, 10) : 0;
  badges.push({
    id: "streak-3",
    icon: Flame,
    iconColor: "#ea580c",
    name: "3-Day Streak",
    description: "Checked in 3 days in a row",
    earned: streak >= 3,
    tier: "bronze",
    category: "Streaks",
    progress: Math.min(1, streak / 3),
  });
  badges.push({
    id: "streak-7",
    icon: Star,
    iconColor: "#d4a017",
    name: "Weekly Warrior",
    description: "7-day analysis streak",
    earned: streak >= 7,
    tier: "silver",
    category: "Streaks",
    progress: Math.min(1, streak / 7),
  });
  badges.push({
    id: "streak-14",
    icon: Sparkles,
    iconColor: "#a78bfa",
    name: "Two-Week Titan",
    description: "14-day analysis streak",
    earned: streak >= 14,
    tier: "silver",
    category: "Streaks",
    progress: Math.min(1, streak / 14),
  });
  badges.push({
    id: "streak-30",
    icon: Trophy,
    iconColor: "#d4a017",
    name: "Monthly Champion",
    description: "30-day analysis streak",
    earned: streak >= 30,
    tier: "gold",
    category: "Streaks",
    progress: Math.min(1, streak / 30),
  });

  // ── Milestones category ──

  const onboarded = localStorage.getItem("ndw_onboarding_complete") === "true";
  badges.push({
    id: "onboarded",
    icon: GraduationCap,
    iconColor: "#0891b2",
    name: "All Set Up",
    description: "Completed app onboarding",
    earned: onboarded,
    tier: "bronze",
    category: "Milestones",
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
      tier: "gold",
      category: "Milestones",
      progress: Math.min(1, seen.filter(e => ALL_EMOTIONS.includes(e)).length / ALL_EMOTIONS.length),
    });
  } catch {
    badges.push({
      id: "emotion-explorer",
      icon: Palette,
      iconColor: "#4ade80",
      name: "Emotion Explorer",
      description: "Logged all 6 emotions",
      earned: false,
      tier: "gold",
      category: "Milestones",
      progress: 0,
    });
  }

  // ── Wellness category ──

  const healthConnected = localStorage.getItem("ndw_health_connect_granted") === "true"
    || localStorage.getItem("ndw_apple_health_granted") === "true";
  badges.push({
    id: "health-sync",
    icon: Heart,
    iconColor: "#e879a8",
    name: "Health Synced",
    description: "Connected a health data source",
    earned: healthConnected,
    tier: "silver",
    category: "Wellness",
  });

  const mealLogged = !!localStorage.getItem("ndw_meal_logged");
  badges.push({
    id: "first-meal",
    icon: UtensilsCrossed,
    iconColor: "#ea580c",
    name: "First Meal",
    description: "Logged your first meal",
    earned: mealLogged,
    tier: "bronze",
    category: "Wellness",
  });

  // ── Brain category ──

  const museConnected = localStorage.getItem("ndw_muse_connected") === "true";
  badges.push({
    id: "muse-connected",
    icon: Brain,
    iconColor: "#6366f1",
    name: "Brain Reader",
    description: "Connected an EEG headband",
    earned: museConnected,
    tier: "gold",
    category: "Brain",
  });

  return badges;
}

// ── Category label chip ─────────────────────────────────────────────────

const CATEGORY_COLORS: Record<AchievementCategory, string> = {
  Sessions: "#0891b2",
  Streaks: "#ea580c",
  Milestones: "#d4a017",
  Wellness: "#e879a8",
  Brain: "#6366f1",
};

function CategoryChip({ category }: { category: AchievementCategory }) {
  const color = CATEGORY_COLORS[category];
  return (
    <span
      data-testid={`category-${category.toLowerCase()}`}
      style={{
        fontSize: 8,
        fontWeight: 700,
        color,
        textTransform: "uppercase" as const,
        letterSpacing: "0.8px",
        background: `${color}15`,
        padding: "2px 6px",
        borderRadius: 4,
      }}
    >
      {category}
    </span>
  );
}

// ── Tier badge label ────────────────────────────────────────────────────

function TierLabel({ tier }: { tier: AchievementTier }) {
  const label = tier.charAt(0).toUpperCase() + tier.slice(1);
  return (
    <span
      data-testid={`tier-${tier}`}
      style={{
        fontSize: 8,
        fontWeight: 700,
        letterSpacing: "0.5px",
        background: TIER_GRADIENTS[tier],
        WebkitBackgroundClip: "text",
        WebkitTextFillColor: "transparent",
        backgroundClip: "text",
      }}
    >
      {label}
    </span>
  );
}

// ── Progress bar ────────────────────────────────────────────────────────

function ProgressBar({ progress, tier }: { progress: number; tier: AchievementTier }) {
  const percent = Math.round(progress * 100);
  return (
    <div style={{ width: "100%", marginTop: 6 }}>
      <div
        style={{
          height: 4,
          borderRadius: 2,
          background: "rgba(255,255,255,0.06)",
          overflow: "hidden",
        }}
      >
        <div
          data-testid="progress-bar-fill"
          style={{
            height: "100%",
            width: `${percent}%`,
            background: TIER_GRADIENTS[tier],
            borderRadius: 2,
            transition: "width 0.6s cubic-bezier(0.22, 1, 0.36, 1)",
          }}
        />
      </div>
      <div style={{ fontSize: 8, color: "var(--muted-foreground)", marginTop: 3, textAlign: "right" as const }}>
        {percent}%
      </div>
    </div>
  );
}

// ── Earned badge card — with shimmer glow ───────────────────────────────

function EarnedBadgeCard({ badge }: { badge: Badge }) {
  injectShimmer();
  const IconComp = badge.icon;
  return (
    <div
      data-testid={`badge-${badge.id}`}
      style={{
        position: "relative" as const,
        background: "var(--card)",
        border: `1px solid ${TIER_BORDER[badge.tier]}`,
        borderRadius: 16,
        padding: "12px 14px",
        display: "flex",
        alignItems: "center",
        gap: 10,
        boxShadow: TIER_GLOW[badge.tier],
        overflow: "hidden" as const,
        transition: "transform 0.2s ease, box-shadow 0.2s ease",
      }}
    >
      {/* Shimmer overlay */}
      <div
        aria-hidden="true"
        style={{
          position: "absolute" as const,
          inset: 0,
          background: `linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.04) 40%, rgba(255,255,255,0.08) 50%, rgba(255,255,255,0.04) 60%, transparent 100%)`,
          backgroundSize: "200% 100%",
          animation: "ndw-achievement-shimmer 3s ease-in-out infinite",
          pointerEvents: "none" as const,
        }}
      />
      {/* Icon with gradient background circle */}
      <div
        style={{
          width: 36,
          height: 36,
          borderRadius: "50%",
          background: `${badge.iconColor}18`,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexShrink: 0,
        }}
      >
        <IconComp style={{ width: 18, height: 18, color: badge.iconColor }} />
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 2 }}>
          <span style={{ fontSize: 12, fontWeight: 600, color: "var(--foreground)" }}>{badge.name}</span>
          <TierLabel tier={badge.tier} />
        </div>
        <div style={{ fontSize: 9, color: "var(--muted-foreground)", lineHeight: 1.3 }}>{badge.description}</div>
        <div style={{ marginTop: 4 }}>
          <CategoryChip category={badge.category} />
        </div>
      </div>
      {/* Earned indicator */}
      <Zap style={{ width: 14, height: 14, color: badge.iconColor, flexShrink: 0, opacity: 0.7 }} />
    </div>
  );
}

// ── Locked badge card — greyed, with progress ───────────────────────────

function LockedBadgeCard({ badge }: { badge: Badge }) {
  return (
    <div
      data-testid={`badge-${badge.id}`}
      style={{
        background: "var(--card)",
        border: "1px solid var(--border)",
        borderRadius: 16,
        padding: "12px 14px",
        display: "flex",
        alignItems: "center",
        gap: 10,
        opacity: 0.55,
        filter: "saturate(0.3)",
      }}
    >
      <div
        style={{
          width: 36,
          height: 36,
          borderRadius: "50%",
          background: "rgba(255,255,255,0.04)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexShrink: 0,
        }}
      >
        <Lock style={{ width: 16, height: 16, color: "var(--muted-foreground)" }} />
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 2 }}>
          <span style={{ fontSize: 12, fontWeight: 600, color: "var(--foreground)" }}>{badge.name}</span>
          <TierLabel tier={badge.tier} />
        </div>
        <div style={{ fontSize: 9, color: "var(--muted-foreground)", lineHeight: 1.3 }}>{badge.description}</div>
        <div style={{ marginTop: 4 }}>
          <CategoryChip category={badge.category} />
        </div>
        {badge.progress != null && badge.progress > 0 && badge.progress < 1 && (
          <ProgressBar progress={badge.progress} tier={badge.tier} />
        )}
      </div>
    </div>
  );
}

// ── Main exported component ─────────────────────────────────────────────

export function AchievementBadges() {
  const [badges, setBadges] = useState<Badge[]>([]);

  useEffect(() => {
    setBadges(checkBadges());
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
      {/* Header */}
      <div style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "baseline",
        marginBottom: 10,
      }}>
        <div style={{
          fontSize: 11, fontWeight: 600, color: "var(--muted-foreground)",
          textTransform: "uppercase" as const, letterSpacing: "0.5px",
        }}>
          Achievements
        </div>
        <div style={{ fontSize: 11, color: "var(--muted-foreground)" }}>
          {earned.length}/{badges.length} earned
        </div>
      </div>

      {/* Overall progress bar */}
      <div style={{
        height: 4,
        borderRadius: 2,
        background: "rgba(255,255,255,0.06)",
        overflow: "hidden",
        marginBottom: 12,
      }}>
        <div
          data-testid="overall-progress"
          style={{
            height: "100%",
            width: badges.length > 0 ? `${Math.round((earned.length / badges.length) * 100)}%` : "0%",
            background: "linear-gradient(90deg, #7c3aed, #e879a8)",
            borderRadius: 2,
            transition: "width 0.8s cubic-bezier(0.22, 1, 0.36, 1)",
          }}
        />
      </div>

      {/* Earned badges */}
      {earned.length > 0 && (
        <div style={{
          display: "flex", flexDirection: "column" as const, gap: 8,
          marginBottom: locked.length > 0 ? 12 : 0,
        }}>
          {earned.map(b => <EarnedBadgeCard key={b.id} badge={b} />)}
        </div>
      )}

      {/* Locked badges */}
      {locked.length > 0 && (
        <>
          <div style={{
            fontSize: 10, fontWeight: 500, color: "var(--muted-foreground)",
            marginBottom: 8, opacity: 0.7,
          }}>
            Locked
          </div>
          <div style={{
            display: "flex", flexDirection: "column" as const, gap: 8,
          }}>
            {locked.map(b => <LockedBadgeCard key={b.id} badge={b} />)}
          </div>
        </>
      )}
    </div>
  );
}

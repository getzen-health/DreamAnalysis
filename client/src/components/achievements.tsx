/**
 * Achievements — premium badge/milestone gallery.
 *
 * Badges are computed from localStorage + API data, not stored server-side.
 * This keeps it lightweight and privacy-first.
 *
 * Visual tiers: bronze (easy), silver (medium), gold (hard).
 * Categories: Sessions, Streaks, Milestones, Wellness, Brain.
 *
 * Premium gallery design with:
 * - Hero section (total earned, completion %, motivational tagline)
 * - Gradient cards per tier with shimmer/glow animations
 * - Locked section with lock overlay and progress bars
 * - Horizontal scrollable category filter pills
 * - Framer-motion stagger animations
 */

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { cardVariants } from "@/lib/animations";
import {
  Mic, Flame, Star, Sparkles, Trophy, Sunrise, Moon, Heart,
  UtensilsCrossed, GraduationCap, Palette, Brain, Lock, Zap,
  type LucideIcon,
} from "lucide-react";
import { sbGetGeneric, sbGetSetting } from "../lib/supabase-store";

// ── Visual tier system ──────────────────────────────────────────────────

export type AchievementTier = "bronze" | "silver" | "gold";
export type AchievementCategory = "All" | "Sessions" | "Streaks" | "Milestones" | "Wellness" | "Brain";

const BADGE_CATEGORIES: AchievementCategory[] = ["All", "Sessions", "Streaks", "Milestones", "Wellness", "Brain"];

const TIER_GRADIENTS: Record<AchievementTier, string> = {
  bronze: "linear-gradient(135deg, #92400e 0%, #b45309 50%, #d97706 100%)",
  silver: "linear-gradient(135deg, #6b7280 0%, #9ca3af 50%, #d1d5db 100%)",
  gold:   "linear-gradient(135deg, #92400e 0%, #d4a017 40%, #fde68a 70%, #d4a017 100%)",
};

const TIER_CARD_BG: Record<AchievementTier, string> = {
  bronze: "linear-gradient(135deg, rgba(146,64,14,0.15) 0%, rgba(217,119,6,0.08) 100%)",
  silver: "linear-gradient(135deg, rgba(107,114,128,0.15) 0%, rgba(209,213,219,0.08) 100%)",
  gold:   "linear-gradient(135deg, rgba(212,160,23,0.18) 0%, rgba(253,230,138,0.06) 100%)",
};

const TIER_BORDER: Record<AchievementTier, string> = {
  bronze: "rgba(217, 119, 6, 0.35)",
  silver: "rgba(156, 163, 175, 0.35)",
  gold:   "rgba(212, 160, 23, 0.45)",
};

const TIER_GLOW: Record<AchievementTier, string> = {
  bronze: "0 4px 20px rgba(217, 119, 6, 0.15), 0 0 12px rgba(217, 119, 6, 0.08)",
  silver: "0 4px 20px rgba(156, 163, 175, 0.15), 0 0 14px rgba(156, 163, 175, 0.08)",
  gold:   "0 4px 24px rgba(212, 160, 23, 0.2), 0 0 18px rgba(212, 160, 23, 0.12)",
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
    @keyframes ndw-achievement-glow {
      0%, 100% { opacity: 0.6; }
      50%      { opacity: 1; }
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
  category: Exclude<AchievementCategory, "All">;
  /** Progress toward earning (0-1). Shown as bar for locked badges. */
  progress?: number;
  /** ISO date when earned. */
  earnedDate?: string;
  /** Human-readable progress label (e.g., "2/3 days"). */
  progressLabel?: string;
}

// ── Badge definitions ───────────────────────────────────────────────────

export function checkBadges(): Badge[] {
  const badges: Badge[] = [];
  const now = new Date();

  // ── Sessions category ──

  const hasCheckin = !!sbGetSetting("ndw_last_emotion");
  badges.push({
    id: "first-checkin",
    icon: Mic,
    iconColor: "#0891b2",
    name: "First Voice",
    description: "Completed your first voice analysis",
    earned: hasCheckin,
    tier: "bronze",
    category: "Sessions",
    earnedDate: hasCheckin ? now.toISOString() : undefined,
  });

  // Early Bird / Night Owl
  try {
    const raw = sbGetSetting("ndw_last_emotion");
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
          earnedDate: hour < 8 ? new Date(ts).toISOString() : undefined,
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
          earnedDate: hour >= 22 ? new Date(ts).toISOString() : undefined,
        });
      }
    }
  } catch { /* ignore */ }

  // ── Streaks category ──

  const streakStr = sbGetSetting("ndw_streak_count");
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
    progressLabel: `${Math.min(streak, 3)}/3 days`,
    earnedDate: streak >= 3 ? now.toISOString() : undefined,
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
    progressLabel: `${Math.min(streak, 7)}/7 days`,
    earnedDate: streak >= 7 ? now.toISOString() : undefined,
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
    progressLabel: `${Math.min(streak, 14)}/14 days`,
    earnedDate: streak >= 14 ? now.toISOString() : undefined,
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
    progressLabel: `${Math.min(streak, 30)}/30 days`,
    earnedDate: streak >= 30 ? now.toISOString() : undefined,
  });

  // ── Milestones category ──

  const onboarded = sbGetSetting("ndw_onboarding_complete") === "true";
  badges.push({
    id: "onboarded",
    icon: GraduationCap,
    iconColor: "#0891b2",
    name: "All Set Up",
    description: "Completed app onboarding",
    earned: onboarded,
    tier: "bronze",
    category: "Milestones",
    earnedDate: onboarded ? now.toISOString() : undefined,
  });

  // Emotion Explorer — logged all 6 emotions
  const ALL_EMOTIONS = ["happy", "sad", "angry", "fear", "surprise", "neutral"];
  try {
    const seen = sbGetGeneric("ndw_emotions_seen") ?? [] as string[];
    const allSeen = ALL_EMOTIONS.every(e => seen.includes(e));
    const matchCount = seen.filter(e => ALL_EMOTIONS.includes(e)).length;
    badges.push({
      id: "emotion-explorer",
      icon: Palette,
      iconColor: "#4ade80",
      name: "Emotion Explorer",
      description: "Logged all 6 emotions",
      earned: allSeen,
      tier: "gold",
      category: "Milestones",
      progress: Math.min(1, matchCount / ALL_EMOTIONS.length),
      progressLabel: `${matchCount}/${ALL_EMOTIONS.length} emotions`,
      earnedDate: allSeen ? now.toISOString() : undefined,
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
      progressLabel: "0/6 emotions",
    });
  }

  // ── Wellness category ──

  const healthConnected = sbGetSetting("ndw_health_connect_granted") === "true"
    || sbGetSetting("ndw_apple_health_granted") === "true";
  badges.push({
    id: "health-sync",
    icon: Heart,
    iconColor: "#e879a8",
    name: "Health Synced",
    description: "Connected a health data source",
    earned: healthConnected,
    tier: "silver",
    category: "Wellness",
    earnedDate: healthConnected ? now.toISOString() : undefined,
  });

  const mealLogged = !!sbGetSetting("ndw_meal_logged");
  badges.push({
    id: "first-meal",
    icon: UtensilsCrossed,
    iconColor: "#ea580c",
    name: "First Meal",
    description: "Logged your first meal",
    earned: mealLogged,
    tier: "bronze",
    category: "Wellness",
    earnedDate: mealLogged ? now.toISOString() : undefined,
  });

  // ── Brain category ──

  const museConnected = sbGetSetting("ndw_muse_connected") === "true";
  badges.push({
    id: "muse-connected",
    icon: Brain,
    iconColor: "#6366f1",
    name: "Brain Reader",
    description: "Connected an EEG headband",
    earned: museConnected,
    tier: "gold",
    category: "Brain",
    earnedDate: museConnected ? now.toISOString() : undefined,
  });

  return badges;
}

// ── Motivational taglines ───────────────────────────────────────────────

function getTagline(earnedCount: number, totalCount: number): string {
  const pct = totalCount > 0 ? earnedCount / totalCount : 0;
  if (pct === 0) return "Your journey begins here";
  if (pct < 0.25) return "Every achievement starts with a single step";
  if (pct < 0.5) return "Building momentum, keep going";
  if (pct < 0.75) return "You are on a remarkable path";
  if (pct < 1) return "Almost there, the summit is in sight";
  return "You have unlocked every achievement";
}

// ── Category filter pill colors ─────────────────────────────────────────

const CATEGORY_PILL_COLORS: Record<AchievementCategory, string> = {
  All: "#94a3b8",
  Sessions: "#0891b2",
  Streaks: "#ea580c",
  Milestones: "#d4a017",
  Wellness: "#e879a8",
  Brain: "#6366f1",
};

// ── Category filter pills ───────────────────────────────────────────────

function CategoryFilterPills({
  active,
  onChange,
  categories,
}: {
  active: AchievementCategory;
  onChange: (cat: AchievementCategory) => void;
  categories: AchievementCategory[];
}) {
  return (
    <div
      data-testid="category-filters"
      style={{
        display: "flex",
        gap: 8,
        overflowX: "auto",
        paddingBottom: 4,
        marginBottom: 20,
        WebkitOverflowScrolling: "touch",
        scrollbarWidth: "none",
        msOverflowStyle: "none",
      }}
    >
      {categories.map((cat) => {
        const isActive = cat === active;
        const color = CATEGORY_PILL_COLORS[cat];
        return (
          <button
            key={cat}
            data-testid={`filter-${cat.toLowerCase()}`}
            onClick={() => onChange(cat)}
            style={{
              flexShrink: 0,
              fontSize: 12,
              fontWeight: 600,
              padding: "6px 14px",
              borderRadius: 20,
              border: `1px solid ${isActive ? color : "var(--border)"}`,
              background: isActive ? `${color}18` : "transparent",
              color: isActive ? color : "var(--muted-foreground)",
              cursor: "pointer",
              transition: "all 0.2s ease",
              letterSpacing: "0.2px",
            }}
          >
            {cat}
          </button>
        );
      })}
    </div>
  );
}

// ── Tier badge label ────────────────────────────────────────────────────

function TierBadge({ tier }: { tier: AchievementTier }) {
  const label = tier.charAt(0).toUpperCase() + tier.slice(1);
  return (
    <span
      data-testid={`tier-${tier}`}
      style={{
        position: "absolute",
        top: 10,
        right: 10,
        fontSize: 9,
        fontWeight: 700,
        letterSpacing: "0.6px",
        textTransform: "uppercase",
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

function ProgressBar({ progress, tier, label }: { progress: number; tier: AchievementTier; label?: string }) {
  const percent = Math.round(progress * 100);
  return (
    <div style={{ width: "100%", marginTop: 8 }}>
      <div
        style={{
          height: 5,
          borderRadius: 3,
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
            borderRadius: 3,
            transition: "width 0.6s cubic-bezier(0.22, 1, 0.36, 1)",
          }}
        />
      </div>
      <div style={{
        fontSize: 10,
        color: "var(--muted-foreground)",
        marginTop: 4,
        textAlign: "center",
      }}>
        {label ?? `${percent}%`}
      </div>
    </div>
  );
}

// ── Earned badge card — large card with gradient bg, shimmer, glow ──────

function EarnedBadgeCard({ badge, index }: { badge: Badge; index: number }) {
  injectShimmer();
  const IconComp = badge.icon;
  return (
    <motion.div
      custom={index}
      initial="hidden"
      animate="visible"
      variants={cardVariants}
      data-testid={`badge-${badge.id}`}
      style={{
        position: "relative",
        background: TIER_CARD_BG[badge.tier],
        border: `1px solid ${TIER_BORDER[badge.tier]}`,
        borderRadius: 20,
        padding: "20px 16px 16px",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        textAlign: "center",
        boxShadow: TIER_GLOW[badge.tier],
        overflow: "hidden",
        transition: "transform 0.2s ease, box-shadow 0.2s ease",
        minHeight: 160,
      }}
    >
      {/* Shimmer overlay */}
      <div
        aria-hidden="true"
        style={{
          position: "absolute",
          inset: 0,
          background: "linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.04) 40%, rgba(255,255,255,0.1) 50%, rgba(255,255,255,0.04) 60%, transparent 100%)",
          backgroundSize: "200% 100%",
          animation: "ndw-achievement-shimmer 3s ease-in-out infinite",
          pointerEvents: "none",
        }}
      />

      {/* Tier badge in corner */}
      <TierBadge tier={badge.tier} />

      {/* Large centered icon */}
      <div
        style={{
          width: 52,
          height: 52,
          borderRadius: "50%",
          background: `${badge.iconColor}20`,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          marginBottom: 10,
          animation: "ndw-achievement-glow 3s ease-in-out infinite",
        }}
      >
        <IconComp style={{ width: 26, height: 26, color: badge.iconColor }} />
      </div>

      {/* Name */}
      <div style={{
        fontSize: 13,
        fontWeight: 700,
        color: "var(--foreground)",
        marginBottom: 4,
        lineHeight: 1.2,
      }}>
        {badge.name}
      </div>

      {/* Description */}
      <div style={{
        fontSize: 10,
        color: "var(--muted-foreground)",
        lineHeight: 1.4,
        marginBottom: 6,
      }}>
        {badge.description}
      </div>

      {/* Earned date */}
      {badge.earnedDate && (
        <div style={{ fontSize: 9, color: "var(--muted-foreground)", opacity: 0.6 }}>
          {new Date(badge.earnedDate).toLocaleDateString("en-US", { month: "short", day: "numeric" })}
        </div>
      )}
    </motion.div>
  );
}

// ── Locked badge card — greyed, with lock overlay and progress ──────────

function LockedBadgeCard({ badge, index }: { badge: Badge; index: number }) {
  const IconComp = badge.icon;
  return (
    <motion.div
      custom={index}
      initial="hidden"
      animate="visible"
      variants={cardVariants}
      data-testid={`badge-${badge.id}`}
      style={{
        position: "relative",
        background: "var(--card)",
        border: "1px solid var(--border)",
        borderRadius: 20,
        padding: "20px 16px 16px",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        textAlign: "center",
        overflow: "hidden",
        minHeight: 160,
      }}
    >
      {/* Lock overlay */}
      <div
        aria-hidden="true"
        data-testid="lock-overlay"
        style={{
          position: "absolute",
          inset: 0,
          background: "rgba(0,0,0,0.35)",
          display: "flex",
          alignItems: "flex-start",
          justifyContent: "flex-end",
          padding: 10,
          pointerEvents: "none",
          zIndex: 1,
        }}
      >
        <Lock style={{ width: 14, height: 14, color: "rgba(255,255,255,0.5)" }} />
      </div>

      {/* Tier badge in corner */}
      <div style={{ position: "relative", zIndex: 0 }}>
        <TierBadge tier={badge.tier} />
      </div>

      {/* Icon — dimmed */}
      <div
        style={{
          width: 52,
          height: 52,
          borderRadius: "50%",
          background: "rgba(255,255,255,0.04)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          marginBottom: 10,
          opacity: 0.4,
          filter: "grayscale(0.8)",
        }}
      >
        <IconComp style={{ width: 26, height: 26, color: badge.iconColor }} />
      </div>

      {/* Name */}
      <div style={{
        fontSize: 13,
        fontWeight: 700,
        color: "var(--foreground)",
        marginBottom: 4,
        lineHeight: 1.2,
        opacity: 0.5,
      }}>
        {badge.name}
      </div>

      {/* Description */}
      <div style={{
        fontSize: 10,
        color: "var(--muted-foreground)",
        lineHeight: 1.4,
        marginBottom: 4,
        opacity: 0.5,
      }}>
        {badge.description}
      </div>

      {/* Progress bar if partially complete */}
      {badge.progress != null && badge.progress > 0 && badge.progress < 1 && (
        <div style={{ width: "100%", position: "relative", zIndex: 2 }}>
          <ProgressBar progress={badge.progress} tier={badge.tier} label={badge.progressLabel} />
        </div>
      )}
    </motion.div>
  );
}

// ── Hero Section ────────────────────────────────────────────────────────

function HeroSection({ earned, total }: { earned: number; total: number }) {
  const pct = total > 0 ? Math.round((earned / total) * 100) : 0;
  const tagline = getTagline(earned, total);

  return (
    <div
      data-testid="achievements-hero"
      style={{
        background: "linear-gradient(135deg, rgba(99,102,241,0.12) 0%, rgba(212,160,23,0.08) 100%)",
        border: "1px solid rgba(99,102,241,0.15)",
        borderRadius: 20,
        padding: "22px 20px",
        marginBottom: 20,
        textAlign: "center",
      }}
    >
      {/* Stats row */}
      <div style={{
        display: "flex",
        justifyContent: "center",
        gap: 32,
        marginBottom: 12,
      }}>
        <div>
          <div data-testid="hero-earned-count" style={{
            fontSize: 32,
            fontWeight: 800,
            color: "#d4a017",
            lineHeight: 1,
          }}>
            {earned}
          </div>
          <div style={{ fontSize: 11, color: "var(--muted-foreground)", marginTop: 4 }}>
            Earned
          </div>
        </div>
        <div style={{
          width: 1,
          background: "var(--border)",
          alignSelf: "stretch",
        }} />
        <div>
          <div data-testid="hero-completion-pct" style={{
            fontSize: 32,
            fontWeight: 800,
            color: "#6366f1",
            lineHeight: 1,
          }}>
            {pct}%
          </div>
          <div style={{ fontSize: 11, color: "var(--muted-foreground)", marginTop: 4 }}>
            Complete
          </div>
        </div>
      </div>

      {/* Overall progress bar */}
      <div style={{
        height: 6,
        borderRadius: 3,
        background: "rgba(255,255,255,0.06)",
        overflow: "hidden",
        marginBottom: 12,
      }}>
        <div
          data-testid="overall-progress"
          style={{
            height: "100%",
            width: `${pct}%`,
            background: "linear-gradient(90deg, #6366f1, #d4a017, #e879a8)",
            borderRadius: 3,
            transition: "width 0.8s cubic-bezier(0.22, 1, 0.36, 1)",
          }}
        />
      </div>

      {/* Tagline */}
      <div data-testid="hero-tagline" style={{
        fontSize: 13,
        color: "var(--muted-foreground)",
        fontStyle: "italic",
        letterSpacing: "0.1px",
      }}>
        {tagline}
      </div>
    </div>
  );
}

// ── Main exported component ─────────────────────────────────────────────

export function AchievementBadges() {
  const [badges, setBadges] = useState<Badge[]>([]);
  const [activeCategory, setActiveCategory] = useState<AchievementCategory>("All");

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

  const filteredBadges = activeCategory === "All"
    ? badges
    : badges.filter(b => b.category === activeCategory);

  const earned = filteredBadges.filter(b => b.earned);
  const locked = filteredBadges.filter(b => !b.earned);
  const totalEarned = badges.filter(b => b.earned).length;

  if (badges.length === 0) return null;

  return (
    <div style={{ marginBottom: 14 }}>
      {/* Hero section — always shows totals regardless of filter */}
      <HeroSection earned={totalEarned} total={badges.length} />

      {/* Category filter pills */}
      <CategoryFilterPills
        active={activeCategory}
        onChange={setActiveCategory}
        categories={BADGE_CATEGORIES}
      />

      {/* Earned badges — large card grid */}
      {earned.length > 0 && (
        <>
          <div style={{
            fontSize: 11,
            fontWeight: 600,
            color: "var(--muted-foreground)",
            textTransform: "uppercase",
            letterSpacing: "0.5px",
            marginBottom: 10,
          }}>
            Earned
          </div>
          <div
            data-testid="earned-grid"
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 12,
              marginBottom: locked.length > 0 ? 20 : 0,
            }}
          >
            {earned.map((b, i) => <EarnedBadgeCard key={b.id} badge={b} index={i} />)}
          </div>
        </>
      )}

      {/* Locked badges */}
      {locked.length > 0 && (
        <>
          <div style={{
            fontSize: 11,
            fontWeight: 500,
            color: "var(--muted-foreground)",
            marginBottom: 10,
            opacity: 0.7,
          }}>
            Locked
          </div>
          <div
            data-testid="locked-grid"
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 12,
            }}
          >
            {locked.map((b, i) => <LockedBadgeCard key={b.id} badge={b} index={i} />)}
          </div>
        </>
      )}
    </div>
  );
}

import { useState, useEffect, useMemo } from "react";
import { useLocation } from "wouter";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { pageTransition, cardVariants } from "@/lib/animations";
import { getParticipantId } from "@/lib/participant";
import { useHealthSync } from "@/hooks/use-health-sync";
import { useFusedState } from "@/hooks/use-fused-state";
import { detectMoodPatterns, type EmotionReading, type MoodInsight } from "@/lib/mood-patterns";
import { listSessions, type SessionSummary } from "@/lib/ml-api";
import { getEmotionHistory as sbGetEmotionHistory, saveEmotionHistory as sbSaveEmotionHistory, sbGetSetting, sbSaveGeneric } from "../lib/supabase-store";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, ReferenceLine,
} from "recharts";
import {
  BedDouble, Brain, Heart, Sparkles, Wind, Moon, Target, BookOpen, Dumbbell,
  Smile, Activity, BarChart3, MessageCircle, Lightbulb, Star, TrendingDown,
  TrendingUp, Sunrise, CloudMoon as CloudMoonIcon, Palette, RefreshCw, Waves,
  HeartHandshake,
  type LucideIcon,
} from "lucide-react";

// ── Emotion data from localStorage ──────────────────────────────────────

interface CheckinData {
  emotion?: string;
  valence?: number;
  stress_index?: number;
  focus_index?: number;
  relaxation_index?: number;
  confidence?: number;
}

const EMOTION_COLOR: Record<string, string> = {
  happy: "#10b981", sad: "#6b7280", angry: "#ef4444", fear: "#6b7280",
  surprise: "#10b981", neutral: "#94a3b8",
};

// ── localStorage emotion history ────────────────────────────────────────
// Accumulates emotion readings locally so the EmotionsOverview chart can
// display data even when the server API returns empty (e.g. new user,
// offline, no DB connection). Capped at 200 entries, 7 days max.

const EMOTION_HISTORY_KEY = "ndw_emotion_history";
const EMOTION_HISTORY_MAX = 200;
const EMOTION_HISTORY_MAX_AGE_MS = 7 * 24 * 60 * 60 * 1000;

interface LocalEmotionEntry {
  stress: number;
  happiness: number;
  focus: number;
  dominantEmotion: string;
  timestamp: string;
}

function appendToEmotionHistory(checkin: CheckinData): void {
  if (!checkin?.emotion) return;
  try {
    const raw = sbGetSetting(EMOTION_HISTORY_KEY);
    const history: LocalEmotionEntry[] = raw ? JSON.parse(raw) : [];
    const now = new Date();
    const cutoff = now.getTime() - EMOTION_HISTORY_MAX_AGE_MS;

    // Deduplicate: skip if last entry was within 30 seconds
    if (history.length > 0) {
      const last = new Date(history[history.length - 1].timestamp).getTime();
      if (now.getTime() - last < 30_000) return;
    }

    const entry = {
      stress: checkin.stress_index ?? 0,
      happiness: checkin.valence != null ? Math.max(0, checkin.valence) : 0.5,
      focus: checkin.focus_index ?? 0.5,
      dominantEmotion: checkin.emotion ?? "neutral",
      timestamp: now.toISOString(),
    };

    history.push(entry);

    // Prune old entries and cap size
    const pruned = history
      .filter(e => new Date(e.timestamp).getTime() > cutoff)
      .slice(-EMOTION_HISTORY_MAX);

    sbSaveGeneric(EMOTION_HISTORY_KEY, pruned);

    // Also persist to Supabase (fire-and-forget)
    sbSaveEmotionHistory("local", {
      stress: entry.stress,
      focus: entry.focus,
      mood: entry.happiness,
      source: "voice",
      dominantEmotion: entry.dominantEmotion,
      created_at: entry.timestamp,
    }).catch(() => {});
  } catch { /* storage quota or parse error */ }
}

function getLocalEmotionHistory(): LocalEmotionEntry[] {
  try {
    const raw = sbGetSetting(EMOTION_HISTORY_KEY);
    if (!raw) return [];
    const history: LocalEmotionEntry[] = JSON.parse(raw);
    const cutoff = Date.now() - EMOTION_HISTORY_MAX_AGE_MS;
    return history.filter(e => new Date(e.timestamp).getTime() > cutoff);
  } catch {
    return [];
  }
}

function useCheckinData(): CheckinData | null {
  const [data, setData] = useState<CheckinData | null>(null);
  useEffect(() => {
    const read = (): CheckinData | null => {
      try {
        const raw = sbGetSetting("ndw_last_emotion");
        if (raw) return JSON.parse(raw)?.result ?? JSON.parse(raw);
      } catch { /* ignore */ }
      return null;
    };
    const initial = read();
    setData(initial);
    if (initial) appendToEmotionHistory(initial);

    const handler = () => {
      const updated = read();
      setData(updated);
      if (updated) appendToEmotionHistory(updated);
    };
    window.addEventListener("ndw-voice-updated", handler);
    window.addEventListener("ndw-emotion-update", handler);
    window.addEventListener("storage", handler);
    return () => {
      window.removeEventListener("ndw-voice-updated", handler);
      window.removeEventListener("ndw-emotion-update", handler);
      window.removeEventListener("storage", handler);
    };
  }, []);
  return data;
}

// ── Emotion Timeline Component ─────────────────────────────────────────────

const TIMELINE_COLORS: Record<string, string> = {
  happy: "#10b981", sad: "#6b7280", angry: "#ef4444", fear: "#6b7280",
  surprise: "#10b981", neutral: "#94a3b8",
};

function EmotionTimeline({ userId }: { userId: string }) {
  const { data } = useQuery<Array<{ dominantEmotion: string; timestamp: string }>>({
    queryKey: [`/api/brain/history/${userId}?days=7`],
    retry: false,
    staleTime: 5 * 60 * 1000,
  });

  if (!data || data.length === 0) return null;

  // Group by day, take last emotion per day
  const dayMap = new Map<string, { emotion: string; label: string }>();
  for (const r of data) {
    const d = new Date(r.timestamp);
    const key = d.toISOString().slice(0, 10);
    const label = d.toLocaleDateString(undefined, { weekday: "short" });
    dayMap.set(key, { emotion: r.dominantEmotion, label });
  }

  const days = Array.from(dayMap.entries())
    .sort(([a], [b]) => a.localeCompare(b))
    .slice(-7);

  if (days.length === 0) return null;

  return (
    <div className="rounded-[14px] bg-card border border-border p-4 mb-3.5">
      <div className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-2.5">
        Your week in emotions
      </div>
      <div className="flex justify-between items-center">
        {days.map(([key, { emotion, label }]) => {
          const color = TIMELINE_COLORS[emotion] ?? "#94a3b8";
          return (
            <div key={key} className="flex flex-col items-center gap-1">
              <div
                role="img"
                aria-label={`${label}: ${emotion}`}
                className="w-7 h-7 rounded-full opacity-85 transition-transform duration-300 ease-[cubic-bezier(0.22,1,0.36,1)]"
                style={{ background: color }}
              />
              <span className="text-[9px] text-foreground/35">{label}</span>
            </div>
          );
        })}
      </div>
      {days.length >= 3 && (
        <div className="text-[10px] text-foreground/35 mt-2 text-center">
          {(() => {
            const counts: Record<string, number> = {};
            days.forEach(([, { emotion }]) => { counts[emotion] = (counts[emotion] || 0) + 1; });
            const top = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
            return top ? `Mostly ${top[0]} this week` : "";
          })()}
        </div>
      )}
    </div>
  );
}

// ── Recommended Section -- personalized suggestions based on emotion ────────

interface Recommendation {
  icon: LucideIcon;
  iconColor: string;
  title: string;
  reason: string;
  route: string;
}

// All possible features for progressive discovery
const ALL_FEATURES = [
  { route: "/biofeedback", title: "Breathing Exercise", icon: Wind as LucideIcon, iconColor: "#10b981", category: "calm" },
  { route: "/sleep-session", title: "Sleep Music", icon: Moon as LucideIcon, iconColor: "#10b981", category: "calm" },
  { route: "/inner-energy", title: "Inner Energy", icon: Sparkles as LucideIcon, iconColor: "#10b981", category: "energy" },
  { route: "/neurofeedback", title: "Neurofeedback", icon: Target as LucideIcon, iconColor: "#10b981", category: "focus" },
  { route: "/dreams", title: "Dream Journal", icon: BookOpen as LucideIcon, iconColor: "#10b981", category: "insight" },
  { route: "/workout", title: "Workout", icon: Dumbbell as LucideIcon, iconColor: "#10b981", category: "energy" },
  { route: "/brain-monitor", title: "Brain Monitor", icon: Brain as LucideIcon, iconColor: "#10b981", category: "insight" },
  { route: "/habits", title: "Habit Tracker", icon: BarChart3 as LucideIcon, iconColor: "#10b981", category: "insight" },
  { route: "/ai-companion", title: "AI Companion", icon: MessageCircle as LucideIcon, iconColor: "#10b981", category: "support" },
  { route: "/insights", title: "Wellness Insights", icon: Lightbulb as LucideIcon, iconColor: "#10b981", category: "insight" },
];

function getUsedFeatures(): Set<string> {
  try {
    const raw = sbGetSetting("ndw_feature_usage");
    return raw ? new Set(JSON.parse(raw)) : new Set();
  } catch {
    return new Set();
  }
}

function trackFeatureUsage(route: string) {
  try {
    const used = getUsedFeatures();
    used.add(route);
    sbSaveGeneric("ndw_feature_usage", Array.from(used));
  } catch { /* ignore */ }
}

function getRecommendations(stress: number, valence: number, focus: number): Recommendation[] {
  const recs: Recommendation[] = [];
  const hour = new Date().getHours();
  const isEvening = hour >= 17;
  const isMorning = hour >= 5 && hour < 12;
  const usedFeatures = getUsedFeatures();

  // Emotion-based (highest priority)
  if (stress > 0.5) {
    recs.push({ icon: Wind, iconColor: "#10b981", title: "Breathing Exercise", reason: "Your stress is elevated", route: "/biofeedback" });
    if (isEvening) {
      recs.push({ icon: Moon, iconColor: "#10b981", title: "Sleep Music", reason: "Wind down before bed", route: "/sleep-session" });
    }
  }
  if (valence < -0.1) {
    recs.push({ icon: MessageCircle, iconColor: "#10b981", title: "AI Companion", reason: "Talk through how you're feeling", route: "/ai-companion" });
    recs.push({ icon: Sparkles, iconColor: "#10b981", title: "Inner Energy", reason: "Rebalance your energy centers", route: "/inner-energy" });
  }
  if (focus < 0.4) {
    recs.push({ icon: Target, iconColor: "#10b981", title: "Neurofeedback", reason: "Train your focus", route: "/neurofeedback" });
  }
  if (valence > 0.3 && stress < 0.3) {
    if (isMorning) {
      recs.push({ icon: Dumbbell, iconColor: "#10b981", title: "Workout", reason: "Ride this morning energy", route: "/workout" });
    }
    recs.push({ icon: BookOpen, iconColor: "#10b981", title: "Dream Journal", reason: "Great mood -- capture your dreams", route: "/dreams" });
  }

  // Time-of-day suggestions (if we don't have enough yet)
  if (recs.length < 2) {
    if (isMorning) {
      recs.push({ icon: Lightbulb, iconColor: "#10b981", title: "Wellness Insights", reason: "Start your day with awareness", route: "/insights" });
    } else if (isEvening) {
      recs.push({ icon: Moon, iconColor: "#10b981", title: "Sleep Music", reason: "Prepare for a restful night", route: "/sleep-session" });
    } else {
      recs.push({ icon: Smile, iconColor: "#10b981", title: "Wellness", reason: "See your patterns", route: "/wellness" });
    }
  }

  // Progressive discovery -- suggest an unused feature
  if (recs.length < 3) {
    const unused = ALL_FEATURES.filter(f =>
      !usedFeatures.has(f.route) && !recs.some(r => r.route === f.route)
    );
    if (unused.length > 0) {
      // Pick a random unused feature
      const pick = unused[Math.floor(Math.random() * unused.length)];
      recs.push({
        icon: pick.icon,
        iconColor: pick.iconColor,
        title: pick.title,
        reason: "Try something new",
        route: pick.route,
      });
    }
  }

  // Fallback
  if (recs.length === 0) {
    recs.push({ icon: Brain, iconColor: "#10b981", title: "Brain Monitor", reason: "Explore your brainwaves", route: "/brain-monitor" });
  }

  return recs.slice(0, 3);
}

function RecommendedSection({ stress, valence, focus, navigate }: {
  stress: number; valence: number; focus: number; navigate: (path: string) => void;
}) {
  const recs = getRecommendations(stress, valence, focus);
  if (recs.length === 0) return null;

  return (
    <div className="mb-4">
      <div className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-2">
        Recommended for you
      </div>
      <div className="flex gap-2 overflow-x-auto pb-1">
        {recs.map((rec) => (
          <button
            key={rec.route}
            onClick={() => { trackFeatureUsage(rec.route); navigate(rec.route); }}
            aria-label={`${rec.title}: ${rec.reason}`}
            className="rounded-[14px] bg-card border border-border p-3 min-w-[150px] shrink-0 text-left cursor-pointer"
          >
            <rec.icon className="w-[22px] h-[22px]" style={{ color: rec.iconColor }} aria-hidden="true" />
            <div className="text-xs font-semibold text-foreground mt-1.5">{rec.title}</div>
            <div className="text-[10px] text-foreground/35 mt-0.5">{rec.reason}</div>
          </button>
        ))}
      </div>
    </div>
  );
}

// ── Mood Insights Card ────────────────────────────────────────────────────

function MoodInsightsCard({ userId }: { userId: string }) {
  const { data } = useQuery<EmotionReading[]>({
    queryKey: [`/api/brain/history/${userId}?days=7`],
    retry: false,
    staleTime: 10 * 60 * 1000,
  });

  if (!data || data.length < 3) return null;

  // eslint-disable-next-line react-hooks/rules-of-hooks
  const insights = useMemo(() => detectMoodPatterns(data), [data]);
  if (insights.length === 0) return null;

  const borderColors: Record<string, string> = {
    positive: "var(--border)",
    warning: "var(--border)",
    neutral: "var(--border)",
  };

  const INSIGHT_ICONS: Record<string, { Icon: LucideIcon; color: string }> = {
    "star": { Icon: Star, color: "#10b981" },
    "heart": { Icon: Heart, color: "#10b981" },
    "waves": { Icon: Waves, color: "#10b981" },
    "trending-down": { Icon: TrendingDown, color: "#10b981" },
    "trending-up": { Icon: TrendingUp, color: "#10b981" },
    "sunrise": { Icon: Sunrise, color: "#10b981" },
    "cloud-moon": { Icon: CloudMoonIcon, color: "#10b981" },
    "palette": { Icon: Palette, color: "#10b981" },
    "refresh": { Icon: RefreshCw, color: "#94a3b8" },
  };

  return (
    <div className="mb-3.5">
      <div className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-2">
        Mood insights
      </div>
      <div className="flex flex-col gap-2">
        {insights.map((insight, i) => {
          const iconEntry = INSIGHT_ICONS[insight.icon];
          const IconComp = iconEntry?.Icon ?? Activity;
          const iconColor = iconEntry?.color ?? "var(--muted-foreground)";
          return (
          <div
            key={i}
            className="bg-card rounded-xl p-2.5 flex items-start gap-2"
            style={{ border: `1px solid ${borderColors[insight.type] ?? "var(--border)"}` }}
          >
            <IconComp className="w-[18px] h-[18px] shrink-0 mt-px" style={{ color: iconColor }} />
            <div>
              <div className="text-xs font-semibold text-foreground">{insight.title}</div>
              <div className="text-[10px] text-foreground/35 mt-0.5 leading-relaxed">
                {insight.description}
              </div>
            </div>
          </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Types ──────────────────────────────────────────────────────────────────

interface NavCard {
  icon: LucideIcon;
  title: string;
  subtitle: string;
  route: string;
  accentColor: string;
}

// ── Data -- 8 main navigation cards (2x4 grid) ─────────────────────────────

const NAV_CARDS: NavCard[] = [
  { icon: BedDouble,  title: "Sleep",        subtitle: "Sleep data, dreams, music",           route: "/sleep",          accentColor: "#10b981" },
  { icon: Brain,      title: "Brain",        subtitle: "EEG, neurofeedback, connectivity",   route: "/brain-monitor",  accentColor: "#10b981" },
  { icon: Heart,      title: "Health Scores",subtitle: "Recovery, strain, sleep, stress",    route: "/scores",         accentColor: "#10b981" },
  { icon: Dumbbell,   title: "Workout",      subtitle: "Exercises, templates, history",       route: "/workout",        accentColor: "#10b981" },
  { icon: Target,     title: "Habits",       subtitle: "Streaks, heatmap, analytics",         route: "/habits",         accentColor: "#10b981" },
  { icon: Sparkles,   title: "Inner Energy", subtitle: "Energy centers, spiritual wellness",  route: "/inner-energy",   accentColor: "#10b981" },
  { icon: Wind,       title: "Wellness",     subtitle: "Body metrics, menstrual, supplements",route: "/wellness",       accentColor: "#10b981" },
  { icon: HeartHandshake, title: "Couples Meditation", subtitle: "Brain synchrony with your partner", route: "/couples-meditation", accentColor: "#10b981" },
];

// Sample sparkline points (normalized 0-40 in Y space, 0-280 in X)
const SPARKLINE_POINTS = [
  [0, 32],
  [40, 24],
  [80, 30],
  [120, 14],
  [160, 22],
  [200, 10],
  [240, 18],
  [280, 8],
] as [number, number][];

function pointsToPolyline(pts: [number, number][]): string {
  return pts.map(([x, y]) => `${x},${y}`).join(" ");
}

function pointsToArea(pts: [number, number][]): string {
  if (pts.length === 0) return "";
  const first = pts[0];
  const last = pts[pts.length - 1];
  const line = pts.map(([x, y]) => `${x},${y}`).join(" L ");
  return `M ${first[0]},${first[1]} L ${line} L ${last[0]},40 L ${first[0]},40 Z`;
}

// ── Emotions Overview -- combined stress/focus/mood chart ──────────────────

interface HistoryRow {
  stress: number;
  happiness: number;
  focus: number;
  dominantEmotion: string;
  timestamp: string;
}

type DiscoverTimeRange = "today" | "week" | "month";

function EmotionsOverview({ userId, navigate, checkin }: { userId: string; navigate: (p: string) => void; checkin: CheckinData | null }) {
  const [range, setRange] = useState<DiscoverTimeRange>("today");

  const { data } = useQuery<HistoryRow[]>({
    queryKey: [`/api/brain/history/${userId}?days=30`],
    retry: false,
    staleTime: 5 * 60 * 1000,
  });

  // Fetch session data for trend comparison
  const { data: sessions } = useQuery<SessionSummary[]>({
    queryKey: ["sessions", userId],
    queryFn: () => listSessions(userId),
    staleTime: 30_000,
    retry: false,
  });

  // Compute trend deltas from last two sessions
  const trends = useMemo(() => {
    if (!sessions || sessions.length < 2) return null;
    const sorted = [...sessions]
      .filter((s) => s.summary && (s.summary.avg_stress != null || s.summary.avg_focus != null))
      .sort((a, b) => (a.start_time ?? 0) - (b.start_time ?? 0));
    if (sorted.length < 2) return null;
    const prev = sorted[sorted.length - 2].summary;
    const cur = sorted[sorted.length - 1].summary;
    const stressDelta = (cur.avg_stress != null && prev.avg_stress != null)
      ? (cur.avg_stress - prev.avg_stress) * 100
      : null;
    const focusDelta = (cur.avg_focus != null && prev.avg_focus != null)
      ? (cur.avg_focus - prev.avg_focus) * 100
      : null;
    return { stressDelta, focusDelta };
  }, [sessions]);

  const allPoints = useMemo(() => {
    // Continuous data points -- every reading as an individual point, not daily averages
    const points: { time: string; ts: number; stress: number; focus: number; mood: number }[] = [];

    // From API history
    const rows: HistoryRow[] =
      data && Array.isArray(data) && data.length > 0 ? data : getLocalEmotionHistory();

    for (const r of rows) {
      const d = new Date(r.timestamp);
      points.push({
        time: d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" }),
        ts: d.getTime(),
        stress: Math.round((r.stress ?? 0) * 100),
        focus: Math.round((r.focus ?? 0) * 100),
        mood: Math.round((r.happiness ?? 0.5) * 100),
      });
    }

    // From session history
    if (sessions && sessions.length > 0) {
      const monthAgo = Date.now() / 1000 - 30 * 86400;
      for (const s of sessions) {
        if (!s.summary || s.summary.avg_focus == null) continue;
        if ((s.start_time ?? 0) < monthAgo) continue;
        const d = new Date((s.start_time ?? 0) * 1000);
        points.push({
          time: d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" }),
          ts: d.getTime(),
          stress: Math.round((s.summary.avg_stress ?? 0) * 100),
          focus: Math.round((s.summary.avg_focus ?? 0) * 100),
          mood: Math.round((s.summary.avg_relaxation ?? 0.5) * 100),
        });
      }
    }

    // Sort by timestamp, show all individual data points
    points.sort((a, b) => a.ts - b.ts);
    return points;
  }, [data, sessions]);

  // Filter by selected time range and reformat time labels
  const chartData = useMemo(() => {
    const now = Date.now();
    const filtered = allPoints.filter((p) => {
      switch (range) {
        case "today":
          return new Date(p.ts).toDateString() === new Date().toDateString();
        case "week":
          return now - p.ts < 7 * 86_400_000;
        case "month":
        default:
          return true;
      }
    });
    // Reformat time labels based on range
    return filtered.map((p) => {
      const d = new Date(p.ts);
      let time: string;
      if (range === "today") {
        time = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      } else if (range === "week") {
        time = d.toLocaleDateString([], { weekday: "short", hour: "2-digit" });
      } else {
        time = d.toLocaleDateString([], { month: "short", day: "numeric" });
      }
      return { ...p, time };
    });
  }, [allPoints, range]);

  // Personal baseline — computed from ALL historical points (not just current range)
  // This mirrors Oura's "relative to your own average" approach
  const baseline = useMemo(() => {
    if (allPoints.length < 5) return null;
    const avgStress = Math.round(allPoints.reduce((s, p) => s + p.stress, 0) / allPoints.length);
    const avgFocus = Math.round(allPoints.reduce((s, p) => s + p.focus, 0) / allPoints.length);
    return { avgStress, avgFocus };
  }, [allPoints]);

  // Current values from checkin data (reactive via useCheckinData hook)
  const current = useMemo(() => {
    if (!checkin) return null;
    return {
      stress: Math.round((checkin.stress_index ?? 0) * 100),
      focus: Math.round((checkin.focus_index ?? 0) * 100),
      mood: checkin.emotion ?? "neutral",
    };
  }, [checkin]);

  return (
    <button
      onClick={() => navigate("/mood")}
      aria-label="View Emotions: Stress & Focus trends"
      role="link"
      className="rounded-[14px] bg-card border border-border w-full p-4 mb-3.5 cursor-pointer text-left transition-all duration-200 ease-out"
    >
      <div className="flex justify-between items-center mb-3">
        <div>
          <p className="text-sm font-semibold text-foreground m-0">
            {current?.mood ? `Mood: ${String(current.mood).charAt(0).toUpperCase() + String(current.mood).slice(1)}` : "Emotions"}
          </p>
          <p className="text-[10px] text-foreground/35 mt-0.5 m-0">Stress & Focus — continuous trend</p>
        </div>
        {current && (
          <div className="flex gap-2.5">
            <div className="text-center">
              <div className="text-base font-bold text-emerald-500">{current.stress}%</div>
              <div className="text-[9px] text-muted-foreground">Stress</div>
              {baseline ? (
                <div className="flex items-center justify-center gap-0.5 mt-0.5">
                  {current.stress > baseline.avgStress ? (
                    <TrendingUp className="w-3 h-3 text-emerald-500" />
                  ) : (
                    <TrendingDown className="w-3 h-3 text-emerald-500" />
                  )}
                  <span className="text-[8px] text-muted-foreground">avg {baseline.avgStress}%</span>
                </div>
              ) : trends?.stressDelta != null && Math.abs(trends.stressDelta) > 2 ? (
                <div className="flex items-center justify-center gap-0.5 mt-0.5">
                  {trends.stressDelta > 0 ? <TrendingUp className="w-3 h-3 text-emerald-500" /> : <TrendingDown className="w-3 h-3 text-emerald-500" />}
                  <span className="text-[8px] text-muted-foreground">{Math.abs(Math.round(trends.stressDelta))}% vs last</span>
                </div>
              ) : null}
            </div>
            <div className="text-center">
              <div className="text-base font-bold text-emerald-500">{current.focus}%</div>
              <div className="text-[9px] text-muted-foreground">Focus</div>
              {baseline ? (
                <div className="flex items-center justify-center gap-0.5 mt-0.5">
                  {current.focus > baseline.avgFocus ? (
                    <TrendingUp className="w-3 h-3 text-emerald-500" />
                  ) : (
                    <TrendingDown className="w-3 h-3 text-emerald-500" />
                  )}
                  <span className="text-[8px] text-muted-foreground">avg {baseline.avgFocus}%</span>
                </div>
              ) : trends?.focusDelta != null && Math.abs(trends.focusDelta) > 2 ? (
                <div className="flex items-center justify-center gap-0.5 mt-0.5">
                  {trends.focusDelta > 0 ? <TrendingUp className="w-3 h-3 text-emerald-500" /> : <TrendingDown className="w-3 h-3 text-emerald-500" />}
                  <span className="text-[8px] text-muted-foreground">{Math.abs(Math.round(trends.focusDelta))}% vs last</span>
                </div>
              ) : null}
            </div>
          </div>
        )}
      </div>

      {/* Time range tabs */}
      <div className="flex gap-1.5 mb-2.5" onClick={(e) => e.stopPropagation()}>
        {(["today", "week", "month"] as DiscoverTimeRange[]).map((r) => (
          <button
            key={r}
            onClick={(e) => { e.preventDefault(); e.stopPropagation(); setRange(r); }}
            className={`flex-1 py-1.5 rounded-[10px] text-[11px] font-semibold border-none cursor-pointer transition-colors duration-200 ${
              range === r
                ? "bg-primary text-primary-foreground"
                : "bg-foreground/[0.06] text-foreground/60"
            }`}
          >
            {r === "today" ? "Today" : r === "week" ? "Week" : "Month"}
          </button>
        ))}
      </div>

      {chartData.length > 1 ? (
        <div className="overflow-x-auto overflow-y-hidden h-[220px]" style={{ WebkitOverflowScrolling: "touch" }} onClick={(e) => e.stopPropagation()}>
          <div style={{ width: Math.max(chartData.length * 40, 320), minWidth: "100%", height: "100%" }}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ left: 0, right: 4, top: 4, bottom: 0 }}>
              <defs>
                <linearGradient id="discStressG" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#10b981" stopOpacity={0.2} />
                  <stop offset="100%" stopColor="#10b981" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="discFocusG" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#6b7280" stopOpacity={0.2} />
                  <stop offset="100%" stopColor="#6b7280" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="discMoodG" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#0891b2" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#0891b2" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" strokeOpacity={0.5} />
              <XAxis dataKey="time" tick={{ fontSize: 9, fill: "var(--muted-foreground)" }} axisLine={false} tickLine={false} interval="preserveStartEnd" />
              <YAxis domain={[0, 100]} tick={{ fontSize: 9, fill: "var(--muted-foreground)" }} axisLine={false} tickLine={false} width={28} tickFormatter={(v) => `${v}`} />
              <Tooltip
                contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", borderRadius: 14, fontSize: 11 }}
                labelStyle={{ color: "var(--muted-foreground)" }}
                formatter={(v: number, name: string) => [`${v}%`, name]}
              />
              <Area type="monotone" dataKey="stress" stroke="#10b981" fill="url(#discStressG)" strokeWidth={2} dot={false} activeDot={false} name="Stress" isAnimationActive={true} animationDuration={1200} animationEasing="ease-out" />
              <Area type="monotone" dataKey="focus" stroke="#6b7280" fill="url(#discFocusG)" strokeWidth={2} dot={false} activeDot={false} name="Focus" isAnimationActive={true} animationDuration={1200} animationEasing="ease-out" />
              {/* Personal baseline reference lines — your average vs today (Oura-style) */}
              {baseline && (
                <>
                  <ReferenceLine y={baseline.avgStress} stroke="#10b981" strokeDasharray="4 3" strokeOpacity={0.5} strokeWidth={1.5} label={{ value: `avg ${baseline.avgStress}%`, position: "insideTopRight", fontSize: 8, fill: "#10b98180" }} />
                  <ReferenceLine y={baseline.avgFocus} stroke="#6b7280" strokeDasharray="4 3" strokeOpacity={0.5} strokeWidth={1.5} label={{ value: `avg ${baseline.avgFocus}%`, position: "insideBottomRight", fontSize: 8, fill: "#6b728080" }} />
                </>
              )}
              {/* Mood removed from chart -- shown as text label at top instead */}
            </AreaChart>
          </ResponsiveContainer>
          </div>
        </div>
      ) : (
        <div className="h-20 flex items-center justify-center">
          <p className="text-[11px] text-foreground/35">Complete a voice analysis to see trends</p>
        </div>
      )}

      {/* Legend */}
      <div className="flex gap-3.5 mt-2">
        <div className="flex items-center gap-1">
          <div className="w-2 h-[3px] rounded-sm bg-emerald-500" />
          <span className="text-[9px] text-muted-foreground">Stress</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-2 h-[3px] rounded-sm bg-gray-500" />
          <span className="text-[9px] text-muted-foreground">Focus</span>
        </div>
      </div>
    </button>
  );
}

// ── Main Component ─────────────────────────────────────────────────────────

export default function Discover() {
  const [, navigate] = useLocation();
  const checkin = useCheckinData();
  const { latestPayload } = useHealthSync();
  const { fusedState } = useFusedState(); // Data fusion bus: auto-updates from EEG/voice/health
  const userId = getParticipantId();

  // Fetch today's food logs for calorie sum
  const { data: foodLogs } = useQuery<{ calories?: number }[]>({
    queryKey: [`/api/food/logs/${userId}`],
    retry: false,
  });

  // Use fused state when available, fall back to checkin data
  const emotion = fusedState?.emotion ?? checkin?.emotion ?? "\u2014";
  const emoColor = EMOTION_COLOR[emotion] ?? "var(--muted-foreground)";
  const stress = fusedState?.stress ?? checkin?.stress_index ?? 0;
  const focus = fusedState?.focus ?? checkin?.focus_index ?? 0;
  const relaxation = checkin?.relaxation_index ?? (1 - stress);
  const valence = fusedState?.valence ?? checkin?.valence ?? 0;
  const hasData = !!fusedState || !!checkin?.emotion;

  // ── Health metric derived values ─────────────────────────────────────────
  const heartRate = latestPayload?.current_heart_rate ?? latestPayload?.resting_heart_rate ?? null;
  const steps = latestPayload?.steps_today ?? null;
  const stepsGoal = 10000;
  const stepsPercent = steps != null ? Math.round((steps / stepsGoal) * 100) : null;

  const sleepHours = latestPayload?.sleep_total_hours ?? null;
  const sleepEfficiency = latestPayload?.sleep_efficiency ?? null;
  const sleepLabel: string | null = sleepHours != null
    ? `${Math.floor(sleepHours)}h ${Math.round((sleepHours % 1) * 60)}m`
    : null;

  // Calorie sum from today's logs
  const caloriesToday = foodLogs
    ? foodLogs.reduce((sum: number, log: { calories?: number }) => sum + (log.calories ?? 0), 0)
    : null;

  // Emotion/readiness score: use valence remapped to 0-100 if available
  const emotionScore = hasData
    ? Math.round(((valence + 1) / 2) * 100)
    : null;
  const emotionLabel = hasData ? emotion : null;

  return (
    <motion.main
      initial={pageTransition.initial}
      animate={pageTransition.animate}
      transition={pageTransition.transition}
      className="bg-background p-4 pb-4 font-sans"
    >
      {/* ── Header ── */}
      <div className="mb-[18px]">
        <p className="text-xl font-bold text-foreground m-0 mb-1 -tracking-[0.3px]">
          Discover
        </p>
        <p className="text-xs text-foreground/60 m-0 tracking-[0.1px]">
          Your scores at a glance
        </p>
      </div>

      {/* ── Emotions Overview -- combined stress, focus, mood graph ── */}
      <EmotionsOverview userId={userId} navigate={navigate} checkin={checkin} />


      {/* ── Emotion Timeline -- color-coded dots for last 7 days ── */}
      <EmotionTimeline userId={userId} />

      {/* ── Mood Insights -- pattern detection from emotion history ── */}
      <MoodInsightsCard userId={userId} />

      {/* ── Section label ── */}
      <div className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-3">
        Explore
      </div>

      {/* ── 2-column navigation grid ── */}
      <div
        role="navigation"
        aria-label="Explore features"
        className="grid grid-cols-2 gap-2.5 mb-5"
      >
        {NAV_CARDS.map((card, index) => (
          <motion.button
            key={card.route}
            custom={index}
            variants={cardVariants}
            initial="hidden"
            animate="visible"
            onClick={() => navigate(card.route)}
            aria-label={`${card.title}: ${card.subtitle}`}
            className="rounded-[14px] bg-card border border-border p-4 text-left cursor-pointer w-full"
            style={{
              borderLeft: `3px solid ${card.accentColor}`,
              WebkitTapHighlightColor: "transparent",
            }}
          >
            <card.icon className="w-7 h-7 mb-2" style={{ color: card.accentColor }} aria-hidden="true" />
            <p className="text-sm font-bold text-foreground m-0 mb-1 -tracking-[0.2px]">
              {card.title}
            </p>
            <p className="text-[10px] text-foreground/35 m-0 leading-relaxed">
              {card.subtitle}
            </p>
          </motion.button>
        ))}
      </div>
    </motion.main>
  );
}

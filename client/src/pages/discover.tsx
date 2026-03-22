import { useState, useEffect, useMemo } from "react";
import { useLocation } from "wouter";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { pageTransition, cardVariants } from "@/lib/animations";
import { resolveUrl } from "@/lib/queryClient";
import { getParticipantId } from "@/lib/participant";
import { useHealthSync } from "@/hooks/use-health-sync";
import { useFusedState } from "@/hooks/use-fused-state";
import { ConfidenceMeter } from "@/components/confidence-meter";
import { InterventionSuggestion } from "@/components/intervention-suggestion";
import { detectMoodPatterns, type EmotionReading, type MoodInsight } from "@/lib/mood-patterns";
import { listSessions, type SessionSummary } from "@/lib/ml-api";
import {
  saveEmotionHistory as sbSaveEmotionHistory,
  getEmotionHistory as sbGetEmotionHistory,
} from "@/lib/supabase-store";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
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
  happy: "#0891b2", sad: "#6366f1", angry: "#ea580c", fear: "#7c3aed",
  surprise: "#d4a017", neutral: "#94a3b8",
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
    const raw = localStorage.getItem(EMOTION_HISTORY_KEY);
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

    localStorage.setItem(EMOTION_HISTORY_KEY, JSON.stringify(pruned));

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
    const raw = localStorage.getItem(EMOTION_HISTORY_KEY);
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
        const raw = localStorage.getItem("ndw_last_emotion");
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
  happy: "#0891b2", sad: "#6366f1", angry: "#ea580c", fear: "#7c3aed",
  surprise: "#d4a017", neutral: "#94a3b8",
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
    <div style={{
      background: "linear-gradient(135deg, var(--card) 0%, rgba(124,58,237,0.03) 100%)",
      border: "1px solid rgba(255,255,255,0.08)",
      borderRadius: 20, padding: "14px 16px", marginBottom: 14,
      boxShadow: "0 2px 16px rgba(0,0,0,0.06), 0 0 0 0.5px rgba(255,255,255,0.04)",
    }}>
      <div style={{
        fontSize: 11, fontWeight: 700, color: "var(--muted-foreground)",
        textTransform: "uppercase" as const, letterSpacing: "0.8px", marginBottom: 10,
      }}>
        Your week in emotions
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        {days.map(([key, { emotion, label }]) => {
          const color = TIMELINE_COLORS[emotion] ?? "#94a3b8";
          return (
            <div key={key} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
              <div style={{
                width: 28, height: 28, borderRadius: "50%", background: color,
                opacity: 0.85, transition: "transform 0.3s cubic-bezier(0.22, 1, 0.36, 1)",
              }} />
              <span style={{ fontSize: 9, color: "var(--muted-foreground)" }}>{label}</span>
            </div>
          );
        })}
      </div>
      {days.length >= 3 && (
        <div style={{ fontSize: 10, color: "var(--muted-foreground)", marginTop: 8, textAlign: "center" }}>
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

// ── Recommended Section — personalized suggestions based on emotion ────────

interface Recommendation {
  icon: LucideIcon;
  iconColor: string;
  title: string;
  reason: string;
  route: string;
}

// All possible features for progressive discovery
const ALL_FEATURES = [
  { route: "/biofeedback", title: "Breathing Exercise", icon: Wind as LucideIcon, iconColor: "#0891b2", category: "calm" },
  { route: "/sleep-session", title: "Sleep Music", icon: Moon as LucideIcon, iconColor: "#7c3aed", category: "calm" },
  { route: "/inner-energy", title: "Inner Energy", icon: Sparkles as LucideIcon, iconColor: "#4ade80", category: "energy" },
  { route: "/mood", title: "Mood Trends", icon: Smile as LucideIcon, iconColor: "#0891b2", category: "insight" },
  { route: "/neurofeedback", title: "Neurofeedback", icon: Target as LucideIcon, iconColor: "#6366f1", category: "focus" },
  { route: "/dreams", title: "Dream Journal", icon: BookOpen as LucideIcon, iconColor: "#a78bfa", category: "insight" },
  { route: "/workout", title: "Workout", icon: Dumbbell as LucideIcon, iconColor: "#ea580c", category: "energy" },
  { route: "/brain-monitor", title: "Brain Monitor", icon: Brain as LucideIcon, iconColor: "#6366f1", category: "insight" },
  { route: "/habits", title: "Habit Tracker", icon: BarChart3 as LucideIcon, iconColor: "#d4a017", category: "insight" },
  { route: "/ai-companion", title: "AI Companion", icon: MessageCircle as LucideIcon, iconColor: "#0891b2", category: "support" },
  { route: "/insights", title: "Wellness Insights", icon: Lightbulb as LucideIcon, iconColor: "#d4a017", category: "insight" },
];

function getUsedFeatures(): Set<string> {
  try {
    const raw = localStorage.getItem("ndw_feature_usage");
    return raw ? new Set(JSON.parse(raw)) : new Set();
  } catch {
    return new Set();
  }
}

function trackFeatureUsage(route: string) {
  try {
    const used = getUsedFeatures();
    used.add(route);
    localStorage.setItem("ndw_feature_usage", JSON.stringify(Array.from(used)));
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
    recs.push({ icon: Wind, iconColor: "#0891b2", title: "Breathing Exercise", reason: "Your stress is elevated", route: "/biofeedback" });
    if (isEvening) {
      recs.push({ icon: Moon, iconColor: "#7c3aed", title: "Sleep Music", reason: "Wind down before bed", route: "/sleep-session" });
    }
  }
  if (valence < -0.1) {
    recs.push({ icon: MessageCircle, iconColor: "#0891b2", title: "AI Companion", reason: "Talk through how you're feeling", route: "/ai-companion" });
    recs.push({ icon: Sparkles, iconColor: "#4ade80", title: "Inner Energy", reason: "Rebalance your energy centers", route: "/inner-energy" });
  }
  if (focus < 0.4) {
    recs.push({ icon: Target, iconColor: "#6366f1", title: "Neurofeedback", reason: "Train your focus", route: "/neurofeedback" });
  }
  if (valence > 0.3 && stress < 0.3) {
    if (isMorning) {
      recs.push({ icon: Dumbbell, iconColor: "#ea580c", title: "Workout", reason: "Ride this morning energy", route: "/workout" });
    }
    recs.push({ icon: BookOpen, iconColor: "#a78bfa", title: "Dream Journal", reason: "Great mood -- capture your dreams", route: "/dreams" });
  }

  // Time-of-day suggestions (if we don't have enough yet)
  if (recs.length < 2) {
    if (isMorning) {
      recs.push({ icon: Lightbulb, iconColor: "#d4a017", title: "Wellness Insights", reason: "Start your day with awareness", route: "/insights" });
    } else if (isEvening) {
      recs.push({ icon: Moon, iconColor: "#7c3aed", title: "Sleep Music", reason: "Prepare for a restful night", route: "/sleep-session" });
    } else {
      recs.push({ icon: Smile, iconColor: "#0891b2", title: "Mood Trends", reason: "See your patterns", route: "/mood" });
    }
  }

  // Progressive discovery — suggest an unused feature
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
    recs.push({ icon: Brain, iconColor: "#6366f1", title: "Brain Monitor", reason: "Explore your brainwaves", route: "/brain-monitor" });
  }

  return recs.slice(0, 3);
}

function RecommendedSection({ stress, valence, focus, navigate }: {
  stress: number; valence: number; focus: number; navigate: (path: string) => void;
}) {
  const recs = getRecommendations(stress, valence, focus);
  if (recs.length === 0) return null;

  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{
        fontSize: 11, fontWeight: 700, color: "var(--muted-foreground)",
        textTransform: "uppercase" as const, letterSpacing: "0.8px", marginBottom: 8,
      }}>
        Recommended for you
      </div>
      <div style={{ display: "flex", gap: 8, overflowX: "auto" as const, paddingBottom: 4 }}>
        {recs.map((rec) => (
          <button
            key={rec.route}
            onClick={() => { trackFeatureUsage(rec.route); navigate(rec.route); }}
            style={{
              background: "linear-gradient(135deg, var(--card) 0%, rgba(124,58,237,0.03) 100%)",
              border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: 20, padding: "12px 14px", minWidth: 150, flexShrink: 0,
              textAlign: "left" as const, cursor: "pointer",
              boxShadow: "0 2px 16px rgba(0,0,0,0.06), 0 0 0 0.5px rgba(255,255,255,0.04)",
              transition: "transform 0.2s ease, box-shadow 0.2s ease",
            }}
          >
            <rec.icon style={{ width: 22, height: 22, color: rec.iconColor }} />
            <div style={{ fontSize: 12, fontWeight: 600, color: "var(--foreground)", marginTop: 6 }}>{rec.title}</div>
            <div style={{ fontSize: 10, color: "var(--muted-foreground)", marginTop: 2 }}>{rec.reason}</div>
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

  const insights = detectMoodPatterns(data);
  if (insights.length === 0) return null;

  const borderColors: Record<string, string> = {
    positive: "rgba(74, 222, 128, 0.2)",
    warning: "rgba(232, 185, 74, 0.2)",
    neutral: "var(--border)",
  };

  const INSIGHT_ICONS: Record<string, { Icon: LucideIcon; color: string }> = {
    "star": { Icon: Star, color: "#d4a017" },
    "heart": { Icon: Heart, color: "#6366f1" },
    "waves": { Icon: Waves, color: "#0891b2" },
    "trending-down": { Icon: TrendingDown, color: "#4ade80" },
    "trending-up": { Icon: TrendingUp, color: "#e879a8" },
    "sunrise": { Icon: Sunrise, color: "#d4a017" },
    "cloud-moon": { Icon: CloudMoonIcon, color: "#7c3aed" },
    "palette": { Icon: Palette, color: "#4ade80" },
    "refresh": { Icon: RefreshCw, color: "#94a3b8" },
  };

  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{
        fontSize: 11, fontWeight: 700, color: "var(--muted-foreground)",
        textTransform: "uppercase" as const, letterSpacing: "0.8px", marginBottom: 8,
      }}>
        Mood insights
      </div>
      <div style={{ display: "flex", flexDirection: "column" as const, gap: 8 }}>
        {insights.map((insight, i) => {
          const iconEntry = INSIGHT_ICONS[insight.icon];
          const IconComp = iconEntry?.Icon ?? Activity;
          const iconColor = iconEntry?.color ?? "var(--muted-foreground)";
          return (
          <div key={i} style={{
            background: "var(--card)",
            border: `1px solid ${borderColors[insight.type] ?? "var(--border)"}`,
            borderRadius: 12, padding: "10px 12px",
            display: "flex", alignItems: "flex-start", gap: 8,
          }}>
            <IconComp style={{ width: 18, height: 18, flexShrink: 0, color: iconColor, marginTop: 1 }} />
            <div>
              <div style={{ fontSize: 12, fontWeight: 600, color: "var(--foreground)" }}>{insight.title}</div>
              <div style={{ fontSize: 10, color: "var(--muted-foreground)", marginTop: 2, lineHeight: 1.4 }}>
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

// ── Data — 8 main navigation cards (2x4 grid) ─────────────────────────────

const NAV_CARDS: NavCard[] = [
  { icon: BedDouble,  title: "Sleep",        subtitle: "Sleep data, dreams, music",           route: "/sleep",          accentColor: "#7c3aed" },
  { icon: Brain,      title: "Brain",        subtitle: "EEG, neurofeedback, connectivity",   route: "/brain-monitor",  accentColor: "#6366f1" },
  { icon: Heart,      title: "Health",       subtitle: "Body metrics, workouts, scores",      route: "/health",         accentColor: "#e879a8" },
  { icon: Sparkles,   title: "Inner Energy", subtitle: "Energy centers, spiritual wellness",  route: "/inner-energy",   accentColor: "#4ade80" },
  { icon: Wind,       title: "Wellness",     subtitle: "Habits, menstrual cycle tracking",    route: "/wellness",       accentColor: "#ea580c" },
  { icon: HeartHandshake, title: "Couples Meditation", subtitle: "Brain synchrony with your partner", route: "/couples-meditation", accentColor: "#e879a8" },
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

// ── Emotions Overview — combined stress/focus/mood chart ──────────────────

interface HistoryRow {
  stress: number;
  happiness: number;
  focus: number;
  dominantEmotion: string;
  timestamp: string;
}

function EmotionsOverview({ userId, navigate, checkin }: { userId: string; navigate: (p: string) => void; checkin: CheckinData | null }) {
  const { data } = useQuery<HistoryRow[]>({
    queryKey: [`/api/brain/history/${userId}?days=7`],
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

  const chartData = useMemo(() => {
    // Use API data if available, otherwise fall back to localStorage history
    const rows: HistoryRow[] =
      data && Array.isArray(data) && data.length > 0
        ? data
        : getLocalEmotionHistory();

    // Group by day, average values
    const dayMap = new Map<string, { stress: number[]; focus: number[]; mood: number[]; ts: number }>();

    // Add rows from API / localStorage
    for (const r of rows) {
      const d = new Date(r.timestamp);
      const day = d.toLocaleDateString(undefined, { weekday: "short" });
      const ts = new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
      if (!dayMap.has(day)) dayMap.set(day, { stress: [], focus: [], mood: [], ts });
      const bucket = dayMap.get(day)!;
      bucket.stress.push((r.stress ?? 0) * 100);
      bucket.focus.push((r.focus ?? 0) * 100);
      bucket.mood.push((r.happiness ?? 0.5) * 100);
    }

    // Also incorporate session history data (listSessions) when available
    if (sessions && sessions.length > 0) {
      const weekAgo = Date.now() / 1000 - 7 * 86400;
      for (const s of sessions) {
        if (!s.summary || s.summary.avg_focus == null) continue;
        if ((s.start_time ?? 0) < weekAgo) continue;
        const d = new Date((s.start_time ?? 0) * 1000);
        const day = d.toLocaleDateString(undefined, { weekday: "short" });
        const ts = new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
        if (!dayMap.has(day)) dayMap.set(day, { stress: [], focus: [], mood: [], ts });
        const bucket = dayMap.get(day)!;
        bucket.stress.push((s.summary.avg_stress ?? 0) * 100);
        bucket.focus.push((s.summary.avg_focus ?? 0) * 100);
        // Use relaxation as a mood proxy (higher relaxation = more positive mood)
        bucket.mood.push((s.summary.avg_relaxation ?? 0.5) * 100);
      }
    }

    if (dayMap.size === 0) return [];

    return Array.from(dayMap.entries())
      .sort(([, a], [, b]) => a.ts - b.ts)
      .map(([day, v]) => ({
        day,
        stress: Math.round(v.stress.reduce((a, b) => a + b, 0) / v.stress.length),
        focus: Math.round(v.focus.reduce((a, b) => a + b, 0) / v.focus.length),
        mood: Math.round(v.mood.reduce((a, b) => a + b, 0) / v.mood.length),
      }));
  }, [data, sessions]);

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
      style={{
        width: "100%",
        background: "linear-gradient(135deg, var(--card) 0%, rgba(124,58,237,0.03) 100%)",
        border: "1px solid rgba(255,255,255,0.08)",
        borderRadius: 20, padding: 16, marginBottom: 14, cursor: "pointer",
        textAlign: "left" as const,
        boxShadow: "0 2px 16px rgba(0,0,0,0.06), 0 0 0 0.5px rgba(255,255,255,0.04)",
        transition: "transform 0.2s ease, box-shadow 0.2s ease",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <div>
          <p style={{ fontSize: 14, fontWeight: 600, color: "var(--foreground)", margin: 0 }}>Emotions</p>
          <p style={{ fontSize: 10, color: "var(--muted-foreground)", margin: "2px 0 0" }}>Stress, Focus, Mood — 7 day trends</p>
        </div>
        {current && (
          <div style={{ display: "flex", gap: 10 }}>
            <div style={{ textAlign: "center" }}>
              <div style={{ fontSize: 16, fontWeight: 700, color: "#e879a8" }}>{current.stress}%</div>
              <div style={{ fontSize: 9, color: "var(--muted-foreground)" }}>Stress</div>
              {trends?.stressDelta != null && Math.abs(trends.stressDelta) > 2 && (
                <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 2, marginTop: 2 }}>
                  {trends.stressDelta > 0 ? (
                    <TrendingUp style={{ width: 12, height: 12, color: "#e879a8" }} />
                  ) : (
                    <TrendingDown style={{ width: 12, height: 12, color: "#0891b2" }} />
                  )}
                  <span style={{ fontSize: 8, color: "var(--muted-foreground)" }}>
                    {Math.abs(Math.round(trends.stressDelta))}% vs last
                  </span>
                </div>
              )}
            </div>
            <div style={{ textAlign: "center" }}>
              <div style={{ fontSize: 16, fontWeight: 700, color: "#6366f1" }}>{current.focus}%</div>
              <div style={{ fontSize: 9, color: "var(--muted-foreground)" }}>Focus</div>
              {trends?.focusDelta != null && Math.abs(trends.focusDelta) > 2 && (
                <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 2, marginTop: 2 }}>
                  {trends.focusDelta > 0 ? (
                    <TrendingUp style={{ width: 12, height: 12, color: "#0891b2" }} />
                  ) : (
                    <TrendingDown style={{ width: 12, height: 12, color: "#e879a8" }} />
                  )}
                  <span style={{ fontSize: 8, color: "var(--muted-foreground)" }}>
                    {Math.abs(Math.round(trends.focusDelta))}% vs last
                  </span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {chartData.length > 1 ? (
        <div style={{ height: 220 }}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ left: 0, right: 4, top: 4, bottom: 0 }}>
              <defs>
                <linearGradient id="discStressG" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#e879a8" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#e879a8" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="discFocusG" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#6366f1" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#6366f1" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="discMoodG" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#0891b2" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#0891b2" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" strokeOpacity={0.5} />
              <XAxis dataKey="day" tick={{ fontSize: 10, fill: "var(--muted-foreground)" }} axisLine={false} tickLine={false} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 9, fill: "var(--muted-foreground)" }} axisLine={false} tickLine={false} width={28} tickFormatter={(v) => `${v}`} />
              <Tooltip
                contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", borderRadius: 12, fontSize: 11 }}
                labelStyle={{ color: "var(--muted-foreground)" }}
                formatter={(v: number, name: string) => [`${v}%`, name]}
              />
              <Area type="monotone" dataKey="stress" stroke="#e879a8" fill="url(#discStressG)" strokeWidth={2.5} dot={{ r: 3, fill: "#e879a8" }} activeDot={{ r: 5 }} name="Stress" isAnimationActive={true} animationDuration={1200} animationEasing="ease-out" />
              <Area type="monotone" dataKey="focus" stroke="#6366f1" fill="url(#discFocusG)" strokeWidth={2.5} dot={{ r: 3, fill: "#6366f1" }} activeDot={{ r: 5 }} name="Focus" isAnimationActive={true} animationDuration={1200} animationEasing="ease-out" />
              <Area type="monotone" dataKey="mood" stroke="#0891b2" fill="url(#discMoodG)" strokeWidth={2.5} dot={{ r: 3, fill: "#0891b2" }} activeDot={{ r: 5 }} name="Mood" isAnimationActive={true} animationDuration={1200} animationEasing="ease-out" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div style={{ height: 80, display: "flex", alignItems: "center", justifyContent: "center" }}>
          <p style={{ fontSize: 11, color: "var(--muted-foreground)" }}>Complete a voice analysis to see trends</p>
        </div>
      )}

      {/* Legend */}
      <div style={{ display: "flex", gap: 14, marginTop: 8 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <div style={{ width: 8, height: 3, borderRadius: 2, background: "#e879a8" }} />
          <span style={{ fontSize: 9, color: "var(--muted-foreground)" }}>Stress</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <div style={{ width: 8, height: 3, borderRadius: 2, background: "#6366f1" }} />
          <span style={{ fontSize: 9, color: "var(--muted-foreground)" }}>Focus</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <div style={{ width: 8, height: 3, borderRadius: 2, background: "#0891b2" }} />
          <span style={{ fontSize: 9, color: "var(--muted-foreground)" }}>Mood</span>
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
    queryKey: [resolveUrl(`/api/food/logs/${userId}`)],
    retry: false,
  });

  // Use fused state when available, fall back to checkin data
  const emotion = fusedState?.emotion ?? checkin?.emotion ?? "—";
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
      style={{
        background: "var(--background)",
        padding: 16,
        paddingBottom: 16,
        fontFamily: "system-ui, -apple-system, sans-serif",
      }}
    >
      {/* ── Header ── */}
      <div style={{ marginBottom: 18 }}>
        <p style={{
          fontSize: 20, fontWeight: 700, color: "var(--foreground)", margin: "0 0 4px 0",
          letterSpacing: "-0.3px",
        }}>
          Discover
        </p>
        <p style={{ fontSize: 12, color: "var(--muted-foreground)", margin: 0, letterSpacing: "0.1px" }}>
          Your scores at a glance
        </p>
      </div>

      {/* ── Emotions Overview — combined stress, focus, mood graph ── */}
      <EmotionsOverview userId={userId} navigate={navigate} checkin={checkin} />

      {/* ── Confidence + Intervention below emotion section ── */}
      {hasData && (
        <div style={{ marginBottom: 14, display: "flex", flexDirection: "column", gap: 10 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ fontSize: 11, color: "var(--muted-foreground)", whiteSpace: "nowrap" }}>
              Confidence
            </span>
            <div style={{ flex: 1 }}>
              <ConfidenceMeter confidence={checkin?.confidence ?? 0.5} size="sm" />
            </div>
          </div>
          <InterventionSuggestion
            emotion={emotion}
            stressIndex={stress}
            valence={valence}
            compact
          />
        </div>
      )}

      {/* ── Emotion Timeline — color-coded dots for last 7 days ── */}
      <EmotionTimeline userId={userId} />

      {/* ── Mood Insights — pattern detection from emotion history ── */}
      <MoodInsightsCard userId={userId} />

      {/* ── Section label ── */}
      <div style={{
        fontSize: 11, fontWeight: 700, color: "var(--muted-foreground)", textTransform: "uppercase" as const,
        letterSpacing: "0.8px", marginBottom: 12,
      }}>
        Explore
      </div>

      {/* ── 2-column navigation grid ── */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 10,
          marginBottom: 20,
        }}
      >
        {NAV_CARDS.map((card, index) => (
          <motion.button
            key={card.route}
            custom={index}
            variants={cardVariants}
            initial="hidden"
            animate="visible"
            onClick={() => navigate(card.route)}
            style={{
              background: `linear-gradient(135deg, var(--card) 0%, ${card.accentColor}06 100%)`,
              border: "1px solid rgba(255,255,255,0.08)",
              borderLeft: `3px solid ${card.accentColor}`,
              borderRadius: 20,
              padding: 16,
              textAlign: "left",
              cursor: "pointer",
              width: "100%",
              WebkitTapHighlightColor: "transparent",
              boxShadow: `0 2px 16px rgba(0,0,0,0.06), 0 0 0 0.5px rgba(255,255,255,0.04)`,
              transition: "transform 0.2s ease, box-shadow 0.2s ease",
            }}
          >
            <card.icon style={{ width: 28, height: 28, marginBottom: 8, color: card.accentColor }} />
            <p
              style={{
                fontSize: 14,
                fontWeight: 700,
                color: "var(--foreground)",
                margin: "0 0 4px 0",
                letterSpacing: "-0.2px",
              }}
            >
              {card.title}
            </p>
            <p style={{ fontSize: 10, color: "var(--muted-foreground)", margin: 0, lineHeight: 1.4 }}>
              {card.subtitle}
            </p>
          </motion.button>
        ))}
      </div>
    </motion.main>
  );
}

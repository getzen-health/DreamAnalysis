import { useState, useEffect, useMemo } from "react";
import { useLocation } from "wouter";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { pageTransition, cardVariants } from "@/lib/animations";
import { resolveUrl } from "@/lib/queryClient";
import { getParticipantId } from "@/lib/participant";
import { useHealthSync } from "@/hooks/use-health-sync";
import { detectMoodPatterns, type EmotionReading, type MoodInsight } from "@/lib/mood-patterns";
import { CommunityMood } from "@/components/community-mood";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from "recharts";

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

function useCheckinData(): CheckinData | null {
  const [data, setData] = useState<CheckinData | null>(null);
  useEffect(() => {
    try {
      const raw = localStorage.getItem("ndw_last_emotion");
      if (raw) {
        const parsed = JSON.parse(raw);
        setData(parsed?.result ?? parsed);
      }
    } catch { /* ignore */ }
    const handler = () => {
      try {
        const raw = localStorage.getItem("ndw_last_emotion");
        if (raw) setData(JSON.parse(raw)?.result ?? JSON.parse(raw));
      } catch { /* ignore */ }
    };
    window.addEventListener("ndw-voice-updated", handler);
    window.addEventListener("ndw-emotion-update", handler);
    return () => {
      window.removeEventListener("ndw-voice-updated", handler);
      window.removeEventListener("ndw-emotion-update", handler);
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
      background: "var(--card)", border: "1px solid var(--border)",
      borderRadius: 20, padding: "14px 16px", marginBottom: 14,
      boxShadow: "0 2px 16px rgba(0,0,0,0.06)",
    }}>
      <div style={{
        fontSize: 11, fontWeight: 600, color: "var(--muted-foreground)",
        textTransform: "uppercase" as const, letterSpacing: "0.5px", marginBottom: 10,
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
  emoji: string;
  title: string;
  reason: string;
  route: string;
}

// All possible features for progressive discovery
const ALL_FEATURES = [
  { route: "/biofeedback", title: "Breathing Exercise", emoji: "🧘", category: "calm" },
  { route: "/sleep-session", title: "Sleep Music", emoji: "🎵", category: "calm" },
  { route: "/inner-energy", title: "Inner Energy", emoji: "✨", category: "energy" },
  { route: "/mood", title: "Mood Trends", emoji: "😊", category: "insight" },
  { route: "/neurofeedback", title: "Neurofeedback", emoji: "🎯", category: "focus" },
  { route: "/dreams", title: "Dream Journal", emoji: "📝", category: "insight" },
  { route: "/workout", title: "Workout", emoji: "🏋️", category: "energy" },
  { route: "/brain-monitor", title: "Brain Monitor", emoji: "🧠", category: "insight" },
  { route: "/habits", title: "Habit Tracker", emoji: "📊", category: "insight" },
  { route: "/ai-companion", title: "AI Companion", emoji: "💬", category: "support" },
  { route: "/insights", title: "Wellness Insights", emoji: "💡", category: "insight" },
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
    recs.push({ emoji: "🧘", title: "Breathing Exercise", reason: "Your stress is elevated", route: "/biofeedback" });
    if (isEvening) {
      recs.push({ emoji: "🎵", title: "Sleep Music", reason: "Wind down before bed", route: "/sleep-session" });
    }
  }
  if (valence < -0.1) {
    recs.push({ emoji: "💬", title: "AI Companion", reason: "Talk through how you're feeling", route: "/ai-companion" });
    recs.push({ emoji: "✨", title: "Inner Energy", reason: "Rebalance your energy centers", route: "/inner-energy" });
  }
  if (focus < 0.4) {
    recs.push({ emoji: "🎯", title: "Neurofeedback", reason: "Train your focus", route: "/neurofeedback" });
  }
  if (valence > 0.3 && stress < 0.3) {
    if (isMorning) {
      recs.push({ emoji: "🏋️", title: "Workout", reason: "Ride this morning energy", route: "/workout" });
    }
    recs.push({ emoji: "📝", title: "Dream Journal", reason: "Great mood — capture your dreams", route: "/dreams" });
  }

  // Time-of-day suggestions (if we don't have enough yet)
  if (recs.length < 2) {
    if (isMorning) {
      recs.push({ emoji: "💡", title: "Wellness Insights", reason: "Start your day with awareness", route: "/insights" });
    } else if (isEvening) {
      recs.push({ emoji: "🎵", title: "Sleep Music", reason: "Prepare for a restful night", route: "/sleep-session" });
    } else {
      recs.push({ emoji: "😊", title: "Mood Trends", reason: "See your patterns", route: "/mood" });
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
        emoji: pick.emoji,
        title: pick.title,
        reason: "Try something new",
        route: pick.route,
      });
    }
  }

  // Fallback
  if (recs.length === 0) {
    recs.push({ emoji: "🧠", title: "Brain Monitor", reason: "Explore your brainwaves", route: "/brain-monitor" });
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
        fontSize: 11, fontWeight: 600, color: "var(--muted-foreground)",
        textTransform: "uppercase" as const, letterSpacing: "0.5px", marginBottom: 8,
      }}>
        Recommended for you
      </div>
      <div style={{ display: "flex", gap: 8, overflowX: "auto" as const, paddingBottom: 4 }}>
        {recs.map((rec) => (
          <button
            key={rec.route}
            onClick={() => { trackFeatureUsage(rec.route); navigate(rec.route); }}
            style={{
              background: "var(--card)", border: "1px solid var(--border)",
              borderRadius: 20, padding: "12px 14px", minWidth: 150, flexShrink: 0,
              textAlign: "left" as const, cursor: "pointer",
              boxShadow: "0 2px 16px rgba(0,0,0,0.06)",
              transition: "transform 0.2s ease, box-shadow 0.2s ease",
            }}
          >
            <span style={{ fontSize: 22 }}>{rec.emoji}</span>
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

  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{
        fontSize: 11, fontWeight: 600, color: "var(--muted-foreground)",
        textTransform: "uppercase" as const, letterSpacing: "0.5px", marginBottom: 8,
      }}>
        Mood insights
      </div>
      <div style={{ display: "flex", flexDirection: "column" as const, gap: 8 }}>
        {insights.map((insight, i) => (
          <div key={i} style={{
            background: "var(--card)",
            border: `1px solid ${borderColors[insight.type] ?? "var(--border)"}`,
            borderRadius: 12, padding: "10px 12px",
            display: "flex", alignItems: "flex-start", gap: 8,
          }}>
            <span style={{ fontSize: 18, flexShrink: 0, lineHeight: 1.2 }}>{insight.emoji}</span>
            <div>
              <div style={{ fontSize: 12, fontWeight: 600, color: "var(--foreground)" }}>{insight.title}</div>
              <div style={{ fontSize: 10, color: "var(--muted-foreground)", marginTop: 2, lineHeight: 1.4 }}>
                {insight.description}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Types ──────────────────────────────────────────────────────────────────

interface NavCard {
  emoji: string;
  title: string;
  subtitle: string;
  route: string;
  accentColor: string;
}

// ── Data — 8 main navigation cards (2x4 grid) ─────────────────────────────

const NAV_CARDS: NavCard[] = [
  { emoji: "😴", title: "Sleep",        subtitle: "Sleep data, dreams, music",           route: "/sleep",          accentColor: "#7c3aed" },
  { emoji: "🧠", title: "Brain",        subtitle: "EEG, neurofeedback, connectivity",   route: "/brain-monitor",  accentColor: "#6366f1" },
  { emoji: "💪", title: "Health",       subtitle: "Body metrics, workouts, scores",      route: "/health",         accentColor: "#e879a8" },
  { emoji: "✨", title: "Inner Energy", subtitle: "Energy centers, spiritual wellness",  route: "/inner-energy",   accentColor: "#4ade80" },
  { emoji: "🧘", title: "Wellness",     subtitle: "Habits, menstrual cycle tracking",    route: "/wellness",       accentColor: "#ea580c" },
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

function EmotionsOverview({ userId, navigate }: { userId: string; navigate: (p: string) => void }) {
  const { data } = useQuery<HistoryRow[]>({
    queryKey: [`/api/brain/history/${userId}?days=7`],
    retry: false,
    staleTime: 5 * 60 * 1000,
  });

  const chartData = useMemo(() => {
    if (!data || !Array.isArray(data) || data.length === 0) return [];
    // Group by day, average values
    const dayMap = new Map<string, { stress: number[]; focus: number[]; mood: number[] }>();
    for (const r of data) {
      const day = new Date(r.timestamp).toLocaleDateString(undefined, { weekday: "short" });
      if (!dayMap.has(day)) dayMap.set(day, { stress: [], focus: [], mood: [] });
      const d = dayMap.get(day)!;
      d.stress.push((r.stress ?? 0) * 100);
      d.focus.push((r.focus ?? 0) * 100);
      d.mood.push((r.happiness ?? 0.5) * 100);
    }
    return Array.from(dayMap.entries()).map(([day, v]) => ({
      day,
      stress: Math.round(v.stress.reduce((a, b) => a + b, 0) / v.stress.length),
      focus: Math.round(v.focus.reduce((a, b) => a + b, 0) / v.focus.length),
      mood: Math.round(v.mood.reduce((a, b) => a + b, 0) / v.mood.length),
    }));
  }, [data]);

  // Current values from localStorage
  const current = useMemo(() => {
    try {
      const raw = localStorage.getItem("ndw_last_emotion");
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      const r = parsed?.result ?? parsed;
      return {
        stress: Math.round((r?.stress_index ?? 0) * 100),
        focus: Math.round((r?.focus_index ?? 0) * 100),
        mood: r?.emotion ?? "neutral",
      };
    } catch { return null; }
  }, []);

  return (
    <button
      onClick={() => navigate("/mood")}
      style={{
        width: "100%", background: "var(--card)", border: "1px solid var(--border)",
        borderRadius: 20, padding: 16, marginBottom: 14, cursor: "pointer",
        textAlign: "left" as const, boxShadow: "0 2px 16px rgba(0,0,0,0.06)",
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
            </div>
            <div style={{ textAlign: "center" }}>
              <div style={{ fontSize: 16, fontWeight: 700, color: "#6366f1" }}>{current.focus}%</div>
              <div style={{ fontSize: 9, color: "var(--muted-foreground)" }}>Focus</div>
            </div>
          </div>
        )}
      </div>

      {chartData.length > 1 ? (
        <div style={{ height: 140 }}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="discStressG" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#e879a8" stopOpacity={0.25} />
                  <stop offset="100%" stopColor="#e879a8" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="discFocusG" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#6366f1" stopOpacity={0.25} />
                  <stop offset="100%" stopColor="#6366f1" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="discMoodG" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#0891b2" stopOpacity={0.25} />
                  <stop offset="100%" stopColor="#0891b2" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" strokeOpacity={0.5} />
              <XAxis dataKey="day" tick={{ fontSize: 10, fill: "var(--muted-foreground)" }} axisLine={false} tickLine={false} />
              <YAxis domain={[0, 100]} hide />
              <Tooltip
                contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", borderRadius: 12, fontSize: 11 }}
                labelStyle={{ color: "var(--muted-foreground)" }}
              />
              <Area type="monotone" dataKey="stress" stroke="#e879a8" fill="url(#discStressG)" strokeWidth={2.5} dot={false} name="Stress" />
              <Area type="monotone" dataKey="focus" stroke="#6366f1" fill="url(#discFocusG)" strokeWidth={2.5} dot={false} name="Focus" />
              <Area type="monotone" dataKey="mood" stroke="#0891b2" fill="url(#discMoodG)" strokeWidth={2.5} dot={false} name="Mood" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div style={{ height: 80, display: "flex", alignItems: "center", justifyContent: "center" }}>
          <p style={{ fontSize: 11, color: "var(--muted-foreground)" }}>Do a voice analysis to see emotion trends</p>
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
  const userId = getParticipantId();

  // Fetch today's food logs for calorie sum
  const { data: foodLogs } = useQuery<{ calories?: number }[]>({
    queryKey: [resolveUrl(`/api/food/logs/${userId}`)],
    retry: false,
  });

  const emotion = checkin?.emotion ?? "—";
  const emoColor = EMOTION_COLOR[emotion] ?? "var(--muted-foreground)";
  const stress = checkin?.stress_index ?? 0;
  const focus = checkin?.focus_index ?? 0;
  const relaxation = checkin?.relaxation_index ?? (1 - stress);
  const valence = checkin?.valence ?? 0;
  const hasData = !!checkin?.emotion;

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
        minHeight: "100vh",
        padding: 16,
        paddingBottom: 100,
        fontFamily: "system-ui, -apple-system, sans-serif",
      }}
    >
      {/* ── Header ── */}
      <div style={{ marginBottom: 16 }}>
        <p style={{ fontSize: 18, fontWeight: 600, color: "var(--foreground)", margin: "0 0 3px 0" }}>
          Discover
        </p>
        <p style={{ fontSize: 12, color: "var(--muted-foreground)", margin: 0 }}>
          Your scores at a glance
        </p>
      </div>

      {/* ── Emotions Overview — combined stress, focus, mood graph ── */}
      <EmotionsOverview userId={userId} navigate={navigate} />

      {/* ── Emotion Timeline — color-coded dots for last 7 days ── */}
      <EmotionTimeline userId={userId} />

      {/* ── Mood Insights — pattern detection from emotion history ── */}
      <MoodInsightsCard userId={userId} />

      {/* ── Community Mood — anonymous peer support ── */}
      <CommunityMood />

      {/* ── Section label ── */}
      <div style={{
        fontSize: 11, fontWeight: 600, color: "var(--muted-foreground)", textTransform: "uppercase" as const,
        letterSpacing: "0.5px", marginBottom: 10,
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
              background: "var(--card)",
              border: "1px solid var(--border)",
              borderLeft: `3px solid ${card.accentColor}`,
              borderRadius: 20,
              padding: 16,
              textAlign: "left",
              cursor: "pointer",
              width: "100%",
              WebkitTapHighlightColor: "transparent",
              boxShadow: "0 2px 16px rgba(0,0,0,0.06)",
              transition: "transform 0.2s ease, box-shadow 0.2s ease",
            }}
          >
            <div style={{ fontSize: 28, marginBottom: 8 }}>{card.emoji}</div>
            <p
              style={{
                fontSize: 13,
                fontWeight: 600,
                color: "var(--foreground)",
                margin: "0 0 3px 0",
              }}
            >
              {card.title}
            </p>
            <p style={{ fontSize: 10, color: "var(--muted-foreground)", margin: 0 }}>
              {card.subtitle}
            </p>
          </motion.button>
        ))}
      </div>
    </motion.main>
  );
}

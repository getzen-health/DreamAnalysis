/**
 * MoodTrends -- Mood detail page showing emotional state over time.
 *
 * Shows:
 * 1. Hero mood score (0-100) derived from valence: (valence + 1) * 50
 * 2. Time range tabs: Today | Week | Month
 * 3. Smooth AreaChart of mood score over time
 * 4. Morning / Afternoon / Evening averages
 * 5. Recent emotion labels
 *
 * Data: /api/brain/history/:userId?days=30, localStorage ndw_last_emotion
 */

import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { pageTransition, cardVariants } from "@/lib/animations";
import { getParticipantId } from "@/lib/participant";
import { useCurrentEmotion } from "@/hooks/use-current-emotion";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { Smile, Sun, Sunset, Moon } from "lucide-react";

/* ---------- constants ---------- */

const MOOD_CYAN = "#0891b2";
const MOOD_AMBER = "#d4a017";
const MOOD_ROSE = "#e879a8";

type TimeRange = "today" | "week" | "month";

/* ---------- types ---------- */

interface HistoryEntry {
  stress: number;
  happiness: number;
  focus: number;
  dominantEmotion: string;
  valence: number | null;
  timestamp: string;
}

/* ---------- helpers ---------- */

function valenceToMood(valence: number | null): number {
  if (valence == null) return 50;
  return Math.round((valence + 1) * 50);
}

function getMoodColor(score: number): string {
  if (score >= 65) return MOOD_CYAN;
  if (score >= 40) return MOOD_AMBER;
  return MOOD_ROSE;
}

function getMoodLabel(score: number): string {
  if (score >= 80) return "Great";
  if (score >= 65) return "Good";
  if (score >= 45) return "Okay";
  if (score >= 30) return "Low";
  return "Poor";
}

function filterByRange(entries: HistoryEntry[], range: TimeRange): HistoryEntry[] {
  const now = Date.now();
  return entries.filter((e) => {
    const ts = new Date(e.timestamp).getTime();
    switch (range) {
      case "today":
        return new Date(ts).toDateString() === new Date().toDateString();
      case "week":
        return now - ts < 7 * 86_400_000;
      case "month":
      default:
        return true;
    }
  });
}

function buildChartData(
  entries: HistoryEntry[],
  range: TimeRange,
): { time: string; value: number }[] {
  if (entries.length === 0) return [];

  if (range === "today") {
    return entries.map((e) => ({
      time: new Date(e.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      value: valenceToMood(e.valence),
    }));
  }

  // Group by day for week/month
  const dayMap = new Map<string, { values: number[]; ts: number }>();
  for (const e of entries) {
    const d = new Date(e.timestamp);
    const key =
      range === "week"
        ? d.toLocaleDateString("en-US", { weekday: "short" })
        : `${d.getMonth() + 1}/${d.getDate()}`;
    const ts = new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
    if (!dayMap.has(key)) dayMap.set(key, { values: [], ts });
    dayMap.get(key)!.values.push(valenceToMood(e.valence));
  }

  return Array.from(dayMap.entries())
    .sort(([, a], [, b]) => a.ts - b.ts)
    .map(([time, { values }]) => ({
      time,
      value: Math.round(values.reduce((a, b) => a + b, 0) / values.length),
    }));
}

function buildPeriodAverages(
  entries: HistoryEntry[],
): { period: string; score: number; icon: "sun" | "sunset" | "moon" }[] {
  const buckets: Record<string, number[]> = {
    Morning: [],
    Afternoon: [],
    Evening: [],
  };

  for (const e of entries) {
    const hour = new Date(e.timestamp).getHours();
    const score = valenceToMood(e.valence);
    if (hour >= 5 && hour < 12) buckets.Morning.push(score);
    else if (hour >= 12 && hour < 18) buckets.Afternoon.push(score);
    else buckets.Evening.push(score);
  }

  const icons: Record<string, "sun" | "sunset" | "moon"> = {
    Morning: "sun",
    Afternoon: "sunset",
    Evening: "moon",
  };

  return Object.entries(buckets)
    .filter(([, values]) => values.length > 0)
    .map(([period, values]) => ({
      period,
      score: Math.round(values.reduce((a, b) => a + b, 0) / values.length),
      icon: icons[period],
    }));
}

function PeriodIcon({ name }: { name: string }) {
  switch (name) {
    case "sun":
      return <Sun className="h-4 w-4 text-amber-400" />;
    case "sunset":
      return <Sunset className="h-4 w-4 text-orange-400" />;
    case "moon":
      return <Moon className="h-4 w-4 text-indigo-400" />;
    default:
      return <Smile className="h-4 w-4 text-muted-foreground" />;
  }
}

const EMOTION_COLORS: Record<string, string> = {
  happy: "#0891b2",
  sad: "#6366f1",
  angry: "#ea580c",
  fear: "#7c3aed",
  surprise: "#d4a017",
  neutral: "#94a3b8",
};

/* ---------- component ---------- */

export default function MoodTrends() {
  const userId = getParticipantId();
  const [range, setRange] = useState<TimeRange>("week");
  const { emotion: currentEmotion } = useCurrentEmotion();

  const { data } = useQuery<
    Array<{
      stress: number;
      happiness: number;
      focus: number;
      dominantEmotion: string;
      valence: number | null;
      timestamp: string;
    }>
  >({
    queryKey: [`/api/brain/history/${userId}?days=30`],
    retry: false,
    staleTime: 60_000,
  });

  const moodScore = currentEmotion?.valence != null
    ? valenceToMood(currentEmotion.valence)
    : null;

  const filtered = useMemo(
    () => filterByRange(data ?? [], range),
    [data, range],
  );

  const chartData = useMemo(
    () => buildChartData(filtered, range),
    [filtered, range],
  );

  const periodAverages = useMemo(
    () => buildPeriodAverages(filtered),
    [filtered],
  );

  // Recent emotion labels (last 10 unique entries)
  const recentEmotions = useMemo(() => {
    if (!data || data.length === 0) return [];
    const seen = new Set<string>();
    const result: { emotion: string; time: string }[] = [];
    for (let i = data.length - 1; i >= 0 && result.length < 8; i--) {
      const e = data[i];
      if (!e.dominantEmotion) continue;
      const key = `${e.dominantEmotion}-${new Date(e.timestamp).toDateString()}`;
      if (seen.has(key)) continue;
      seen.add(key);
      result.push({
        emotion: e.dominantEmotion,
        time: new Date(e.timestamp).toLocaleDateString(undefined, {
          weekday: "short",
          hour: "2-digit",
          minute: "2-digit",
        }),
      });
    }
    return result;
  }, [data]);

  const heroColor = moodScore !== null ? getMoodColor(moodScore) : "var(--muted-foreground)";
  const heroLabel = moodScore !== null ? getMoodLabel(moodScore) : "Unknown";

  return (
    <div className="max-w-2xl mx-auto px-4 py-6 space-y-5 pb-4">
      {/* Header */}
      <motion.div
        initial={pageTransition.initial}
        animate={pageTransition.animate}
        transition={pageTransition.transition}
      >
        <h1 className="text-xl font-bold tracking-tight text-foreground">
          Mood
        </h1>
        <p className="text-sm mt-1 text-muted-foreground">
          How you've been feeling over time
        </p>
      </motion.div>

      {/* Hero — mood score */}
      <motion.div
        className="rounded-2xl p-6 border border-border bg-card flex flex-col items-center"
        style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}
        initial={{ opacity: 0, y: 12, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
      >
        {moodScore !== null ? (
          <>
            <span
              className="text-5xl font-bold tabular-nums"
              style={{ color: heroColor }}
            >
              {moodScore}
            </span>
            <span className="text-xs text-muted-foreground mt-1">/ 100</span>
            <div
              className="text-sm font-semibold mt-2"
              style={{ color: heroColor }}
            >
              {heroLabel}
            </div>
            {currentEmotion?.emotion && (
              <span className="text-xs text-muted-foreground mt-1 capitalize">
                Feeling {currentEmotion.emotion}
              </span>
            )}
          </>
        ) : (
          <div className="text-center py-4">
            <Smile className="h-10 w-10 mx-auto text-muted-foreground/30 mb-3" />
            <p className="text-sm text-muted-foreground">
              Complete a check-in to see your mood score
            </p>
          </div>
        )}
      </motion.div>

      {/* Time range tabs */}
      <div className="flex gap-2">
        {(["today", "week", "month"] as TimeRange[]).map((r) => (
          <button
            key={r}
            onClick={() => setRange(r)}
            className={`flex-1 py-2 rounded-xl text-xs font-semibold transition-colors ${
              range === r
                ? "bg-primary text-primary-foreground"
                : "bg-muted/40 text-muted-foreground hover:bg-muted/60"
            }`}
          >
            {r === "today" ? "Today" : r === "week" ? "Week" : "Month"}
          </button>
        ))}
      </div>

      {/* Mood chart */}
      <motion.div
        className="rounded-2xl p-4 border border-border bg-card"
        style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}
        custom={1}
        initial="hidden"
        animate="visible"
        variants={cardVariants}
      >
        <div className="flex items-center gap-2 mb-4">
          <Smile className="h-4 w-4" style={{ color: MOOD_CYAN }} />
          <span className="text-sm font-semibold text-foreground">
            Mood Over Time
          </span>
        </div>

        {chartData.length >= 2 ? (
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart
              data={chartData}
              margin={{ left: 0, right: 4, top: 4, bottom: 0 }}
            >
              <defs>
                <linearGradient id="moodGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={MOOD_CYAN} stopOpacity={0.3} />
                  <stop offset="50%" stopColor={MOOD_CYAN} stopOpacity={0.12} />
                  <stop offset="100%" stopColor={MOOD_ROSE} stopOpacity={0.05} />
                </linearGradient>
              </defs>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="hsl(220,18%,14%)"
                opacity={0.5}
              />
              <XAxis
                dataKey="time"
                tick={{ fontSize: 9, fill: "hsl(220,12%,42%)" }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                domain={[0, 100]}
                tick={{ fontSize: 9, fill: "hsl(220,12%,42%)" }}
                axisLine={false}
                tickLine={false}
                width={24}
              />
              <Tooltip
                contentStyle={{
                  background: "hsl(220,18%,10%)",
                  border: "1px solid hsl(220,18%,20%)",
                  borderRadius: 10,
                  fontSize: 11,
                  color: "hsl(220,12%,80%)",
                }}
                formatter={(v: number) => [`${v}`, "Mood"]}
              />
              <Area
                type="natural"
                dataKey="value"
                name="Mood"
                stroke={MOOD_CYAN}
                fill="url(#moodGrad)"
                strokeWidth={2.5}
                dot={false}
                activeDot={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-32 flex items-center justify-center text-sm text-muted-foreground">
            Not enough data to show a trend yet
          </div>
        )}
      </motion.div>

      {/* Morning / Afternoon / Evening */}
      {periodAverages.length > 0 && (
        <motion.div
          className="rounded-2xl p-4 border border-border bg-card"
          style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}
          custom={2}
          initial="hidden"
          animate="visible"
          variants={cardVariants}
        >
          <span className="text-sm font-semibold text-foreground mb-3 block">
            Time of Day
          </span>
          <div className="space-y-3">
            {periodAverages.map((p) => {
              const color = getMoodColor(p.score);
              return (
                <div key={p.period} className="flex items-center gap-3">
                  <PeriodIcon name={p.icon} />
                  <span className="text-xs text-foreground w-20">{p.period}</span>
                  <div className="flex-1 bg-muted/30 rounded-full h-2">
                    <motion.div
                      className="h-2 rounded-full"
                      style={{ backgroundColor: color }}
                      initial={{ width: 0 }}
                      animate={{ width: `${p.score}%` }}
                      transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
                    />
                  </div>
                  <span
                    className="text-xs font-mono font-semibold w-10 text-right"
                    style={{ color }}
                  >
                    {p.score}
                  </span>
                </div>
              );
            })}
          </div>
        </motion.div>
      )}

      {/* Recent emotions */}
      {recentEmotions.length > 0 && (
        <motion.div
          className="rounded-2xl p-4 border border-border bg-card"
          style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}
          custom={3}
          initial="hidden"
          animate="visible"
          variants={cardVariants}
        >
          <span className="text-sm font-semibold text-foreground mb-3 block">
            Recent Feelings
          </span>
          <div className="flex flex-wrap gap-2">
            {recentEmotions.map((e, i) => (
              <div
                key={i}
                className="flex items-center gap-1.5 rounded-full px-3 py-1.5 bg-muted/30"
              >
                <span
                  className="inline-block w-2 h-2 rounded-full"
                  style={{
                    backgroundColor:
                      EMOTION_COLORS[e.emotion.toLowerCase()] ?? "#94a3b8",
                  }}
                />
                <span className="text-xs text-foreground capitalize">
                  {e.emotion}
                </span>
                <span className="text-[10px] text-muted-foreground">
                  {e.time}
                </span>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Empty state */}
      {chartData.length === 0 && moodScore === null && (
        <div
          className="rounded-2xl p-8 border border-border bg-card text-center"
          style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}
        >
          <Smile className="h-10 w-10 mx-auto text-muted-foreground/30 mb-3" />
          <p className="text-sm text-muted-foreground">
            No mood data yet. Complete a voice check-in to start tracking how
            you feel.
          </p>
        </div>
      )}
    </div>
  );
}

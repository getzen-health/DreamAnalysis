/**
 * MoodTrends -- Dedicated mood dashboard page.
 *
 * Shows ONLY mood/emotion content:
 * 1. Current detected emotion with label
 * 2. Mood valence score (positive/negative)
 * 3. 7-day mood trend line chart
 * 4. Emotion distribution bar chart
 * 5. Weekly mood summary text
 *
 * Data: useCurrentEmotion, /api/brain/history/:userId?days=7
 */

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { pageTransition, cardVariants } from "@/lib/animations";
import { useCurrentEmotion } from "@/hooks/use-current-emotion";
import { getParticipantId } from "@/lib/participant";
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { TrendingUp, BarChart3, Activity, Calendar } from "lucide-react";

/* ---------- constants ---------- */

const EMOTION_COLORS: Record<string, string> = {
  happy: "#0891b2",
  sad: "#6366f1",
  angry: "#ea580c",
  fear: "#7c3aed",
  fearful: "#7c3aed",
  surprise: "#d4a017",
  neutral: "#94a3b8",
};

const EMOTION_LABELS: Record<string, string> = {
  happy: "Happy",
  sad: "Sad",
  angry: "Angry",
  fear: "Fearful",
  fearful: "Fearful",
  surprise: "Surprised",
  neutral: "Neutral",
};

/* ---------- types ---------- */

interface HistoryEntry {
  dominantEmotion: string;
  timestamp: string;
  valence?: number;
  arousal?: number;
}

/* ---------- helpers ---------- */

function buildDistribution(
  data: HistoryEntry[]
): { name: string; count: number; color: string }[] {
  const counts: Record<string, number> = {};
  for (const d of data) {
    const e = d.dominantEmotion ?? "neutral";
    counts[e] = (counts[e] || 0) + 1;
  }
  return Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .map(([name, count]) => ({
      name: EMOTION_LABELS[name] ?? name,
      count,
      color: EMOTION_COLORS[name] ?? "#94a3b8",
    }));
}

function buildDailyValenceTrend(
  data: HistoryEntry[]
): { date: string; valence: number }[] {
  const dayMap = new Map<string, { values: number[]; ts: number }>();
  for (const r of data) {
    const d = new Date(r.timestamp);
    const key = d.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
    });
    const ts = new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
    if (!dayMap.has(key)) dayMap.set(key, { values: [], ts });
    const entry = dayMap.get(key)!;
    if (r.valence != null) entry.values.push(r.valence);
  }

  return Array.from(dayMap.entries())
    .sort(([, a], [, b]) => a.ts - b.ts)
    .slice(-7)
    .map(([date, { values }]) => ({
      date,
      valence: values.length
        ? Math.round(
            (values.reduce((a, b) => a + b, 0) / values.length) * 100
          ) / 100
        : 0,
    }));
}

function valenceLabel(v: number): string {
  if (v > 0.5) return "Very Positive";
  if (v > 0.2) return "Positive";
  if (v > -0.2) return "Neutral";
  if (v > -0.5) return "Slightly Negative";
  return "Negative";
}

function valenceColor(v: number): string {
  if (v > 0.2) return "#0891b2";
  if (v > -0.2) return "#94a3b8";
  return "#6366f1";
}

function generateWeeklySummary(
  data: HistoryEntry[],
  distribution: { name: string; count: number }[]
): string {
  if (data.length === 0) return "No mood data recorded this week.";

  const avgValence =
    data.filter((d) => d.valence != null).reduce((sum, d) => sum + (d.valence ?? 0), 0) /
    (data.filter((d) => d.valence != null).length || 1);

  const topEmotion = distribution[0]?.name ?? "neutral";
  const totalReadings = data.length;

  let tone = "balanced";
  if (avgValence > 0.3) tone = "positive";
  else if (avgValence > 0.1) tone = "slightly positive";
  else if (avgValence < -0.3) tone = "challenging";
  else if (avgValence < -0.1) tone = "slightly low";

  return `Over ${totalReadings} reading${totalReadings !== 1 ? "s" : ""} this week, your mood has been ${tone} overall. Your most frequent emotion was ${topEmotion}, with an average valence of ${avgValence >= 0 ? "+" : ""}${avgValence.toFixed(2)}.`;
}

/* ---------- custom tooltip ---------- */

interface ValenceTooltipProps {
  active?: boolean;
  payload?: Array<{ value: number }>;
  label?: string;
}

function ValenceTooltip({ active, payload, label }: ValenceTooltipProps) {
  if (!active || !payload || payload.length === 0) return null;
  const val = payload[0].value;
  return (
    <div className="rounded-lg bg-card border border-border px-3 py-2 text-xs shadow-lg">
      <p className="text-muted-foreground">{label}</p>
      <p className="text-foreground font-medium mt-0.5">
        Valence: {val >= 0 ? "+" : ""}
        {val.toFixed(2)}
      </p>
      <p className="text-muted-foreground mt-0.5">{valenceLabel(val)}</p>
    </div>
  );
}

/* ---------- component ---------- */

export default function MoodTrends() {
  const userId = getParticipantId();
  const { emotion: currentEmotion } = useCurrentEmotion();

  const { data: history } = useQuery<HistoryEntry[]>({
    queryKey: [`/api/brain/history/${userId}?days=7`],
    retry: false,
    staleTime: 5 * 60 * 1000,
  });

  const distribution = useMemo(
    () => (history ? buildDistribution(history) : []),
    [history]
  );
  const valenceTrend = useMemo(
    () => (history ? buildDailyValenceTrend(history) : []),
    [history]
  );
  const weeklySummary = useMemo(
    () => generateWeeklySummary(history ?? [], distribution),
    [history, distribution]
  );
  const hasHistory = history && history.length > 0;

  const emotionLabel = currentEmotion?.emotion ?? "neutral";
  const emotionColor = EMOTION_COLORS[emotionLabel] ?? "#94a3b8";
  const emotionDisplayName = EMOTION_LABELS[emotionLabel] ?? emotionLabel;

  return (
    <div className="max-w-2xl mx-auto px-4 py-6 space-y-5 pb-24">
      {/* Page header */}
      <motion.div
        initial={pageTransition.initial}
        animate={pageTransition.animate}
        transition={pageTransition.transition}
      >
        <h1 className="text-xl font-bold tracking-tight text-foreground">
          Mood Trends
        </h1>
        <p className="text-sm mt-1 text-muted-foreground">
          Your emotional patterns over time
        </p>
      </motion.div>

      {/* Current Emotion -- hero card */}
      <motion.div
        className="rounded-[14px] p-6 border border-border bg-card"
        initial={{ opacity: 0, y: 12, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
      >
        <div className="flex items-center gap-2 mb-4">
          <Activity className="h-4 w-4 text-cyan-500" />
          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Current Emotion
          </span>
        </div>

        {currentEmotion ? (
          <div className="space-y-4">
            <div className="flex items-center gap-4">
              <div
                className="w-14 h-14 rounded-2xl flex items-center justify-center shrink-0"
                style={{
                  background: `${emotionColor}18`,
                  border: `2px solid ${emotionColor}50`,
                }}
              >
                <span
                  className="w-6 h-6 rounded-full"
                  style={{ backgroundColor: emotionColor }}
                />
              </div>
              <div className="flex-1 min-w-0">
                <p
                  className="text-2xl font-bold capitalize"
                  style={{ color: emotionColor }}
                >
                  {emotionDisplayName}
                </p>
                <p className="text-xs text-muted-foreground mt-0.5">
                  {currentEmotion.confidence
                    ? `${Math.round(currentEmotion.confidence * 100)}% confidence`
                    : "Detected from latest analysis"}
                </p>
              </div>
            </div>

            {/* Valence indicator */}
            <div className="rounded-xl bg-muted/20 border border-border/50 p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-muted-foreground">
                  Mood Valence
                </span>
                <span
                  className="text-sm font-semibold"
                  style={{ color: valenceColor(currentEmotion.valence) }}
                >
                  {valenceLabel(currentEmotion.valence)}
                </span>
              </div>
              <div className="relative h-2 rounded-full bg-muted/40 overflow-hidden">
                <div
                  className="absolute top-0 left-1/2 h-full rounded-full transition-all duration-700"
                  style={{
                    width: `${Math.abs(currentEmotion.valence) * 50}%`,
                    transform:
                      currentEmotion.valence >= 0
                        ? "translateX(0)"
                        : `translateX(-100%)`,
                    background: valenceColor(currentEmotion.valence),
                  }}
                />
                <div className="absolute top-0 left-1/2 w-px h-full bg-muted-foreground/30" />
              </div>
              <div className="flex justify-between mt-1">
                <span className="text-[9px] text-muted-foreground/60">
                  Negative
                </span>
                <span className="text-[9px] text-muted-foreground/60">
                  Positive
                </span>
              </div>
            </div>
          </div>
        ) : (
          <div className="text-center py-4">
            <div
              className="w-14 h-14 rounded-2xl mx-auto flex items-center justify-center mb-3"
              style={{
                background: `${emotionColor}18`,
                border: `2px solid ${emotionColor}30`,
              }}
            >
              <span
                className="w-6 h-6 rounded-full"
                style={{ backgroundColor: emotionColor, opacity: 0.5 }}
              />
            </div>
            <p className="text-sm text-muted-foreground">
              No emotion detected yet. Run a voice analysis to start tracking.
            </p>
          </div>
        )}
      </motion.div>

      {/* 7-Day Mood Trend -- line chart */}
      {valenceTrend.length >= 2 && (
        <motion.div
          className="rounded-[14px] p-4 border border-border bg-card"
          custom={1}
          initial="hidden"
          animate="visible"
          variants={cardVariants}
        >
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="h-4 w-4 text-cyan-500" />
            <span className="text-sm font-semibold text-foreground">
              7-Day Mood Trend
            </span>
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart
              data={valenceTrend}
              margin={{ left: 0, right: 8, top: 8, bottom: 0 }}
            >
              <defs>
                <linearGradient id="moodValenceGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#0891b2" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#0891b2" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="hsl(220,18%,14%)"
                opacity={0.5}
              />
              <XAxis
                dataKey="date"
                tick={{ fontSize: 10, fill: "hsl(220,12%,42%)" }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                domain={[-1, 1]}
                tick={{ fontSize: 9, fill: "hsl(220,12%,42%)" }}
                axisLine={false}
                tickLine={false}
                width={32}
                tickFormatter={(v: number) =>
                  v === 0 ? "0" : v > 0 ? `+${v}` : `${v}`
                }
              />
              <Tooltip content={<ValenceTooltip />} />
              <Area
                type="monotone"
                dataKey="valence"
                name="Valence"
                stroke="#0891b2"
                fill="url(#moodValenceGrad)"
                strokeWidth={2.5}
                dot={{ fill: "#0891b2", r: 4, strokeWidth: 2, stroke: "#0e1117" }}
                activeDot={{ r: 6, fill: "#0891b2", stroke: "#0e1117", strokeWidth: 2 }}
              />
            </AreaChart>
          </ResponsiveContainer>
          <p className="text-[10px] text-muted-foreground/50 text-center mt-2">
            Daily average valence score
          </p>
        </motion.div>
      )}

      {/* Emotion Distribution -- bar chart */}
      {distribution.length > 0 && (
        <motion.div
          className="rounded-[14px] p-4 border border-border bg-card"
          custom={2}
          initial="hidden"
          animate="visible"
          variants={cardVariants}
        >
          <div className="flex items-center gap-2 mb-4">
            <BarChart3 className="h-4 w-4 text-rose-400" />
            <span className="text-sm font-semibold text-foreground">
              Emotion Distribution
            </span>
            <span className="text-[10px] text-muted-foreground ml-auto">
              Last 7 days
            </span>
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart
              data={distribution}
              margin={{ left: 0, right: 0, top: 4, bottom: 0 }}
            >
              <defs>
                {distribution.map((entry, index) => (
                  <linearGradient key={index} id={`moodBarGrad-${index}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={entry.color} stopOpacity={0.9} />
                    <stop offset="100%" stopColor={entry.color} stopOpacity={0.5} />
                  </linearGradient>
                ))}
              </defs>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="hsl(220,18%,14%)"
                opacity={0.5}
              />
              <XAxis
                dataKey="name"
                tick={{ fontSize: 10, fill: "hsl(220,12%,42%)" }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                tick={{ fontSize: 9, fill: "hsl(220,12%,42%)" }}
                axisLine={false}
                tickLine={false}
                width={24}
                allowDecimals={false}
              />
              <Tooltip
                contentStyle={{
                  background: "var(--popover)",
                  border: "1px solid var(--border)",
                  borderRadius: 10,
                  fontSize: 11,
                }}
              />
              <Bar dataKey="count" name="Occurrences" radius={[4, 4, 0, 0]}>
                {distribution.map((_, index) => (
                  <Cell key={index} fill={`url(#moodBarGrad-${index})`} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </motion.div>
      )}

      {/* Your Week in Emotions -- dot timeline */}
      {hasHistory && (
        <motion.div
          className="rounded-[14px] p-4 border border-border bg-card"
          custom={3}
          initial="hidden"
          animate="visible"
          variants={cardVariants}
        >
          <div className="flex items-center gap-2 mb-3">
            <Calendar className="h-4 w-4 text-amber-500" />
            <span className="text-sm font-semibold text-foreground">
              Your Week in Emotions
            </span>
          </div>
          <div className="flex justify-between items-center">
            {(() => {
              const dayMap = new Map<
                string,
                { emotion: string; label: string }
              >();
              for (const r of history!) {
                const d = new Date(r.timestamp);
                const key = d.toISOString().slice(0, 10);
                const label = d.toLocaleDateString(undefined, {
                  weekday: "short",
                });
                dayMap.set(key, { emotion: r.dominantEmotion, label });
              }
              const days = Array.from(dayMap.entries())
                .sort(([a], [b]) => a.localeCompare(b))
                .slice(-7);
              return days.map(([key, { emotion, label }]) => {
                const color = EMOTION_COLORS[emotion] ?? "#94a3b8";
                const displayName = EMOTION_LABELS[emotion] ?? emotion;
                return (
                  <div
                    key={key}
                    className="flex flex-col items-center gap-1.5"
                  >
                    <div
                      className="w-8 h-8 rounded-full"
                      style={{ background: color, opacity: 0.85 }}
                    />
                    <span className="text-[9px] text-muted-foreground capitalize">
                      {displayName.slice(0, 3)}
                    </span>
                    <span className="text-[9px] text-muted-foreground/60">
                      {label}
                    </span>
                  </div>
                );
              });
            })()}
          </div>
        </motion.div>
      )}

      {/* Weekly Summary */}
      {hasHistory && (
        <motion.div
          className="rounded-[14px] p-4 border border-border bg-card"
          custom={4}
          initial="hidden"
          animate="visible"
          variants={cardVariants}
        >
          <div className="flex items-center gap-2 mb-3">
            <TrendingUp className="h-4 w-4 text-indigo-400" />
            <span className="text-sm font-semibold text-foreground">
              Weekly Summary
            </span>
          </div>
          <p className="text-sm leading-relaxed text-muted-foreground">
            {weeklySummary}
          </p>
        </motion.div>
      )}

      {/* Empty state */}
      {!hasHistory && !currentEmotion && (
        <motion.div
          className="rounded-[14px] p-8 border border-border bg-card text-center"
          custom={1}
          initial="hidden"
          animate="visible"
          variants={cardVariants}
        >
          <Activity className="h-10 w-10 mx-auto text-muted-foreground/30 mb-3" />
          <p className="text-sm text-muted-foreground">
            No mood data yet. Run a voice analysis to start tracking your
            emotional patterns.
          </p>
        </motion.div>
      )}
    </div>
  );
}

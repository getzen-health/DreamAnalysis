/**
 * MoodTrends -- Emotion/Mood focused page.
 *
 * Shows:
 * 1. Current detected emotion (prominently)
 * 2. Emotion distribution pie/bar chart (last 7 days)
 * 3. Valence/arousal trend lines
 * 4. Emotion history over time
 *
 * Data: useCurrentEmotion, /api/brain/history/:userId
 */

import { useState } from "react";
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
import { Brain, Smile, TrendingUp } from "lucide-react";

/* ---------- constants ---------- */

const EMOTION_COLORS: Record<string, string> = {
  happy: "#0891b2",
  sad: "#6366f1",
  angry: "#ea580c",
  fear: "#7c3aed",
  fearful: "#7c3aed",
  surprise: "#d946ef",
  neutral: "#94a3b8",
  relaxed: "#2dd4bf",
  focused: "#3b82f6",
};

const EMOTION_EMOJI: Record<string, string> = {
  happy: "😊",
  sad: "😢",
  angry: "😠",
  fear: "😨",
  fearful: "😨",
  surprise: "😮",
  neutral: "😐",
  relaxed: "😌",
  focused: "🎯",
};

/* ---------- types ---------- */

interface HistoryEntry {
  dominantEmotion: string;
  timestamp: string;
  valence?: number;
  arousal?: number;
  stress_index?: number;
  focus_index?: number;
}

/* ---------- helpers ---------- */

function buildDistribution(data: HistoryEntry[]): { name: string; count: number; color: string }[] {
  const counts: Record<string, number> = {};
  for (const d of data) {
    const e = d.dominantEmotion ?? "neutral";
    counts[e] = (counts[e] || 0) + 1;
  }
  return Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .map(([name, count]) => ({
      name,
      count,
      color: EMOTION_COLORS[name] ?? "#94a3b8",
    }));
}

function buildValenceTrend(data: HistoryEntry[]): { date: string; valence: number; arousal: number }[] {
  // Group by day, average
  const dayMap = new Map<string, { valences: number[]; arousals: number[]; ts: number }>();
  for (const r of data) {
    const d = new Date(r.timestamp);
    const key = d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
    const ts = new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
    if (!dayMap.has(key)) dayMap.set(key, { valences: [], arousals: [], ts });
    const entry = dayMap.get(key)!;
    if (r.valence != null) entry.valences.push(r.valence);
    if (r.arousal != null) entry.arousals.push(r.arousal);
  }

  return Array.from(dayMap.entries())
    .sort(([, a], [, b]) => a.ts - b.ts)
    .slice(-7)
    .map(([date, { valences, arousals }]) => ({
      date,
      valence: valences.length
        ? Math.round((valences.reduce((a, b) => a + b, 0) / valences.length) * 100)
        : 0,
      arousal: arousals.length
        ? Math.round((arousals.reduce((a, b) => a + b, 0) / arousals.length) * 100)
        : 0,
    }));
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

  const distribution = history ? buildDistribution(history) : [];
  const valenceTrend = history ? buildValenceTrend(history) : [];
  const hasHistory = history && history.length > 0;

  const emotionLabel = currentEmotion?.emotion ?? "neutral";
  const emotionColor = EMOTION_COLORS[emotionLabel] ?? "#94a3b8";
  const emotionEmoji = EMOTION_EMOJI[emotionLabel] ?? "😐";

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
          Emotion history and patterns
        </p>
      </motion.div>

      {/* Current Emotion — hero card */}
      <motion.div
        className="rounded-[14px] p-6 border border-border bg-card text-center"
        initial={{ opacity: 0, y: 12, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
      >
        <div className="text-5xl mb-3">{emotionEmoji}</div>
        <div
          className="text-2xl font-bold capitalize mb-1"
          style={{ color: emotionColor }}
        >
          {emotionLabel}
        </div>
        {currentEmotion ? (
          <div className="flex items-center justify-center gap-4 mt-3">
            <div className="text-center">
              <div className="text-xs text-muted-foreground">Valence</div>
              <div className="text-sm font-mono font-semibold text-foreground">
                {currentEmotion.valence > 0 ? "+" : ""}
                {currentEmotion.valence.toFixed(2)}
              </div>
            </div>
            <div className="w-px h-8 bg-border" />
            <div className="text-center">
              <div className="text-xs text-muted-foreground">Arousal</div>
              <div className="text-sm font-mono font-semibold text-foreground">
                {(currentEmotion.arousal * 100).toFixed(0)}%
              </div>
            </div>
            <div className="w-px h-8 bg-border" />
            <div className="text-center">
              <div className="text-xs text-muted-foreground">Confidence</div>
              <div className="text-sm font-mono font-semibold text-foreground">
                {(currentEmotion.confidence * 100).toFixed(0)}%
              </div>
            </div>
          </div>
        ) : (
          <p className="text-sm text-muted-foreground mt-2">
            Run a voice analysis to detect your current mood
          </p>
        )}
      </motion.div>

      {/* Emotion Distribution — bar chart */}
      {distribution.length > 0 && (
        <motion.div
          className="rounded-[14px] p-4 border border-border bg-card"
          custom={1}
          initial="hidden"
          animate="visible"
          variants={cardVariants}
        >
          <div className="flex items-center gap-2 mb-4">
            <Smile className="h-4 w-4 text-primary" />
            <span className="text-sm font-semibold text-foreground">
              Emotion Distribution (7 days)
            </span>
          </div>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={distribution} margin={{ left: 0, right: 0, top: 4, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(220,18%,14%)" opacity={0.5} />
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
              />
              <Tooltip
                contentStyle={{
                  background: "var(--popover)",
                  border: "1px solid var(--border)",
                  borderRadius: 10,
                  fontSize: 11,
                }}
              />
              <Bar dataKey="count" name="Count" radius={[4, 4, 0, 0]}>
                {distribution.map((entry, index) => (
                  <Cell key={index} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </motion.div>
      )}

      {/* Valence & Arousal Trends */}
      {valenceTrend.length >= 2 && (
        <motion.div
          className="rounded-[14px] p-4 border border-border bg-card"
          custom={2}
          initial="hidden"
          animate="visible"
          variants={cardVariants}
        >
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="h-4 w-4 text-primary" />
            <span className="text-sm font-semibold text-foreground">
              Valence & Arousal Trends
            </span>
          </div>
          <ResponsiveContainer width="100%" height={180}>
            <AreaChart
              data={valenceTrend}
              margin={{ left: 0, right: 4, top: 4, bottom: 0 }}
            >
              <defs>
                <linearGradient id="valenceGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#0891b2" stopOpacity={0.25} />
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
                tick={{ fontSize: 9, fill: "hsl(220,12%,42%)" }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                domain={[-100, 100]}
                tick={{ fontSize: 9, fill: "hsl(220,12%,42%)" }}
                axisLine={false}
                tickLine={false}
                width={30}
                tickFormatter={(v) => `${v}`}
              />
              <Tooltip
                cursor={{ stroke: "hsl(220,14%,55%)", strokeWidth: 1 }}
                contentStyle={{
                  background: "var(--popover)",
                  border: "1px solid var(--border)",
                  borderRadius: 10,
                  fontSize: 11,
                }}
              />
              <Area
                type="monotone"
                dataKey="valence"
                name="Valence"
                stroke="#0891b2"
                fill="url(#valenceGrad)"
                strokeWidth={2}
                dot={false}
              />
              <Area
                type="monotone"
                dataKey="arousal"
                name="Arousal"
                stroke="#d4a017"
                fill="none"
                strokeWidth={1.5}
                strokeDasharray="4 3"
                dot={false}
              />
            </AreaChart>
          </ResponsiveContainer>
          <div className="flex gap-4 mt-2 justify-center">
            {[
              { label: "Valence", color: "#0891b2" },
              { label: "Arousal", color: "#d4a017", dashed: true },
            ].map((l) => (
              <div key={l.label} className="flex items-center gap-1">
                <svg width="14" height="8">
                  <line
                    x1="0"
                    y1="4"
                    x2="14"
                    y2="4"
                    stroke={l.color}
                    strokeWidth="2"
                    strokeDasharray={l.dashed ? "4 3" : "0"}
                  />
                </svg>
                <span className="text-[10px] text-muted-foreground">
                  {l.label}
                </span>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Emotion timeline — dots for last 7 days */}
      {hasHistory && (
        <motion.div
          className="rounded-[14px] p-4 border border-border bg-card"
          custom={3}
          initial="hidden"
          animate="visible"
          variants={cardVariants}
        >
          <div className="flex items-center gap-2 mb-3">
            <Brain className="h-4 w-4 text-primary" />
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
                return (
                  <div
                    key={key}
                    className="flex flex-col items-center gap-1.5"
                  >
                    <div
                      className="w-8 h-8 rounded-full"
                      style={{
                        background: color,
                        opacity: 0.85,
                      }}
                    />
                    <span className="text-[9px] text-muted-foreground capitalize">
                      {emotion.slice(0, 3)}
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

      {/* Empty state */}
      {!hasHistory && !currentEmotion && (
        <div className="rounded-[14px] p-8 border border-border bg-card text-center">
          <Smile className="h-10 w-10 mx-auto text-muted-foreground/30 mb-3" />
          <p className="text-sm text-muted-foreground">
            No emotion data yet. Run a voice analysis to start tracking your
            mood.
          </p>
        </div>
      )}
    </div>
  );
}

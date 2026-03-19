/**
 * StressTrends -- Stress focused page.
 *
 * Shows:
 * 1. Stress Score gauge (from useScores)
 * 2. Stress trend line (last 7/30 days)
 * 3. HRV trend (inversely correlated with stress)
 * 4. Stress triggers analysis
 * 5. Recovery recommendations when stress is high
 *
 * Data: useScores, useCurrentEmotion, /api/brain/history/:userId
 */

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { pageTransition, cardVariants } from "@/lib/animations";
import { useScores } from "@/hooks/use-scores";
import { useCurrentEmotion } from "@/hooks/use-current-emotion";
import { useAuth } from "@/hooks/use-auth";
import { useHealthSync } from "@/hooks/use-health-sync";
import { getParticipantId } from "@/lib/participant";
import { ScoreGauge } from "@/components/score-gauge";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { Brain, Heart, Shield, Lightbulb } from "lucide-react";
import { listSessions, type SessionSummary } from "@/lib/ml-api";

/* ---------- types ---------- */

interface HistoryEntry {
  dominantEmotion: string;
  timestamp: string;
  stress_index?: number;
}

/* ---------- helpers ---------- */

function buildStressTrend(
  sessions: SessionSummary[],
  days: number
): { date: string; stress: number }[] {
  const map: Record<string, { values: number[]; ts: number }> = {};
  const cutoff = Date.now() / 1000 - days * 86400;

  for (const s of sessions) {
    if ((s.start_time ?? 0) < cutoff) continue;
    if (s.summary?.avg_stress == null) continue;
    const d = new Date((s.start_time ?? 0) * 1000);
    const key =
      days <= 7
        ? d.toLocaleDateString("en-US", { weekday: "short" })
        : d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
    const ts = new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
    if (!map[key]) map[key] = { values: [], ts };
    map[key].values.push(s.summary.avg_stress * 100);
  }

  return Object.entries(map)
    .sort(([, a], [, b]) => a.ts - b.ts)
    .map(([date, { values }]) => ({
      date,
      stress: Math.round(values.reduce((a, b) => a + b, 0) / values.length),
    }));
}

/* ---------- component ---------- */

export default function StressTrends() {
  const { user } = useAuth();
  const userId = user?.id?.toString();
  const participantId = getParticipantId();
  const { scores, loading: scoresLoading } = useScores(userId);
  const { emotion: currentEmotion } = useCurrentEmotion();
  const { latestPayload } = useHealthSync();
  const [periodDays, setPeriodDays] = useState(7);

  const { data: allSessions = [] } = useQuery<SessionSummary[]>({
    queryKey: ["sessions"],
    queryFn: () => listSessions(),
    retry: false,
    staleTime: 2 * 60 * 1000,
  });

  const stressTrend = buildStressTrend(allSessions, periodDays);
  const currentStress = currentEmotion?.stress ?? null;
  const stressPercent = currentStress !== null ? Math.round(currentStress * 100) : null;
  const hrv = latestPayload?.hrv_sdnn ?? null;
  const restingHR = latestPayload?.resting_heart_rate ?? null;

  const stressLevel =
    stressPercent === null
      ? "Unknown"
      : stressPercent < 30
      ? "Low"
      : stressPercent < 60
      ? "Moderate"
      : "High";

  const stressColor =
    stressPercent === null
      ? "var(--muted-foreground)"
      : stressPercent < 30
      ? "#0891b2"
      : stressPercent < 60
      ? "#d4a017"
      : "#e879a8";

  // Recovery tips based on stress level
  const tips =
    stressPercent !== null && stressPercent >= 50
      ? [
          { emoji: "🧘", text: "Try a 5-minute breathing exercise" },
          { emoji: "🚶", text: "Take a short walk outside" },
          { emoji: "🎵", text: "Listen to calming music" },
          { emoji: "💤", text: "Consider an early bedtime tonight" },
        ]
      : stressPercent !== null && stressPercent >= 30
      ? [
          { emoji: "☕", text: "Take a mindful break" },
          { emoji: "📝", text: "Journal about what's on your mind" },
        ]
      : [];

  return (
    <div className="max-w-2xl mx-auto px-4 py-6 space-y-5 pb-24">
      {/* Header */}
      <motion.div
        initial={pageTransition.initial}
        animate={pageTransition.animate}
        transition={pageTransition.transition}
      >
        <h1 className="text-xl font-bold tracking-tight text-foreground">
          Stress
        </h1>
        <p className="text-sm mt-1 text-muted-foreground">
          Stress levels, HRV, and recovery
        </p>
      </motion.div>

      {/* Stress Gauge — hero */}
      <motion.div
        className="rounded-[14px] p-6 border border-border bg-card flex flex-col items-center"
        initial={{ opacity: 0, y: 12, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
      >
        {stressPercent !== null ? (
          <>
            <div className="relative w-32 h-32 mb-3">
              <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
                <circle
                  cx="50"
                  cy="50"
                  r="42"
                  fill="none"
                  stroke="hsl(220,18%,14%)"
                  strokeWidth="8"
                />
                <motion.circle
                  cx="50"
                  cy="50"
                  r="42"
                  fill="none"
                  stroke={stressColor}
                  strokeWidth="8"
                  strokeLinecap="round"
                  strokeDasharray={`${(stressPercent / 100) * 264} 264`}
                  initial={{ strokeDasharray: "0 264" }}
                  animate={{
                    strokeDasharray: `${(stressPercent / 100) * 264} 264`,
                  }}
                  transition={{ duration: 1, ease: [0.22, 1, 0.36, 1] }}
                />
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <span
                  className="text-3xl font-bold"
                  style={{ color: stressColor }}
                >
                  {stressPercent}
                </span>
                <span className="text-[10px] text-muted-foreground">/ 100</span>
              </div>
            </div>
            <div
              className="text-sm font-semibold"
              style={{ color: stressColor }}
            >
              {stressLevel} Stress
            </div>
          </>
        ) : (
          <div className="text-center py-4">
            <Brain className="h-10 w-10 mx-auto text-muted-foreground/30 mb-3" />
            <p className="text-sm text-muted-foreground">
              Run a voice analysis to measure your stress level
            </p>
          </div>
        )}
      </motion.div>

      {/* HRV & Heart Rate cards */}
      <div className="grid grid-cols-2 gap-3">
        <motion.div
          className="rounded-[14px] p-4 border border-border bg-card"
          custom={1}
          initial="hidden"
          animate="visible"
          variants={cardVariants}
        >
          <div className="flex items-center gap-2 mb-2">
            <Heart className="h-3.5 w-3.5 text-rose-400" />
            <span className="text-xs text-muted-foreground">HRV</span>
          </div>
          <div className="text-xl font-bold text-foreground font-mono">
            {hrv !== null ? `${Math.round(hrv)}` : "\u2014"}
            {hrv !== null && (
              <span className="text-xs font-normal text-muted-foreground ml-1">
                ms
              </span>
            )}
          </div>
          <div className="text-[10px] text-muted-foreground mt-1">
            {hrv !== null
              ? hrv > 50
                ? "Good recovery"
                : hrv > 30
                ? "Moderate"
                : "Low — consider rest"
              : "No data"}
          </div>
        </motion.div>

        <motion.div
          className="rounded-[14px] p-4 border border-border bg-card"
          custom={2}
          initial="hidden"
          animate="visible"
          variants={cardVariants}
        >
          <div className="flex items-center gap-2 mb-2">
            <Heart className="h-3.5 w-3.5 text-rose-400" />
            <span className="text-xs text-muted-foreground">Resting HR</span>
          </div>
          <div className="text-xl font-bold text-foreground font-mono">
            {restingHR !== null ? `${Math.round(restingHR)}` : "\u2014"}
            {restingHR !== null && (
              <span className="text-xs font-normal text-muted-foreground ml-1">
                bpm
              </span>
            )}
          </div>
          <div className="text-[10px] text-muted-foreground mt-1">
            {restingHR !== null
              ? restingHR < 60
                ? "Athletic"
                : restingHR < 80
                ? "Normal"
                : "Elevated"
              : "No data"}
          </div>
        </motion.div>
      </div>

      {/* Stress Trend */}
      <motion.div
        className="rounded-[14px] p-4 border border-border bg-card"
        custom={3}
        initial="hidden"
        animate="visible"
        variants={cardVariants}
      >
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Shield className="h-4 w-4 text-primary" />
            <span className="text-sm font-semibold text-foreground">
              Stress Trend
            </span>
          </div>
          <div className="flex gap-1">
            {[
              { label: "7d", days: 7 },
              { label: "30d", days: 30 },
            ].map((tab) => (
              <button
                key={tab.days}
                onClick={() => setPeriodDays(tab.days)}
                className={`px-2.5 py-1 text-[11px] rounded-full transition-colors ${
                  periodDays === tab.days
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:bg-muted"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {stressTrend.length >= 2 ? (
          <ResponsiveContainer width="100%" height={160}>
            <AreaChart
              data={stressTrend}
              margin={{ left: 0, right: 4, top: 4, bottom: 0 }}
            >
              <defs>
                <linearGradient id="stressGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#e879a8" stopOpacity={0.25} />
                  <stop offset="95%" stopColor="#e879a8" stopOpacity={0} />
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
                domain={[0, 100]}
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
                formatter={(v: number) => [`${v}%`, "Stress"]}
              />
              <Area
                type="monotone"
                dataKey="stress"
                name="Stress"
                stroke="#e879a8"
                fill="url(#stressGrad)"
                strokeWidth={2}
                dot={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-32 flex items-center justify-center text-sm text-muted-foreground">
            Not enough sessions to show trend
          </div>
        )}
      </motion.div>

      {/* Recovery Recommendations */}
      {tips.length > 0 && (
        <motion.div
          className="rounded-[14px] p-4 border border-border bg-card"
          custom={4}
          initial="hidden"
          animate="visible"
          variants={cardVariants}
        >
          <div className="flex items-center gap-2 mb-3">
            <Lightbulb className="h-4 w-4 text-primary" />
            <span className="text-sm font-semibold text-foreground">
              Recovery Tips
            </span>
          </div>
          <div className="space-y-2">
            {tips.map((tip, i) => (
              <div
                key={i}
                className="flex items-center gap-3 rounded-xl p-3 bg-muted/30"
              >
                <span className="text-lg">{tip.emoji}</span>
                <span className="text-xs text-foreground">{tip.text}</span>
              </div>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  );
}

/**
 * StressTrends -- Dedicated stress dashboard.
 *
 * Shows ONLY stress-related content:
 * 1. Current stress level as a percentage with color coding
 * 2. Stress classification label (Low/Moderate/High/Critical)
 * 3. 7-day stress trend chart (AreaChart with gradient fill)
 * 4. HRV data from health sync
 * 5. Tips for stress management based on current level
 * 6. Daily stress pattern (morning/afternoon/evening breakdown)
 *
 * Data: /api/brain/history/:userId?days=7, localStorage ndw_last_emotion
 */

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { pageTransition, cardVariants } from "@/lib/animations";
import { getParticipantId } from "@/lib/participant";
import { useHealthSync } from "@/hooks/use-health-sync";
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
import {
  Activity,
  TrendingDown,
  Heart,
  Shield,
  Lightbulb,
  Sun,
  Sunset,
  Moon,
} from "lucide-react";

/* ---------- constants ---------- */

const STRESS_LOW = "#0891b2"; // cyan
const STRESS_MODERATE = "#d4a017"; // amber
const STRESS_HIGH = "#e879a8"; // rose

/* ---------- types ---------- */

interface HistoryEntry {
  timestamp: string;
  stress: number; // 0-1
  dominantEmotion?: string;
}

/* ---------- helpers ---------- */

function getStressColor(percent: number): string {
  if (percent < 30) return STRESS_LOW;
  if (percent < 60) return STRESS_MODERATE;
  return STRESS_HIGH;
}

function getStressLabel(percent: number): string {
  if (percent < 25) return "Low";
  if (percent < 50) return "Moderate";
  if (percent < 75) return "High";
  return "Critical";
}

function readCurrentStress(): number | null {
  try {
    const raw = localStorage.getItem("ndw_last_emotion");
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    const data = parsed?.result ?? parsed;
    const stress = data?.stress_index ?? data?.stress ?? null;
    if (stress === null || stress === undefined) return null;
    return Math.round(stress * 100);
  } catch {
    return null;
  }
}

function buildDailyTrend(
  entries: HistoryEntry[]
): { date: string; stress: number }[] {
  const map: Record<string, { values: number[]; ts: number }> = {};

  for (const entry of entries) {
    if (entry.stress == null) continue;
    const d = new Date(entry.timestamp);
    const key = d.toLocaleDateString("en-US", { weekday: "short" });
    const ts = new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
    if (!map[key]) map[key] = { values: [], ts };
    map[key].values.push(entry.stress * 100);
  }

  return Object.entries(map)
    .sort(([, a], [, b]) => a.ts - b.ts)
    .map(([date, { values }]) => ({
      date,
      stress: Math.round(values.reduce((a, b) => a + b, 0) / values.length),
    }));
}

function buildDailyPattern(
  entries: HistoryEntry[]
): { period: string; stress: number; icon: string }[] {
  const buckets: Record<string, number[]> = {
    Morning: [],
    Afternoon: [],
    Evening: [],
  };

  for (const entry of entries) {
    if (entry.stress == null) continue;
    const hour = new Date(entry.timestamp).getHours();
    if (hour >= 5 && hour < 12) {
      buckets.Morning.push(entry.stress * 100);
    } else if (hour >= 12 && hour < 18) {
      buckets.Afternoon.push(entry.stress * 100);
    } else {
      buckets.Evening.push(entry.stress * 100);
    }
  }

  return Object.entries(buckets)
    .filter(([, values]) => values.length > 0)
    .map(([period, values]) => ({
      period,
      stress: Math.round(values.reduce((a, b) => a + b, 0) / values.length),
      icon: period === "Morning" ? "sun" : period === "Afternoon" ? "sunset" : "moon",
    }));
}

function getTips(percent: number): { icon: string; text: string }[] {
  if (percent >= 75) {
    return [
      { icon: "breathe", text: "Try box breathing: inhale 4s, hold 4s, exhale 4s, hold 4s" },
      { icon: "walk", text: "Step away from screens and take a 10-minute walk" },
      { icon: "music", text: "Listen to calming music or nature sounds" },
      { icon: "sleep", text: "Prioritize an early bedtime tonight" },
      { icon: "water", text: "Hydrate — dehydration amplifies stress responses" },
    ];
  }
  if (percent >= 50) {
    return [
      { icon: "breathe", text: "Try a 5-minute breathing exercise" },
      { icon: "walk", text: "Take a short walk outside" },
      { icon: "music", text: "Listen to calming music" },
      { icon: "sleep", text: "Consider winding down earlier tonight" },
    ];
  }
  if (percent >= 25) {
    return [
      { icon: "break", text: "Take a mindful break between tasks" },
      { icon: "journal", text: "Journal about what's on your mind" },
    ];
  }
  return [
    { icon: "maintain", text: "Your stress is low — keep up whatever you're doing" },
  ];
}

/* ---------- period icon component ---------- */

function PeriodIcon({ name }: { name: string }) {
  switch (name) {
    case "sun":
      return <Sun className="h-4 w-4 text-amber-400" />;
    case "sunset":
      return <Sunset className="h-4 w-4 text-orange-400" />;
    case "moon":
      return <Moon className="h-4 w-4 text-indigo-400" />;
    default:
      return <Activity className="h-4 w-4 text-muted-foreground" />;
  }
}

/* ---------- component ---------- */

export default function StressTrends() {
  const userId = getParticipantId();
  const { latestPayload } = useHealthSync();
  const stressPercent = useMemo(() => readCurrentStress(), []);

  const { data: historyData = [] } = useQuery<HistoryEntry[]>({
    queryKey: ["brain-history-stress", userId],
    queryFn: async () => {
      const res = await fetch(`/api/brain/history/${userId}?days=7`);
      if (!res.ok) return [];
      const json = await res.json();
      return Array.isArray(json) ? json : json?.entries ?? json?.data ?? [];
    },
    retry: false,
    staleTime: 2 * 60 * 1000,
  });

  const stressTrend = useMemo(() => buildDailyTrend(historyData), [historyData]);
  const dailyPattern = useMemo(() => buildDailyPattern(historyData), [historyData]);

  const stressColor =
    stressPercent !== null ? getStressColor(stressPercent) : "var(--muted-foreground)";
  const stressLabel = stressPercent !== null ? getStressLabel(stressPercent) : "Unknown";
  const tips = stressPercent !== null ? getTips(stressPercent) : [];
  const hrv = latestPayload?.hrv_sdnn ?? null;

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
          Your stress levels, trends, and recovery
        </p>
      </motion.div>

      {/* Stress Gauge — hero card */}
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
                <defs>
                  <linearGradient id="stressGaugeGrad" x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0%" stopColor="#e879a8" />
                    <stop offset="100%" stopColor="#be185d" />
                  </linearGradient>
                </defs>
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
                  stroke="url(#stressGaugeGrad)"
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
                  {stressPercent}%
                </span>
              </div>
            </div>
            <div
              className="text-sm font-semibold"
              style={{ color: stressColor }}
            >
              {stressLabel} Stress
            </div>
          </>
        ) : (
          <div className="text-center py-4">
            <Activity className="h-10 w-10 mx-auto text-muted-foreground/30 mb-3" />
            <p className="text-sm text-muted-foreground">
              Run a voice or EEG analysis to measure your stress level
            </p>
          </div>
        )}
      </motion.div>

      {/* HRV card */}
      <motion.div
        className="rounded-[14px] p-4 border border-border bg-card"
        custom={1}
        initial="hidden"
        animate="visible"
        variants={cardVariants}
      >
        <div className="flex items-center gap-2 mb-2">
          <Heart className="h-3.5 w-3.5 text-rose-400" />
          <span className="text-xs text-muted-foreground">
            Heart Rate Variability (HRV)
          </span>
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
              ? "Good recovery — lower stress reactivity"
              : hrv > 30
              ? "Moderate — some stress load present"
              : "Low HRV — elevated stress, consider rest"
            : "Sync a health device to see HRV data"}
        </div>
        {hrv !== null && (
          <div className="mt-2 w-full bg-muted/30 rounded-full h-1.5">
            <div
              className="h-1.5 rounded-full transition-all duration-500"
              style={{
                width: `${Math.min(100, (hrv / 80) * 100)}%`,
                backgroundColor:
                  hrv > 50
                    ? STRESS_LOW
                    : hrv > 30
                    ? STRESS_MODERATE
                    : STRESS_HIGH,
              }}
            />
          </div>
        )}
      </motion.div>

      {/* 7-Day Stress Trend */}
      <motion.div
        className="rounded-[14px] p-4 border border-border bg-card"
        custom={2}
        initial="hidden"
        animate="visible"
        variants={cardVariants}
      >
        <div className="flex items-center gap-2 mb-4">
          <TrendingDown className="h-4 w-4 text-primary" />
          <span className="text-sm font-semibold text-foreground">
            7-Day Stress Trend
          </span>
        </div>

        {stressTrend.length >= 2 ? (
          <ResponsiveContainer width="100%" height={240}>
            <AreaChart
              data={stressTrend}
              margin={{ left: 0, right: 4, top: 4, bottom: 0 }}
            >
              <defs>
                <linearGradient id="stressGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#e879a8" stopOpacity={0.3} />
                  <stop offset="50%" stopColor="#be185d" stopOpacity={0.12} />
                  <stop offset="100%" stopColor="#be185d" stopOpacity={0} />
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
                stroke={STRESS_HIGH}
                fill="url(#stressGrad)"
                strokeWidth={2.5}
                dot={false}
                activeDot={{
                  r: 4,
                  stroke: STRESS_HIGH,
                  strokeWidth: 2,
                  fill: "var(--card)",
                }}
              />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-32 flex items-center justify-center text-sm text-muted-foreground">
            Not enough data to show a trend yet
          </div>
        )}
      </motion.div>

      {/* Daily Stress Pattern */}
      {dailyPattern.length > 0 && (
        <motion.div
          className="rounded-[14px] p-4 border border-border bg-card"
          custom={3}
          initial="hidden"
          animate="visible"
          variants={cardVariants}
        >
          <div className="flex items-center gap-2 mb-4">
            <Shield className="h-4 w-4 text-primary" />
            <span className="text-sm font-semibold text-foreground">
              Daily Stress Pattern
            </span>
          </div>

          <div className="space-y-3">
            {dailyPattern.map((bucket) => {
              const color = getStressColor(bucket.stress);
              return (
                <div key={bucket.period} className="flex items-center gap-3">
                  <PeriodIcon name={bucket.icon} />
                  <span className="text-xs text-foreground w-20">
                    {bucket.period}
                  </span>
                  <div className="flex-1 bg-muted/30 rounded-full h-2">
                    <motion.div
                      className="h-2 rounded-full"
                      style={{ backgroundColor: color }}
                      initial={{ width: 0 }}
                      animate={{ width: `${bucket.stress}%` }}
                      transition={{
                        duration: 0.8,
                        ease: [0.22, 1, 0.36, 1],
                      }}
                    />
                  </div>
                  <span
                    className="text-xs font-mono font-semibold w-10 text-right"
                    style={{ color }}
                  >
                    {bucket.stress}%
                  </span>
                </div>
              );
            })}
          </div>

          <div className="mt-3 flex gap-3 justify-center">
            {[
              { label: "Low", color: STRESS_LOW },
              { label: "Moderate", color: STRESS_MODERATE },
              { label: "High", color: STRESS_HIGH },
            ].map((l) => (
              <div key={l.label} className="flex items-center gap-1">
                <span
                  className="inline-block w-2 h-2 rounded-full"
                  style={{ backgroundColor: l.color }}
                />
                <span className="text-[10px] text-muted-foreground">
                  {l.label}
                </span>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Stress Management Tips */}
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
              {stressPercent !== null && stressPercent >= 50
                ? "Recovery Tips"
                : "Stress Management"}
            </span>
          </div>
          <div className="space-y-2">
            {tips.map((tip, i) => (
              <motion.div
                key={i}
                className="flex items-start gap-3 rounded-xl p-3 bg-muted/30"
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{
                  delay: i * 0.06,
                  duration: 0.3,
                  ease: "easeOut",
                }}
              >
                <Activity className="h-4 w-4 mt-0.5 text-muted-foreground shrink-0" />
                <span className="text-xs text-foreground">{tip.text}</span>
              </motion.div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Empty state — no data at all */}
      {stressTrend.length === 0 && stressPercent === null && (
        <div className="rounded-[14px] p-8 border border-border bg-card text-center">
          <Activity className="h-10 w-10 mx-auto text-muted-foreground/30 mb-3" />
          <p className="text-sm text-muted-foreground">
            No stress data yet. Complete a voice or EEG session to start
            tracking your stress patterns.
          </p>
        </div>
      )}
    </div>
  );
}

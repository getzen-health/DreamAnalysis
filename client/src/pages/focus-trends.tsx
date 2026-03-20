/**
 * FocusTrends -- Focus-only dashboard page.
 *
 * Shows:
 * 1. Current focus level as percentage with color-coded gauge
 * 2. Focus classification label (Diffuse/Moderate/Sharp/Deep)
 * 3. 7-day focus trend chart (AreaChart with gradient fill)
 * 4. Focus tips based on current level
 * 5. Best focus times of day (when data available)
 *
 * Data: /api/brain/history/:userId?days=7, localStorage "ndw_last_emotion"
 */

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { pageTransition, cardVariants } from "@/lib/animations";
import { getParticipantId } from "@/lib/participant";
import { useCurrentEmotion } from "@/hooks/use-current-emotion";
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
import { Target, Zap, TrendingUp, TrendingDown, Lightbulb, Clock, Sun, Sunset, Moon } from "lucide-react";

/* ---------- constants ---------- */

const FOCUS_PRIMARY = "#6366f1"; // indigo
const FOCUS_ACCENT = "#0891b2"; // cyan

/* ---------- types ---------- */

interface HistoryEntry {
  dominantEmotion: string;
  timestamp: string;
  focus_index?: number;
}

/* ---------- helpers ---------- */

function getCurrentFocus(): { focusIndex: number; timestamp: number } | null {
  try {
    const raw = localStorage.getItem("ndw_last_emotion");
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed?.timestamp || Date.now() - parsed.timestamp > 86_400_000) return null;
    const fi = parsed.result?.focus_index;
    if (fi == null || typeof fi !== "number") return null;
    return { focusIndex: fi, timestamp: parsed.timestamp };
  } catch {
    return null;
  }
}

function classifyFocus(percent: number): {
  label: string;
  color: string;
  description: string;
} {
  if (percent >= 80)
    return {
      label: "Deep",
      color: FOCUS_PRIMARY,
      description: "Exceptional concentration — you are in the zone.",
    };
  if (percent >= 55)
    return {
      label: "Sharp",
      color: FOCUS_ACCENT,
      description: "Strong focus — ideal for complex tasks.",
    };
  if (percent >= 30)
    return {
      label: "Moderate",
      color: "#d4a017",
      description: "Baseline attention — fine for routine work.",
    };
  return {
    label: "Diffuse",
    color: "#94a3b8",
    description: "Scattered attention — consider a short break.",
  };
}

function buildDailyFocusTrend(
  data: HistoryEntry[]
): { date: string; focus: number; ts: number }[] {
  const dayMap = new Map<
    string,
    { values: number[]; ts: number }
  >();

  for (const entry of data) {
    if (entry.focus_index == null) continue;
    const d = new Date(entry.timestamp);
    const key = d.toLocaleDateString("en-US", { weekday: "short" });
    const ts = new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
    if (!dayMap.has(key)) dayMap.set(key, { values: [], ts });
    dayMap.get(key)!.values.push(entry.focus_index * 100);
  }

  return Array.from(dayMap.entries())
    .sort(([, a], [, b]) => a.ts - b.ts)
    .map(([date, { values, ts }]) => ({
      date,
      focus: Math.round(values.reduce((a, b) => a + b, 0) / values.length),
      ts,
    }));
}

function buildFocusByHour(
  data: HistoryEntry[]
): { hour: string; focus: number; hourNum: number }[] {
  const hourMap: Record<number, number[]> = {};

  for (const entry of data) {
    if (entry.focus_index == null) continue;
    const h = new Date(entry.timestamp).getHours();
    if (!hourMap[h]) hourMap[h] = [];
    hourMap[h].push(entry.focus_index * 100);
  }

  return Object.entries(hourMap)
    .map(([hour, values]) => ({
      hourNum: parseInt(hour),
      hour: `${parseInt(hour) % 12 || 12}${parseInt(hour) < 12 ? "a" : "p"}`,
      focus: Math.round(values.reduce((a, b) => a + b, 0) / values.length),
    }))
    .sort((a, b) => a.hourNum - b.hourNum);
}

function getFocusTips(percent: number | null): { icon: React.ReactNode; text: string }[] {
  if (percent === null) return [];
  if (percent >= 80)
    return [
      { icon: <Zap className="h-4 w-4 text-indigo-400" />, text: "You are in deep focus. Tackle your hardest task now." },
      { icon: <Target className="h-4 w-4 text-cyan-400" />, text: "Block notifications to maintain this state." },
    ];
  if (percent >= 55)
    return [
      { icon: <Zap className="h-4 w-4 text-cyan-400" />, text: "Good focus level. Prioritize complex work while it lasts." },
      { icon: <Target className="h-4 w-4 text-indigo-400" />, text: "A short walk after 25 minutes keeps focus sharp." },
    ];
  if (percent >= 30)
    return [
      { icon: <Target className="h-4 w-4 text-amber-400" />, text: "Try a 5-minute breathing exercise to sharpen attention." },
      { icon: <Zap className="h-4 w-4 text-amber-400" />, text: "Handle simpler tasks first, then ramp up." },
      { icon: <Lightbulb className="h-4 w-4 text-amber-400" />, text: "Reduce open browser tabs and background noise." },
    ];
  return [
    { icon: <Target className="h-4 w-4 text-slate-400" />, text: "Step away from screens for a few minutes." },
    { icon: <Zap className="h-4 w-4 text-slate-400" />, text: "Hydrate and do some light stretching." },
    { icon: <Lightbulb className="h-4 w-4 text-slate-400" />, text: "This may not be the time for deep work — save it for later." },
  ];
}

function timeOfDayLabel(hourNum: number): { label: string; icon: React.ReactNode } {
  if (hourNum >= 5 && hourNum < 12)
    return { label: "Morning", icon: <Sun className="h-4 w-4 text-amber-400" /> };
  if (hourNum >= 12 && hourNum < 17)
    return { label: "Afternoon", icon: <Sun className="h-4 w-4 text-orange-400" /> };
  if (hourNum >= 17 && hourNum < 21)
    return { label: "Evening", icon: <Sunset className="h-4 w-4 text-rose-400" /> };
  return { label: "Night", icon: <Moon className="h-4 w-4 text-indigo-300" /> };
}

/* ---------- component ---------- */

export default function FocusTrends() {
  const userId = getParticipantId();

  const { data: history } = useQuery<HistoryEntry[]>({
    queryKey: [`/api/brain/history/${userId}?days=7`],
    retry: false,
    staleTime: 5 * 60 * 1000,
  });

  const { emotion: currentEmotion } = useCurrentEmotion();
  const focusPercent = currentEmotion?.focus != null
    ? Math.round(currentEmotion.focus * 100)
    : null;
  const classification = focusPercent !== null ? classifyFocus(focusPercent) : null;

  const focusTrend = useMemo(
    () => (history ? buildDailyFocusTrend(history) : []),
    [history]
  );

  const focusByHour = useMemo(
    () => (history ? buildFocusByHour(history) : []),
    [history]
  );

  const bestHour = useMemo(() => {
    if (focusByHour.length === 0) return null;
    return focusByHour.reduce((best, h) => (h.focus > best.focus ? h : best));
  }, [focusByHour]);

  const weekAvg = useMemo(() => {
    if (focusTrend.length === 0) return null;
    const sum = focusTrend.reduce((s, d) => s + d.focus, 0);
    return Math.round(sum / focusTrend.length);
  }, [focusTrend]);

  const tips = getFocusTips(focusPercent);

  return (
    <div className="max-w-2xl mx-auto px-4 py-6 space-y-5 pb-24">
      {/* Header */}
      <motion.div
        initial={pageTransition.initial}
        animate={pageTransition.animate}
        transition={pageTransition.transition}
      >
        <h1 className="text-xl font-bold tracking-tight text-foreground">
          Focus
        </h1>
        <p className="text-sm mt-1 text-muted-foreground">
          Concentration levels, trends, and peak times
        </p>
      </motion.div>

      {/* Focus Gauge — hero card */}
      <motion.div
        className="rounded-2xl p-6 border border-border bg-card flex flex-col items-center" style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}
        initial={{ opacity: 0, y: 12, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
      >
        {focusPercent !== null && classification ? (
          <>
            <div className="relative w-32 h-32 mb-3">
              <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
                <defs>
                  <linearGradient id="focusGaugeGrad" x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0%" stopColor="#6366f1" />
                    <stop offset="100%" stopColor="#818cf8" />
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
                  stroke="url(#focusGaugeGrad)"
                  strokeWidth="8"
                  strokeLinecap="round"
                  strokeDasharray={`${(focusPercent / 100) * 264} 264`}
                  initial={{ strokeDasharray: "0 264" }}
                  animate={{
                    strokeDasharray: `${(focusPercent / 100) * 264} 264`,
                  }}
                  transition={{ duration: 1, ease: [0.22, 1, 0.36, 1] }}
                />
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <span
                  className="text-3xl font-bold"
                  style={{ color: classification.color }}
                >
                  {focusPercent}
                </span>
                <span className="text-[10px] text-muted-foreground">/ 100</span>
              </div>
            </div>
            <div
              className="text-sm font-semibold"
              style={{ color: classification.color }}
            >
              {classification.label} Focus
            </div>
            <p className="text-xs text-muted-foreground mt-1 text-center max-w-[240px]">
              {classification.description}
            </p>
            {(() => {
              const entries = (history ?? []).filter((e) => e.focus_index != null);
              const prevFocus = entries.length >= 2 ? entries[entries.length - 2]?.focus_index : null;
              const curFocus = entries.length >= 1 ? entries[entries.length - 1]?.focus_index : null;
              const focusDelta = prevFocus != null && curFocus != null ? (curFocus - prevFocus) * 100 : null;
              if (focusDelta == null || Math.abs(focusDelta) <= 2) return null;
              return (
                <div style={{ display: "flex", alignItems: "center", gap: 4, marginTop: 6 }}>
                  {focusDelta > 0 ? (
                    <TrendingUp className="h-3.5 w-3.5 text-cyan-400" />
                  ) : (
                    <TrendingDown className="h-3.5 w-3.5 text-rose-400" />
                  )}
                  <span className="text-xs text-muted-foreground">
                    {Math.abs(Math.round(focusDelta))}% {focusDelta > 0 ? "higher" : "lower"} vs last session
                  </span>
                </div>
              );
            })()}
          </>
        ) : (
          <div className="text-center py-4">
            <Target className="h-10 w-10 mx-auto text-muted-foreground/30 mb-3" />
            <p className="text-sm text-muted-foreground">
              Run a voice analysis to measure your focus level
            </p>
          </div>
        )}
      </motion.div>

      {/* Week average stat */}
      {weekAvg !== null && (
        <motion.div
          className="rounded-2xl p-4 border border-border bg-card flex items-center gap-4" style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}
          custom={1}
          initial="hidden"
          animate="visible"
          variants={cardVariants}
        >
          <div
            className="w-10 h-10 rounded-xl flex items-center justify-center"
            style={{ background: `${FOCUS_PRIMARY}20` }}
          >
            <TrendingUp className="h-5 w-5" style={{ color: FOCUS_PRIMARY }} />
          </div>
          <div>
            <div className="text-xs text-muted-foreground">7-Day Average</div>
            <div className="text-2xl font-bold text-foreground font-mono">
              {weekAvg}%
            </div>
          </div>
          <div className="ml-auto text-[10px] text-muted-foreground">
            {focusTrend.length} {focusTrend.length === 1 ? "day" : "days"} of data
          </div>
        </motion.div>
      )}

      {/* 7-Day Focus Trend */}
      <motion.div
        className="rounded-2xl p-4 border border-border bg-card" style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}
        custom={2}
        initial="hidden"
        animate="visible"
        variants={cardVariants}
      >
        <div className="flex items-center gap-2 mb-4">
          <Target className="h-4 w-4" style={{ color: FOCUS_PRIMARY }} />
          <span className="text-sm font-semibold text-foreground">
            Focus Trend (7 days)
          </span>
        </div>

        {focusTrend.length >= 2 ? (
          <ResponsiveContainer width="100%" height={240}>
            <AreaChart
              data={focusTrend}
              margin={{ left: 0, right: 4, top: 4, bottom: 0 }}
            >
              <defs>
                <linearGradient id="focusGradFT" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#6366f1" stopOpacity={0.3} />
                  <stop offset="50%" stopColor="#818cf8" stopOpacity={0.12} />
                  <stop offset="100%" stopColor="#818cf8" stopOpacity={0} />
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
                formatter={(v: number) => [`${v}%`, "Focus"]}
              />
              <Area
                type="monotone"
                dataKey="focus"
                name="Focus"
                stroke={FOCUS_PRIMARY}
                fill="url(#focusGradFT)"
                strokeWidth={2.5}
                dot={{ r: 3, fill: FOCUS_PRIMARY, strokeWidth: 0 }}
                activeDot={{ r: 5, fill: FOCUS_PRIMARY, strokeWidth: 2, stroke: "#fff" }}
              />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-32 flex items-center justify-center text-sm text-muted-foreground">
            Not enough data to show a trend yet
          </div>
        )}
      </motion.div>

      {/* Best Focus Times */}
      {focusByHour.length >= 3 && (
        <motion.div
          className="rounded-2xl p-4 border border-border bg-card" style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}
          custom={3}
          initial="hidden"
          animate="visible"
          variants={cardVariants}
        >
          <div className="flex items-center gap-2 mb-4">
            <Clock className="h-4 w-4" style={{ color: FOCUS_ACCENT }} />
            <span className="text-sm font-semibold text-foreground">
              Focus by Time of Day
            </span>
          </div>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart
              data={focusByHour}
              margin={{ left: 0, right: 0, top: 4, bottom: 0 }}
            >
              <defs>
                <linearGradient id="focusBarGradPeak" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#6366f1" stopOpacity={0.9} />
                  <stop offset="100%" stopColor="#818cf8" stopOpacity={0.5} />
                </linearGradient>
                <linearGradient id="focusBarGradDim" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="hsl(220,18%,24%)" stopOpacity={0.9} />
                  <stop offset="100%" stopColor="hsl(220,18%,18%)" stopOpacity={0.5} />
                </linearGradient>
              </defs>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="hsl(220,18%,14%)"
                opacity={0.5}
              />
              <XAxis
                dataKey="hour"
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
                formatter={(v: number) => [`${v}%`, "Focus"]}
              />
              <Bar dataKey="focus" radius={[4, 4, 0, 0]}>
                {focusByHour.map((entry, index) => (
                  <Cell
                    key={index}
                    fill={
                      bestHour && entry.hour === bestHour.hour
                        ? "url(#focusBarGradPeak)"
                        : "url(#focusBarGradDim)"
                    }
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          {bestHour && (
            <div className="mt-3 flex items-center gap-2 rounded-xl p-3 bg-muted/30">
              {timeOfDayLabel(bestHour.hourNum).icon}
              <span className="text-xs text-foreground">
                Peak focus at{" "}
                <strong style={{ color: FOCUS_PRIMARY }}>{bestHour.hour}</strong>{" "}
                ({bestHour.focus}% avg) &mdash;{" "}
                {timeOfDayLabel(bestHour.hourNum).label.toLowerCase()} works best
                for you
              </span>
            </div>
          )}
        </motion.div>
      )}

      {/* Focus Tips */}
      {tips.length > 0 && (
        <motion.div
          className="rounded-2xl p-4 border border-border bg-card" style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}
          custom={4}
          initial="hidden"
          animate="visible"
          variants={cardVariants}
        >
          <div className="flex items-center gap-2 mb-3">
            <Lightbulb className="h-4 w-4" style={{ color: FOCUS_ACCENT }} />
            <span className="text-sm font-semibold text-foreground">
              Focus Tips
            </span>
          </div>
          <div className="space-y-2">
            {tips.map((tip, i) => (
              <div
                key={i}
                className="flex items-center gap-3 rounded-xl p-3 bg-muted/30"
              >
                {tip.icon}
                <span className="text-xs text-foreground">{tip.text}</span>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Empty state — no data at all */}
      {focusTrend.length === 0 && focusPercent === null && (
        <div className="rounded-2xl p-8 border border-border bg-card text-center" style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}>
          <Target className="h-10 w-10 mx-auto text-muted-foreground/30 mb-3" />
          <p className="text-sm text-muted-foreground">
            No focus data yet. Complete a voice check-in to start tracking your
            concentration patterns.
          </p>
        </div>
      )}
    </div>
  );
}

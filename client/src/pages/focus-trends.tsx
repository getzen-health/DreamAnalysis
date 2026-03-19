/**
 * FocusTrends -- Focus focused page.
 *
 * Shows:
 * 1. Focus Score gauge (from useCurrentEmotion)
 * 2. Focus trend line (last 7/30 days)
 * 3. Cognitive load history
 * 4. Best focus times of day
 *
 * Data: useScores, useCurrentEmotion, sessions
 */

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { pageTransition, cardVariants } from "@/lib/animations";
import { useScores } from "@/hooks/use-scores";
import { useCurrentEmotion } from "@/hooks/use-current-emotion";
import { useAuth } from "@/hooks/use-auth";
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
import { Target, Clock, Brain, Zap } from "lucide-react";
import { listSessions, type SessionSummary } from "@/lib/ml-api";

/* ---------- helpers ---------- */

function buildFocusTrend(
  sessions: SessionSummary[],
  days: number
): { date: string; focus: number; cogLoad: number }[] {
  const map: Record<
    string,
    { focuses: number[]; cogLoads: number[]; ts: number }
  > = {};
  const cutoff = Date.now() / 1000 - days * 86400;

  for (const s of sessions) {
    if ((s.start_time ?? 0) < cutoff) continue;
    if (s.summary?.avg_focus == null) continue;
    const d = new Date((s.start_time ?? 0) * 1000);
    const key =
      days <= 7
        ? d.toLocaleDateString("en-US", { weekday: "short" })
        : d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
    const ts = new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
    if (!map[key]) map[key] = { focuses: [], cogLoads: [], ts };
    map[key].focuses.push((s.summary.avg_focus ?? 0) * 100);
    // Cognitive load approximated from stress (inverse of focus)
    const cogLoad = 1 - (s.summary.avg_focus ?? 0);
    map[key].cogLoads.push(cogLoad * 100);
  }

  return Object.entries(map)
    .sort(([, a], [, b]) => a.ts - b.ts)
    .map(([date, { focuses, cogLoads }]) => ({
      date,
      focus: Math.round(
        focuses.reduce((a, b) => a + b, 0) / focuses.length
      ),
      cogLoad: cogLoads.length
        ? Math.round(cogLoads.reduce((a, b) => a + b, 0) / cogLoads.length)
        : 0,
    }));
}

function buildFocusByHour(
  sessions: SessionSummary[]
): { hour: string; focus: number }[] {
  const hourMap: Record<number, number[]> = {};

  for (const s of sessions) {
    if (s.summary?.avg_focus == null) continue;
    const d = new Date((s.start_time ?? 0) * 1000);
    const h = d.getHours();
    if (!hourMap[h]) hourMap[h] = [];
    hourMap[h].push(s.summary.avg_focus * 100);
  }

  return Object.entries(hourMap)
    .map(([hour, values]) => ({
      hourNum: parseInt(hour),
      hour: `${parseInt(hour) % 12 || 12}${parseInt(hour) < 12 ? "a" : "p"}`,
      focus: Math.round(values.reduce((a, b) => a + b, 0) / values.length),
    }))
    .sort((a, b) => a.hourNum - b.hourNum)
    .map(({ hour, focus }) => ({ hour, focus }));
}

/* ---------- component ---------- */

export default function FocusTrends() {
  const { user } = useAuth();
  const userId = user?.id?.toString();
  const { scores } = useScores(userId);
  const { emotion: currentEmotion } = useCurrentEmotion();
  const [periodDays, setPeriodDays] = useState(7);

  const { data: allSessions = [] } = useQuery<SessionSummary[]>({
    queryKey: ["sessions"],
    queryFn: () => listSessions(),
    retry: false,
    staleTime: 2 * 60 * 1000,
  });

  const focusTrend = buildFocusTrend(allSessions, periodDays);
  const focusByHour = buildFocusByHour(allSessions);
  const currentFocus = currentEmotion?.focus ?? null;
  const focusPercent =
    currentFocus !== null ? Math.round(currentFocus * 100) : null;

  const focusLevel =
    focusPercent === null
      ? "Unknown"
      : focusPercent >= 70
      ? "Sharp"
      : focusPercent >= 40
      ? "Moderate"
      : "Diffuse";

  const focusColor =
    focusPercent === null
      ? "var(--muted-foreground)"
      : focusPercent >= 70
      ? "#3b82f6"
      : focusPercent >= 40
      ? "#d4a017"
      : "#94a3b8";

  // Find best focus hour
  const bestHour = focusByHour.length
    ? focusByHour.reduce((best, h) => (h.focus > best.focus ? h : best))
    : null;

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
          Concentration, cognitive load, and peak times
        </p>
      </motion.div>

      {/* Focus Gauge — hero */}
      <motion.div
        className="rounded-[14px] p-6 border border-border bg-card flex flex-col items-center"
        initial={{ opacity: 0, y: 12, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
      >
        {focusPercent !== null ? (
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
                  stroke={focusColor}
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
                  style={{ color: focusColor }}
                >
                  {focusPercent}
                </span>
                <span className="text-[10px] text-muted-foreground">/ 100</span>
              </div>
            </div>
            <div
              className="text-sm font-semibold"
              style={{ color: focusColor }}
            >
              {focusLevel} Focus
            </div>
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

      {/* Focus Trend */}
      <motion.div
        className="rounded-[14px] p-4 border border-border bg-card"
        custom={1}
        initial="hidden"
        animate="visible"
        variants={cardVariants}
      >
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Target className="h-4 w-4 text-primary" />
            <span className="text-sm font-semibold text-foreground">
              Focus Trend
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

        {focusTrend.length >= 2 ? (
          <>
            <ResponsiveContainer width="100%" height={160}>
              <AreaChart
                data={focusTrend}
                margin={{ left: 0, right: 4, top: 4, bottom: 0 }}
              >
                <defs>
                  <linearGradient id="focusGradFT" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.25} />
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
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
                  formatter={(v: number) => [`${v}%`]}
                />
                <Area
                  type="monotone"
                  dataKey="focus"
                  name="Focus"
                  stroke="#3b82f6"
                  fill="url(#focusGradFT)"
                  strokeWidth={2}
                  dot={false}
                />
                <Area
                  type="monotone"
                  dataKey="cogLoad"
                  name="Cog. Load"
                  stroke="#7c3aed"
                  fill="none"
                  strokeWidth={1.5}
                  strokeDasharray="4 3"
                  dot={false}
                />
              </AreaChart>
            </ResponsiveContainer>
            <div className="flex gap-4 mt-2 justify-center">
              {[
                { label: "Focus", color: "#3b82f6" },
                { label: "Cog. Load", color: "#7c3aed", dashed: true },
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
          </>
        ) : (
          <div className="h-32 flex items-center justify-center text-sm text-muted-foreground">
            Not enough sessions to show trend
          </div>
        )}
      </motion.div>

      {/* Best Focus Times */}
      {focusByHour.length >= 3 && (
        <motion.div
          className="rounded-[14px] p-4 border border-border bg-card"
          custom={2}
          initial="hidden"
          animate="visible"
          variants={cardVariants}
        >
          <div className="flex items-center gap-2 mb-4">
            <Clock className="h-4 w-4 text-primary" />
            <span className="text-sm font-semibold text-foreground">
              Focus by Time of Day
            </span>
          </div>
          <ResponsiveContainer width="100%" height={140}>
            <BarChart
              data={focusByHour}
              margin={{ left: 0, right: 0, top: 4, bottom: 0 }}
            >
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
                        ? "#3b82f6"
                        : "hsl(220,18%,20%)"
                    }
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          {bestHour && (
            <div className="mt-3 flex items-center gap-2 rounded-xl p-3 bg-muted/30">
              <Zap className="h-4 w-4 text-blue-400" />
              <span className="text-xs text-foreground">
                Your peak focus time is around{" "}
                <strong className="text-blue-400">{bestHour.hour}</strong> (
                {bestHour.focus}% avg)
              </span>
            </div>
          )}
        </motion.div>
      )}

      {/* Empty state */}
      {focusTrend.length === 0 && focusPercent === null && (
        <div className="rounded-[14px] p-8 border border-border bg-card text-center">
          <Target className="h-10 w-10 mx-auto text-muted-foreground/30 mb-3" />
          <p className="text-sm text-muted-foreground">
            No focus data yet. Complete some sessions to track your focus
            patterns.
          </p>
        </div>
      )}
    </div>
  );
}

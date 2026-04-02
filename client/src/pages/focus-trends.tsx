/**
 * FocusTrends -- Focus detail page with clean data visualization.
 *
 * Shows:
 * 1. Hero focus percentage (0-100%), color-coded rose/amber/cyan
 * 2. Time range tabs: Today | Week | Month
 * 3. Smooth AreaChart of focus % over time
 * 4. Morning / Afternoon / Evening breakdown
 *
 * Data: /api/brain/history/:userId?days=30, localStorage ndw_last_emotion
 */

import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { pageTransition, cardVariants } from "@/lib/animations";
import { getParticipantId } from "@/lib/participant";
import { useCurrentEmotion } from "@/hooks/use-current-emotion";
import { sbGetGeneric } from "@/lib/supabase-store";
import { Skeleton } from "@/components/ui/skeleton";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { Target, Sun, Sunset, Moon } from "lucide-react";

/* ---------- constants ---------- */

const FOCUS_INDIGO = "#6366f1";
const FOCUS_AMBER = "#d4a017";
const FOCUS_ROSE = "#e879a8";
const FOCUS_CYAN = "#0891b2";

type TimeRange = "today" | "week" | "month";

/* ---------- types ---------- */

interface HistoryEntry {
  stress: number;
  happiness: number;
  focus: number;
  dominantEmotion: string;
  timestamp: string;
}

/* ---------- helpers ---------- */

function getFocusColor(percent: number): string {
  if (percent < 30) return FOCUS_ROSE;
  if (percent < 60) return FOCUS_AMBER;
  return FOCUS_CYAN;
}

function getFocusLabel(percent: number): string {
  if (percent >= 80) return "Deep";
  if (percent >= 55) return "Sharp";
  if (percent >= 30) return "Moderate";
  return "Diffuse";
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

  const sorted = entries
    .map((e) => ({ value: Math.round((e.focus ?? 0) * 100), ts: new Date(e.timestamp).getTime() }))
    .sort((a, b) => a.ts - b.ts);

  // Thin: max 1 point per minute
  const thinned: typeof sorted = [];
  for (const p of sorted) {
    if (thinned.length === 0 || p.ts - thinned[thinned.length - 1].ts > 60000) {
      thinned.push(p);
    }
  }

  return thinned.map((p) => {
    const d = new Date(p.ts);
    const time = range === "today"
      ? d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })
      : range === "week"
      ? d.toLocaleDateString([], { weekday: "short" }) + " " + d.toLocaleTimeString([], { hour: "numeric" })
      : d.toLocaleDateString([], { month: "short", day: "numeric" });
    return { time, value: p.value };
  });
}

function buildPeriodAverages(
  entries: HistoryEntry[],
): { period: string; focus: number; icon: "sun" | "sunset" | "moon" }[] {
  const buckets: Record<string, number[]> = {
    Morning: [],
    Afternoon: [],
    Evening: [],
  };

  for (const e of entries) {
    const hour = new Date(e.timestamp).getHours();
    const val = Math.round(e.focus * 100);
    if (hour >= 5 && hour < 12) buckets.Morning.push(val);
    else if (hour >= 12 && hour < 18) buckets.Afternoon.push(val);
    else buckets.Evening.push(val);
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
      focus: Math.round(values.reduce((a, b) => a + b, 0) / values.length),
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
      return <Target className="h-4 w-4 text-muted-foreground" />;
  }
}

/* ---------- component ---------- */

export default function FocusTrends() {
  const userId = getParticipantId();
  const [range, setRange] = useState<TimeRange>("today");
  const { emotion: currentEmotion } = useCurrentEmotion();

  const { data, isLoading } = useQuery<HistoryEntry[]>({
    queryKey: [`/api/brain/history/${userId}?days=90`],
    queryFn: async () => {
      let all: HistoryEntry[] = [];
      try {
        const res = await fetch(`/api/brain/history/${userId}?days=90`);
        if (res.ok) {
          const json = await res.json();
          if (Array.isArray(json)) all = json;
        }
      } catch { /* API unavailable */ }
      try {
        const { getSupabase } = await import("@/lib/supabase-browser");
        const sb = await getSupabase();
        if (sb) {
          const since = new Date(Date.now() - 30 * 86400000).toISOString();
          const { data: rows } = await sb.from("emotion_history").select("*")
            .eq("user_id", userId).gte("created_at", since).order("created_at", { ascending: false }).limit(500);
          if (rows) {
            for (const r of rows) {
              all.push({ stress: r.stress ?? 0, happiness: r.mood ?? 0, focus: r.focus ?? 0, dominantEmotion: r.dominant_emotion ?? "neutral", timestamp: r.created_at });
            }
          }
        }
      } catch { /* Supabase unavailable */ }
      // 3. Supabase-backed local cache fallback
      const cached = sbGetGeneric<any[]>("ndw_emotion_history");
      if (Array.isArray(cached)) all.push(...cached);
      all.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
      const deduped: HistoryEntry[] = [];
      for (const e of all) {
        const ts = new Date(e.timestamp).getTime();
        if (deduped.length === 0 || ts - new Date(deduped[deduped.length - 1].timestamp).getTime() > 3000) {
          deduped.push(e);
        }
      }
      return deduped;
    },
    retry: false,
    staleTime: 60_000,
  });

  const focusPercent = currentEmotion?.focus != null
    ? Math.round(currentEmotion.focus * 100)
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

  const heroColor =
    focusPercent !== null ? getFocusColor(focusPercent) : "var(--muted-foreground)";
  const heroLabel =
    focusPercent !== null ? getFocusLabel(focusPercent) : "Unknown";

  return (
    <div className="max-w-2xl mx-auto px-4 py-6 space-y-5 pb-4">
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
          Your concentration levels over time
        </p>
      </motion.div>

      {/* Hero — focus percentage */}
      <motion.div
        className="rounded-2xl p-6 border border-border bg-card flex flex-col items-center"
        style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}
        initial={{ opacity: 0, y: 12, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
      >
        {focusPercent !== null ? (
          <>
            <span
              className="text-5xl font-bold tabular-nums"
              style={{ color: heroColor }}
            >
              {focusPercent}%
            </span>
            <div
              className="text-sm font-semibold mt-2"
              style={{ color: heroColor }}
            >
              {heroLabel} Focus
            </div>
          </>
        ) : (
          <div className="text-center py-4">
            <Target className="h-10 w-10 mx-auto text-muted-foreground/30 mb-3" />
            <p className="text-sm text-muted-foreground">
              Complete a check-in to measure your focus level
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

      {/* Focus chart */}
      <motion.div
        className="rounded-2xl p-4 border border-border bg-card"
        style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}
        custom={1}
        initial="hidden"
        animate="visible"
        variants={cardVariants}
      >
        <div className="flex items-center gap-2 mb-4">
          <Target className="h-4 w-4" style={{ color: FOCUS_INDIGO }} />
          <span className="text-sm font-semibold text-foreground">
            Focus Over Time
          </span>
        </div>

        {isLoading ? (
          <div className="h-[220px] space-y-2 pt-2">
            <Skeleton className="h-full w-full rounded-xl" />
          </div>
        ) : chartData.length >= 2 ? (
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart
              data={chartData}
              margin={{ left: 0, right: 4, top: 4, bottom: 0 }}
            >
              <defs>
                <linearGradient id="focusGradNew" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={FOCUS_INDIGO} stopOpacity={0.3} />
                  <stop offset="50%" stopColor={FOCUS_INDIGO} stopOpacity={0.12} />
                  <stop offset="100%" stopColor={FOCUS_INDIGO} stopOpacity={0} />
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
                interval="preserveStartEnd"
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
                formatter={(v: number) => [`${v}%`, "Focus"]}
              />
              <Area
                type="monotone"
                dataKey="value"
                name="Focus"
                stroke={FOCUS_INDIGO}
                fill="url(#focusGradNew)"
                strokeWidth={2.5}
                dot={false}
                activeDot={false}
                connectNulls
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
              const color = getFocusColor(p.focus);
              return (
                <div key={p.period} className="flex items-center gap-3">
                  <PeriodIcon name={p.icon} />
                  <span className="text-xs text-foreground w-20">{p.period}</span>
                  <div className="flex-1 bg-muted/30 rounded-full h-2">
                    <motion.div
                      className="h-2 rounded-full"
                      style={{ backgroundColor: color }}
                      initial={{ width: 0 }}
                      animate={{ width: `${p.focus}%` }}
                      transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
                    />
                  </div>
                  <span
                    className="text-xs font-mono font-semibold w-10 text-right"
                    style={{ color }}
                  >
                    {p.focus}%
                  </span>
                </div>
              );
            })}
          </div>
        </motion.div>
      )}

      {/* Empty state */}
      {chartData.length === 0 && focusPercent === null && (
        <div
          className="rounded-2xl p-8 border border-border bg-card text-center"
          style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}
        >
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

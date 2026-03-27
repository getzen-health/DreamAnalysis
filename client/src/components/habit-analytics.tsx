/**
 * HabitAnalytics -- weekly completion rate chart, best/worst day analysis,
 * and overall completion rate.
 *
 * Uses Recharts BarChart for the 8-week trend.
 * All computation is client-side from habit logs data.
 */

import { useMemo } from "react";
import { motion } from "framer-motion";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { TrendingUp, TrendingDown, CalendarCheck, Target } from "lucide-react";
import { AnimatedNumber } from "@/components/animated-number";
import { cardVariants } from "@/lib/animations";

/* ---------- types ---------- */

interface Habit {
  id: string;
  name: string;
}

interface HabitLog {
  id: string;
  habitId: string | null;
  loggedAt: string | null;
}

interface HabitAnalyticsProps {
  habits: Habit[];
  logs: HabitLog[];
}

/* ---------- helpers ---------- */

const DAY_NAMES = [
  "Sunday",
  "Monday",
  "Tuesday",
  "Wednesday",
  "Thursday",
  "Friday",
  "Saturday",
];
const DAY_SHORT = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

/** Get the Monday of the week containing a given date */
function getWeekMonday(date: Date): string {
  const d = new Date(date);
  d.setHours(12, 0, 0, 0);
  const day = d.getDay();
  const diff = day === 0 ? 6 : day - 1;
  d.setDate(d.getDate() - diff);
  return d.toISOString().slice(0, 10);
}

/** Build weekly completion rate data for last 8 weeks */
function buildWeeklyData(
  habits: Habit[],
  logs: HabitLog[],
): { week: string; label: string; rate: number }[] {
  if (habits.length === 0) return [];

  const totalHabits = habits.length;
  const today = new Date();
  today.setHours(12, 0, 0, 0);

  // Get Monday of current week
  const currentDay = today.getDay();
  const mondayOffset = currentDay === 0 ? 6 : currentDay - 1;
  const currentMonday = new Date(today);
  currentMonday.setDate(today.getDate() - mondayOffset);

  // Build 8 week keys
  const weekKeys: string[] = [];
  for (let i = 7; i >= 0; i--) {
    const monday = new Date(currentMonday);
    monday.setDate(currentMonday.getDate() - i * 7);
    weekKeys.push(monday.toISOString().slice(0, 10));
  }

  // Count unique habit-date combinations per week
  const weekCounts = new Map<string, Set<string>>();
  for (const key of weekKeys) weekCounts.set(key, new Set());

  for (const log of logs) {
    if (!log.habitId || !log.loggedAt) continue;
    const logDate = new Date(log.loggedAt);
    const weekKey = getWeekMonday(logDate);
    if (weekCounts.has(weekKey)) {
      const dateStr = logDate.toISOString().slice(0, 10);
      weekCounts.get(weekKey)!.add(`${log.habitId}|${dateStr}`);
    }
  }

  // Each week has 7 days * totalHabits possible completions
  const maxPerWeek = 7 * totalHabits;

  return weekKeys.map((key) => {
    const count = weekCounts.get(key)?.size ?? 0;
    const d = new Date(key + "T12:00:00");
    return {
      week: key,
      label: `${d.toLocaleDateString(undefined, { month: "short", day: "numeric" })}`,
      rate: Math.round((count / maxPerWeek) * 100),
    };
  });
}

/** Compute per-day-of-week completion rates */
function computeDayOfWeekRates(
  habits: Habit[],
  logs: HabitLog[],
): { day: string; dayIndex: number; rate: number }[] {
  if (habits.length === 0) return [];

  const totalHabits = habits.length;
  // Count unique habit completions per day-of-week
  const dayCounts: number[] = [0, 0, 0, 0, 0, 0, 0];
  const dayTotals: number[] = [0, 0, 0, 0, 0, 0, 0];

  // Look at last 8 weeks
  const eightWeeksAgo = new Date();
  eightWeeksAgo.setDate(eightWeeksAgo.getDate() - 56);
  const cutoff = eightWeeksAgo.toISOString().slice(0, 10);

  // Count how many of each day-of-week exist in the range
  const today = new Date();
  const cursor = new Date(eightWeeksAgo);
  while (cursor <= today) {
    dayTotals[cursor.getDay()] += totalHabits;
    cursor.setDate(cursor.getDate() + 1);
  }

  // Count actual completions per day-of-week
  const seen = new Set<string>();
  for (const log of logs) {
    if (!log.habitId || !log.loggedAt) continue;
    const logDate = new Date(log.loggedAt);
    const dateStr = logDate.toISOString().slice(0, 10);
    if (dateStr < cutoff) continue;
    const key = `${log.habitId}|${dateStr}`;
    if (seen.has(key)) continue;
    seen.add(key);
    dayCounts[logDate.getDay()]++;
  }

  return dayCounts.map((count, i) => ({
    day: DAY_SHORT[i],
    dayIndex: i,
    rate: dayTotals[i] > 0 ? Math.round((count / dayTotals[i]) * 100) : 0,
  }));
}

/** Compute overall completion rate (last 30 days) */
function computeOverallRate(habits: Habit[], logs: HabitLog[]): number {
  if (habits.length === 0) return 0;

  const thirtyDaysAgo = new Date();
  thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);
  const cutoff = thirtyDaysAgo.toISOString().slice(0, 10);

  const seen = new Set<string>();
  for (const log of logs) {
    if (!log.habitId || !log.loggedAt) continue;
    const dateStr = new Date(log.loggedAt).toISOString().slice(0, 10);
    if (dateStr < cutoff) continue;
    seen.add(`${log.habitId}|${dateStr}`);
  }

  const maxCompletions = habits.length * 30;
  return Math.round((seen.size / maxCompletions) * 100);
}

/** Compute last 7 days completion rate */
function computeLast7DaysRate(habits: Habit[], logs: HabitLog[]): number {
  if (habits.length === 0) return 0;

  const sevenDaysAgo = new Date();
  sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);
  const cutoff = sevenDaysAgo.toISOString().slice(0, 10);

  const seen = new Set<string>();
  for (const log of logs) {
    if (!log.habitId || !log.loggedAt) continue;
    const dateStr = new Date(log.loggedAt).toISOString().slice(0, 10);
    if (dateStr < cutoff) continue;
    seen.add(`${log.habitId}|${dateStr}`);
  }

  const maxCompletions = habits.length * 7;
  return Math.round((seen.size / maxCompletions) * 100);
}

/* ========== Component ========== */

export function HabitAnalytics({ habits, logs }: HabitAnalyticsProps) {
  const weeklyData = useMemo(
    () => buildWeeklyData(habits, logs),
    [habits, logs],
  );
  const dayOfWeekRates = useMemo(
    () => computeDayOfWeekRates(habits, logs),
    [habits, logs],
  );
  const overallRate = useMemo(
    () => computeOverallRate(habits, logs),
    [habits, logs],
  );
  const last7Rate = useMemo(
    () => computeLast7DaysRate(habits, logs),
    [habits, logs],
  );

  const bestDay = useMemo(() => {
    if (dayOfWeekRates.length === 0) return null;
    return dayOfWeekRates.reduce((best, curr) =>
      curr.rate > best.rate ? curr : best,
    );
  }, [dayOfWeekRates]);

  const worstDay = useMemo(() => {
    if (dayOfWeekRates.length === 0) return null;
    return dayOfWeekRates.reduce((worst, curr) =>
      curr.rate < worst.rate ? curr : worst,
    );
  }, [dayOfWeekRates]);

  const isOnARoll = last7Rate >= 60;

  if (habits.length === 0) return null;

  return (
    <motion.div
      className="bevel-card p-4 space-y-4"
      custom={2}
      initial="hidden"
      animate="visible"
      variants={cardVariants}
    >
      <h3 className="text-sm font-semibold text-foreground">Analytics</h3>

      {/* Top stats row */}
      <div className="grid grid-cols-2 gap-3">
        {/* Overall completion rate */}
        <div className="rounded-xl bg-muted/30 p-3 text-center space-y-1">
          <Target className="h-4 w-4 mx-auto text-emerald-400" />
          <div className="flex items-baseline justify-center gap-0.5">
            <AnimatedNumber
              value={overallRate}
              duration={1000}
              className="text-2xl font-extrabold tabular-nums text-foreground"
              suffix="%"
            />
          </div>
          <p className="text-[10px] text-muted-foreground">30-day rate</p>
        </div>

        {/* Status */}
        <div className="rounded-xl bg-muted/30 p-3 text-center space-y-1">
          {isOnARoll ? (
            <TrendingUp className="h-4 w-4 mx-auto text-emerald-400" />
          ) : (
            <TrendingDown className="h-4 w-4 mx-auto text-amber-400" />
          )}
          <p
            className={`text-sm font-bold ${
              isOnARoll ? "text-emerald-400" : "text-amber-400"
            }`}
          >
            {isOnARoll ? "On a roll" : "Needs attention"}
          </p>
          <p className="text-[10px] text-muted-foreground">
            {last7Rate}% last 7 days
          </p>
        </div>
      </div>

      {/* Weekly bar chart */}
      <div className="space-y-2">
        <p className="text-xs text-muted-foreground flex items-center gap-1.5">
          <CalendarCheck className="h-3.5 w-3.5" />
          Weekly completion rate
        </p>
        <div className="h-32">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={weeklyData}
              margin={{ top: 4, right: 4, left: -20, bottom: 0 }}
            >
              <XAxis
                dataKey="label"
                tick={{ fontSize: 9, fill: "hsl(var(--muted-foreground))" }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                domain={[0, 100]}
                tick={{ fontSize: 9, fill: "hsl(var(--muted-foreground))" }}
                axisLine={false}
                tickLine={false}
                tickFormatter={(v: number) => `${v}%`}
              />
              <Bar dataKey="rate" radius={[4, 4, 0, 0]} maxBarSize={24}>
                {weeklyData.map((entry, i) => (
                  <Cell
                    key={`cell-${i}`}
                    fill={
                      entry.rate >= 60
                        ? "hsl(160 84% 39%)"
                        : entry.rate >= 30
                        ? "hsl(45 93% 47%)"
                        : "hsl(var(--muted-foreground) / 0.3)"
                    }
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Best / worst day */}
      <div className="grid grid-cols-2 gap-3 text-xs">
        {bestDay && (
          <div className="flex items-center gap-2 rounded-lg bg-emerald-500/10 px-3 py-2">
            <TrendingUp className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
            <div>
              <p className="font-semibold text-emerald-400">
                {DAY_NAMES[bestDay.dayIndex]}s
              </p>
              <p className="text-[10px] text-muted-foreground">
                Most consistent ({bestDay.rate}%)
              </p>
            </div>
          </div>
        )}
        {worstDay && (
          <div className="flex items-center gap-2 rounded-lg bg-amber-500/10 px-3 py-2">
            <TrendingDown className="h-3.5 w-3.5 text-amber-400 shrink-0" />
            <div>
              <p className="font-semibold text-amber-400">
                {DAY_NAMES[worstDay.dayIndex]}s
              </p>
              <p className="text-[10px] text-muted-foreground">
                Least consistent ({worstDay.rate}%)
              </p>
            </div>
          </div>
        )}
      </div>
    </motion.div>
  );
}

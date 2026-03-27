/**
 * HabitHeatmap -- GitHub-style contribution graph for habit completions.
 *
 * Shows a 90-day grid with 7 rows (Mon-Sun) and ~13 columns (weeks).
 * Cell color intensity reflects daily completion rate across all habits.
 * Tap a cell to see date + which habits were completed.
 */

import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cardVariants } from "@/lib/animations";

/* ---------- types ---------- */

interface Habit {
  id: string;
  name: string;
  icon: string | null;
}

interface HabitLog {
  id: string;
  habitId: string | null;
  value: string;
  loggedAt: string | null;
}

interface HabitHeatmapProps {
  habits: Habit[];
  logs: HabitLog[];
}

/* ---------- helpers ---------- */

function getDateStr(date: Date): string {
  return date.toISOString().slice(0, 10);
}

/** Build a map of date -> { habitIds that logged, completionRate } */
function buildDayMap(
  habits: Habit[],
  logs: HabitLog[],
): Map<string, { habitIds: Set<string>; rate: number }> {
  const logsByDate = new Map<string, Set<string>>();

  for (const log of logs) {
    if (!log.habitId || !log.loggedAt) continue;
    const dateStr = new Date(log.loggedAt).toISOString().slice(0, 10);
    if (!logsByDate.has(dateStr)) logsByDate.set(dateStr, new Set());
    logsByDate.get(dateStr)!.add(log.habitId);
  }

  const totalHabits = Math.max(habits.length, 1);
  const result = new Map<string, { habitIds: Set<string>; rate: number }>();

  for (const [date, habitIds] of logsByDate) {
    result.set(date, {
      habitIds,
      rate: habitIds.size / totalHabits,
    });
  }

  return result;
}

/** Build the 90-day grid: an array of weeks, each week an array of 7 days */
function buildGrid(): { date: Date; dateStr: string }[][] {
  const today = new Date();
  today.setHours(12, 0, 0, 0);

  // Find the Monday of the current week
  const dayOfWeek = today.getDay(); // 0=Sun, 1=Mon...
  const mondayOffset = dayOfWeek === 0 ? 6 : dayOfWeek - 1;
  const endMonday = new Date(today);
  endMonday.setDate(today.getDate() - mondayOffset);

  // Go back 12 more weeks (13 total including current)
  const startMonday = new Date(endMonday);
  startMonday.setDate(endMonday.getDate() - 12 * 7);

  const weeks: { date: Date; dateStr: string }[][] = [];
  const cursor = new Date(startMonday);

  for (let w = 0; w < 13; w++) {
    const week: { date: Date; dateStr: string }[] = [];
    for (let d = 0; d < 7; d++) {
      week.push({ date: new Date(cursor), dateStr: getDateStr(cursor) });
      cursor.setDate(cursor.getDate() + 1);
    }
    weeks.push(week);
  }

  return weeks;
}

/** Get intensity class based on completion rate */
function getCellColor(rate: number): string {
  if (rate === 0) return "bg-gray-800";
  if (rate <= 0.25) return "bg-emerald-900";
  if (rate <= 0.5) return "bg-emerald-700";
  if (rate <= 0.75) return "bg-emerald-500";
  return "bg-emerald-400";
}

/** Format date for tooltip display */
function formatDate(date: Date): string {
  return date.toLocaleDateString(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
  });
}

/* ---------- month labels ---------- */

function getMonthLabels(weeks: { date: Date; dateStr: string }[][]): { label: string; colIndex: number }[] {
  const labels: { label: string; colIndex: number }[] = [];
  let lastMonth = -1;

  for (let w = 0; w < weeks.length; w++) {
    // Use the Monday of each week
    const month = weeks[w][0].date.getMonth();
    if (month !== lastMonth) {
      labels.push({
        label: weeks[w][0].date.toLocaleDateString(undefined, { month: "short" }),
        colIndex: w,
      });
      lastMonth = month;
    }
  }

  return labels;
}

/* ---------- day labels ---------- */

const DAY_LABELS: { index: number; label: string }[] = [
  { index: 0, label: "M" },
  { index: 2, label: "W" },
  { index: 4, label: "F" },
];

/* ========== Component ========== */

export function HabitHeatmap({ habits, logs }: HabitHeatmapProps) {
  const dayMap = useMemo(() => buildDayMap(habits, logs), [habits, logs]);
  const weeks = useMemo(() => buildGrid(), []);
  const monthLabels = useMemo(() => getMonthLabels(weeks), [weeks]);
  const today = getDateStr(new Date());

  const [activeCell, setActiveCell] = useState<string | null>(null);

  const habitNameMap = useMemo(() => {
    const map = new Map<string, string>();
    for (const h of habits) map.set(h.id, h.name);
    return map;
  }, [habits]);

  if (habits.length === 0) return null;

  return (
    <TooltipProvider delayDuration={100}>
      <motion.div
        className="bevel-card p-4 space-y-2"
        custom={0}
        initial="hidden"
        animate="visible"
        variants={cardVariants}
      >
        <div className="flex items-center justify-between mb-1">
          <h3 className="text-sm font-semibold text-foreground">Activity</h3>
          <span className="text-[10px] text-muted-foreground">Last 90 days</span>
        </div>

        {/* Month labels */}
        <div className="flex">
          {/* Spacer for day labels column */}
          <div className="w-5 shrink-0" />
          <div className="flex-1 grid" style={{ gridTemplateColumns: `repeat(13, 1fr)` }}>
            {Array.from({ length: 13 }).map((_, i) => {
              const ml = monthLabels.find((m) => m.colIndex === i);
              return (
                <span
                  key={i}
                  className="text-[9px] text-muted-foreground/60 leading-none"
                >
                  {ml?.label ?? ""}
                </span>
              );
            })}
          </div>
        </div>

        {/* Grid body */}
        <div className="flex gap-0">
          {/* Day labels */}
          <div className="flex flex-col w-5 shrink-0">
            {Array.from({ length: 7 }).map((_, row) => {
              const dayLabel = DAY_LABELS.find((d) => d.index === row);
              return (
                <div
                  key={row}
                  className="flex items-center justify-end pr-1"
                  style={{ height: "calc((100% - 0px) / 7)" }}
                >
                  <span className="text-[9px] text-muted-foreground/60 leading-none">
                    {dayLabel?.label ?? ""}
                  </span>
                </div>
              );
            })}
          </div>

          {/* Heatmap cells */}
          <div
            className="flex-1 grid gap-[2px]"
            style={{
              gridTemplateColumns: `repeat(13, 1fr)`,
              gridTemplateRows: `repeat(7, 1fr)`,
            }}
          >
            {/* Render column by column (week by week), row by row (day by day) */}
            {Array.from({ length: 7 }).map((_, row) =>
              weeks.map((week, col) => {
                const day = week[row];
                const dayData = dayMap.get(day.dateStr);
                const rate = dayData?.rate ?? 0;
                const isFuture = day.dateStr > today;
                const isToday = day.dateStr === today;
                const completedHabits = dayData?.habitIds
                  ? Array.from(dayData.habitIds)
                      .map((id) => habitNameMap.get(id))
                      .filter(Boolean)
                  : [];

                if (isFuture) {
                  return (
                    <div
                      key={`${row}-${col}`}
                      className="aspect-square rounded-[3px] bg-transparent"
                    />
                  );
                }

                return (
                  <Tooltip key={`${row}-${col}`}>
                    <TooltipTrigger asChild>
                      <button
                        className={`aspect-square rounded-[3px] transition-all duration-150 ${getCellColor(
                          rate,
                        )} ${
                          isToday ? "ring-1 ring-emerald-400/50" : ""
                        } ${
                          activeCell === day.dateStr
                            ? "ring-1 ring-white/30 scale-125 z-10"
                            : ""
                        } hover:scale-110 focus:outline-none`}
                        onClick={() =>
                          setActiveCell(
                            activeCell === day.dateStr ? null : day.dateStr,
                          )
                        }
                        aria-label={`${formatDate(day.date)}: ${
                          completedHabits.length
                        } habits completed`}
                      />
                    </TooltipTrigger>
                    <TooltipContent
                      side="top"
                      className="max-w-[200px] text-xs"
                    >
                      <p className="font-semibold">{formatDate(day.date)}</p>
                      {completedHabits.length > 0 ? (
                        <ul className="mt-1 space-y-0.5">
                          {completedHabits.map((name) => (
                            <li key={name} className="text-emerald-400">
                              {name}
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <p className="text-muted-foreground mt-0.5">
                          No habits logged
                        </p>
                      )}
                    </TooltipContent>
                  </Tooltip>
                );
              }),
            )}
          </div>
        </div>

        {/* Legend */}
        <div className="flex items-center justify-end gap-1.5 pt-1">
          <span className="text-[9px] text-muted-foreground/50">Less</span>
          {[0, 0.25, 0.5, 0.75, 1].map((rate) => (
            <div
              key={rate}
              className={`w-2.5 h-2.5 rounded-[2px] ${getCellColor(rate)}`}
            />
          ))}
          <span className="text-[9px] text-muted-foreground/50">More</span>
        </div>
      </motion.div>
    </TooltipProvider>
  );
}

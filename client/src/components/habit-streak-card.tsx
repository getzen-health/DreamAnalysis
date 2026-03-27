/**
 * HabitStreakCard -- per-habit streak display in a compact card.
 *
 * Shows: habit name + icon, current streak, longest streak, completion rate.
 * Fire icon for active streaks (>= 3 days), trophy for record-breaking.
 * Animated counter on mount. Color-coded by streak health.
 */

import { useMemo } from "react";
import { motion } from "framer-motion";
import {
  Flame,
  Trophy,
  Droplets,
  Sun,
  Monitor,
  Coffee,
  Brain,
  Dumbbell,
  ListChecks,
} from "lucide-react";
import { AnimatedNumber } from "@/components/animated-number";
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
  loggedAt: string | null;
}

interface HabitStreakCardProps {
  habit: Habit;
  currentStreak: number;
  logs: HabitLog[];
  index: number;
}

/* ---------- icon map ---------- */

const ICON_MAP: Record<string, typeof Droplets> = {
  droplets: Droplets,
  sun: Sun,
  monitor: Monitor,
  coffee: Coffee,
  brain: Brain,
  dumbbell: Dumbbell,
  flame: Flame,
  check: ListChecks,
};

function getIcon(iconName: string | null) {
  if (!iconName) return ListChecks;
  return ICON_MAP[iconName] ?? ListChecks;
}

/* ---------- helpers ---------- */

/** Compute longest streak from logs for a specific habit */
function computeLongestStreak(habitId: string, logs: HabitLog[]): number {
  const dates = new Set<string>();
  for (const log of logs) {
    if (log.habitId !== habitId || !log.loggedAt) continue;
    dates.add(new Date(log.loggedAt).toISOString().slice(0, 10));
  }

  if (dates.size === 0) return 0;

  const sorted = Array.from(dates).sort();
  let longest = 1;
  let current = 1;

  for (let i = 1; i < sorted.length; i++) {
    const prev = new Date(sorted[i - 1] + "T12:00:00");
    const curr = new Date(sorted[i] + "T12:00:00");
    const diffDays = Math.round(
      (curr.getTime() - prev.getTime()) / (24 * 60 * 60 * 1000),
    );

    if (diffDays === 1) {
      current++;
      longest = Math.max(longest, current);
    } else {
      current = 1;
    }
  }

  return longest;
}

/** Compute completion rate over last 30 days */
function computeCompletionRate(habitId: string, logs: HabitLog[]): number {
  const thirtyDaysAgo = new Date();
  thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);
  const cutoff = thirtyDaysAgo.toISOString().slice(0, 10);

  const dates = new Set<string>();
  for (const log of logs) {
    if (log.habitId !== habitId || !log.loggedAt) continue;
    const dateStr = new Date(log.loggedAt).toISOString().slice(0, 10);
    if (dateStr >= cutoff) dates.add(dateStr);
  }

  return Math.round((dates.size / 30) * 100);
}

/** Get streak status color */
function getStreakColor(streak: number): {
  bg: string;
  text: string;
  border: string;
} {
  if (streak === 0) {
    return {
      bg: "bg-gray-800/50",
      text: "text-muted-foreground",
      border: "border-border/30",
    };
  }
  if (streak === 1) {
    return {
      bg: "bg-amber-500/10",
      text: "text-amber-400",
      border: "border-amber-500/20",
    };
  }
  return {
    bg: "bg-emerald-500/10",
    text: "text-emerald-400",
    border: "border-emerald-500/20",
  };
}

/* ========== Component ========== */

export function HabitStreakCard({
  habit,
  currentStreak,
  logs,
  index,
}: HabitStreakCardProps) {
  const longestStreak = useMemo(
    () => computeLongestStreak(habit.id, logs),
    [habit.id, logs],
  );
  const completionRate = useMemo(
    () => computeCompletionRate(habit.id, logs),
    [habit.id, logs],
  );

  const Icon = getIcon(habit.icon);
  const isActive = currentStreak >= 3;
  const isRecord = currentStreak > 0 && currentStreak >= longestStreak;
  const colors = getStreakColor(currentStreak);

  return (
    <motion.div
      className={`premium-card p-3 min-w-[140px] max-w-[160px] shrink-0 border ${colors.border} space-y-2`}
      custom={index}
      initial="hidden"
      animate="visible"
      variants={cardVariants}
    >
      {/* Habit icon + name */}
      <div className="flex items-center gap-2">
        <div
          className={`w-7 h-7 rounded-lg flex items-center justify-center shrink-0 ${colors.bg}`}
        >
          <Icon className={`h-3.5 w-3.5 ${colors.text}`} />
        </div>
        <span className="text-xs font-semibold text-foreground truncate">
          {habit.name}
        </span>
      </div>

      {/* Streak number */}
      <div className="flex items-baseline gap-1.5">
        <AnimatedNumber
          value={currentStreak}
          duration={800}
          className={`text-2xl font-extrabold tabular-nums ${colors.text}`}
        />
        <span className="text-[10px] text-muted-foreground">days</span>
        {isActive && (
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: "spring", stiffness: 300, damping: 15, delay: 0.3 }}
          >
            <Flame className="h-4 w-4 text-orange-400" />
          </motion.div>
        )}
        {isRecord && currentStreak > 0 && (
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: "spring", stiffness: 300, damping: 15, delay: 0.4 }}
          >
            <Trophy className="h-3.5 w-3.5 text-amber-400" />
          </motion.div>
        )}
      </div>

      {/* Stats row */}
      <div className="flex items-center justify-between text-[10px] text-muted-foreground">
        <span>Best: {longestStreak}d</span>
        <span>{completionRate}%</span>
      </div>
    </motion.div>
  );
}

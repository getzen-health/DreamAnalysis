import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { useAuth } from "@/hooks/use-auth";
import { useHealthSync } from "@/hooks/use-health-sync";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Dumbbell,
  Clock,
  Flame,
  Heart,
  RefreshCw,
  Smartphone,
  TrendingUp,
} from "lucide-react";
import { motion } from "framer-motion";
import { EmotionStrip } from "@/components/emotion-strip";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

/* ---------- types ---------- */

interface Workout {
  id: string;
  userId: string;
  name: string | null;
  workoutType: string;
  startedAt: string;
  endedAt: string | null;
  durationMin: string | null;
  totalStrain: string | null;
  avgHr: string | null;
  maxHr: string | null;
  caloriesBurned: string | null;
  hrZones: unknown;
  hrRecovery: string | null;
  source: string;
  eegSessionId: string | null;
  notes: string | null;
  createdAt: string;
}

/* ---------- helpers ---------- */

function formatDurationMin(minutes: number): string {
  if (minutes < 60) return `${Math.round(minutes)}m`;
  const h = Math.floor(minutes / 60);
  const m = Math.round(minutes % 60);
  return m > 0 ? `${h}h ${m}m` : `${h}h`;
}

function formatWorkoutType(type: string): string {
  // Clean up HealthKit workout types like "HKWorkoutActivityTypeRunning"
  const cleaned = type
    .replace(/^HKWorkoutActivityType/, "")
    .replace(/([A-Z])/g, " $1")
    .trim();
  return cleaned.charAt(0).toUpperCase() + cleaned.slice(1);
}

function getWorkoutEmoji(type: string): string {
  const lower = type.toLowerCase();
  if (lower.includes("run")) return "🏃";
  if (lower.includes("walk")) return "🚶";
  if (lower.includes("cycl") || lower.includes("bik")) return "🚴";
  if (lower.includes("swim")) return "🏊";
  if (lower.includes("yoga")) return "🧘";
  if (lower.includes("strength") || lower.includes("weight") || lower.includes("functional")) return "🏋️";
  if (lower.includes("hiit") || lower.includes("interval") || lower.includes("cross")) return "⚡";
  if (lower.includes("pilates")) return "🤸";
  if (lower.includes("dance")) return "💃";
  if (lower.includes("hik")) return "🥾";
  if (lower.includes("row")) return "🚣";
  return "🏋️";
}

function isToday(iso: string): boolean {
  return new Date(iso).toDateString() === new Date().toDateString();
}

function isThisWeek(iso: string): boolean {
  const d = new Date(iso);
  const now = new Date();
  const weekStart = new Date(now);
  weekStart.setDate(now.getDate() - now.getDay());
  weekStart.setHours(0, 0, 0, 0);
  return d >= weekStart;
}

function formatSyncTime(date: Date | null): string {
  if (!date) return "Never";
  const mins = Math.floor((Date.now() - date.getTime()) / 60000);
  if (mins < 1) return "Just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

/* ---------- animation variants ---------- */

const fadeInUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.35, ease: "easeOut" as const },
};

/* ---------- Workout Card ---------- */

function WorkoutCard({ workout }: { workout: Workout }) {
  const durationMin = workout.durationMin ? parseFloat(workout.durationMin) : null;
  const calories = workout.caloriesBurned ? parseFloat(workout.caloriesBurned) : null;
  const avgHr = workout.avgHr ? parseFloat(workout.avgHr) : null;
  const strain = workout.totalStrain ? parseFloat(workout.totalStrain) : null;
  const startDate = new Date(workout.startedAt);
  const dateStr = startDate.toLocaleDateString("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
  });
  const timeStr = startDate.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const emoji = getWorkoutEmoji(workout.workoutType);
  const typeName = formatWorkoutType(workout.workoutType);

  return (
    <div
      style={{
        background: "var(--card)",
        border: "1px solid var(--border)",
        boxShadow: "0 2px 16px rgba(0,0,0,0.06)",
        borderRadius: 20,
        padding: 14,
      }}
    >
      {/* Header row */}
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
        <span style={{ fontSize: 24 }}>{emoji}</span>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 14, fontWeight: 600, color: "var(--foreground)" }}>
            {workout.name || typeName}
          </div>
          <div style={{ fontSize: 11, color: "var(--muted-foreground)", marginTop: 2 }}>
            {dateStr} at {timeStr}
          </div>
        </div>
        <Badge variant="outline" className="text-[10px]">
          {workout.source === "apple_health" ? "Apple Health" : workout.source === "google_fit" ? "Google Health" : workout.source}
        </Badge>
      </div>

      {/* Stats row */}
      <div style={{ display: "flex", gap: 12 }}>
        {durationMin !== null && (
          <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <Clock className="h-3.5 w-3.5 text-muted-foreground" />
            <span style={{ fontSize: 12, fontWeight: 600, color: "var(--foreground)" }}>
              {formatDurationMin(durationMin)}
            </span>
          </div>
        )}
        {calories !== null && calories > 0 && (
          <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <Flame className="h-3.5 w-3.5 text-orange-400" />
            <span style={{ fontSize: 12, fontWeight: 600, color: "var(--foreground)" }}>
              {Math.round(calories)} kcal
            </span>
          </div>
        )}
        {avgHr !== null && avgHr > 0 && (
          <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <Heart className="h-3.5 w-3.5 text-rose-400" />
            <span style={{ fontSize: 12, fontWeight: 600, color: "var(--foreground)" }}>
              {Math.round(avgHr)} bpm
            </span>
          </div>
        )}
        {strain !== null && strain > 0 && (
          <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <TrendingUp className="h-3.5 w-3.5 text-amber-400" />
            <span style={{ fontSize: 12, fontWeight: 600, color: "var(--foreground)" }}>
              {strain.toFixed(1)} strain
            </span>
          </div>
        )}
      </div>

      {/* Notes */}
      {workout.notes && (
        <p style={{
          fontSize: 11, color: "var(--muted-foreground)", marginTop: 8,
          fontStyle: "italic", lineHeight: 1.4,
        }}>
          {workout.notes}
        </p>
      )}
    </div>
  );
}

/* ========== Component ========== */

export default function WorkoutPage() {
  const { user } = useAuth();
  const { lastSyncAt, syncNow, status, isAvailable } = useHealthSync();

  const isSyncing = status === "syncing";

  /* ---- Query workout history ---- */
  const { data: workoutHistory = [] } = useQuery<Workout[]>({
    queryKey: [`/api/workouts/${user?.id}`],
    enabled: !!user?.id,
  });

  /* ---- Derived data ---- */
  const todayWorkouts = useMemo(
    () => workoutHistory.filter((w) => isToday(w.startedAt)),
    [workoutHistory],
  );

  const thisWeekWorkouts = useMemo(
    () => workoutHistory.filter((w) => isThisWeek(w.startedAt)),
    [workoutHistory],
  );

  const weeklyStats = useMemo(() => {
    let totalMin = 0;
    let totalCal = 0;
    let count = 0;
    for (const w of thisWeekWorkouts) {
      count++;
      if (w.durationMin) totalMin += parseFloat(w.durationMin);
      if (w.caloriesBurned) totalCal += parseFloat(w.caloriesBurned);
    }
    return { totalMin, totalCal, count };
  }, [thisWeekWorkouts]);

  // Weekly exercise minutes chart (last 4 weeks)
  const weeklyChart = useMemo(() => {
    const now = new Date();
    const weeks: { label: string; minutes: number }[] = [];
    for (let i = 3; i >= 0; i--) {
      const weekEnd = new Date(now);
      weekEnd.setDate(now.getDate() - i * 7);
      const weekStart = new Date(weekEnd);
      weekStart.setDate(weekEnd.getDate() - 7);
      const mins = workoutHistory
        .filter((w) => {
          const d = new Date(w.startedAt);
          return d >= weekStart && d < weekEnd;
        })
        .reduce((sum, w) => sum + (w.durationMin ? parseFloat(w.durationMin) : 0), 0);
      weeks.push({
        label: weekStart.toLocaleDateString("en-US", { month: "short", day: "numeric" }),
        minutes: Math.round(mins),
      });
    }
    return weeks;
  }, [workoutHistory]);

  const hasChartData = weeklyChart.some((w) => w.minutes > 0);

  return (
    <div className="max-w-lg mx-auto px-4 py-6 space-y-4 pb-24">
      {/* Header */}
      <motion.div
        className="space-y-2 mb-2"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl flex items-center justify-center bg-gradient-to-br from-ndw-recovery to-ndw-stress">
            <Dumbbell className="h-5 w-5 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-foreground">Workouts</h1>
            <p className="text-xs text-muted-foreground">Imported from your health app</p>
          </div>
        </div>
        <EmotionStrip />
      </motion.div>

      {/* Sync Status + Button */}
      <motion.div
        className="rounded-2xl p-4 bg-card border border-border shadow-[0_2px_16px_rgba(0,0,0,0.06)]"
        {...fadeInUp}
        transition={{ ...fadeInUp.transition, delay: 0.05 }}
      >
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Smartphone className="h-4 w-4 text-primary" />
            <p className="text-[13px] font-semibold text-foreground">Health Sync</p>
          </div>
          <span className="text-[10px] text-muted-foreground">
            Last synced: {formatSyncTime(lastSyncAt)}
          </span>
        </div>

        {isAvailable ? (
          <Button
            onClick={() => syncNow()}
            disabled={isSyncing}
            className="w-full h-11 text-sm font-semibold gap-2"
            variant="outline"
          >
            <RefreshCw className={`h-4 w-4 ${isSyncing ? "animate-spin" : ""}`} />
            {isSyncing ? "Syncing Workouts..." : "Sync Workouts"}
          </Button>
        ) : (
          <div className="space-y-2">
            <p className="text-sm text-muted-foreground">
              Your workouts are automatically imported from Google Health or Apple Health.
            </p>
            <p className="text-xs text-muted-foreground/60">
              Connect your health app in Settings to see workout data here.
            </p>
          </div>
        )}
      </motion.div>

      {/* Weekly Summary */}
      {weeklyStats.count > 0 && (
        <motion.div
          className="rounded-2xl p-4 bg-card border border-border shadow-[0_2px_16px_rgba(0,0,0,0.06)]"
          {...fadeInUp}
          transition={{ ...fadeInUp.transition, delay: 0.1 }}
        >
          <p className="text-[11px] font-semibold text-muted-foreground/60 uppercase tracking-[0.08em] mb-3">
            This Week
          </p>
          <div className="grid grid-cols-3 gap-3 text-center">
            <div>
              <p className="text-2xl font-bold text-foreground">{weeklyStats.count}</p>
              <p className="text-[10px] text-muted-foreground">Workouts</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-foreground">
                {formatDurationMin(weeklyStats.totalMin)}
              </p>
              <p className="text-[10px] text-muted-foreground">Active Time</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-orange-400">
                {Math.round(weeklyStats.totalCal).toLocaleString()}
              </p>
              <p className="text-[10px] text-muted-foreground">Calories</p>
            </div>
          </div>
        </motion.div>
      )}

      {/* Exercise Minutes Chart (4 weeks) */}
      {hasChartData && (
        <motion.div
          className="rounded-2xl p-4 bg-card border border-border shadow-[0_2px_16px_rgba(0,0,0,0.06)]"
          {...fadeInUp}
          transition={{ ...fadeInUp.transition, delay: 0.15 }}
        >
          <div className="flex items-center gap-2 mb-3">
            <TrendingUp className="h-4 w-4 text-primary" />
            <p className="text-[13px] font-semibold text-foreground">Weekly Activity</p>
            <span className="text-[10px] text-muted-foreground ml-auto">Last 4 weeks</span>
          </div>
          <ResponsiveContainer width="100%" height={240}>
            <AreaChart data={weeklyChart} margin={{ left: 0, right: 4, top: 4, bottom: 0 }}>
              <defs>
                <linearGradient id="workoutGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#0891b2" stopOpacity={0.3} />
                  <stop offset="50%" stopColor="#0891b2" stopOpacity={0.12} />
                  <stop offset="100%" stopColor="#0891b2" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.5} />
              <XAxis
                dataKey="label"
                tick={{ fontSize: 9, fill: "var(--muted-foreground)" }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                tick={{ fontSize: 9, fill: "var(--muted-foreground)" }}
                axisLine={false}
                tickLine={false}
                width={28}
                tickFormatter={(v) => `${v}m`}
              />
              <Tooltip
                contentStyle={{
                  background: "var(--popover)",
                  border: "1px solid var(--border)",
        boxShadow: "0 2px 16px rgba(0,0,0,0.06)",
                  borderRadius: 10,
                  fontSize: 11,
                  color: "var(--foreground)",
                }}
                formatter={(v: number) => [`${v} min`, "Exercise"]}
              />
              <Area
                type="monotone"
                dataKey="minutes"
                stroke="#0891b2"
                strokeWidth={2}
                fill="url(#workoutGrad)"
                dot={{ r: 3, fill: "#0891b2" }}
                activeDot={{ r: 5 }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </motion.div>
      )}

      {/* Today's Workouts */}
      {todayWorkouts.length > 0 && (
        <motion.div
          {...fadeInUp}
          transition={{ ...fadeInUp.transition, delay: 0.2 }}
        >
          <p className="text-[11px] font-semibold text-muted-foreground/60 uppercase tracking-[0.08em] mb-2">
            Today
          </p>
          <div className="space-y-2">
            {todayWorkouts.map((w) => (
              <WorkoutCard key={w.id} workout={w} />
            ))}
          </div>
        </motion.div>
      )}

      {/* Workout History */}
      <motion.div
        {...fadeInUp}
        transition={{ ...fadeInUp.transition, delay: 0.25 }}
      >
        <p className="text-[11px] font-semibold text-muted-foreground/60 uppercase tracking-[0.08em] mb-2">
          {todayWorkouts.length > 0 ? "Previous Workouts" : "Workout History"}
        </p>
        {workoutHistory.length === 0 ? (
          <div className="rounded-2xl p-6 text-center bg-card border border-border shadow-[0_2px_16px_rgba(0,0,0,0.06)]">
            <Dumbbell className="h-8 w-8 mx-auto mb-3 text-muted-foreground/40" />
            <p className="text-[13px] font-medium text-foreground/70">
              No workouts yet
            </p>
            <p className="text-[11px] text-muted-foreground mt-1">
              Your workouts are automatically imported from Google Health or Apple Health
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            {workoutHistory
              .filter((w) => !isToday(w.startedAt))
              .slice(0, 20)
              .map((w) => (
                <WorkoutCard key={w.id} workout={w} />
              ))}
          </div>
        )}
      </motion.div>
    </div>
  );
}

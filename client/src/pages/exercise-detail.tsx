import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { useLocation, useParams } from "wouter";
import { useAuth } from "@/hooks/use-auth";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  ArrowLeft,
  Dumbbell,
  Trophy,
  TrendingUp,
  Calendar,
  Weight,
} from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { motion } from "framer-motion";
import { pageTransition, cardVariants } from "@/lib/animations";
import type { Exercise, ExerciseHistoryRecord, PersonalRecord } from "@/lib/workout-types";
import { getMuscleGroupColor, getCategoryColor } from "@/lib/workout-types";

/* ---------- Chart Tooltip ---------- */

function CustomTooltip({ active, payload, label }: { active?: boolean; payload?: Array<{ value: number; dataKey: string }>; label?: string }) {
  if (!active || !payload?.length) return null;

  return (
    <div className="rounded-lg border border-border bg-card p-2 shadow-lg text-xs">
      <p className="text-muted-foreground mb-1">{label}</p>
      {payload.map((p) => (
        <p key={p.dataKey} className="text-foreground font-medium">
          {p.dataKey === "estimated1rm"
            ? `Est. 1RM: ${p.value.toFixed(1)} kg`
            : p.dataKey === "bestWeightKg"
            ? `Best Weight: ${p.value.toFixed(1)} kg`
            : `Volume: ${Math.round(p.value)} kg`}
        </p>
      ))}
    </div>
  );
}

/* ========== Component ========== */

export default function ExerciseDetail() {
  const { user } = useAuth();
  const [, setLocation] = useLocation();
  const params = useParams<{ id: string }>();
  const exerciseId = params.id;

  // Fetch exercise details
  const { data: exercise, isLoading: exerciseLoading } = useQuery<Exercise>({
    queryKey: [`/api/exercises/${exerciseId}`],
    enabled: !!exerciseId,
    staleTime: 5 * 60_000,
  });

  // Fetch exercise progression history
  const { data: history = [] } = useQuery<ExerciseHistoryRecord[]>({
    queryKey: [`/api/exercise-history/${user?.id}/${exerciseId}`],
    enabled: !!user?.id && !!exerciseId,
    staleTime: 30_000,
  });

  // Fetch all personal records to find this exercise's PR
  const { data: allPRs = [] } = useQuery<PersonalRecord[]>({
    queryKey: [`/api/exercise-history/${user?.id}/prs`],
    enabled: !!user?.id,
    staleTime: 60_000,
  });

  // Find this exercise's PR
  const pr = useMemo(
    () => allPRs.find((p) => p.exerciseId === exerciseId) ?? null,
    [allPRs, exerciseId]
  );

  // Chart data
  const chartData = useMemo(
    () =>
      history.map((h) => ({
        date: new Date(h.date).toLocaleDateString("en-US", {
          month: "short",
          day: "numeric",
        }),
        estimated1rm: h.estimated1rm ?? 0,
        bestWeightKg: h.bestWeightKg ?? 0,
        totalVolume: h.totalVolume ?? 0,
      })),
    [history]
  );

  if (exerciseLoading) {
    return (
      <div className="max-w-lg mx-auto px-4 py-6 space-y-4">
        <div className="animate-pulse space-y-4">
          <div className="h-6 bg-muted rounded w-1/3" />
          <div className="h-4 bg-muted rounded w-2/3" />
          <div className="h-48 bg-muted rounded" />
        </div>
      </div>
    );
  }

  if (!exercise) {
    return (
      <div className="max-w-lg mx-auto px-4 py-6 text-center">
        <p className="text-muted-foreground">Exercise not found</p>
        <Button
          variant="outline"
          size="sm"
          className="mt-4"
          onClick={() => setLocation("/exercises")}
        >
          Back to Library
        </Button>
      </div>
    );
  }

  return (
    <div className="max-w-lg mx-auto px-4 py-6 space-y-6 pb-24">
      {/* Header */}
      <motion.div
        className="space-y-4"
        initial={pageTransition.initial}
        animate={pageTransition.animate}
        transition={pageTransition.transition}
      >
        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            size="sm"
            className="h-9 w-9 p-0"
            onClick={() => setLocation("/exercises")}
          >
            <ArrowLeft className="h-5 w-5" />
          </Button>
          <div className="flex-1 min-w-0">
            <h1 className="text-xl font-bold tracking-tight text-foreground truncate">
              {exercise.name}
            </h1>
          </div>
        </div>

        {/* Category + Equipment */}
        <div className="flex flex-wrap gap-2">
          <Badge
            variant="secondary"
            className={getCategoryColor(exercise.category)}
          >
            {exercise.category}
          </Badge>
          {exercise.equipment && (
            <Badge variant="outline" className="text-xs">
              {exercise.equipment}
            </Badge>
          )}
        </div>

        {/* Muscle Groups */}
        <div className="flex flex-wrap gap-1.5">
          {exercise.muscleGroups.map((mg) => (
            <span
              key={mg}
              className={`inline-flex items-center px-2 py-1 rounded-md text-xs font-medium border ${getMuscleGroupColor(mg)}`}
            >
              {mg}
            </span>
          ))}
        </div>
      </motion.div>

      {/* Instructions */}
      {exercise.instructions && (
        <motion.div
          className="rounded-xl border border-border bg-card p-4"
          custom={0}
          initial="hidden"
          animate="visible"
          variants={cardVariants}
        >
          <p className="text-[11px] font-semibold text-muted-foreground/60 uppercase tracking-[0.08em] mb-2">
            Instructions
          </p>
          <p className="text-sm text-muted-foreground leading-relaxed whitespace-pre-line">
            {exercise.instructions}
          </p>
        </motion.div>
      )}

      {/* Personal Records */}
      {pr && (
        <motion.div
          className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-4"
          custom={1}
          initial="hidden"
          animate="visible"
          variants={cardVariants}
        >
          <p className="text-[11px] font-semibold text-amber-400 uppercase tracking-[0.08em] mb-3 flex items-center gap-1">
            <Trophy className="h-3.5 w-3.5" />
            Personal Records
          </p>
          <div className="grid grid-cols-3 gap-3 text-center">
            <div>
              <p className="text-xl font-bold text-foreground">
                {pr.estimated1rm != null
                  ? `${pr.estimated1rm.toFixed(1)}`
                  : "---"}
              </p>
              <p className="text-[10px] text-muted-foreground">Est. 1RM (kg)</p>
            </div>
            <div>
              <p className="text-xl font-bold text-foreground">
                {pr.bestWeightKg != null
                  ? `${pr.bestWeightKg.toFixed(1)}`
                  : "---"}
              </p>
              <p className="text-[10px] text-muted-foreground">Max Weight (kg)</p>
            </div>
            <div>
              <p className="text-xl font-bold text-foreground">
                {pr.bestReps ?? "---"}
              </p>
              <p className="text-[10px] text-muted-foreground">Max Reps</p>
            </div>
          </div>
          {pr.date && (
            <p className="text-[10px] text-muted-foreground mt-2 text-center flex items-center justify-center gap-1">
              <Calendar className="h-3 w-3" />
              PR set on{" "}
              {new Date(pr.date).toLocaleDateString("en-US", {
                month: "short",
                day: "numeric",
                year: "numeric",
              })}
            </p>
          )}
        </motion.div>
      )}

      {/* Weight Progression Chart */}
      <motion.div
        className="rounded-xl border border-border bg-card p-4"
        custom={2}
        initial="hidden"
        animate="visible"
        variants={cardVariants}
      >
        <p className="text-[11px] font-semibold text-muted-foreground/60 uppercase tracking-[0.08em] mb-3 flex items-center gap-1">
          <TrendingUp className="h-3.5 w-3.5" />
          Weight Progression
        </p>

        {chartData.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            <Weight className="h-8 w-8 mx-auto mb-2 opacity-40" />
            <p className="text-xs font-medium">No history yet</p>
            <p className="text-[10px] mt-1">
              Complete a workout with this exercise to see progression
            </p>
          </div>
        ) : (
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="hsl(var(--border))"
                  vertical={false}
                />
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                  tickLine={false}
                  axisLine={false}
                />
                <YAxis
                  tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                  tickLine={false}
                  axisLine={false}
                  width={40}
                />
                <Tooltip content={<CustomTooltip />} />
                <Line
                  type="monotone"
                  dataKey="estimated1rm"
                  stroke="hsl(var(--primary))"
                  strokeWidth={2}
                  dot={{ r: 3, fill: "hsl(var(--primary))" }}
                  activeDot={{ r: 5 }}
                  name="Est. 1RM"
                />
                <Line
                  type="monotone"
                  dataKey="bestWeightKg"
                  stroke="hsl(var(--chart-2))"
                  strokeWidth={1.5}
                  strokeDasharray="4 4"
                  dot={{ r: 2 }}
                  name="Best Weight"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </motion.div>

      {/* Volume Chart */}
      {chartData.length > 0 && (
        <motion.div
          className="rounded-xl border border-border bg-card p-4"
          custom={3}
          initial="hidden"
          animate="visible"
          variants={cardVariants}
        >
          <p className="text-[11px] font-semibold text-muted-foreground/60 uppercase tracking-[0.08em] mb-3">
            Total Volume per Session
          </p>
          <div className="h-36">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="hsl(var(--border))"
                  vertical={false}
                />
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                  tickLine={false}
                  axisLine={false}
                />
                <YAxis
                  tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                  tickLine={false}
                  axisLine={false}
                  width={50}
                />
                <Tooltip content={<CustomTooltip />} />
                <Line
                  type="monotone"
                  dataKey="totalVolume"
                  stroke="#22c55e"
                  strokeWidth={2}
                  dot={{ r: 3, fill: "#22c55e" }}
                  activeDot={{ r: 5 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </motion.div>
      )}

      {/* Session History */}
      {history.length > 0 && (
        <motion.div
          className="space-y-2"
          custom={4}
          initial="hidden"
          animate="visible"
          variants={cardVariants}
        >
          <p className="text-[11px] font-semibold text-muted-foreground/60 uppercase tracking-[0.08em]">
            Session History
          </p>
          {history
            .slice()
            .reverse()
            .slice(0, 10)
            .map((record) => (
              <div
                key={record.date}
                className="rounded-xl border border-border bg-card p-3 flex items-center justify-between"
              >
                <div>
                  <p className="text-xs font-medium text-foreground">
                    {new Date(record.date).toLocaleDateString("en-US", {
                      weekday: "short",
                      month: "short",
                      day: "numeric",
                    })}
                  </p>
                  <p className="text-[10px] text-muted-foreground mt-0.5">
                    {record.bestWeightKg != null
                      ? `${record.bestWeightKg} kg`
                      : "---"}{" "}
                    / {record.bestReps ?? "---"} reps
                  </p>
                </div>
                <div className="text-right">
                  {record.estimated1rm != null && (
                    <p className="text-xs font-semibold text-primary">
                      1RM: {record.estimated1rm.toFixed(1)} kg
                    </p>
                  )}
                  {record.totalVolume != null && (
                    <p className="text-[10px] text-muted-foreground">
                      Vol: {Math.round(record.totalVolume)} kg
                    </p>
                  )}
                </div>
              </div>
            ))}
        </motion.div>
      )}

      {/* Add to Workout Button */}
      <div className="fixed bottom-20 left-0 right-0 px-4 pb-4 bg-gradient-to-t from-background via-background to-transparent pt-6">
        <div className="max-w-lg mx-auto">
          <Button
            className="w-full h-12 gap-2"
            onClick={() => setLocation("/active-workout")}
          >
            <Dumbbell className="h-4 w-4" />
            Start Workout with This Exercise
          </Button>
        </div>
      </div>
    </div>
  );
}

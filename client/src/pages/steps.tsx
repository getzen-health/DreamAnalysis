/**
 * Steps -- Dedicated steps/activity page.
 *
 * Shows today's step count with progress toward 10K goal,
 * circular progress ring, 7-day bar chart, distance,
 * calories burned, and daily average.
 */

import { useMemo } from "react";
import { motion } from "framer-motion";
import { pageTransition, cardVariants } from "@/lib/animations";
import { useHealthSync } from "@/hooks/use-health-sync";
import { useQuery } from "@tanstack/react-query";
import { getParticipantId } from "@/lib/participant";
import { Footprints, TrendingUp, Flame, MapPin } from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

/* ---------- Constants ---------- */

const STEPS_GOAL = 10_000;
const STEP_COLOR = "#0891b2";
const STEP_COLOR_LIGHT = "#06b6d4";

/* ---------- Circular Progress Ring ---------- */

function ProgressRing({ percent }: { percent: number }) {
  const size = 140;
  const stroke = 10;
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (Math.min(percent, 100) / 100) * circumference;

  return (
    <svg width={size} height={size} className="drop-shadow-sm">
      <defs>
        <linearGradient id="stepsRingGrad" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor={STEP_COLOR} />
          <stop offset="100%" stopColor={STEP_COLOR_LIGHT} />
        </linearGradient>
      </defs>
      {/* Background track */}
      <circle
        cx={size / 2}
        cy={size / 2}
        r={radius}
        fill="none"
        stroke="var(--border)"
        strokeWidth={stroke}
        opacity={0.4}
      />
      {/* Progress arc */}
      <motion.circle
        cx={size / 2}
        cy={size / 2}
        r={radius}
        fill="none"
        stroke="url(#stepsRingGrad)"
        strokeWidth={stroke}
        strokeLinecap="round"
        strokeDasharray={circumference}
        initial={{ strokeDashoffset: circumference }}
        animate={{ strokeDashoffset: offset }}
        transition={{ duration: 1.2, ease: [0.22, 1, 0.36, 1] }}
        transform={`rotate(-90 ${size / 2} ${size / 2})`}
      />
      {/* Center text */}
      <text
        x="50%"
        y="46%"
        textAnchor="middle"
        dominantBaseline="central"
        className="fill-foreground"
        style={{ fontSize: 28, fontWeight: 700 }}
      >
        {percent}%
      </text>
      <text
        x="50%"
        y="62%"
        textAnchor="middle"
        dominantBaseline="central"
        className="fill-muted-foreground"
        style={{ fontSize: 10, fontWeight: 500 }}
      >
        of goal
      </text>
    </svg>
  );
}

/* ---------- Stat Card ---------- */

function StatCard({
  icon: Icon,
  label,
  value,
  unit,
  sub,
  index,
}: {
  icon: React.ElementType;
  label: string;
  value: string | number | null;
  unit: string;
  sub?: string;
  index: number;
}) {
  return (
    <motion.div
      custom={index}
      variants={cardVariants}
      initial="hidden"
      animate="visible"
      className="rounded-2xl border border-border bg-card p-3.5" style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}
    >
      <div className="flex items-center gap-1.5 mb-1">
        <Icon className="h-3.5 w-3.5" style={{ color: STEP_COLOR }} />
        <p className="text-[10px] text-muted-foreground uppercase tracking-wide font-medium">
          {label}
        </p>
      </div>
      <div className="flex items-baseline gap-1">
        <span className="text-2xl font-bold" style={{ color: STEP_COLOR }}>
          {value ?? "--"}
        </span>
        <span className="text-xs text-muted-foreground">{unit}</span>
      </div>
      {sub && (
        <p className="text-[10px] text-muted-foreground mt-1">{sub}</p>
      )}
    </motion.div>
  );
}

/* ---------- Main Component ---------- */

export default function Steps() {
  const { latestPayload } = useHealthSync();
  const userId = useMemo(() => getParticipantId(), []);

  // Fetch 7-day steps history
  const { data: stepsHistory } = useQuery<
    Array<{ value: number; recorded_at: string }>
  >({
    queryKey: [`/api/health/steps/${userId}?days=7`],
    retry: false,
    staleTime: 5 * 60 * 1000,
  });

  // Today's steps from health sync
  const stepsToday = latestPayload?.steps_today ?? 0;
  const stepsPct = Math.min(100, Math.round((stepsToday / STEPS_GOAL) * 100));

  // Derived metrics
  const distanceKm = stepsToday > 0 ? (stepsToday * 0.0008).toFixed(1) : null;
  const caloriesBurned = stepsToday > 0 ? Math.round(stepsToday * 0.04) : null;

  // Build weekly chart data grouped by day
  const weeklyChartData = useMemo(() => {
    if (!stepsHistory || !Array.isArray(stepsHistory) || stepsHistory.length === 0) return [];
    const dayMap = new Map<string, number>();
    for (const s of stepsHistory) {
      const day = new Date(s.recorded_at).toLocaleDateString(undefined, {
        weekday: "short",
      });
      dayMap.set(day, (dayMap.get(day) || 0) + s.value);
    }
    return Array.from(dayMap.entries()).map(([day, steps]) => ({ day, steps }));
  }, [stepsHistory]);

  // Daily average over the week
  const dailyAverage = useMemo(() => {
    if (weeklyChartData.length === 0) return null;
    const total = weeklyChartData.reduce((sum, d) => sum + d.steps, 0);
    return Math.round(total / weeklyChartData.length);
  }, [weeklyChartData]);

  return (
    <div className="max-w-2xl mx-auto px-4 py-6 pb-24">
      {/* Header */}
      <motion.div
        initial={pageTransition.initial}
        animate={pageTransition.animate}
        transition={pageTransition.transition}
        className="mb-5"
      >
        <div className="flex items-center gap-2">
          <Footprints className="h-5 w-5" style={{ color: STEP_COLOR }} />
          <h1 className="text-xl font-bold tracking-tight text-foreground">
            Steps
          </h1>
        </div>
        <p className="text-sm mt-1 text-muted-foreground">
          Daily activity and walking progress
        </p>
      </motion.div>

      {/* Hero: step count + circular ring */}
      <motion.div
        custom={0}
        variants={cardVariants}
        initial="hidden"
        animate="visible"
        className="rounded-2xl border border-border bg-card p-5 mb-4" style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}
      >
        <div className="flex items-center justify-between">
          {/* Left: step count */}
          <div>
            <p className="text-[10px] text-muted-foreground uppercase tracking-wide font-medium mb-1">
              Today
            </p>
            <div className="flex items-baseline gap-2">
              <span
                className="text-4xl font-bold tabular-nums"
                style={{ color: STEP_COLOR }}
              >
                {stepsToday > 0 ? stepsToday.toLocaleString() : "--"}
              </span>
              <span className="text-sm text-muted-foreground">
                / {STEPS_GOAL.toLocaleString()}
              </span>
            </div>
            <p className="text-[10px] text-muted-foreground mt-2">
              {stepsToday >= STEPS_GOAL
                ? "Goal reached"
                : stepsToday > 0
                  ? `${(STEPS_GOAL - stepsToday).toLocaleString()} steps to go`
                  : "No steps recorded yet"}
            </p>
          </div>

          {/* Right: circular ring */}
          <ProgressRing percent={stepsPct} />
        </div>

        {/* Linear progress bar */}
        <div className="h-2 bg-border rounded-full overflow-hidden mt-4">
          <motion.div
            className="h-full rounded-full"
            style={{
              background: `linear-gradient(90deg, ${STEP_COLOR}, ${STEP_COLOR_LIGHT})`,
            }}
            initial={{ width: 0 }}
            animate={{ width: `${stepsPct}%` }}
            transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
          />
        </div>
      </motion.div>

      {/* Stat cards grid */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        <StatCard
          icon={MapPin}
          label="Distance"
          value={distanceKm}
          unit="km"
          sub="Estimated from steps"
          index={1}
        />
        <StatCard
          icon={Flame}
          label="Calories"
          value={caloriesBurned}
          unit="kcal"
          sub="Burned from walking"
          index={2}
        />
        <StatCard
          icon={TrendingUp}
          label="Daily Average"
          value={
            dailyAverage !== null ? dailyAverage.toLocaleString() : null
          }
          unit="steps"
          sub={dailyAverage !== null ? `Over ${weeklyChartData.length} days` : undefined}
          index={3}
        />
        <StatCard
          icon={Footprints}
          label="Goal Progress"
          value={stepsPct}
          unit="%"
          sub={
            stepsToday >= STEPS_GOAL
              ? "Target met"
              : `${STEPS_GOAL.toLocaleString()} step target`
          }
          index={4}
        />
      </div>

      {/* 7-day bar chart */}
      {weeklyChartData.length > 0 ? (
        <motion.div
          custom={5}
          variants={cardVariants}
          initial="hidden"
          animate="visible"
          className="rounded-2xl border border-border bg-card p-4" style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}
        >
          <p className="text-[10px] text-muted-foreground uppercase tracking-wide font-semibold mb-3">
            Steps over the last 7 days
          </p>
          <div className="h-44">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={weeklyChartData}>
                <defs>
                  <linearGradient
                    id="stepsPageBarGrad"
                    x1="0"
                    y1="0"
                    x2="0"
                    y2="1"
                  >
                    <stop
                      offset="0%"
                      stopColor={STEP_COLOR}
                      stopOpacity={0.9}
                    />
                    <stop
                      offset="100%"
                      stopColor={STEP_COLOR_LIGHT}
                      stopOpacity={0.5}
                    />
                  </linearGradient>
                </defs>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="var(--border)"
                  opacity={0.5}
                />
                <XAxis
                  dataKey="day"
                  tick={{
                    fontSize: 9,
                    fill: "var(--muted-foreground)",
                  }}
                  tickLine={false}
                  axisLine={false}
                />
                <YAxis
                  tick={{
                    fontSize: 9,
                    fill: "var(--muted-foreground)",
                  }}
                  tickLine={false}
                  axisLine={false}
                  width={35}
                />
                <Tooltip
                  contentStyle={{
                    background: "var(--card)",
                    border: "1px solid var(--border)",
                    borderRadius: 8,
                    fontSize: 11,
                    color: "var(--foreground)",
                  }}
                />
                <Bar
                  dataKey="steps"
                  fill="url(#stepsPageBarGrad)"
                  radius={[4, 4, 0, 0]}
                  name="Steps"
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
          {dailyAverage !== null && (
            <p className="text-[10px] text-muted-foreground mt-2 text-center">
              Avg: {dailyAverage.toLocaleString()} steps/day
            </p>
          )}
        </motion.div>
      ) : (
        <motion.div
          custom={5}
          variants={cardVariants}
          initial="hidden"
          animate="visible"
          className="rounded-2xl border border-border bg-card p-6 text-center" style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}
        >
          <Footprints className="h-8 w-8 text-muted-foreground/30 mx-auto mb-2" />
          <p className="text-sm text-muted-foreground">
            No weekly step data yet
          </p>
          <p className="text-[10px] text-muted-foreground mt-1">
            Step history syncs automatically from Apple Health or Google Health
            Connect
          </p>
        </motion.div>
      )}
    </div>
  );
}

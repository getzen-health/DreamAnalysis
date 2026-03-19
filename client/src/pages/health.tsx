/**
 * Health -- Consolidated health page with tabs.
 *
 * Tabs:
 * - Body (body metrics from body-metrics.tsx)
 * - Heart (heart rate, HRV, BP, SpO2 trends)
 * - Activity (steps, distance, flights, standing, exercise)
 * - Workouts (workout from workout.tsx)
 */

import { lazy, Suspense, useMemo } from "react";
import { motion } from "framer-motion";
import { pageTransition } from "@/lib/animations";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Scale, Heart as HeartIcon, Footprints, Dumbbell } from "lucide-react";
import { useHealthSync } from "@/hooks/use-health-sync";
import { useQuery } from "@tanstack/react-query";
import { getParticipantId } from "@/lib/participant";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
} from "recharts";

// Lazy-load tab contents from existing pages
const BodyMetrics = lazy(() => import("@/pages/body-metrics"));
const WorkoutPage = lazy(() => import("@/pages/workout"));

/* ---------- Loading fallback ---------- */

function TabLoader() {
  return (
    <div className="flex items-center justify-center py-12">
      <div className="h-5 w-5 rounded-full border-2 border-primary/30 border-t-primary animate-spin" />
    </div>
  );
}

/* ---------- Metric Card ---------- */

function MetricCard({
  label,
  value,
  unit,
  color,
  sub,
}: {
  label: string;
  value: string | number | null | undefined;
  unit: string;
  color: string;
  sub?: string;
}) {
  return (
    <div className="rounded-[14px] border border-border bg-card p-3.5">
      <p className="text-[10px] text-muted-foreground uppercase tracking-wide font-medium mb-1">
        {label}
      </p>
      <div className="flex items-baseline gap-1">
        <span className="text-xl font-bold" style={{ color }}>
          {value ?? "--"}
        </span>
        <span className="text-xs text-muted-foreground">{unit}</span>
      </div>
      {sub && (
        <p className="text-[10px] text-muted-foreground mt-1">{sub}</p>
      )}
    </div>
  );
}

/* ---------- Heart Tab ---------- */

function HeartTab() {
  const { latestPayload } = useHealthSync();
  const userId = useMemo(() => getParticipantId(), []);

  // Fetch 7-day heart rate history from Supabase health samples
  const { data: hrHistory } = useQuery<Array<{ value: number; recorded_at: string }>>({
    queryKey: [`/api/health-samples/${userId}?metric=heart_rate&days=7`],
    retry: false,
    staleTime: 5 * 60 * 1000,
  });

  const { data: hrvHistory } = useQuery<Array<{ value: number; recorded_at: string }>>({
    queryKey: [`/api/health-samples/${userId}?metric=hrv_rmssd&days=7`],
    retry: false,
    staleTime: 5 * 60 * 1000,
  });

  const hrChartData = useMemo(() => {
    if (!hrHistory || hrHistory.length === 0) return [];
    return hrHistory.map((s) => ({
      time: new Date(s.recorded_at).toLocaleDateString(undefined, {
        weekday: "short",
      }),
      hr: Math.round(s.value),
    }));
  }, [hrHistory]);

  const hrvChartData = useMemo(() => {
    if (!hrvHistory || hrvHistory.length === 0) return [];
    return hrvHistory.map((s) => ({
      time: new Date(s.recorded_at).toLocaleDateString(undefined, {
        weekday: "short",
      }),
      hrv: Math.round(s.value),
    }));
  }, [hrvHistory]);

  const hr = latestPayload?.current_heart_rate;
  const rhr = latestPayload?.resting_heart_rate;
  const spo2 = latestPayload?.spo2;
  const bpSys = latestPayload?.blood_pressure_systolic;
  const bpDia = latestPayload?.blood_pressure_diastolic;

  return (
    <div className="space-y-4 mt-4">
      {/* Current metrics grid */}
      <div className="grid grid-cols-2 gap-3">
        <MetricCard
          label="Heart Rate"
          value={hr ? Math.round(hr) : null}
          unit="bpm"
          color="#e879a8"
          sub={hr ? (hr < 60 ? "Low" : hr < 100 ? "Normal" : "Elevated") : undefined}
        />
        <MetricCard
          label="Resting HR"
          value={rhr ? Math.round(rhr) : null}
          unit="bpm"
          color="#ea580c"
          sub={rhr ? (rhr < 50 ? "Athletic" : rhr < 70 ? "Good" : "Average") : undefined}
        />
        <MetricCard
          label="SpO2"
          value={spo2 ? Math.round(spo2) : null}
          unit="%"
          color="#6366f1"
          sub={spo2 ? (spo2 >= 95 ? "Normal" : "Low") : undefined}
        />
        <MetricCard
          label="Blood Pressure"
          value={bpSys && bpDia ? `${Math.round(bpSys)}/${Math.round(bpDia)}` : null}
          unit="mmHg"
          color="#8b5cf6"
          sub={bpSys ? (bpSys < 120 ? "Normal" : bpSys < 140 ? "Elevated" : "High") : undefined}
        />
      </div>

      {/* HR Trend (7d) */}
      {hrChartData.length > 1 && (
        <div className="rounded-[14px] border border-border bg-card p-4">
          <p className="text-[10px] text-muted-foreground uppercase tracking-wide font-semibold mb-3">
            Heart rate trend (7d)
          </p>
          <div className="h-40">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={hrChartData}>
                <defs>
                  <linearGradient id="hrGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#e879a8" stopOpacity={0.3} />
                    <stop offset="50%" stopColor="#be185d" stopOpacity={0.12} />
                    <stop offset="100%" stopColor="#be185d" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.5} />
                <XAxis dataKey="time" tick={{ fontSize: 9, fill: "var(--muted-foreground)" }} tickLine={false} axisLine={false} />
                <YAxis tick={{ fontSize: 9, fill: "var(--muted-foreground)" }} tickLine={false} axisLine={false} width={30} />
                <Tooltip contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 11, color: "var(--foreground)" }} />
                <Area type="monotone" dataKey="hr" stroke="#e879a8" fill="url(#hrGrad)" strokeWidth={2} dot={{ r: 2, fill: "#e879a8" }} name="HR (bpm)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* HRV Trend (7d) */}
      {hrvChartData.length > 1 && (
        <div className="rounded-[14px] border border-border bg-card p-4">
          <p className="text-[10px] text-muted-foreground uppercase tracking-wide font-semibold mb-3">
            HRV trend (7d)
          </p>
          <div className="h-40">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={hrvChartData}>
                <defs>
                  <linearGradient id="hrvGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#06b6d4" stopOpacity={0.3} />
                    <stop offset="50%" stopColor="#0891b2" stopOpacity={0.12} />
                    <stop offset="100%" stopColor="#0891b2" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.5} />
                <XAxis dataKey="time" tick={{ fontSize: 9, fill: "var(--muted-foreground)" }} tickLine={false} axisLine={false} />
                <YAxis tick={{ fontSize: 9, fill: "var(--muted-foreground)" }} tickLine={false} axisLine={false} width={30} />
                <Tooltip contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 11, color: "var(--foreground)" }} />
                <Area type="monotone" dataKey="hrv" stroke="#06b6d4" fill="url(#hrvGrad)" strokeWidth={2} dot={{ r: 2, fill: "#06b6d4" }} name="HRV (ms)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* No data state */}
      {!hr && !rhr && !spo2 && !bpSys && hrChartData.length === 0 && (
        <div className="rounded-[14px] border border-border bg-card p-6 text-center">
          <HeartIcon className="h-8 w-8 text-muted-foreground/30 mx-auto mb-2" />
          <p className="text-sm text-muted-foreground">No heart data yet</p>
          <p className="text-[10px] text-muted-foreground mt-1">
            Heart metrics sync automatically from Apple Health or Google Health Connect
          </p>
        </div>
      )}
    </div>
  );
}

/* ---------- Activity Tab ---------- */

function ActivityTab() {
  const { latestPayload } = useHealthSync();
  const userId = useMemo(() => getParticipantId(), []);

  // Weekly activity summary
  const { data: stepsHistory } = useQuery<Array<{ value: number; recorded_at: string }>>({
    queryKey: [`/api/health-samples/${userId}?metric=steps&days=7`],
    retry: false,
    staleTime: 5 * 60 * 1000,
  });

  const weeklyChartData = useMemo(() => {
    if (!stepsHistory || stepsHistory.length === 0) return [];
    // Group by day
    const dayMap = new Map<string, number>();
    for (const s of stepsHistory) {
      const day = new Date(s.recorded_at).toLocaleDateString(undefined, { weekday: "short" });
      dayMap.set(day, (dayMap.get(day) || 0) + s.value);
    }
    return Array.from(dayMap.entries()).map(([day, steps]) => ({ day, steps }));
  }, [stepsHistory]);

  const steps = latestPayload?.steps_today ?? 0;
  const distance = latestPayload?.walking_distance_km;
  const flights = latestPayload?.flights_climbed;
  const standing = latestPayload?.standing_hours;
  const calories = latestPayload?.active_energy_kcal;
  const exercise = latestPayload?.exercise_minutes_today;

  const stepsGoal = 10000;
  const stepsPct = Math.min(100, Math.round((steps / stepsGoal) * 100));

  return (
    <div className="space-y-4 mt-4">
      {/* Steps progress */}
      <div className="rounded-[14px] border border-border bg-card p-4">
        <div className="flex items-center justify-between mb-3">
          <div>
            <p className="text-[10px] text-muted-foreground uppercase tracking-wide font-medium">Steps</p>
            <div className="flex items-baseline gap-1.5">
              <span className="text-2xl font-bold text-foreground">{steps > 0 ? steps.toLocaleString() : "--"}</span>
              <span className="text-xs text-muted-foreground">/ {stepsGoal.toLocaleString()}</span>
            </div>
          </div>
          <div className="text-right">
            <span className="text-lg font-bold text-primary">{stepsPct}%</span>
            <p className="text-[9px] text-muted-foreground">of goal</p>
          </div>
        </div>
        <div className="h-2 bg-border rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-700"
            style={{ width: `${stepsPct}%`, background: "linear-gradient(90deg, #0891b2, #06b6d4)" }}
          />
        </div>
      </div>

      {/* Metrics grid */}
      <div className="grid grid-cols-2 gap-3">
        <MetricCard
          label="Distance"
          value={distance ? distance.toFixed(1) : null}
          unit="km"
          color="#0891b2"
        />
        <MetricCard
          label="Flights Climbed"
          value={flights ? Math.round(flights) : null}
          unit="floors"
          color="#ea580c"
        />
        <MetricCard
          label="Standing Hours"
          value={standing ? Math.round(standing) : null}
          unit="hrs"
          color="#0891b2"
        />
        <MetricCard
          label="Active Calories"
          value={calories ? Math.round(calories) : null}
          unit="kcal"
          color="#e879a8"
        />
      </div>

      {/* Exercise minutes */}
      <MetricCard
        label="Exercise Minutes"
        value={exercise ? Math.round(exercise) : null}
        unit="min"
        color="#8b5cf6"
        sub={exercise ? (exercise >= 30 ? "Great job! Goal met" : `${Math.round(30 - exercise)} min to daily goal`) : undefined}
      />

      {/* Weekly activity summary chart */}
      {weeklyChartData.length > 1 && (
        <div className="rounded-[14px] border border-border bg-card p-4">
          <p className="text-[10px] text-muted-foreground uppercase tracking-wide font-semibold mb-3">
            Weekly activity
          </p>
          <div className="h-36">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={weeklyChartData}>
                <defs>
                  <linearGradient id="stepsBarGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#0891b2" stopOpacity={0.9} />
                    <stop offset="100%" stopColor="#06b6d4" stopOpacity={0.5} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.5} />
                <XAxis dataKey="day" tick={{ fontSize: 9, fill: "var(--muted-foreground)" }} tickLine={false} axisLine={false} />
                <YAxis tick={{ fontSize: 9, fill: "var(--muted-foreground)" }} tickLine={false} axisLine={false} width={30} />
                <Tooltip contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 11, color: "var(--foreground)" }} />
                <Bar dataKey="steps" fill="url(#stepsBarGrad)" radius={[4, 4, 0, 0]} name="Steps" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* No data state */}
      {steps === 0 && !distance && !flights && weeklyChartData.length === 0 && (
        <div className="rounded-[14px] border border-border bg-card p-6 text-center">
          <Footprints className="h-8 w-8 text-muted-foreground/30 mx-auto mb-2" />
          <p className="text-sm text-muted-foreground">No activity data yet</p>
          <p className="text-[10px] text-muted-foreground mt-1">
            Activity data syncs automatically from your health app
          </p>
        </div>
      )}
    </div>
  );
}

/* ---------- Main component ---------- */

export default function Health() {
  return (
    <div className="max-w-2xl mx-auto px-4 py-6 pb-24">
      {/* Header */}
      <motion.div
        initial={pageTransition.initial}
        animate={pageTransition.animate}
        transition={pageTransition.transition}
        className="mb-5"
      >
        <h1 className="text-xl font-bold tracking-tight text-foreground">
          Health
        </h1>
        <p className="text-sm mt-1 text-muted-foreground">
          Body composition, heart, activity, and workouts
        </p>
      </motion.div>

      <Tabs defaultValue="body" className="w-full">
        <TabsList className="w-full">
          <TabsTrigger value="body" className="flex-1 gap-1.5">
            <Scale className="h-3.5 w-3.5" />
            Body
          </TabsTrigger>
          <TabsTrigger value="heart" className="flex-1 gap-1.5">
            <HeartIcon className="h-3.5 w-3.5" />
            Heart
          </TabsTrigger>
          <TabsTrigger value="activity" className="flex-1 gap-1.5">
            <Footprints className="h-3.5 w-3.5" />
            Activity
          </TabsTrigger>
          <TabsTrigger value="workouts" className="flex-1 gap-1.5">
            <Dumbbell className="h-3.5 w-3.5" />
            Workouts
          </TabsTrigger>
        </TabsList>

        <TabsContent value="body">
          <Suspense fallback={<TabLoader />}>
            <BodyMetrics />
          </Suspense>
        </TabsContent>

        <TabsContent value="heart">
          <HeartTab />
        </TabsContent>

        <TabsContent value="activity">
          <ActivityTab />
        </TabsContent>

        <TabsContent value="workouts">
          <Suspense fallback={<TabLoader />}>
            <WorkoutPage />
          </Suspense>
        </TabsContent>
      </Tabs>
    </div>
  );
}

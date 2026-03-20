/**
 * Heart Rate — Dedicated page for heart rate analytics.
 *
 * Shows current resting HR, heart rate zones, 7-day trend chart,
 * aggregate stats, and contextual tips.
 */

import { useMemo } from "react";
import { motion } from "framer-motion";
import { pageTransition, cardVariants } from "@/lib/animations";
import { useHealthSync } from "@/hooks/use-health-sync";
import { useQuery } from "@tanstack/react-query";
import { getParticipantId } from "@/lib/participant";
import { Heart, Activity, TrendingUp } from "lucide-react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

/* ---------- Constants ---------- */

const HR_COLOR = "#e879a8";

const ZONES = [
  {
    name: "Resting",
    range: "< 60 bpm",
    description:
      "Your heart at rest. A lower resting heart rate generally indicates better cardiovascular fitness and more efficient heart function.",
    color: "#6366f1",
  },
  {
    name: "Fat Burn",
    range: "60 - 70% max HR",
    description:
      "Low-to-moderate intensity zone where the body primarily uses fat as fuel. Ideal for longer, steady-state workouts.",
    color: "#22c55e",
  },
  {
    name: "Cardio",
    range: "70 - 85% max HR",
    description:
      "Moderate-to-high intensity zone that strengthens the cardiovascular system and improves aerobic capacity.",
    color: "#f59e0b",
  },
  {
    name: "Peak",
    range: "85 - 100% max HR",
    description:
      "Maximum effort zone for short bursts. Builds speed and power but should be used sparingly to avoid overtraining.",
    color: "#ef4444",
  },
];

/* ---------- Helpers ---------- */

function getHrLevel(rhr: number): { label: string; description: string } {
  if (rhr < 50) {
    return {
      label: "Athletic",
      description:
        "Your resting heart rate is in the athletic range. This typically reflects excellent cardiovascular conditioning.",
    };
  }
  if (rhr < 60) {
    return {
      label: "Excellent",
      description:
        "Your resting heart rate indicates strong cardiovascular health. Keep up your current activity level.",
    };
  }
  if (rhr < 70) {
    return {
      label: "Good",
      description:
        "Your resting heart rate is within a healthy range. Regular aerobic exercise can help lower it further.",
    };
  }
  if (rhr < 80) {
    return {
      label: "Average",
      description:
        "Your resting heart rate is average. Consider adding more cardiovascular exercise to improve it over time.",
    };
  }
  return {
    label: "Elevated",
    description:
      "Your resting heart rate is above average. Stress management, better sleep, and regular exercise can help bring it down.",
  };
}

function getTips(rhr: number | null): string[] {
  const tips: string[] = [];

  if (rhr === null) {
    tips.push("Wear your device consistently to track resting heart rate trends over time.");
    tips.push("Measure resting HR first thing in the morning for the most accurate baseline.");
    tips.push("Hydration and sleep quality both directly affect your heart rate readings.");
    return tips;
  }

  if (rhr >= 80) {
    tips.push("Try deep breathing exercises — 4 seconds in, 7 seconds hold, 8 seconds out — to activate the parasympathetic nervous system.");
    tips.push("Reduce caffeine and alcohol intake, both of which can elevate resting heart rate.");
    tips.push("Aim for 7-9 hours of quality sleep. Poor sleep is one of the most common causes of elevated resting HR.");
  } else if (rhr >= 70) {
    tips.push("Start with 30 minutes of brisk walking five days a week to gradually lower your resting HR.");
    tips.push("Stay hydrated throughout the day — dehydration forces the heart to work harder.");
    tips.push("Track your HR trends weekly. Consistency matters more than any single reading.");
  } else if (rhr >= 60) {
    tips.push("Your cardiovascular fitness is solid. Adding interval training can push it further.");
    tips.push("Monitor morning resting HR — a sudden spike can signal overtraining or illness.");
    tips.push("Cold exposure (cold showers, cold water immersion) can improve vagal tone over time.");
  } else {
    tips.push("Excellent cardiovascular conditioning. Ensure you are getting adequate recovery between intense sessions.");
    tips.push("A very low resting HR is normal for trained athletes but consult a doctor if you experience dizziness or fatigue.");
    tips.push("Continue tracking trends — even small changes can reveal shifts in fitness or recovery.");
  }

  return tips;
}

/* ---------- Component ---------- */

export default function HeartRate() {
  const { latestPayload } = useHealthSync();
  const userId = useMemo(() => getParticipantId(), []);

  const hr = latestPayload?.current_heart_rate ?? null;
  const rhr = latestPayload?.resting_heart_rate ?? null;

  // Fetch 7-day heart rate history
  const { data: hrHistory } = useQuery<
    Array<{ value: number; recorded_at: string }>
  >({
    queryKey: [`/api/health/heart-rate/${userId}?days=7`],
    retry: false,
    staleTime: 5 * 60 * 1000,
  });

  // Transform history for the chart
  const chartData = useMemo(() => {
    if (!hrHistory || !Array.isArray(hrHistory) || hrHistory.length === 0) return [];
    return hrHistory.map((s) => ({
      time: new Date(s.recorded_at).toLocaleDateString(undefined, {
        weekday: "short",
      }),
      hr: Math.round(s.value),
    }));
  }, [hrHistory]);

  // Compute aggregate stats from history
  const stats = useMemo(() => {
    if (!chartData || chartData.length === 0) return null;
    const values = chartData.map((d) => d.hr);
    const sum = values.reduce((a, b) => a + b, 0);
    return {
      avg: Math.round(sum / values.length),
      min: Math.min(...values),
      max: Math.max(...values),
    };
  }, [chartData]);

  const hrLevel = rhr !== null ? getHrLevel(rhr) : null;
  const tips = getTips(rhr);

  return (
    <div className="max-w-2xl mx-auto px-4 py-6 pb-24">
      {/* Header */}
      <motion.div
        initial={pageTransition.initial}
        animate={pageTransition.animate}
        transition={pageTransition.transition}
        className="mb-6"
      >
        <div className="flex items-center gap-2.5 mb-1">
          <Heart className="h-5 w-5" style={{ color: HR_COLOR }} />
          <h1 className="text-xl font-bold tracking-tight text-foreground">
            Heart Rate
          </h1>
        </div>
        <p className="text-sm text-muted-foreground">
          Resting heart rate, zones, and trends
        </p>
      </motion.div>

      {/* Current Resting HR — hero card */}
      <motion.div
        custom={0}
        variants={cardVariants}
        initial="hidden"
        animate="visible"
        className="rounded-2xl border border-border bg-card p-6 mb-4" style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}
      >
        <p className="text-[10px] text-muted-foreground uppercase tracking-wide font-medium mb-2">
          Resting Heart Rate
        </p>
        <div className="flex items-baseline gap-2">
          <span
            className="text-5xl font-bold tabular-nums"
            style={{ color: HR_COLOR }}
          >
            {rhr !== null ? Math.round(rhr) : "--"}
          </span>
          <span className="text-base text-muted-foreground">bpm</span>
        </div>
        {hrLevel && (
          <div className="mt-3">
            <span
              className="inline-block text-xs font-semibold px-2 py-0.5 rounded-full"
              style={{
                color: HR_COLOR,
                backgroundColor: `${HR_COLOR}18`,
              }}
            >
              {hrLevel.label}
            </span>
            <p className="text-xs text-muted-foreground mt-2 leading-relaxed">
              {hrLevel.description}
            </p>
          </div>
        )}
        {hr !== null && (
          <div className="mt-4 pt-3 border-t border-border">
            <p className="text-[10px] text-muted-foreground uppercase tracking-wide font-medium mb-1">
              Current Heart Rate
            </p>
            <div className="flex items-baseline gap-1.5">
              <span
                className="text-2xl font-bold tabular-nums"
                style={{ color: HR_COLOR }}
              >
                {Math.round(hr)}
              </span>
              <span className="text-xs text-muted-foreground">bpm</span>
              <span className="text-[10px] text-muted-foreground ml-1">
                {hr < 60 ? "Low" : hr < 100 ? "Normal" : "Elevated"}
              </span>
            </div>
          </div>
        )}
      </motion.div>

      {/* Heart Rate Zones */}
      <motion.div
        custom={1}
        variants={cardVariants}
        initial="hidden"
        animate="visible"
        className="rounded-2xl border border-border bg-card p-4 mb-4" style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}
      >
        <div className="flex items-center gap-2 mb-3">
          <Activity className="h-3.5 w-3.5 text-muted-foreground" />
          <p className="text-[10px] text-muted-foreground uppercase tracking-wide font-semibold">
            Heart Rate Zones
          </p>
        </div>
        <div className="space-y-3">
          {ZONES.map((zone) => (
            <div key={zone.name} className="flex gap-3">
              <div
                className="w-1 rounded-full flex-shrink-0 mt-0.5"
                style={{ backgroundColor: zone.color }}
              />
              <div className="min-w-0">
                <div className="flex items-baseline gap-2">
                  <span className="text-sm font-semibold text-foreground">
                    {zone.name}
                  </span>
                  <span className="text-[10px] text-muted-foreground">
                    {zone.range}
                  </span>
                </div>
                <p className="text-xs text-muted-foreground leading-relaxed mt-0.5">
                  {zone.description}
                </p>
              </div>
            </div>
          ))}
        </div>
      </motion.div>

      {/* 7-Day Resting HR Trend */}
      {chartData.length > 1 && (
        <motion.div
          custom={2}
          variants={cardVariants}
          initial="hidden"
          animate="visible"
          className="rounded-2xl border border-border bg-card p-4 mb-4" style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}
        >
          <div className="flex items-center gap-2 mb-3">
            <TrendingUp className="h-3.5 w-3.5 text-muted-foreground" />
            <p className="text-[10px] text-muted-foreground uppercase tracking-wide font-semibold">
              Resting HR Trend (7 days)
            </p>
          </div>
          <div className="h-44">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient
                    id="heartRateGrad"
                    x1="0"
                    y1="0"
                    x2="0"
                    y2="1"
                  >
                    <stop
                      offset="0%"
                      stopColor={HR_COLOR}
                      stopOpacity={0.35}
                    />
                    <stop
                      offset="50%"
                      stopColor={HR_COLOR}
                      stopOpacity={0.12}
                    />
                    <stop
                      offset="100%"
                      stopColor={HR_COLOR}
                      stopOpacity={0}
                    />
                  </linearGradient>
                </defs>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="var(--border)"
                  opacity={0.5}
                />
                <XAxis
                  dataKey="time"
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
                  width={30}
                  domain={["dataMin - 5", "dataMax + 5"]}
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
                <Area
                  type="monotone"
                  dataKey="hr"
                  stroke={HR_COLOR}
                  fill="url(#heartRateGrad)"
                  strokeWidth={2.5}
                  dot={{ r: 2.5, fill: HR_COLOR }}
                  activeDot={{ r: 4, fill: HR_COLOR }}
                  name="HR (bpm)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </motion.div>
      )}

      {/* Aggregate Stats */}
      {stats && (
        <motion.div
          custom={3}
          variants={cardVariants}
          initial="hidden"
          animate="visible"
          className="grid grid-cols-3 gap-3 mb-4"
        >
          {[
            { label: "Average", value: stats.avg },
            { label: "Min", value: stats.min },
            { label: "Max", value: stats.max },
          ].map((stat) => (
            <div
              key={stat.label}
              className="rounded-2xl border border-border bg-card p-3.5 text-center" style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}
            >
              <p className="text-[10px] text-muted-foreground uppercase tracking-wide font-medium mb-1">
                {stat.label}
              </p>
              <div className="flex items-baseline justify-center gap-1">
                <span
                  className="text-2xl font-bold tabular-nums"
                  style={{ color: HR_COLOR }}
                >
                  {stat.value}
                </span>
                <span className="text-xs text-muted-foreground">bpm</span>
              </div>
            </div>
          ))}
        </motion.div>
      )}

      {/* Tips */}
      <motion.div
        custom={4}
        variants={cardVariants}
        initial="hidden"
        animate="visible"
        className="rounded-2xl border border-border bg-card p-4" style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}
      >
        <p className="text-[10px] text-muted-foreground uppercase tracking-wide font-semibold mb-3">
          Tips for your heart rate
        </p>
        <ul className="space-y-2.5">
          {tips.map((tip, i) => (
            <li key={i} className="flex gap-2.5 text-xs text-muted-foreground leading-relaxed">
              <span
                className="mt-1 h-1.5 w-1.5 rounded-full flex-shrink-0"
                style={{ backgroundColor: HR_COLOR }}
              />
              {tip}
            </li>
          ))}
        </ul>
      </motion.div>

      {/* Empty state — no data at all */}
      {rhr === null && hr === null && chartData.length === 0 && (
        <motion.div
          custom={5}
          variants={cardVariants}
          initial="hidden"
          animate="visible"
          className="rounded-2xl border border-border bg-card p-6 text-center mt-4" style={{ boxShadow: "0 2px 16px rgba(0,0,0,0.06)" }}
        >
          <Heart className="h-8 w-8 text-muted-foreground/30 mx-auto mb-2" />
          <p className="text-sm text-muted-foreground">
            No heart rate data yet
          </p>
          <p className="text-[10px] text-muted-foreground mt-1">
            Heart rate data syncs automatically from Apple Health or Google
            Health Connect
          </p>
        </motion.div>
      )}
    </div>
  );
}

/**
 * ScoresDashboard -- main health scores overview page.
 *
 * Layout:
 * 1. Hero: Energy Bank battery
 * 2. Primary Scores Grid (2x2): Recovery, Sleep, Strain, Stress
 * 3. Secondary Row: Nutrition Score + Cardio Load
 * 4. Today's Summary Card: steps, calories, weight, HR zones
 * 5. Trend Alerts: dismissable list
 */

import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Heart,
  Moon,
  Flame,
  Brain,
  Apple,
  Footprints,
  Zap,
  Scale,
  X,
  Activity,
  TrendingUp,
  TrendingDown,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { pageTransition, cardVariants } from "@/lib/animations";
import { useAuth } from "@/hooks/use-auth";
import { useScores } from "@/hooks/use-scores";
import { ScoreCard } from "@/components/score-card";
import { ScoreGauge } from "@/components/score-gauge";
import { EnergyBattery } from "@/components/energy-battery";
import { getMLApiUrl, listSessions, type SessionSummary } from "@/lib/ml-api";
import { EmotionStrip } from "@/components/emotion-strip";
import { Skeleton } from "@/components/ui/skeleton";
import { getParticipantId } from "@/lib/participant";

// ── Types ─────────────────────────────────────────────────────────────────────

interface TrendAlert {
  id: string;
  message: string;
  type: "positive" | "negative" | "info";
  timestamp: string;
}

interface TodaySummary {
  steps: number | null;
  stepsGoal: number;
  activeCalories: number | null;
  weight: number | null;
  hrZones?: { zone: string; minutes: number }[];
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function ProgressBar({
  value,
  max,
  color,
}: {
  value: number;
  max: number;
  color: string;
}) {
  const pct = Math.min((value / max) * 100, 100);
  return (
    <div className="w-full h-3 rounded-full bg-muted">
      <motion.div
        className="h-2 rounded-full"
        style={{ background: color }}
        initial={{ width: 0 }}
        animate={{ width: `${pct}%` }}
        transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
      />
    </div>
  );
}

function SummaryRow({
  icon: Icon,
  label,
  value,
  unit,
  color,
}: {
  icon: React.ElementType;
  label: string;
  value: string | number | null;
  unit?: string;
  color: string;
}) {
  return (
    <div className="flex items-center justify-between py-2">
      <div className="flex items-center gap-2.5">
        <Icon className="h-4 w-4" style={{ color }} />
        <span className="text-sm text-muted-foreground">
          {label}
        </span>
      </div>
      <span className="text-sm font-mono font-semibold text-foreground">
        {value === null ? "\u2014" : value}
        {unit && value !== null && (
          <span className="text-xs ml-1 text-muted-foreground">
            {unit}
          </span>
        )}
      </span>
    </div>
  );
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function ScoresDashboard() {
  const { user } = useAuth();
  const userId = user?.id?.toString();
  const { scores, loading: scoresLoading } = useScores(userId);

  const [dismissedAlerts, setDismissedAlerts] = useState<Set<string>>(new Set());

  // Fetch today's summary
  const { data: todaySummary } = useQuery<TodaySummary>({
    queryKey: ["today-summary", userId],
    queryFn: async () => {
      const baseUrl = getMLApiUrl();
      const res = await fetch(
        `${baseUrl}/health/today-summary/${encodeURIComponent(userId!)}`
      );
      if (!res.ok) {
        // Return defaults if endpoint doesn't exist yet
        return { steps: null, stepsGoal: 10000, activeCalories: null, weight: null };
      }
      return res.json();
    },
    enabled: !!userId,
    staleTime: 5 * 60_000,
    retry: false,
  });

  // Fetch trend alerts
  const { data: trendAlerts } = useQuery<TrendAlert[]>({
    queryKey: ["trend-alerts", userId],
    queryFn: async () => {
      const baseUrl = getMLApiUrl();
      const res = await fetch(
        `${baseUrl}/health/trend-alerts/${encodeURIComponent(userId!)}`
      );
      if (!res.ok) return [];
      return res.json();
    },
    enabled: !!userId,
    staleTime: 5 * 60_000,
    retry: false,
  });

  // Fetch session data for trend arrows
  const participantId = getParticipantId();
  const { data: sessions } = useQuery<SessionSummary[]>({
    queryKey: ["sessions", participantId],
    queryFn: () => listSessions(participantId),
    staleTime: 30_000,
    retry: false,
  });

  // Compute trend deltas from last two sessions for each score type
  const scoreTrends = useMemo(() => {
    if (!sessions || sessions.length < 2) return null;
    const sorted = [...sessions]
      .filter((s) => s.summary)
      .sort((a, b) => (a.start_time ?? 0) - (b.start_time ?? 0));
    if (sorted.length < 2) return null;
    const prev = sorted[sorted.length - 2].summary;
    const cur = sorted[sorted.length - 1].summary;

    function computeTrend(curVal: number | undefined | null, prevVal: number | undefined | null): { trend: "up" | "down" | "stable"; trendValue: string } | null {
      if (curVal == null || prevVal == null) return null;
      const delta = (curVal - prevVal) * 100;
      if (Math.abs(delta) <= 2) return null;
      return {
        trend: delta > 0 ? "up" : "down",
        trendValue: `${delta > 0 ? "+" : ""}${Math.round(delta)}% vs last`,
      };
    }

    return {
      // Map session avg fields to score card names
      // Recovery maps to relaxation, Sleep maps to avg_relaxation too (best proxy),
      // Strain maps to arousal, Stress maps to avg_stress, Nutrition is not in sessions
      stress: computeTrend(cur.avg_stress, prev.avg_stress),
      recovery: computeTrend(cur.avg_relaxation, prev.avg_relaxation),
      focus: computeTrend(cur.avg_focus, prev.avg_focus),
      strain: computeTrend(cur.avg_arousal, prev.avg_arousal),
    };
  }, [sessions]);

  const visibleAlerts = (trendAlerts ?? []).filter(
    (a) => !dismissedAlerts.has(a.id)
  );

  const dismissAlert = (id: string) => {
    setDismissedAlerts((prev) => new Set(prev).add(id));
  };

  // Loading state — skeleton that mirrors the real layout
  if (scoresLoading) {
    return (
      <div className="max-w-2xl mx-auto px-4 py-6 space-y-6">
        {/* Header skeleton */}
        <div>
          <Skeleton className="h-6 w-36 mb-2" />
          <Skeleton className="h-4 w-48" />
        </div>

        {/* Energy Bank hero skeleton */}
        <div className="flex justify-center py-6 rounded-2xl bg-card/80 border border-border">
          <Skeleton className="h-28 w-28 rounded-full" />
        </div>

        {/* Primary Scores 2x2 grid skeleton */}
        <div className="grid grid-cols-2 gap-3">
          {Array.from({ length: 4 }).map((_, i) => (
            <div
              key={i}
              className="rounded-2xl p-4 bg-card border border-border"
              style={{ borderLeft: "3px solid var(--border)" }}
            >
              <div className="flex items-center justify-between gap-3">
                <div className="flex-1 space-y-2">
                  <div className="flex items-center gap-2">
                    <Skeleton className="h-7 w-7 rounded-lg" />
                    <Skeleton className="h-4 w-16" />
                  </div>
                </div>
                <Skeleton className="h-14 w-14 rounded-full shrink-0" />
              </div>
            </div>
          ))}
        </div>

        {/* Secondary row skeleton */}
        <div className="grid grid-cols-2 gap-3">
          <div
            className="rounded-2xl p-4 bg-card border border-border"
            style={{ borderLeft: "3px solid var(--border)" }}
          >
            <div className="flex items-center gap-2 mb-2">
              <Skeleton className="h-7 w-7 rounded-lg" />
              <Skeleton className="h-4 w-20" />
            </div>
            <Skeleton className="h-14 w-14 rounded-full ml-auto" />
          </div>
          <div
            className="rounded-2xl p-4 bg-card border border-border"
            style={{ borderLeft: "3px solid var(--border)" }}
          >
            <div className="flex items-center gap-2 mb-2">
              <Skeleton className="h-7 w-7 rounded-lg" />
              <Skeleton className="h-4 w-24" />
            </div>
            <Skeleton className="h-3 w-16 ml-9" />
          </div>
        </div>

        {/* Today's Summary skeleton */}
        <div className="rounded-2xl p-5 bg-card border border-border">
          <Skeleton className="h-4 w-12 mb-4" />
          <div className="space-y-3">
            {Array.from({ length: 3 }).map((_, i) => (
              <div key={i} className="flex items-center justify-between py-1">
                <div className="flex items-center gap-2.5">
                  <Skeleton className="h-4 w-4 rounded" />
                  <Skeleton className="h-3.5 w-24" />
                </div>
                <Skeleton className="h-3.5 w-12" />
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  const summary = todaySummary ?? {
    steps: null,
    stepsGoal: 10000,
    activeCalories: null,
    weight: null,
  };

  return (
    <div className="max-w-2xl mx-auto px-4 py-6 space-y-6">
      {/* Page header */}
      <motion.div
        initial={pageTransition.initial}
        animate={pageTransition.animate}
        transition={pageTransition.transition}
      >
        <h1 className="text-xl font-bold tracking-tight text-foreground">
          Health Scores
        </h1>
        <p className="text-sm mt-1 text-muted-foreground">
          {scores?.computedAt
            ? `Updated ${new Date(scores.computedAt).toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
              })}`
            : "Real-time health overview"}
        </p>
        <div className="mt-2">
          <EmotionStrip />
        </div>
      </motion.div>

      {/* ── Hero: Energy Bank ──────────────────────────────────────────── */}
      <motion.div
        className="flex justify-center py-4 rounded-2xl backdrop-blur-sm bg-card/80 border border-border animate-heartbeat shadow-[0_2px_16px_rgba(0,0,0,0.06)]"
        initial={{ opacity: 0, y: 12, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
      >
        <EnergyBattery value={scores?.energyBank ?? null} />
      </motion.div>

      {/* ── Primary Scores Grid (2x2) ─────────────────────────────────── */}
      <div className="grid grid-cols-2 gap-3">
        {[
          { title: "Recovery", value: scores?.recoveryScore ?? null, color: "recovery" as const, icon: <Heart className="h-3.5 w-3.5 text-ndw-recovery" />, trendKey: "recovery" as const },
          { title: "Sleep", value: scores?.sleepScore ?? null, color: "sleep" as const, icon: <Moon className="h-3.5 w-3.5 text-ndw-sleep" />, trendKey: null },
          { title: "Strain", value: scores?.strainScore ?? null, color: "strain" as const, icon: <Flame className="h-3.5 w-3.5 text-ndw-strain" />, trendKey: "strain" as const },
          { title: "Stress", value: scores?.stressScore ?? null, color: "stress" as const, icon: <Brain className="h-3.5 w-3.5 text-ndw-stress" />, trendKey: "stress" as const },
        ].map((card, i) => {
          const t = card.trendKey && scoreTrends ? scoreTrends[card.trendKey] : null;
          // For stress, invert the arrow color logic: up stress is bad (rose), down stress is good (cyan)
          // ScoreCard already uses cyan for "up" and rose for "down", so for stress we flip the trend direction display
          const trendForStress = card.title === "Stress" && t ? {
            trend: (t.trend === "up" ? "up" : "down") as "up" | "down",
            trendValue: t.trendValue,
          } : t;
          return (
            <motion.div
              key={card.title}
              custom={i}
              initial="hidden"
              animate="visible"
              variants={cardVariants}
            >
              <ScoreCard
                title={card.title}
                value={card.value}
                color={card.color}
                icon={card.icon}
                trend={trendForStress?.trend}
                trendValue={trendForStress?.trendValue}
              />
            </motion.div>
          );
        })}
      </div>

      {/* ── Secondary Row ─────────────────────────────────────────────── */}
      <motion.div
        className="grid grid-cols-2 gap-3"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.3, ease: [0.22, 1, 0.36, 1] }}
      >
        <ScoreCard
          title="Nutrition"
          value={scores?.nutritionScore ?? null}
          color="nutrition"
          icon={<Apple className="h-3.5 w-3.5 text-ndw-nutrition" />}
          // Nutrition has no session-level proxy, no trend arrow
        />
        {/* Cardio Load status card (uses strain color for visual grouping) */}
        <div
          className="rounded-2xl p-4 flex flex-col justify-center bg-card border border-border shadow-[0_2px_16px_rgba(0,0,0,0.06)]"
          style={{ borderLeft: "3px solid #e879a8" }}
        >
          <div className="flex items-center gap-2 mb-2">
            <span
              className="flex items-center justify-center w-7 h-7 rounded-lg"
              style={{ background: "#e879a820" }}
            >
              <Activity className="h-3.5 w-3.5 text-ndw-strain" />
            </span>
            <span className="text-sm font-semibold text-foreground">
              Cardio Load
            </span>
          </div>
          <p className="text-xs font-mono ml-9 text-muted-foreground">
            {scores?.strainScore !== null && scores?.strainScore !== undefined
              ? scores.strainScore > 70
                ? "Overreaching"
                : scores.strainScore > 40
                ? "Optimal"
                : "Detraining"
              : "No data"}
          </p>
        </div>
      </motion.div>

      {/* ── Today's Summary Card ──────────────────────────────────────── */}
      <motion.div
        className="rounded-2xl p-5 bg-card border border-border animate-fade-in-up shadow-[0_2px_16px_rgba(0,0,0,0.06)]"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.4, ease: [0.22, 1, 0.36, 1] }}
      >
        <h2 className="text-sm font-semibold mb-3 text-foreground">
          Today
        </h2>

        {/* Steps with progress bar */}
        <div className="mb-3">
          <SummaryRow
            icon={Footprints}
            label="Steps"
            value={summary.steps !== null ? summary.steps.toLocaleString() : null}
            color="#0891b2"
          />
          {summary.steps !== null && (
            <div className="ml-6 mt-1">
              <ProgressBar
                value={summary.steps}
                max={summary.stepsGoal}
                color="#0891b2"
              />
              <p className="text-[10px] mt-1 text-right text-muted-foreground/50">
                {summary.stepsGoal.toLocaleString()} goal
              </p>
            </div>
          )}
        </div>

        <div className="border-t border-border">
          <SummaryRow
            icon={Zap}
            label="Active calories"
            value={summary.activeCalories}
            unit="kcal"
            color="#d4a017"
          />
        </div>

        <div className="border-t border-border">
          <SummaryRow
            icon={Scale}
            label="Weight"
            value={summary.weight !== null ? summary.weight.toFixed(1) : null}
            unit="kg"
            color="#7c3aed"
          />
        </div>

        {/* HR Zones (if available) */}
        {summary.hrZones && summary.hrZones.length > 0 && (
          <div className="border-t border-border pt-3 mt-1">
            <p className="text-xs font-semibold mb-2 text-muted-foreground">
              HR Zones
            </p>
            <div className="flex gap-1.5">
              {summary.hrZones.map((z) => (
                <div key={z.zone} className="flex-1 text-center">
                  <div className="h-8 rounded-md mb-1 flex items-end justify-center bg-muted/50">
                    <motion.div
                      className="w-full rounded-md"
                      style={{
                        background:
                          z.zone === "Zone 5"
                            ? "#be185d"
                            : z.zone === "Zone 4"
                            ? "#ea580c"
                            : z.zone === "Zone 3"
                            ? "#d4a017"
                            : z.zone === "Zone 2"
                            ? "#0891b2"
                            : "var(--muted-foreground)",
                      }}
                      initial={{ height: 0 }}
                      animate={{
                        height: `${Math.min(
                          (z.minutes / 30) * 100,
                          100
                        )}%`,
                      }}
                      transition={{ duration: 0.6, delay: 0.5, ease: [0.22, 1, 0.36, 1] }}
                    />
                  </div>
                  <p className="text-[9px] text-muted-foreground">
                    {z.minutes}m
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}
      </motion.div>

      {/* ── Trend Alerts ──────────────────────────────────────────────── */}
      <AnimatePresence>
        {visibleAlerts.length > 0 && (
          <motion.div
            className="space-y-2"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <h2 className="text-sm font-semibold text-foreground">
              Trend Alerts
            </h2>
            {visibleAlerts.map((alert) => (
              <motion.div
                key={alert.id}
                className={`flex items-start gap-3 rounded-2xl p-3 bg-card border shadow-[0_2px_16px_rgba(0,0,0,0.06)] ${
                  alert.type === "negative"
                    ? "border-rose-500/30"
                    : alert.type === "positive"
                    ? "border-cyan-500/30"
                    : "border-border"
                }`}
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 8, height: 0 }}
                transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
              >
                {alert.type === "positive" ? (
                  <TrendingUp className="h-4 w-4 shrink-0 mt-0.5 text-cyan-400" />
                ) : alert.type === "negative" ? (
                  <TrendingDown className="h-4 w-4 shrink-0 mt-0.5 text-rose-400" />
                ) : (
                  <Activity className="h-4 w-4 shrink-0 mt-0.5 text-indigo-400" />
                )}
                <p className="text-xs flex-1 text-foreground">
                  {alert.message}
                </p>
                <button
                  onClick={() => dismissAlert(alert.id)}
                  className="shrink-0 p-0.5 rounded hover:bg-muted transition-all duration-200"
                  aria-label="Dismiss alert"
                >
                  <X className="h-3.5 w-3.5 text-muted-foreground" />
                </button>
              </motion.div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

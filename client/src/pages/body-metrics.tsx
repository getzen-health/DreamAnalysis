import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { useAuth } from "@/hooks/use-auth";
import { useHealthSync } from "@/hooks/use-health-sync";
import { Button } from "@/components/ui/button";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import {
  Scale,
  TrendingUp,
  TrendingDown,
  Minus,
  Activity,
  Target,
  RefreshCw,
  Smartphone,
} from "lucide-react";
import { motion } from "framer-motion";
import { cardVariants, pageTransition } from "@/lib/animations";
import { ScoreGauge } from "@/components/score-gauge";
import { EmotionStrip } from "@/components/emotion-strip";

/* ---------- types ---------- */

interface BodyMetric {
  id: string;
  userId: string | null;
  weightKg: string | null;
  bodyFatPct: string | null;
  leanMassKg: string | null;
  bmi: string | null;
  heightCm: string | null;
  source: string;
  recordedAt: string;
  createdAt: string | null;
}

interface ChartPoint {
  date: string;
  ts: number;
  weight: number | null;
  movingAvg: number | null;
  bodyFat: number | null;
  leanMass: number | null;
}

/* ---------- helpers ---------- */

function bmiCategory(bmi: number): { label: string; color: string } {
  if (bmi < 18.5) return { label: "Underweight", color: "text-yellow-400" };
  if (bmi < 25) return { label: "Normal", color: "text-ndw-recovery" };
  if (bmi < 30) return { label: "Overweight", color: "text-orange-400" };
  return { label: "Obese", color: "text-ndw-strain" };
}

function compute7DayMovingAvg(
  data: { ts: number; weight: number | null }[]
): (number | null)[] {
  const result: (number | null)[] = [];
  const SEVEN_DAYS_MS = 7 * 24 * 60 * 60 * 1000;

  for (let i = 0; i < data.length; i++) {
    const currentTs = data[i].ts;
    const windowValues: number[] = [];

    for (let j = 0; j <= i; j++) {
      if (
        data[j].weight !== null &&
        currentTs - data[j].ts <= SEVEN_DAYS_MS
      ) {
        windowValues.push(data[j].weight!);
      }
    }

    if (windowValues.length >= 2) {
      result.push(
        parseFloat(
          (windowValues.reduce((a, b) => a + b, 0) / windowValues.length).toFixed(1)
        )
      );
    } else {
      result.push(null);
    }
  }

  return result;
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
  initial: { opacity: 0, y: 12, scale: 0.98 },
  animate: { opacity: 1, y: 0, scale: 1 },
  transition: { duration: 0.4, ease: [0.22, 1, 0.36, 1] },
};

/* ========== Component ========== */

export default function BodyMetrics() {
  const { user } = useAuth();
  const { latestPayload, lastSyncAt, syncNow, status, isAvailable } = useHealthSync();

  // Fetch latest body metric from DB (populated by health sync)
  const { data: latest } = useQuery<BodyMetric>({
    queryKey: [`/api/body-metrics/${user?.id}/latest`],
    enabled: !!user?.id,
    retry: false,
    staleTime: 30_000,
  });

  // Fetch history (last 90 days)
  const { data: history = [] } = useQuery<BodyMetric[]>({
    queryKey: [`/api/body-metrics/${user?.id}`],
    enabled: !!user?.id,
    retry: false,
    staleTime: 30_000,
  });

  // Build chart data
  const chartData: ChartPoint[] = useMemo(() => {
    const sorted = [...history].sort(
      (a, b) =>
        new Date(a.recordedAt).getTime() - new Date(b.recordedAt).getTime()
    );

    const points = sorted.map((m) => ({
      date: new Date(m.recordedAt).toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
      }),
      ts: new Date(m.recordedAt).getTime(),
      weight: m.weightKg ? parseFloat(m.weightKg) : null,
      bodyFat: m.bodyFatPct ? parseFloat(m.bodyFatPct) : null,
      leanMass: m.leanMassKg ? parseFloat(m.leanMassKg) : null,
      movingAvg: null as number | null,
    }));

    const movingAvgs = compute7DayMovingAvg(points);
    points.forEach((p, i) => {
      p.movingAvg = movingAvgs[i];
    });

    return points;
  }, [history]);

  // Has body fat data?
  const hasBodyFatData = chartData.some((p) => p.bodyFat !== null);

  // Rate of change (weekly)
  const rateOfChange = useMemo(() => {
    const validWeights = chartData.filter((p) => p.weight !== null);
    if (validWeights.length < 2) return null;

    const recent = validWeights[validWeights.length - 1];
    const weekAgoTs = recent.ts - 7 * 24 * 60 * 60 * 1000;

    let closest = validWeights[0];
    let closestDiff = Math.abs(closest.ts - weekAgoTs);
    for (const p of validWeights) {
      const diff = Math.abs(p.ts - weekAgoTs);
      if (diff < closestDiff) {
        closest = p;
        closestDiff = diff;
      }
    }

    if (closest === recent) return null;

    const daysBetween =
      (recent.ts - closest.ts) / (24 * 60 * 60 * 1000);
    if (daysBetween < 1) return null;

    const weeklyChange =
      ((recent.weight! - closest.weight!) / daysBetween) * 7;
    const weeklyPct =
      (weeklyChange / closest.weight!) * 100;

    return { weeklyChange, weeklyPct };
  }, [chartData]);

  // Latest stats — prefer health sync payload for live weight, fall back to DB
  const latestWeight = latestPayload?.weight_kg ?? (latest?.weightKg ? parseFloat(latest.weightKg) : null);
  const latestBmi = latest?.bmi ? parseFloat(latest.bmi) : null;
  const latestBodyFat = latestPayload?.body_fat_pct ?? (latest?.bodyFatPct ? parseFloat(latest.bodyFatPct) : null);
  const latestLeanMass = latestPayload?.lean_mass_kg ?? (latest?.leanMassKg ? parseFloat(latest.leanMassKg) : null);

  // Weight domain for chart (auto-scale with padding)
  const weightDomain = useMemo(() => {
    const weights = chartData
      .map((p) => p.weight)
      .filter((w): w is number => w !== null);
    if (weights.length === 0) return [60, 100] as [number, number];
    const min = Math.min(...weights);
    const max = Math.max(...weights);
    const pad = Math.max((max - min) * 0.15, 2);
    return [
      parseFloat((min - pad).toFixed(0)),
      parseFloat((max + pad).toFixed(0)),
    ] as [number, number];
  }, [chartData]);

  const isSyncing = status === "syncing";

  return (
    <main className="px-4 pt-2 pb-24 space-y-4 max-w-xl mx-auto">
      {/* Page Header */}
      <motion.div className="space-y-2 mb-1" {...fadeInUp}>
        <div className="flex items-center gap-2">
          <Scale className="h-5 w-5 text-primary" />
          <h1 className="text-lg font-semibold text-foreground">Body Metrics</h1>
        </div>
        <EmotionStrip />
      </motion.div>

      {/* Synced from Health Card */}
      <motion.div
        className="rounded-2xl p-5 bg-card border border-border shadow-sm"
        {...fadeInUp}
        transition={{ ...fadeInUp.transition, delay: 0.05 }}
      >
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Smartphone className="h-4 w-4 text-primary" />
            <p className="text-[13px] font-semibold text-foreground">Synced from Health</p>
          </div>
          <span className="text-[10px] text-muted-foreground">
            Last synced: {formatSyncTime(lastSyncAt)}
          </span>
        </div>

        {isAvailable ? (
          <div className="space-y-3">
            {latestWeight !== null ? (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Weight</span>
                  <span className="text-lg font-bold text-foreground">
                    {latestWeight.toFixed(1)} kg
                  </span>
                </div>
                {latestBodyFat !== null && (
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Body Fat</span>
                    <span className="text-lg font-bold text-foreground">
                      {latestBodyFat.toFixed(1)}%
                    </span>
                  </div>
                )}
                {latestLeanMass !== null && (
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Lean Mass</span>
                    <span className="text-lg font-bold text-foreground">
                      {latestLeanMass.toFixed(1)} kg
                    </span>
                  </div>
                )}
                {latestBmi !== null && (
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">BMI</span>
                    <span className={`text-lg font-bold ${bmiCategory(latestBmi).color}`}>
                      {latestBmi.toFixed(1)} ({bmiCategory(latestBmi).label})
                    </span>
                  </div>
                )}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">
                No body metrics synced yet. Tap Sync Now to pull data from your health app.
              </p>
            )}

            <Button
              onClick={() => syncNow()}
              disabled={isSyncing}
              className="w-full h-12 text-base font-semibold gap-2"
              variant="outline"
            >
              <RefreshCw className={`h-4 w-4 ${isSyncing ? "animate-spin" : ""}`} />
              {isSyncing ? "Syncing..." : "Sync Now"}
            </Button>
          </div>
        ) : (
          <div className="space-y-3">
            <p className="text-sm text-muted-foreground">
              Connect Google Health or Apple Health in Settings to see your body metrics.
            </p>
            <p className="text-xs text-muted-foreground/60">
              Body metrics are automatically imported from your health app. No manual entry needed.
            </p>
          </div>
        )}
      </motion.div>

      {/* Current Stats Card */}
      {latestWeight !== null && (
        <motion.div
          className="rounded-2xl p-4 bg-card border border-border shadow-sm"
          {...fadeInUp}
          transition={{ ...fadeInUp.transition, delay: 0.1 }}
        >
          <div className="flex items-center gap-2 mb-3">
            <Activity className="h-4 w-4 text-primary" />
            <p className="text-[13px] font-semibold text-foreground">Current Stats</p>
          </div>

          <div className="grid grid-cols-2 gap-3">
            {/* Weight */}
            <div className="p-3 rounded-xl bg-muted/50 border border-border/50">
              <p className="text-[10px] text-muted-foreground/60 uppercase tracking-wider mb-1">
                Weight
              </p>
              <p className="text-lg font-bold text-foreground">
                {latestWeight.toFixed(1)}{" "}
                <span className="text-[11px] font-normal text-muted-foreground">
                  kg
                </span>
              </p>
            </div>

            {/* BMI with ScoreGauge */}
            {latestBmi !== null && (
              <div className="p-3 rounded-xl bg-muted/50 border border-border/50 flex items-center gap-3">
                <div className="flex-1">
                  <p className="text-[10px] text-muted-foreground/60 uppercase tracking-wider mb-1">
                    BMI
                  </p>
                  <p className={`text-lg font-bold ${bmiCategory(latestBmi).color}`}>
                    {latestBmi.toFixed(1)}
                  </p>
                  <p className={`text-[10px] ${bmiCategory(latestBmi).color}`}>
                    {bmiCategory(latestBmi).label}
                  </p>
                </div>
                <ScoreGauge
                  value={Math.min(latestBmi, 40)}
                  max={40}
                  label=""
                  color="nutrition"
                  size="sm"
                />
              </div>
            )}

            {/* Body Fat */}
            {latestBodyFat !== null && (
              <div className="p-3 rounded-xl bg-muted/50 border border-border/50">
                <p className="text-[10px] text-muted-foreground/60 uppercase tracking-wider mb-1">
                  Body Fat
                </p>
                <p className="text-lg font-bold text-foreground">
                  {latestBodyFat.toFixed(1)}
                  <span className="text-[11px] font-normal text-muted-foreground">
                    %
                  </span>
                </p>
              </div>
            )}

            {/* Lean Mass */}
            {latestLeanMass !== null && (
              <div className="p-3 rounded-xl bg-muted/50 border border-border/50">
                <p className="text-[10px] text-muted-foreground/60 uppercase tracking-wider mb-1">
                  Lean Mass
                </p>
                <p className="text-lg font-bold text-foreground">
                  {latestLeanMass.toFixed(1)}{" "}
                  <span className="text-[11px] font-normal text-muted-foreground">
                    kg
                  </span>
                </p>
              </div>
            )}
          </div>
        </motion.div>
      )}

      {/* Rate of Change Indicator */}
      {rateOfChange && (
        <motion.div
          className="rounded-2xl p-4 bg-card border border-border shadow-sm"
          {...fadeInUp}
          transition={{ ...fadeInUp.transition, delay: 0.15 }}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              {rateOfChange.weeklyChange < -0.05 ? (
                <TrendingDown className="h-4 w-4 text-indigo-400" />
              ) : rateOfChange.weeklyChange > 0.05 ? (
                <TrendingUp className="h-4 w-4 text-orange-400" />
              ) : (
                <Minus className="h-4 w-4 text-ndw-recovery" />
              )}
              <p className="text-[13px] font-semibold text-foreground">Weekly Trend</p>
            </div>

            <div className="text-right">
              <p
                className={`text-sm font-bold ${
                  Math.abs(rateOfChange.weeklyPct) > 2
                    ? "text-ndw-strain"
                    : rateOfChange.weeklyChange < -0.05
                      ? "text-indigo-400"
                      : rateOfChange.weeklyChange > 0.05
                        ? "text-orange-400"
                        : "text-ndw-recovery"
                }`}
              >
                {rateOfChange.weeklyChange > 0 ? "+" : ""}
                {rateOfChange.weeklyChange.toFixed(2)} kg/week
              </p>
              <p className="text-[10px] text-muted-foreground">
                {rateOfChange.weeklyPct > 0 ? "+" : ""}
                {rateOfChange.weeklyPct.toFixed(1)}% per week
              </p>
              {Math.abs(rateOfChange.weeklyPct) > 2 && (
                <p className="text-[10px] text-ndw-strain mt-0.5">
                  Rapid change -- monitor closely
                </p>
              )}
            </div>
          </div>
        </motion.div>
      )}

      {/* Weight Trend Chart */}
      {chartData.length >= 2 && (
        <motion.div
          className="rounded-2xl p-4 bg-card border border-border shadow-sm"
          {...fadeInUp}
          transition={{ ...fadeInUp.transition, delay: 0.2 }}
        >
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="h-4 w-4 text-primary" />
            <p className="text-[13px] font-semibold text-foreground">Weight Trend</p>
            <span className="text-[10px] text-muted-foreground ml-auto">
              Last 90 days
            </span>
          </div>

          <ResponsiveContainer width="100%" height={200}>
            <LineChart
              data={chartData}
              margin={{ left: 0, right: 4, top: 4, bottom: 0 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="var(--border)"
                opacity={0.5}
              />
              <XAxis
                dataKey="date"
                tick={{ fontSize: 9, fill: "var(--muted-foreground)" }}
                axisLine={false}
                tickLine={false}
                interval="preserveStartEnd"
              />
              <YAxis
                domain={weightDomain}
                tick={{ fontSize: 9, fill: "var(--muted-foreground)" }}
                axisLine={false}
                tickLine={false}
                width={32}
                tickFormatter={(v) => `${v}`}
              />
              <Tooltip
                cursor={{
                  stroke: "var(--muted-foreground)",
                  strokeWidth: 1,
                  strokeDasharray: "4 3",
                }}
                contentStyle={{
                  background: "var(--popover)",
                  border: "1px solid var(--border)",
                  borderRadius: 10,
                  fontSize: 11,
                  color: "var(--foreground)",
                }}
                formatter={(v: number, name: string) => [
                  `${v} kg`,
                  name === "movingAvg" ? "7-day avg" : "Weight",
                ]}
              />
              <Line
                type="monotone"
                dataKey="weight"
                name="Weight"
                stroke="#0891b2"
                strokeWidth={2}
                dot={{ r: 3, fill: "#0891b2" }}
                activeDot={{ r: 5 }}
                connectNulls
              />
              <Line
                type="monotone"
                dataKey="movingAvg"
                name="7-day avg"
                stroke="hsl(38,85%,58%)"
                strokeWidth={1.5}
                strokeDasharray="5 3"
                dot={false}
                connectNulls
              />
            </LineChart>
          </ResponsiveContainer>

          <div className="flex gap-4 mt-2 justify-center">
            {[
              { label: "Weight", color: "#0891b2" },
              {
                label: "7-day avg",
                color: "hsl(38,85%,58%)",
                dashed: true,
              },
            ].map((l) => (
              <div key={l.label} className="flex items-center gap-1">
                <svg width="14" height="8">
                  <line
                    x1="0"
                    y1="4"
                    x2="14"
                    y2="4"
                    stroke={l.color}
                    strokeWidth="2"
                    strokeDasharray={l.dashed ? "4 3" : "0"}
                  />
                </svg>
                <span className="text-[10px] text-muted-foreground">
                  {l.label}
                </span>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Body Composition Chart */}
      {hasBodyFatData && chartData.length >= 2 && (
        <motion.div
          className="rounded-2xl p-4 bg-card border border-border shadow-sm"
          {...fadeInUp}
          transition={{ ...fadeInUp.transition, delay: 0.25 }}
        >
          <div className="flex items-center gap-2 mb-4">
            <Target className="h-4 w-4 text-primary" />
            <p className="text-[13px] font-semibold text-foreground">Body Composition</p>
          </div>

          <ResponsiveContainer width="100%" height={180}>
            <AreaChart
              data={chartData}
              margin={{ left: 0, right: 4, top: 4, bottom: 0 }}
            >
              <defs>
                <linearGradient id="bodyFatGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop
                    offset="5%"
                    stopColor="hsl(38,85%,58%)"
                    stopOpacity={0.3}
                  />
                  <stop
                    offset="95%"
                    stopColor="hsl(38,85%,58%)"
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
                dataKey="date"
                tick={{ fontSize: 9, fill: "var(--muted-foreground)" }}
                axisLine={false}
                tickLine={false}
                interval="preserveStartEnd"
              />
              <YAxis
                tick={{ fontSize: 9, fill: "var(--muted-foreground)" }}
                axisLine={false}
                tickLine={false}
                width={28}
                tickFormatter={(v) => `${v}%`}
              />
              <Tooltip
                cursor={{
                  stroke: "var(--muted-foreground)",
                  strokeWidth: 1,
                  strokeDasharray: "4 3",
                }}
                contentStyle={{
                  background: "var(--popover)",
                  border: "1px solid var(--border)",
                  borderRadius: 10,
                  fontSize: 11,
                  color: "var(--foreground)",
                }}
                formatter={(v: number, name: string) => [
                  `${v}%`,
                  name === "bodyFat" ? "Body Fat" : "Lean Mass",
                ]}
              />
              <Area
                type="monotone"
                dataKey="bodyFat"
                name="bodyFat"
                stroke="hsl(38,85%,58%)"
                fill="url(#bodyFatGrad)"
                strokeWidth={2}
                dot={false}
                connectNulls
              />
            </AreaChart>
          </ResponsiveContainer>

          <div className="flex gap-4 mt-2 justify-center">
            <div className="flex items-center gap-1">
              <svg width="14" height="8">
                <line
                  x1="0"
                  y1="4"
                  x2="14"
                  y2="4"
                  stroke="hsl(38,85%,58%)"
                  strokeWidth="2"
                />
              </svg>
              <span className="text-[10px] text-muted-foreground">
                Body Fat %
              </span>
            </div>
          </div>
        </motion.div>
      )}

      {/* Empty state */}
      {history.length === 0 && latestWeight === null && (
        <motion.div
          className="rounded-2xl p-6 text-center bg-card border border-border"
          {...fadeInUp}
          transition={{ ...fadeInUp.transition, delay: 0.1 }}
        >
          <Scale className="h-8 w-8 mx-auto mb-3 text-muted-foreground/40" />
          <p className="text-[13px] font-medium text-foreground/70">
            No body metrics yet
          </p>
          <p className="text-[11px] text-muted-foreground mt-1">
            Your weight, BMI, and body fat are automatically synced from Google Health or Apple Health
          </p>
        </motion.div>
      )}
    </main>
  );
}

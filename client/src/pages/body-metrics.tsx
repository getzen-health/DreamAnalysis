import { useState, useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { useAuth } from "@/hooks/use-auth";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
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
} from "lucide-react";
import { motion } from "framer-motion";
import { useToast } from "@/hooks/use-toast";
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

function kgToLbs(kg: number): number {
  return kg * 2.20462;
}

function lbsToKg(lbs: number): number {
  return lbs / 2.20462;
}

function computeBmi(weightKg: number, heightCm: number): number {
  const heightM = heightCm / 100;
  return weightKg / (heightM * heightM);
}

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

/* ---------- animation variants ---------- */

const fadeInUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.35, ease: "easeOut" },
};

/* ========== Component ========== */

export default function BodyMetrics() {
  const { user } = useAuth();
  const queryClient = useQueryClient();
  const { toast } = useToast();

  // Form state
  const [weightInput, setWeightInput] = useState("");
  const [bodyFatInput, setBodyFatInput] = useState("");
  const [heightInput, setHeightInput] = useState("");
  const [useLbs, setUseLbs] = useState(false);

  // Fetch latest body metric (for prefilling height)
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

  // Prefill height from latest record
  const savedHeight = latest?.heightCm ? parseFloat(latest.heightCm) : null;
  const effectiveHeight = heightInput
    ? parseFloat(heightInput)
    : savedHeight;

  // Log mutation
  const logMutation = useMutation({
    mutationFn: async () => {
      const rawWeight = parseFloat(weightInput);
      if (isNaN(rawWeight) || rawWeight <= 0) {
        throw new Error("Enter a valid weight");
      }

      const weightKg = useLbs ? lbsToKg(rawWeight) : rawWeight;
      const bodyFatPct = bodyFatInput ? parseFloat(bodyFatInput) : undefined;
      const heightCm = heightInput
        ? parseFloat(heightInput)
        : savedHeight ?? undefined;

      if (bodyFatPct !== undefined && (bodyFatPct < 0 || bodyFatPct > 100)) {
        throw new Error("Body fat must be 0-100%");
      }

      const res = await apiRequest("POST", "/api/body-metrics", {
        weightKg: parseFloat(weightKg.toFixed(2)),
        bodyFatPct: bodyFatPct ?? null,
        heightCm: heightCm ?? null,
        source: "manual",
      });

      return res.json();
    },
    onSuccess: (data: BodyMetric) => {
      queryClient.invalidateQueries({
        queryKey: [`/api/body-metrics/${user?.id}/latest`],
      });
      queryClient.invalidateQueries({
        queryKey: [`/api/body-metrics/${user?.id}`],
      });
      setWeightInput("");
      setBodyFatInput("");

      const bmi = data.bmi ? parseFloat(data.bmi) : null;
      const leanMass = data.leanMassKg ? parseFloat(data.leanMassKg) : null;

      toast({
        title: "Weight logged",
        description: [
          bmi ? `BMI: ${bmi.toFixed(1)}` : null,
          leanMass ? `Lean mass: ${leanMass.toFixed(1)} kg` : null,
        ]
          .filter(Boolean)
          .join(" | ") || "Saved successfully",
      });
    },
    onError: (err: Error) => {
      toast({
        title: "Error",
        description: err.message,
        variant: "destructive",
      });
    },
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

    // Find the point closest to 7 days ago
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

  // Latest stats
  const latestWeight = latest?.weightKg ? parseFloat(latest.weightKg) : null;
  const latestBmi = latest?.bmi ? parseFloat(latest.bmi) : null;
  const latestBodyFat = latest?.bodyFatPct
    ? parseFloat(latest.bodyFatPct)
    : null;
  const latestLeanMass = latest?.leanMassKg
    ? parseFloat(latest.leanMassKg)
    : null;

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

      {/* Quick Log Card */}
      <motion.div
        className="rounded-2xl p-5 bg-card border border-border shadow-sm"
        {...fadeInUp}
        transition={{ ...fadeInUp.transition, delay: 0.05 }}
      >
        <p className="text-[11px] font-semibold text-muted-foreground/60 uppercase tracking-[0.08em] mb-3">
          Log Weight
        </p>

        <div className="space-y-3">
          {/* Weight input with unit toggle */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <Label className="text-[12px] text-muted-foreground">
                Weight
              </Label>
              <button
                type="button"
                onClick={() => setUseLbs(!useLbs)}
                className="text-[10px] px-2 py-0.5 rounded-full bg-primary/10 text-primary hover:bg-primary/20 transition-colors"
              >
                {useLbs ? "lbs" : "kg"} (tap to switch)
              </button>
            </div>
            <Input
              type="number"
              step="0.1"
              placeholder={useLbs ? "e.g. 165" : "e.g. 75"}
              value={weightInput}
              onChange={(e) => setWeightInput(e.target.value)}
              className="bg-background/50 border-border/50 text-base h-12"
            />
          </div>

          {/* Body Fat (optional) */}
          <div>
            <Label className="text-[12px] text-muted-foreground mb-1.5 block">
              Body Fat % (optional)
            </Label>
            <Input
              type="number"
              step="0.1"
              placeholder="e.g. 18"
              value={bodyFatInput}
              onChange={(e) => setBodyFatInput(e.target.value)}
              className="bg-background/50 border-border/50 text-base h-12"
            />
          </div>

          {/* Height (set once) */}
          {!savedHeight && (
            <div>
              <Label className="text-[12px] text-muted-foreground mb-1.5 block">
                Height (cm) -- set once for BMI
              </Label>
              <Input
                type="number"
                step="0.1"
                placeholder="e.g. 175"
                value={heightInput}
                onChange={(e) => setHeightInput(e.target.value)}
                className="bg-background/50 border-border/50 text-base h-12"
              />
            </div>
          )}

          {/* Preview BMI before logging */}
          {weightInput && effectiveHeight && (
            <div className="text-[11px] text-muted-foreground/70">
              {(() => {
                const wKg = useLbs
                  ? lbsToKg(parseFloat(weightInput))
                  : parseFloat(weightInput);
                if (isNaN(wKg) || wKg <= 0) return null;
                const bmi = computeBmi(wKg, effectiveHeight);
                const cat = bmiCategory(bmi);
                return (
                  <span>
                    Preview BMI: <span className={cat.color}>{bmi.toFixed(1)} ({cat.label})</span>
                  </span>
                );
              })()}
            </div>
          )}

          <Button
            onClick={() => logMutation.mutate()}
            disabled={logMutation.isPending || !weightInput}
            className="w-full h-12 text-base font-semibold"
          >
            {logMutation.isPending ? "Logging..." : "Log Weight"}
          </Button>
        </div>
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
      {history.length === 0 && (
        <motion.div
          className="rounded-2xl p-6 text-center bg-card border border-border"
          {...fadeInUp}
          transition={{ ...fadeInUp.transition, delay: 0.1 }}
        >
          <Scale className="h-8 w-8 mx-auto mb-3 text-muted-foreground/40" />
          <p className="text-[13px] font-medium text-foreground/70">
            No entries yet
          </p>
          <p className="text-[11px] text-muted-foreground mt-1">
            Log your weight above to start tracking trends
          </p>
        </motion.div>
      )}
    </main>
  );
}

/**
 * /emotional-intelligence — Unified EIQ dashboard (#205)
 *
 * Shows:
 *   • Large EIQ score circle with letter grade
 *   • 5 dimension bars
 *   • Sensor contribution panel (EEG / Voice / Health)
 *   • Strengths & growth areas
 *   • 30-day EIQ trend chart
 */
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import {
  Brain,
  Mic,
  Activity,
  TrendingUp,
  TrendingDown,
  Minus,
  Star,
  Target,
} from "lucide-react";
import {
  getEIQSessionStats,
  getEIQHistory,
  getMultimodalStatus,
} from "@/lib/ml-api";
import { getParticipantId } from "@/lib/participant";
import { Skeleton } from "@/components/ui/skeleton";

const userId = getParticipantId();

// ── Grade colour ──────────────────────────────────────────────────────────────

const GRADE_COLOR: Record<string, string> = {
  A: "text-cyan-400",
  B: "text-cyan-400",
  C: "text-amber-400",
  D: "text-orange-400",
  F: "text-rose-400",
};

// ── Dimension labels ───────────────────────────────────────────────────────────

const DIM_LABELS: Record<string, string> = {
  self_perception: "Self-Perception",
  self_expression: "Self-Expression",
  interpersonal: "Interpersonal",
  decision_making: "Decision-Making",
  stress_management: "Stress-Management",
};

const DIM_COLOR: Record<string, string> = {
  self_perception: "bg-violet-500",
  self_expression: "bg-indigo-500",
  interpersonal: "bg-cyan-500",
  decision_making: "bg-amber-500",
  stress_management: "bg-cyan-500",
};

// ── Sub-components ────────────────────────────────────────────────────────────

const GRADE_GRADIENT: Record<string, { from: string; to: string }> = {
  A: { from: "#0891b2", to: "#06b6d4" },
  B: { from: "#0891b2", to: "#06b6d4" },
  C: { from: "#d4a017", to: "#ea580c" },
  D: { from: "#ea580c", to: "#c2410c" },
  F: { from: "#e879a8", to: "#be185d" },
};

function ScoreCircle({ score, grade }: { score: number; grade: string }) {
  const r = 54;
  const circ = 2 * Math.PI * r;
  const dash = (score / 100) * circ;
  const textClass = GRADE_COLOR[grade] ?? "text-zinc-400";
  const gradColors = GRADE_GRADIENT[grade] ?? { from: "#71717a", to: "#52525b" };

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative w-36 h-36">
        <svg className="w-full h-full -rotate-90" viewBox="0 0 120 120">
          <defs>
            <linearGradient id="eiqScoreGrad" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor={gradColors.from} />
              <stop offset="100%" stopColor={gradColors.to} />
            </linearGradient>
          </defs>
          <circle cx="60" cy="60" r={r} fill="none" stroke="#27272a" strokeWidth="10" />
          <circle
            cx="60" cy="60" r={r}
            fill="none"
            stroke="url(#eiqScoreGrad)"
            strokeWidth="10"
            strokeLinecap="round"
            strokeDasharray={`${dash} ${circ}`}
            className="transition-all duration-700"
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={`text-3xl font-bold tabular-nums ${textClass}`}>{score}</span>
          <span className="text-xs text-zinc-500">/ 100</span>
        </div>
      </div>
      <Badge className={`text-lg px-3 py-0.5 ${GRADE_COLOR[grade]} bg-zinc-800 border-zinc-700`}>
        Grade {grade}
      </Badge>
    </div>
  );
}

function DimensionBar({ name, value }: { name: string; value: number }) {
  const color = DIM_COLOR[name] ?? "bg-zinc-500";
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="text-zinc-300">{DIM_LABELS[name] ?? name}</span>
        <span className="tabular-nums text-zinc-400">{value.toFixed(1)}</span>
      </div>
      <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${color}`}
          style={{ width: `${Math.min(value, 100)}%` }}
        />
      </div>
    </div>
  );
}

const SENSOR_COLOR_MAP: Record<string, { border: string; bg: string; text: string; dot: string }> = {
  violet: { border: "border-violet-500/40", bg: "bg-violet-500/10", text: "text-violet-300", dot: "bg-violet-400" },
  blue:   { border: "border-indigo-500/40",   bg: "bg-indigo-500/10",   text: "text-indigo-300",   dot: "bg-indigo-400" },
  purple: { border: "border-purple-500/40", bg: "bg-purple-500/10", text: "text-purple-300", dot: "bg-purple-400" },
  emerald:{ border: "border-cyan-500/40",bg: "bg-cyan-500/10",text: "text-cyan-300",dot: "bg-cyan-400" },
  cyan:   { border: "border-cyan-500/40",   bg: "bg-cyan-500/10",   text: "text-cyan-300",   dot: "bg-cyan-400" },
  rose:   { border: "border-rose-500/40",   bg: "bg-rose-500/10",   text: "text-rose-300",   dot: "bg-rose-400" },
};

function SensorChip({
  icon: Icon,
  label,
  active,
  weight,
  color,
}: {
  icon: React.ElementType;
  label: string;
  active: boolean;
  weight: number;
  color: string;
}) {
  const c = SENSOR_COLOR_MAP[color] ?? SENSOR_COLOR_MAP["violet"];
  return (
    <div
      className={`flex items-center gap-2 rounded-lg border px-3 py-2 text-sm ${
        active
          ? `${c.border} ${c.bg} ${c.text}`
          : "border-zinc-800 bg-zinc-900/40 text-zinc-500"
      }`}
    >
      <Icon className="h-4 w-4 shrink-0" />
      <div className="flex-1 min-w-0">
        <p className="font-medium leading-tight">{label}</p>
        <p className="text-xs opacity-70">
          {active ? `${Math.round(weight * 100)}% weight` : "offline"}
        </p>
      </div>
      <div className={`h-2 w-2 rounded-full ${active ? c.dot : "bg-zinc-700"}`} />
    </div>
  );
}

function TrendIcon({ trend }: { trend: string | null }) {
  if (trend === "improving")
    return <TrendingUp className="h-4 w-4 text-cyan-400" />;
  if (trend === "declining")
    return <TrendingDown className="h-4 w-4 text-rose-400" />;
  return <Minus className="h-4 w-4 text-zinc-400" />;
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function EmotionalIntelligencePage() {
  const statsQ = useQuery({
    queryKey: ["eiq-stats", userId],
    queryFn: () => getEIQSessionStats(userId),
    refetchInterval: 60_000,
  });

  const histQ = useQuery({
    queryKey: ["eiq-history", userId],
    queryFn: () => getEIQHistory(userId, 30),
    refetchInterval: 60_000,
  });

  const modalQ = useQuery({
    queryKey: ["multimodal-status"],
    queryFn: getMultimodalStatus,
    refetchInterval: 30_000,
  });

  // Latest EIQ entry
  const history = histQ.data?.history ?? [];
  const latest = history.length > 0 ? history[history.length - 1] : null;
  const previous = history.length > 1 ? history[history.length - 2] : null;
  const fw = modalQ.data?.fusion_weights ?? { eeg: 0.5, audio: 0, video: 0 };

  // EIQ trend delta
  const eiqDelta = (latest && previous)
    ? latest.eiq_score - previous.eiq_score
    : null;

  // Chart data
  const chartData = history.map((h, i) => ({
    idx: i + 1,
    eiq: h.eiq_score,
  }));

  const loading = statsQ.isLoading || histQ.isLoading;

  return (
    <div className="max-w-2xl mx-auto p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center gap-2">
        <Brain className="h-5 w-5 text-violet-400" />
        <h1 className="text-xl font-semibold text-white">Emotional Intelligence</h1>
        {statsQ.data?.trend && (
          <TrendIcon trend={statsQ.data.trend} />
        )}
      </div>
      <p className="text-sm text-zinc-400">
        Composite EIQ from EEG brain signals, voice patterns, and health data.
      </p>

      {loading && (
        <div className="space-y-4">
          <div className="flex items-center gap-4">
            <Skeleton className="w-36 h-36 rounded-full" />
            <div className="flex-1 space-y-3">
              <Skeleton className="h-4 w-24 rounded" />
              <Skeleton className="h-3 w-32 rounded" />
              <Skeleton className="h-3 w-28 rounded" />
            </div>
          </div>
          <Skeleton className="h-24 w-full rounded-xl" />
          <Skeleton className="h-20 w-full rounded-xl" />
        </div>
      )}

      {!loading && !latest && (
        <Card className="bg-zinc-900/60 border-zinc-800">
          <CardContent className="p-6 text-center text-sm text-zinc-500">
            No EIQ assessments yet. Start with voice and health inputs; optional EEG can deepen the score later.
          </CardContent>
        </Card>
      )}

      {latest && (
        <>
          {/* Score + stats row */}
          <div className="grid grid-cols-[auto,1fr] gap-4 items-start">
            <div className="flex flex-col items-center gap-1">
              <ScoreCircle score={latest.eiq_score} grade={latest.eiq_grade} />
              {eiqDelta != null && Math.abs(eiqDelta) > 2 && (
                <div className="flex items-center gap-1 mt-1">
                  {eiqDelta > 0 ? (
                    <TrendingUp className="h-3.5 w-3.5 text-cyan-400" />
                  ) : (
                    <TrendingDown className="h-3.5 w-3.5 text-rose-400" />
                  )}
                  <span className={`text-xs ${eiqDelta > 0 ? "text-cyan-400" : "text-rose-400"}`}>
                    {eiqDelta > 0 ? "+" : ""}{Math.round(eiqDelta)} vs last
                  </span>
                </div>
              )}
            </div>
            <div className="space-y-2 pt-2">
              <div className="flex gap-3 text-sm">
                <div>
                  <p className="text-zinc-500 text-xs">Sessions</p>
                  <p className="text-white font-semibold">{statsQ.data?.n_assessments ?? "—"}</p>
                </div>
                <div>
                  <p className="text-zinc-500 text-xs">Average</p>
                  <p className="text-white font-semibold">
                    {statsQ.data?.mean_eiq != null ? statsQ.data.mean_eiq.toFixed(1) : "—"}
                  </p>
                </div>
                <div>
                  <p className="text-zinc-500 text-xs">Trend</p>
                  <p className="text-white font-semibold capitalize">
                    {statsQ.data?.trend ?? "—"}
                  </p>
                </div>
              </div>

              {/* Strengths */}
              {latest.strengths.length > 0 && (
                <div>
                  <p className="text-xs text-zinc-500 mb-1 flex items-center gap-1">
                    <Star className="h-3 w-3 text-amber-400" /> Strengths
                  </p>
                  <div className="flex flex-wrap gap-1">
                    {latest.strengths.map((s) => (
                      <Badge key={s} className="text-xs bg-amber-500/10 text-amber-300 border-amber-500/30">
                        {DIM_LABELS[s] ?? s}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              {/* Growth areas */}
              {latest.growth_areas.length > 0 && (
                <div>
                  <p className="text-xs text-zinc-500 mb-1 flex items-center gap-1">
                    <Target className="h-3 w-3 text-indigo-400" /> Growth areas
                  </p>
                  <div className="flex flex-wrap gap-1">
                    {latest.growth_areas.map((g) => (
                      <Badge key={g} className="text-xs bg-indigo-500/10 text-indigo-300 border-indigo-500/30">
                        {DIM_LABELS[g] ?? g}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* 5 Dimension bars */}
          <Card className="bg-zinc-900/60 border-zinc-800">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-zinc-300">5 EI Dimensions</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {Object.entries(latest.dimensions).map(([dim, val]) => (
                <DimensionBar key={dim} name={dim} value={val} />
              ))}
            </CardContent>
          </Card>

          {/* Sensor contribution */}
          <Card className="bg-zinc-900/60 border-zinc-800">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-zinc-300">Active Sensors</CardTitle>
            </CardHeader>
            <CardContent className="grid grid-cols-3 gap-2">
              <SensorChip
                icon={Brain}
                label="EEG"
                active={modalQ.data?.eeg_model_loaded ?? false}
                weight={fw.eeg}
                color="violet"
              />
              <SensorChip
                icon={Mic}
                label="Voice"
                active={modalQ.data?.audio_model_loaded ?? false}
                weight={fw.audio}
                color="blue"
              />
              <SensorChip
                icon={Activity}
                label="Health"
                active={false}
                weight={0}
                color="purple"
              />
            </CardContent>
          </Card>

          {/* Trend chart */}
          {chartData.length > 1 && (
            <Card className="bg-zinc-900/60 border-zinc-800">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-zinc-300 flex items-center gap-2">
                  <TrendingUp className="h-4 w-4" /> EIQ Trend
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={240}>
                  <AreaChart data={chartData}>
                    <defs>
                      <linearGradient id="eiqTrendGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#7c3aed" stopOpacity={0.3} />
                        <stop offset="50%" stopColor="#6366f1" stopOpacity={0.12} />
                        <stop offset="100%" stopColor="#6366f1" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                    <XAxis dataKey="idx" tick={{ fontSize: 10, fill: "#71717a" }} />
                    <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: "#71717a" }} />
                    <Tooltip
                      contentStyle={{ background: "#18181b", border: "1px solid #3f3f46", borderRadius: 8 }}
                      labelStyle={{ color: "#a1a1aa" }}
                      itemStyle={{ color: "#7c3aed" }}
                      formatter={(v: number) => [v.toFixed(1), "EIQ"]}
                    />
                    <Area
                      type="monotone"
                      dataKey="eiq"
                      stroke="#7c3aed"
                      fill="url(#eiqTrendGrad)"
                      strokeWidth={2.5}
                      dot={false}
                      activeDot={{ r: 4, fill: "#7c3aed" }}
                      isAnimationActive={true}
                      animationDuration={1200}
                      animationEasing="ease-out"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  );
}

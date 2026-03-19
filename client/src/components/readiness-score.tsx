/**
 * ReadinessScore — Brain Readiness Score card.
 *
 * Displays a 0-100 circular score with a colour gradient (red/yellow/green),
 * a 7-day sparkline trend, and an expandable factor breakdown.
 *
 * Score is computed server-side from:
 *   sleep quality (40%), stress avg (25%), HRV trend (20%), voice emotion (15%)
 */

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ChevronDown, ChevronUp } from "lucide-react";
import {
  LineChart,
  Line,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import { Card, CardContent } from "@/components/ui/card";
import { getMLApiUrl } from "@/lib/ml-api";

// ── Types ─────────────────────────────────────────────────────────────────────

interface ReadinessFactors {
  sleep_quality: number | null;
  stress_avg: number | null;
  hrv_trend: number | null;
  voice_emotion: number | null;
}

interface HistoryPoint {
  date: string;
  score: number | null;
}

interface ReadinessScoreData {
  user_id: string;
  score: number;
  factors: ReadinessFactors;
  history: HistoryPoint[];
  color: "red" | "yellow" | "green";
  label: string;
}

interface ReadinessScoreProps {
  userId: string;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function scoreColor(color: "red" | "yellow" | "green"): string {
  if (color === "green") return "#0891b2"; // ocean blue
  if (color === "yellow") return "#d4a017"; // golden honey
  return "#e879a8"; // warm coral
}

function scoreColorClass(color: "red" | "yellow" | "green"): string {
  if (color === "green") return "text-cyan-400";
  if (color === "yellow") return "text-amber-400";
  return "text-rose-400";
}

/** Arc path for the circular score indicator (270-degree arc). */
function ScoreArc({
  score,
  color,
  size = 96,
}: {
  score: number;
  color: "red" | "yellow" | "green";
  size?: number;
}) {
  const strokeWidth = 7;
  const r = (size - strokeWidth * 2) / 2;
  const cx = size / 2;
  const cy = size / 2;
  const circumference = 2 * Math.PI * r;
  const arcLength = circumference * 0.75;
  const gapLength = circumference * 0.25;
  const dashOffset = arcLength * (1 - score / 100);
  const fill = scoreColor(color);

  return (
    <svg
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      aria-label={`Readiness score: ${score}`}
    >
      {/* Background track */}
      <circle
        cx={cx}
        cy={cy}
        r={r}
        fill="none"
        stroke="hsl(var(--border))"
        strokeWidth={strokeWidth}
        strokeDasharray={`${arcLength} ${gapLength}`}
        strokeLinecap="round"
        transform={`rotate(135 ${cx} ${cy})`}
        opacity={0.4}
      />
      {/* Score arc */}
      <circle
        cx={cx}
        cy={cy}
        r={r}
        fill="none"
        stroke={fill}
        strokeWidth={strokeWidth}
        strokeDasharray={`${arcLength} ${gapLength}`}
        strokeDashoffset={dashOffset}
        strokeLinecap="round"
        transform={`rotate(135 ${cx} ${cy})`}
        style={{
          transition: "stroke-dashoffset 1.2s cubic-bezier(0.34, 1.56, 0.64, 1)",
        }}
      />
      {/* Score number */}
      <text
        x={cx}
        y={cy - 2}
        textAnchor="middle"
        dominantBaseline="central"
        fill="hsl(var(--foreground))"
        fontSize={24}
        fontWeight="700"
        fontFamily="Inter, system-ui, sans-serif"
      >
        {score}
      </text>
      {/* Label */}
      <text
        x={cx}
        y={cy + 16}
        textAnchor="middle"
        dominantBaseline="central"
        fill="hsl(var(--muted-foreground))"
        fontSize={9}
        fontFamily="Inter, system-ui, sans-serif"
      >
        / 100
      </text>
    </svg>
  );
}

/** Single factor row inside the expanded breakdown. */
function FactorRow({
  label,
  value,
  weight,
}: {
  label: string;
  value: number | null;
  weight: string;
}) {
  if (value === null) {
    return (
      <div className="flex items-center justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="text-muted-foreground/50 italic">no data</span>
      </div>
    );
  }
  const pct = Math.round(value);
  const barColor =
    pct >= 70 ? "bg-cyan-500" : pct >= 50 ? "bg-amber-500" : "bg-rose-500";
  return (
    <div className="space-y-0.5">
      <div className="flex items-center justify-between text-xs">
        <span className="text-muted-foreground">
          {label}
          <span className="text-muted-foreground/50 ml-1">({weight})</span>
        </span>
        <span className="font-mono text-foreground/80">{pct}</span>
      </div>
      <div className="w-full h-1 rounded-full bg-muted/30">
        <div
          className={`h-1 rounded-full ${barColor} transition-all duration-700`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

// ── Component ─────────────────────────────────────────────────────────────────

export function ReadinessScore({ userId }: ReadinessScoreProps) {
  const [expanded, setExpanded] = useState(false);

  const { data, isLoading, isError } = useQuery<ReadinessScoreData>({
    queryKey: ["readiness-score", userId],
    queryFn: async () => {
      const baseUrl = getMLApiUrl();
      const res = await fetch(
        `${baseUrl}/brain-report/readiness-score/${encodeURIComponent(userId)}`
      );
      if (!res.ok) throw new Error(`Readiness score error: ${res.status}`);
      return res.json() as Promise<ReadinessScoreData>;
    },
    staleTime: 5 * 60_000,
    retry: 1,
  });

  if (isLoading) {
    return (
      <Card className="bg-card rounded-xl border border-border/50 shadow-sm">
        <CardContent className="p-4">
          <div className="h-20 flex items-center justify-center">
            <div className="h-5 w-5 rounded-full border-2 border-primary border-t-transparent animate-spin" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (isError || !data) return null;

  const { score, color, label, factors, history } = data;

  // Filter history to points with actual scores for the sparkline
  const sparkData = history.map((h) => ({ date: h.date, score: h.score ?? 0 }));
  const lineColor = scoreColor(color);

  return (
    <Card className="bg-card rounded-xl border border-border/50 shadow-sm">
      <CardContent className="p-4">
        {/* Header row */}
        <div className="flex items-center justify-between mb-3">
          <div>
            <p className="text-[11px] font-semibold text-muted-foreground/60 uppercase tracking-[0.08em]">
              Brain Readiness
            </p>
            <p className={`text-sm font-semibold mt-0.5 ${scoreColorClass(color)}`}>
              {label}
            </p>
          </div>

          {/* Circular score */}
          <ScoreArc score={score} color={color} size={88} />
        </div>

        {/* 7-day sparkline */}
        <div className="h-10 w-full mb-2" aria-label="7-day readiness trend">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={sparkData}>
              <Line
                type="monotone"
                dataKey="score"
                stroke={lineColor}
                strokeWidth={2.5}
                dot={false}
                isAnimationActive={false}
              />
              <Tooltip
                contentStyle={{
                  background: "hsl(var(--card))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: 6,
                  fontSize: 11,
                  padding: "2px 8px",
                }}
                formatter={(val: number) => [`${val}`, "Readiness"]}
                labelFormatter={(label: string) => label}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Expand / collapse button */}
        <button
          className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors w-full justify-end"
          onClick={() => setExpanded((v) => !v)}
          aria-expanded={expanded}
          aria-controls="readiness-factors"
        >
          {expanded ? (
            <>
              Hide breakdown <ChevronUp className="h-3 w-3" />
            </>
          ) : (
            <>
              Show breakdown <ChevronDown className="h-3 w-3" />
            </>
          )}
        </button>

        {/* Factor breakdown */}
        {expanded && (
          <div id="readiness-factors" className="mt-3 space-y-2.5">
            <FactorRow
              label="Sleep quality"
              value={factors.sleep_quality}
              weight="40%"
            />
            <FactorRow
              label="Stress (inverted)"
              value={
                factors.stress_avg !== null
                  ? 100 - factors.stress_avg
                  : null
              }
              weight="25%"
            />
            <FactorRow
              label="HRV trend"
              value={factors.hrv_trend}
              weight="20%"
            />
            <FactorRow
              label="Voice emotion"
              value={factors.voice_emotion}
              weight="15%"
            />
          </div>
        )}
      </CardContent>
    </Card>
  );
}

/**
 * EFSHeroScore — Large arc gauge showing the Emotional Fitness Score (0-100).
 *
 * Uses the same 270-degree SVG arc math as readiness-score.tsx ScoreArc.
 * When score is null and progress exists, shows a building progress ring.
 * Confidence "early_estimate" shows an indicator badge.
 */

import { Badge } from "@/components/ui/badge";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";

// ── Helpers ───────────────────────────────────────────────────────────────────

function efsColor(color: "green" | "amber" | "red" | null): string {
  if (color === "green") return "#0891b2";
  if (color === "amber") return "#d4a017";
  if (color === "red") return "#e879a8";
  return "#71717a";
}

function efsColorClass(color: "green" | "amber" | "red" | null): string {
  if (color === "green") return "text-cyan-400";
  if (color === "amber") return "text-amber-400";
  if (color === "red") return "text-rose-400";
  return "text-zinc-400";
}

// ── Props ─────────────────────────────────────────────────────────────────────

interface EFSHeroScoreProps {
  score: number | null;
  color: "green" | "amber" | "red" | null;
  label: string | null;
  confidence: "full" | "early_estimate" | "building";
  trend: { direction: "up" | "down" | "stable"; delta: number; period: string } | null;
  progress?: { daysTracked: number; daysRequired: number; percentage: number; message: string };
}

// ── Component ─────────────────────────────────────────────────────────────────

export function EFSHeroScore({ score, color, label, confidence, trend, progress }: EFSHeroScoreProps) {
  const size = 180;
  const strokeWidth = 10;
  const r = (size - strokeWidth * 2) / 2;
  const cx = size / 2;
  const cy = size / 2;
  const circumference = 2 * Math.PI * r;
  const arcLength = circumference * 0.75;
  const gapLength = circumference * 0.25;

  // Building state — score is null, show progress ring
  if (score === null && progress) {
    const progressDash = arcLength * (progress.percentage / 100);
    return (
      <div className="flex flex-col items-center gap-3">
        <div className="relative">
          <svg
            width={size}
            height={size}
            viewBox={`0 0 ${size} ${size}`}
            aria-label={`Building score: ${progress.percentage}%`}
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
            {/* Progress arc */}
            <circle
              cx={cx}
              cy={cy}
              r={r}
              fill="none"
              stroke="#71717a"
              strokeWidth={strokeWidth}
              strokeDasharray={`${progressDash} ${circumference - progressDash}`}
              strokeLinecap="round"
              transform={`rotate(135 ${cx} ${cy})`}
              style={{
                transition: "stroke-dasharray 1.2s cubic-bezier(0.34, 1.56, 0.64, 1)",
              }}
            />
            {/* Percentage text */}
            <text
              x={cx}
              y={cy - 6}
              textAnchor="middle"
              dominantBaseline="central"
              fill="hsl(var(--foreground))"
              fontSize={32}
              fontWeight="700"
              fontFamily="Inter, system-ui, sans-serif"
            >
              {progress.percentage}%
            </text>
            <text
              x={cx}
              y={cy + 18}
              textAnchor="middle"
              dominantBaseline="central"
              fill="hsl(var(--muted-foreground))"
              fontSize={11}
              fontFamily="Inter, system-ui, sans-serif"
            >
              building
            </text>
          </svg>
        </div>
        <p className="text-sm text-muted-foreground text-center max-w-[220px]">
          {progress.message}
        </p>
        <p className="text-xs text-muted-foreground/60">
          {progress.daysTracked} / {progress.daysRequired} days tracked
        </p>
      </div>
    );
  }

  // Score display state
  const displayScore = score ?? 0;
  const fill = efsColor(color);
  const dashOffset = arcLength * (1 - displayScore / 100);

  return (
    <div className="flex flex-col items-center gap-3">
      <div className="relative">
        <svg
          width={size}
          height={size}
          viewBox={`0 0 ${size} ${size}`}
          aria-label={`Emotional Fitness Score: ${displayScore}`}
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
            y={cy - 6}
            textAnchor="middle"
            dominantBaseline="central"
            fill="hsl(var(--foreground))"
            fontSize={40}
            fontWeight="700"
            fontFamily="Inter, system-ui, sans-serif"
          >
            {displayScore}
          </text>
          <text
            x={cx}
            y={cy + 20}
            textAnchor="middle"
            dominantBaseline="central"
            fill="hsl(var(--muted-foreground))"
            fontSize={11}
            fontFamily="Inter, system-ui, sans-serif"
          >
            / 100
          </text>
        </svg>
      </div>

      {/* Label */}
      <p className="text-sm font-medium text-muted-foreground">Emotional Fitness</p>

      {/* Confidence badge */}
      {confidence === "early_estimate" && (
        <Badge variant="outline" className="text-xs text-amber-400 border-amber-500/30">
          Early estimate
        </Badge>
      )}

      {/* Label (e.g. "Strong", "Fair") */}
      {label && (
        <p className={`text-sm font-semibold ${efsColorClass(color)}`}>{label}</p>
      )}

      {/* Trend badge */}
      {trend && trend.delta > 0 && (
        <div className="flex items-center gap-1.5 text-sm">
          {trend.direction === "up" && <TrendingUp className="h-4 w-4 text-cyan-400" />}
          {trend.direction === "down" && <TrendingDown className="h-4 w-4 text-rose-400" />}
          {trend.direction === "stable" && <Minus className="h-4 w-4 text-zinc-400" />}
          <span
            className={
              trend.direction === "up"
                ? "text-cyan-400"
                : trend.direction === "down"
                  ? "text-rose-400"
                  : "text-zinc-400"
            }
          >
            {trend.direction === "up" ? "+" : trend.direction === "down" ? "-" : ""}
            {trend.delta} pts {trend.period}
          </span>
        </div>
      )}
    </div>
  );
}

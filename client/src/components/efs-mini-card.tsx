/**
 * EFSMiniCard — Compact dashboard widget for the Emotional Fitness Score.
 *
 * Shows score number (large) + trend arrow + "Emotional Fitness" label.
 * Links to /emotional-fitness via wouter's useLocation.
 * If score is null, shows progress ring percentage.
 */

import { useQuery } from "@tanstack/react-query";
import { useLocation } from "wouter";
import { Card, CardContent } from "@/components/ui/card";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";
import { getEmotionalFitness } from "@/lib/ml-api";

// ── Props ─────────────────────────────────────────────────────────────────────

interface EFSMiniCardProps {
  userId: string;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function miniScoreColor(color: "green" | "amber" | "red" | null): string {
  if (color === "green") return "text-cyan-400";
  if (color === "amber") return "text-amber-400";
  if (color === "red") return "text-rose-400";
  return "text-zinc-400";
}

// ── Component ─────────────────────────────────────────────────────────────────

export function EFSMiniCard({ userId }: EFSMiniCardProps) {
  const [, navigate] = useLocation();

  const { data, isLoading } = useQuery({
    queryKey: ["efs-mini", userId],
    queryFn: () => getEmotionalFitness(userId),
    staleTime: 5 * 60_000,
    retry: 1,
  });

  if (isLoading) {
    return (
      <Card className="bg-card rounded-xl border border-border/50 shadow-sm cursor-pointer hover:border-border transition-colors">
        <CardContent className="p-4">
          <div className="h-14 flex items-center justify-center">
            <div className="h-4 w-4 rounded-full border-2 border-primary border-t-transparent animate-spin" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!data) return null;

  const hasScore = data.score !== null;

  // Mini progress ring for building state
  const ProgressRing = () => {
    if (!data.progress) return null;
    const ringSize = 40;
    const sw = 3;
    const rr = (ringSize - sw * 2) / 2;
    const cxr = ringSize / 2;
    const cyr = ringSize / 2;
    const circ = 2 * Math.PI * rr;
    const dash = circ * (data.progress.percentage / 100);
    return (
      <svg width={ringSize} height={ringSize} viewBox={`0 0 ${ringSize} ${ringSize}`}>
        <circle cx={cxr} cy={cyr} r={rr} fill="none" stroke="#27272a" strokeWidth={sw} />
        <circle
          cx={cxr}
          cy={cyr}
          r={rr}
          fill="none"
          stroke="#71717a"
          strokeWidth={sw}
          strokeLinecap="round"
          strokeDasharray={`${dash} ${circ - dash}`}
          transform={`rotate(-90 ${cxr} ${cyr})`}
          className="transition-all duration-700"
        />
        <text
          x={cxr}
          y={cyr}
          textAnchor="middle"
          dominantBaseline="central"
          fill="hsl(var(--muted-foreground))"
          fontSize={10}
          fontWeight="600"
          fontFamily="Inter, system-ui, sans-serif"
        >
          {data.progress.percentage}%
        </text>
      </svg>
    );
  };

  return (
    <Card
      className="bg-card rounded-xl border border-border/50 shadow-sm cursor-pointer hover:border-border transition-colors"
      onClick={() => navigate("/emotional-fitness")}
    >
      <CardContent className="p-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-[11px] font-semibold text-muted-foreground/60 uppercase tracking-[0.08em]">
              Emotional Fitness
            </p>
            {hasScore ? (
              <div className="flex items-center gap-2 mt-1">
                <span className={`text-2xl font-bold tabular-nums ${miniScoreColor(data.color)}`}>
                  {data.score}
                </span>
                {data.trend && (
                  <span className="flex items-center">
                    {data.trend.direction === "up" && <TrendingUp className="h-4 w-4 text-cyan-400" />}
                    {data.trend.direction === "down" && <TrendingDown className="h-4 w-4 text-rose-400" />}
                    {data.trend.direction === "stable" && <Minus className="h-4 w-4 text-zinc-400" />}
                  </span>
                )}
              </div>
            ) : (
              <p className="text-xs text-muted-foreground mt-1">
                {data.progress ? "Building..." : "No data yet"}
              </p>
            )}
          </div>
          {!hasScore && data.progress && <ProgressRing />}
        </div>
      </CardContent>
    </Card>
  );
}

/**
 * StreakBadge — compact habit streak counter.
 *
 * Shows the current streak with a flame icon, "Best: X days" subtitle,
 * and milestone indicators at 7, 30, and 100 days.
 *
 * Fetches data from GET /brain-report/streak/{user_id}.
 */

import { useQuery } from "@tanstack/react-query";
import { Flame, Trophy } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { getMLApiUrl } from "@/lib/ml-api";

// ── Types ─────────────────────────────────────────────────────────────────────

interface StreakData {
  user_id: string;
  current_streak: number;
  best_streak: number;
  today_checked_in: boolean;
  milestones: number[];
  next_milestone: number | null;
  total_checkins: number;
}

interface StreakBadgeProps {
  userId: string;
  /** If true, render as a compact inline badge rather than a full Card. */
  compact?: boolean;
}

// ── Milestone dots ────────────────────────────────────────────────────────────

const MILESTONES = [7, 30, 100];

function MilestoneDot({
  milestone,
  current,
}: {
  milestone: number;
  current: number;
}) {
  const reached = current >= milestone;
  return (
    <div className="flex flex-col items-center gap-0.5">
      <div
        className={`w-2 h-2 rounded-full transition-colors ${
          reached ? "bg-orange-400" : "bg-muted/40"
        }`}
        title={`${milestone}-day milestone${reached ? " (reached)" : ""}`}
        aria-label={
          reached
            ? `${milestone}-day milestone reached`
            : `${milestone}-day milestone not yet reached`
        }
      />
      <span className="text-[9px] text-muted-foreground/50 font-mono">
        {milestone}
      </span>
    </div>
  );
}

// ── Component ─────────────────────────────────────────────────────────────────

export function StreakBadge({ userId, compact = false }: StreakBadgeProps) {
  const { data, isLoading, isError } = useQuery<StreakData>({
    queryKey: ["brain-streak", userId],
    queryFn: async () => {
      const baseUrl = getMLApiUrl();
      const res = await fetch(
        `${baseUrl}/brain-report/streak/${encodeURIComponent(userId)}`
      );
      if (!res.ok) throw new Error(`Streak error: ${res.status}`);
      return res.json() as Promise<StreakData>;
    },
    staleTime: 60_000,
    retry: 1,
  });

  if (isLoading) {
    return (
      <Card className="bg-card rounded-xl border border-border/50 shadow-sm">
        <CardContent className="p-3">
          <div className="h-10 flex items-center justify-center">
            <div className="h-4 w-4 rounded-full border-2 border-primary border-t-transparent animate-spin" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (isError || !data) return null;

  const { current_streak, best_streak, today_checked_in, next_milestone } = data;
  const isOnFire = current_streak >= 7;
  const flameColor = isOnFire ? "text-orange-400" : "text-orange-300/60";

  if (compact) {
    return (
      <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-muted/20 border border-border/30">
        <Flame className={`h-3.5 w-3.5 ${flameColor}`} />
        <span className="text-xs font-semibold text-foreground">
          {current_streak}d
        </span>
        {best_streak > current_streak && (
          <span className="text-[10px] text-muted-foreground">
            · best {best_streak}d
          </span>
        )}
      </div>
    );
  }

  return (
    <Card className="bg-card rounded-xl border border-border/50 shadow-sm">
      <CardContent className="p-3">
        <div className="flex items-center justify-between">
          {/* Left: flame + count */}
          <div className="flex items-center gap-2">
            <Flame className={`h-5 w-5 shrink-0 ${flameColor}`} aria-hidden="true" />
            <div>
              <p className="text-sm font-bold text-foreground leading-tight">
                {current_streak > 0
                  ? `${current_streak} day streak`
                  : "Start your streak"}
              </p>
              {best_streak > 0 && (
                <p className="text-[11px] text-muted-foreground leading-tight flex items-center gap-1">
                  <Trophy className="h-2.5 w-2.5 inline" aria-hidden="true" />
                  Best: {best_streak} day{best_streak !== 1 ? "s" : ""}
                </p>
              )}
            </div>
          </div>

          {/* Right: milestone dots */}
          <div className="flex items-end gap-2">
            {MILESTONES.map((m) => (
              <MilestoneDot key={m} milestone={m} current={current_streak} />
            ))}
          </div>
        </div>

        {/* Next milestone hint */}
        {next_milestone !== null && current_streak > 0 && (
          <p className="text-[11px] text-muted-foreground/70 mt-2">
            {next_milestone - current_streak} day
            {next_milestone - current_streak !== 1 ? "s" : ""} to {next_milestone}-day milestone
          </p>
        )}

        {/* Today check-in nudge */}
        {!today_checked_in && current_streak > 0 && (
          <p className="text-[11px] text-muted-foreground/60 mt-1 italic">
            Do a voice analysis today to keep your streak alive.
          </p>
        )}
      </CardContent>
    </Card>
  );
}

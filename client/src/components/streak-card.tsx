/**
 * StreakCard — displays the user's daily check-in streak with ethical,
 * non-addictive framing.  Shows current streak, progress toward the next
 * milestone, and which insight features have been unlocked.
 */

import { useQuery } from "@tanstack/react-query";
import { Flame } from "lucide-react";
import { useLocation } from "wouter";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { getMLApiUrl } from "@/lib/ml-api";

// ── Types ─────────────────────────────────────────────────────────────────────

interface StreakStatus {
  user_id: string;
  streak_days: number;
  longest_streak: number;
  today_checked_in: boolean;
  flexible_days_used: number;
  milestones_achieved: string[];
  next_milestone: number | null;
  unlocked_features: string[];
  last_checkin_date: string | null;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const FEATURE_LABELS: Record<string, string> = {
  weekly_patterns: "Weekly Patterns",
  supplement_correlations: "Supplement Correlations",
  monthly_report: "Monthly Report",
  personal_model: "Personal Model",
};

function progressPercent(streakDays: number, nextMilestone: number | null): number {
  if (!nextMilestone) return 100;
  // Find the previous milestone (or 0)
  const milestones = [0, 3, 7, 14, 30, 60, 90];
  let prev = 0;
  for (const m of milestones) {
    if (m < nextMilestone && streakDays >= m) prev = m;
  }
  const range = nextMilestone - prev;
  if (range <= 0) return 100;
  return Math.min(100, Math.round(((streakDays - prev) / range) * 100));
}

// ── Component ─────────────────────────────────────────────────────────────────

interface StreakCardProps {
  userId: string;
}

export function StreakCard({ userId }: StreakCardProps) {
  const [, navigate] = useLocation();

  const { data, isLoading, isError } = useQuery<StreakStatus>({
    queryKey: ["streak-status", userId],
    queryFn: async () => {
      const baseUrl = getMLApiUrl();
      const res = await fetch(
        `${baseUrl}/api/streaks/status/${encodeURIComponent(userId)}`,
        { headers: { "Content-Type": "application/json" } }
      );
      if (!res.ok) throw new Error(`Streak status error: ${res.status}`);
      return res.json() as Promise<StreakStatus>;
    },
    staleTime: 60_000,
    retry: 1,
  });

  /** Navigate to the emotions page to start a voice check-in. */
  const handleCheckin = () => {
    navigate("/emotions");
  };

  if (isLoading) {
    return (
      <Card className="bg-card rounded-xl border border-border/50 shadow-sm">
        <CardContent className="p-4">
          <div className="h-16 flex items-center justify-center">
            <div className="h-5 w-5 rounded-full border-2 border-primary border-t-transparent animate-spin" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (isError || !data) return null;

  const { streak_days, next_milestone, unlocked_features, today_checked_in } = data;
  const pct = progressPercent(streak_days, next_milestone);
  const hasStreak = streak_days > 0;
  const isGreen = streak_days >= 7;

  // Calculate hours remaining until midnight for streak reset warning
  const now = new Date();
  const midnight = new Date(now);
  midnight.setHours(24, 0, 0, 0);
  const hoursUntilReset = Math.round((midnight.getTime() - now.getTime()) / 3_600_000);

  // Milestone badges — 7d, 14d, 30d, 100d
  const STREAK_MILESTONES = [7, 14, 30, 100];

  return (
    <Card className="bg-card rounded-xl border border-border/50 shadow-sm">
      <CardContent className="p-4 space-y-3">
        {/* Header — prominent streak number */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Flame
              className={`h-6 w-6 ${isGreen ? "text-emerald-500" : "text-orange-400"}`}
            />
            <div>
              <div className="flex items-baseline gap-1.5">
                <span className="text-2xl font-extrabold tabular-nums leading-none text-foreground">
                  {streak_days}
                </span>
                <span className="text-sm font-semibold text-muted-foreground">
                  day{streak_days !== 1 ? "s" : ""}
                </span>
              </div>
              {data.longest_streak > streak_days && (
                <p className="text-[11px] text-muted-foreground mt-0.5">
                  Best: {data.longest_streak} days
                </p>
              )}
            </div>
          </div>

          {!today_checked_in && (
            <Badge
              variant="outline"
              className="text-xs border-primary/40 text-primary cursor-pointer hover:bg-primary/10 transition-colors"
              onClick={handleCheckin}
            >
              Check in today
            </Badge>
          )}
        </div>

        {/* Milestone badges row */}
        <div className="flex items-center gap-2">
          {STREAK_MILESTONES.map((m) => {
            const reached = data.longest_streak >= m;
            return (
              <span
                key={m}
                className={`inline-flex items-center gap-0.5 px-2 py-0.5 rounded-md text-[11px] font-semibold border ${
                  reached
                    ? "bg-emerald-500/15 text-emerald-400 border-emerald-500/30"
                    : "bg-muted/20 text-muted-foreground/40 border-border/30"
                }`}
                title={reached ? `${m}-day milestone reached` : `Reach a ${m}-day streak to unlock`}
              >
                {reached && <span aria-label="unlocked">&#10003;</span>}
                {m}d
              </span>
            );
          })}
        </div>

        {/* Progress to next milestone */}
        {next_milestone !== null && (
          <div className="space-y-1">
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>{streak_days}d</span>
              <span>Next: {next_milestone}d</span>
            </div>
            <Progress value={pct} className="h-1.5" />
          </div>
        )}

        {/* Midnight reset warning */}
        {!today_checked_in && hasStreak && (
          <p className="text-[11px] text-amber-400/80">
            Check in by midnight to keep your streak ({hoursUntilReset}h remaining)
          </p>
        )}

        {/* Unlocked features */}
        {unlocked_features.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {unlocked_features.map((key) => (
              <Badge
                key={key}
                className="text-xs bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 hover:bg-emerald-500/20"
              >
                {FEATURE_LABELS[key] ?? key}
              </Badge>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

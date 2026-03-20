/**
 * NeurofeedbackScheduleCard — shows session scheduling guidance based on
 * evidence-based 2-3 day spacing for optimal neuroplasticity.
 *
 * Never guilt-trips. Always supportive. Encourages rest when needed.
 */

import { Card, CardContent } from "@/components/ui/card";
import { Calendar, Brain, Target } from "lucide-react";
import type { SessionSchedule } from "@/lib/neurofeedback-schedule";

// ── Progress Ring ────────────────────────────────────────────────────────────

function ProgressRing({
  percent,
  size = 64,
  strokeWidth = 5,
}: {
  percent: number;
  size?: number;
  strokeWidth?: number;
}) {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - percent / 100);
  const center = size / 2;

  return (
    <svg width={size} height={size} className="shrink-0">
      <circle
        cx={center}
        cy={center}
        r={radius}
        fill="none"
        stroke="rgba(255,255,255,0.08)"
        strokeWidth={strokeWidth}
      />
      <circle
        cx={center}
        cy={center}
        r={radius}
        fill="none"
        stroke="hsl(var(--primary))"
        strokeWidth={strokeWidth}
        strokeDasharray={circumference}
        strokeDashoffset={offset}
        strokeLinecap="round"
        transform={`rotate(-90 ${center} ${center})`}
        className="transition-all duration-500"
      />
      <text
        x={center}
        y={center}
        textAnchor="middle"
        dominantBaseline="central"
        fill="currentColor"
        fontSize="13"
        fontWeight="bold"
        fontFamily="monospace"
      >
        {percent}%
      </text>
    </svg>
  );
}

// ── Phase Badge ──────────────────────────────────────────────────────────────

const PHASE_CONFIG = {
  beginner: {
    label: "Learning",
    range: "Sessions 1-8",
    color: "bg-blue-500/15 text-blue-400 border-blue-500/30",
  },
  building: {
    label: "Building",
    range: "Sessions 9-20",
    color: "bg-violet-500/15 text-violet-400 border-violet-500/30",
  },
  maintaining: {
    label: "Maintaining",
    range: "Session 21+",
    color: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
  },
} as const;

// ── Timing indicator color ───────────────────────────────────────────────────

function getTimingStyle(schedule: SessionSchedule) {
  if (schedule.totalSessions === 0) {
    return {
      border: "border-primary/30",
      bg: "bg-primary/5",
      textColor: "text-primary",
    };
  }
  if (schedule.isTooSoon) {
    return {
      border: "border-amber-500/30",
      bg: "bg-amber-500/5",
      textColor: "text-amber-400",
    };
  }
  if (schedule.isOptimalWindow) {
    return {
      border: "border-emerald-500/30",
      bg: "bg-emerald-500/5",
      textColor: "text-emerald-400",
    };
  }
  if (schedule.isTooLate) {
    return {
      border: "border-rose-500/30",
      bg: "bg-rose-500/5",
      textColor: "text-rose-300",
    };
  }
  // Getting late (4-5 days)
  return {
    border: "border-primary/30",
    bg: "bg-primary/5",
    textColor: "text-foreground/70",
  };
}

// ── Component ────────────────────────────────────────────────────────────────

interface NeurofeedbackScheduleCardProps {
  schedule: SessionSchedule;
}

export function NeurofeedbackScheduleCard({
  schedule,
}: NeurofeedbackScheduleCardProps) {
  const phase = PHASE_CONFIG[schedule.currentPhase];
  const timing = getTimingStyle(schedule);

  return (
    <Card
      className={`${timing.bg} rounded-xl ${timing.border} border shadow-sm`}
      data-testid="nf-schedule-card"
    >
      <CardContent className="p-4 space-y-4">
        {/* Header row: message + progress ring */}
        <div className="flex items-center gap-4">
          <ProgressRing percent={schedule.progressPercent} />
          <div className="flex-1 min-w-0">
            <p className={`text-sm font-medium ${timing.textColor}`}>
              {schedule.message}
            </p>
            {schedule.totalSessions > 0 && (
              <p className="text-xs text-muted-foreground mt-1">
                {schedule.totalSessions} of 20 sessions toward measurable change
              </p>
            )}
          </div>
        </div>

        {/* Phase + next session row */}
        <div className="flex items-center justify-between gap-3 flex-wrap">
          {/* Phase badge */}
          <div className="flex items-center gap-2">
            <Brain className="h-3.5 w-3.5 text-muted-foreground" />
            <span
              className={`inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-semibold border ${phase.color}`}
            >
              {phase.label}
            </span>
            <span className="text-[11px] text-muted-foreground">
              {phase.range}
            </span>
          </div>

          {/* Next session date */}
          {schedule.nextSessionDate && (
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <Calendar className="h-3.5 w-3.5" />
              <span>
                Next:{" "}
                {schedule.nextSessionDate.toLocaleDateString(undefined, {
                  weekday: "short",
                  month: "short",
                  day: "numeric",
                })}
              </span>
            </div>
          )}
        </div>

        {/* 20-session milestone context for newcomers */}
        {schedule.totalSessions === 0 && (
          <p className="text-[11px] text-muted-foreground leading-relaxed">
            Research shows 20 sessions with 2-3 day spacing produces
            measurable neuroplastic changes. No rush -- go at your own pace.
          </p>
        )}
      </CardContent>
    </Card>
  );
}

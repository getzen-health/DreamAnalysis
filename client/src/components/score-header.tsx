/**
 * ScoreHeader -- compact horizontal strip showing current emotion + health scores.
 *
 * Bevel pattern: scores first, content second. This component sits at the top
 * of every page via app-layout.tsx, below the nav bar but above page content.
 */

import { useState } from "react";
import { useCurrentEmotion } from "@/hooks/use-current-emotion";
import { useScores } from "@/hooks/use-scores";
import { useAuth } from "@/hooks/use-auth";
import { EmotionBadge } from "@/components/emotion-badge";
import { cn } from "@/lib/utils";

// ── Score status helpers ─────────────────────────────────────────────────────

function getRecoveryStatus(value: number): { label: string; color: string } {
  if (value >= 67) return { label: "Great", color: "text-green-400" };
  if (value >= 34) return { label: "Good", color: "text-yellow-400" };
  return { label: "Low", color: "text-red-400" };
}

function getStrainStatus(value: number): { label: string; color: string } {
  if (value >= 67) return { label: "Heavy", color: "text-red-400" };
  if (value >= 34) return { label: "Moderate", color: "text-yellow-400" };
  return { label: "Light", color: "text-green-400" };
}

function getEnergyStatus(value: number): { label: string; color: string } {
  if (value >= 67) return { label: "Full", color: "text-green-400" };
  if (value >= 34) return { label: "Half", color: "text-yellow-400" };
  return { label: "Low", color: "text-red-400" };
}

// ── Score descriptions ───────────────────────────────────────────────────────

const SCORE_INFO: Record<string, { subtitle: string; detail: string }> = {
  Recovery: {
    subtitle: "How well your body has recovered",
    detail:
      "Computed from sleep quality, resting heart rate, and HRV trends. Higher means your body is ready for activity.",
  },
  Strain: {
    subtitle: "Physical and mental load today",
    detail:
      "Based on heart rate, activity intensity, and cumulative stress. Lower strain means less accumulated load on your system.",
  },
  Energy: {
    subtitle: "Your available energy reserve",
    detail:
      "Derived from sleep duration, recovery score, and recent strain history. Indicates how much capacity you have left.",
  },
};

// ── Info Tooltip ──────────────────────────────────────────────────────────────

function InfoTooltip({ scoreKey }: { scoreKey: string }) {
  const [open, setOpen] = useState(false);
  const info = SCORE_INFO[scoreKey];
  if (!info) return null;

  return (
    <div className="relative inline-flex">
      <button
        type="button"
        aria-label={`What is ${scoreKey}?`}
        className="inline-flex items-center justify-center w-3 h-3 rounded-full border border-muted-foreground/40 text-muted-foreground/60 hover:text-foreground hover:border-foreground/60 transition-colors text-[7px] font-bold leading-none cursor-help"
        onClick={(e) => {
          e.stopPropagation();
          setOpen((prev) => !prev);
        }}
        onBlur={() => setOpen(false)}
      >
        ?
      </button>
      {open && (
        <div
          className="absolute bottom-full right-0 mb-1.5 w-48 rounded-md border border-border bg-popover p-2 shadow-lg z-50 text-left"
          role="tooltip"
        >
          <p className="text-[10px] font-medium text-foreground leading-tight mb-0.5">
            {scoreKey}
          </p>
          <p className="text-[9px] text-muted-foreground leading-snug">
            {info.detail}
          </p>
        </div>
      )}
    </div>
  );
}

// ── Mini Score Arc ────────────────────────────────────────────────────────────

interface MiniScoreProps {
  value: number | null;
  label: string;
  colorFrom: string;
  colorTo: string;
  gradientId: string;
  statusFn: (v: number) => { label: string; color: string };
}

function MiniScoreArc({
  value,
  label,
  colorFrom,
  colorTo,
  gradientId,
  statusFn,
}: MiniScoreProps) {
  const size = 44;
  const r = 16;
  const stroke = 3.5;
  const cx = size / 2;
  const cy = size / 2;
  const circumference = 2 * Math.PI * r;
  const arcLength = circumference * 0.75;
  const gapLength = circumference * 0.25;
  const displayValue = value ?? 0;
  const dashOffset = arcLength * (1 - displayValue / 100);

  const status = value !== null ? statusFn(displayValue) : null;
  const info = SCORE_INFO[label];

  return (
    <div
      className="flex flex-col items-center gap-0.5 group relative"
      title={info?.subtitle}
    >
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <defs>
          <linearGradient
            id={gradientId}
            gradientUnits="userSpaceOnUse"
            x1={size * 0.15}
            y1={size * 0.15}
            x2={size * 0.85}
            y2={size * 0.85}
          >
            <stop offset="0%" stopColor={colorFrom} />
            <stop offset="100%" stopColor={colorTo} />
          </linearGradient>
        </defs>
        {/* Background track */}
        <circle
          cx={cx}
          cy={cy}
          r={r}
          fill="none"
          stroke="var(--border)"
          strokeWidth={stroke}
          strokeDasharray={`${arcLength} ${gapLength}`}
          strokeLinecap="round"
          transform={`rotate(135 ${cx} ${cy})`}
          opacity={0.4}
        />
        {/* Score arc */}
        {value !== null && (
          <circle
            cx={cx}
            cy={cy}
            r={r}
            fill="none"
            stroke={`url(#${gradientId})`}
            strokeWidth={stroke}
            strokeDasharray={`${arcLength} ${gapLength}`}
            strokeDashoffset={dashOffset}
            strokeLinecap="round"
            transform={`rotate(135 ${cx} ${cy})`}
            style={{
              transition:
                "stroke-dashoffset 0.8s cubic-bezier(0.34, 1.56, 0.64, 1)",
            }}
          />
        )}
        {/* Number */}
        <text
          x={cx}
          y={cy + 1}
          textAnchor="middle"
          dominantBaseline="central"
          fill="var(--foreground)"
          fontSize={12}
          fontWeight="600"
          fontFamily="Inter, system-ui, sans-serif"
        >
          {value !== null ? Math.round(displayValue) : "--"}
        </text>
      </svg>
      {/* Label row: name + info icon */}
      <span className="flex items-center gap-0.5">
        <span className="text-[9px] font-medium text-muted-foreground leading-none">
          {label}
        </span>
        <InfoTooltip scoreKey={label} />
      </span>
      {/* Status label */}
      {status && (
        <span
          className={cn(
            "text-[8px] font-semibold leading-none -mt-0.5",
            status.color,
          )}
        >
          {status.label}
        </span>
      )}
    </div>
  );
}

// ── ScoreHeader ──────────────────────────────────────────────────────────────

export function ScoreHeader() {
  const { user } = useAuth();
  const userId = user?.id ?? (user as any)?.userId;
  const { scores, loading } = useScores(userId);
  const { emotion } = useCurrentEmotion();

  // Show syncing placeholder if no user or still loading with no data
  if (!userId || (loading && !scores && !emotion)) {
    return (
      <div className="flex items-center justify-center px-4 py-2 border-b border-border/50 bg-card/60 backdrop-blur-sm">
        <span className="text-[11px] text-muted-foreground animate-pulse">
          Syncing health data...
        </span>
      </div>
    );
  }

  return (
    <div
      className={cn(
        "flex items-center justify-between px-4 py-1.5",
        "border-b border-border/40 bg-card/50 backdrop-blur-sm",
      )}
    >
      {/* Left: Emotion Badge */}
      <EmotionBadge
        size="sm"
        showLabel
        onClick={() => window.dispatchEvent(new Event("ndw-open-voice-checkin"))}
      />

      {/* Right: Mini score circles */}
      <div className="flex items-center gap-2">
        <MiniScoreArc
          value={scores?.recoveryScore ?? null}
          label="Recovery"
          colorFrom="hsl(152, 60%, 48%)"
          colorTo="hsl(165, 55%, 38%)"
          gradientId="sh-recovery"
          statusFn={getRecoveryStatus}
        />
        <MiniScoreArc
          value={scores?.strainScore ?? null}
          label="Strain"
          colorFrom="hsl(32, 85%, 55%)"
          colorTo="hsl(0, 65%, 55%)"
          gradientId="sh-strain"
          statusFn={getStrainStatus}
        />
        <MiniScoreArc
          value={scores?.energyBank ?? null}
          label="Energy"
          colorFrom="hsl(200, 70%, 55%)"
          colorTo="hsl(262, 45%, 65%)"
          gradientId="sh-energy"
          statusFn={getEnergyStatus}
        />
      </div>
    </div>
  );
}

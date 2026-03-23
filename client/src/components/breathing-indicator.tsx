/**
 * BreathingIndicator — Visual indicator for breathing state detected from frontal EEG.
 *
 * Shows an animated circle that pulses with the detected breathing rhythm,
 * color-coded by breathing state:
 *   Green  — deep_slow (meditation breathing)
 *   Cyan   — normal
 *   Amber  — shallow_fast (stress/anxiety)
 *   Gray   — holding / unknown
 *
 * Displays breaths/min, coherence score, and a descriptive message.
 *
 * @see Issue #529
 */

import { type BreathingAnalysis, type BreathingState } from "@/lib/breathing-detector";
import { Badge } from "@/components/ui/badge";
import { Wind } from "lucide-react";

export interface BreathingIndicatorProps {
  analysis: BreathingAnalysis;
}

const STATE_COLORS: Record<BreathingState, {
  icon: string;
  badge: string;
  border: string;
  bg: string;
  ring: string;
}> = {
  deep_slow: {
    icon: "text-emerald-400",
    badge: "text-emerald-400 border-emerald-400/30",
    border: "border-emerald-500/30",
    bg: "bg-emerald-500/5",
    ring: "ring-emerald-400/40",
  },
  normal: {
    icon: "text-cyan-400",
    badge: "text-cyan-400 border-cyan-400/30",
    border: "border-border/50",
    bg: "bg-card/50",
    ring: "ring-cyan-400/40",
  },
  shallow_fast: {
    icon: "text-amber-400",
    badge: "text-amber-400 border-amber-400/30",
    border: "border-amber-500/30",
    bg: "bg-amber-500/5",
    ring: "ring-amber-400/40",
  },
  holding: {
    icon: "text-gray-400",
    badge: "text-gray-400 border-gray-400/30",
    border: "border-border/50",
    bg: "bg-card/50",
    ring: "ring-gray-400/40",
  },
  unknown: {
    icon: "text-gray-400",
    badge: "text-gray-400 border-gray-400/30",
    border: "border-border/50",
    bg: "bg-card/50",
    ring: "ring-gray-400/40",
  },
};

const STATE_LABELS: Record<BreathingState, string> = {
  deep_slow: "Deep & Slow",
  normal: "Normal",
  shallow_fast: "Shallow & Fast",
  holding: "Holding",
  unknown: "Unclear",
};

/**
 * Compute the CSS animation duration for the pulsing circle based on
 * the estimated breathing rate. Lower rate = slower pulse.
 */
function getPulseDuration(rate: number): string {
  if (rate <= 0) return "0s"; // no animation for holding/unknown
  const periodSec = 60 / rate; // seconds per breath
  return `${periodSec.toFixed(1)}s`;
}

export function BreathingIndicator({ analysis }: BreathingIndicatorProps) {
  const { state, estimatedRate, coherence, message } = analysis;
  const colors = STATE_COLORS[state];
  const shouldAnimate = state !== "holding" && state !== "unknown" && estimatedRate > 0;
  const pulseDuration = getPulseDuration(estimatedRate);

  return (
    <div
      className={`rounded-xl border px-4 py-3 flex items-center gap-3 ${colors.border} ${colors.bg}`}
      data-testid="breathing-indicator"
    >
      {/* Animated breathing circle */}
      <div className="relative flex items-center justify-center">
        <div
          className={`h-8 w-8 rounded-full flex items-center justify-center ${
            shouldAnimate ? `animate-breathe ${colors.ring}` : ""
          }`}
          style={
            shouldAnimate
              ? ({ "--breathe-duration": pulseDuration } as React.CSSProperties)
              : undefined
          }
        >
          <Wind className={`h-4 w-4 shrink-0 ${colors.icon}`} />
        </div>
      </div>

      {/* Stats and message */}
      <div className="flex items-center gap-4 flex-wrap flex-1">
        <div>
          {estimatedRate > 0 && (
            <span className="text-sm font-medium">
              {Math.round(estimatedRate)} breaths/min
            </span>
          )}
          {coherence > 0 && (
            <span className="text-xs text-muted-foreground ml-2">
              coherence {Math.round(coherence * 100)}%
            </span>
          )}
        </div>
        <Badge
          variant="outline"
          className={`text-[10px] ${colors.badge}`}
        >
          {STATE_LABELS[state]}
        </Badge>
      </div>

      {/* Message */}
      {message && (
        <p className="text-xs text-muted-foreground hidden sm:block max-w-[200px]">
          {message}
        </p>
      )}
    </div>
  );
}

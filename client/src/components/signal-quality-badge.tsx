import { Radio } from "lucide-react";

export interface SignalQualityBadgeProps {
  /** Signal quality score: 0 (worst) to 1 (best), typically from clean_ratio */
  quality: number;
  /** Whether the EEG stream is active */
  isStreaming: boolean;
}

/**
 * Inline badge showing EEG signal quality in three states:
 *   good (>0.7)    — green dot + "Good signal"
 *   moderate (>0.4) — amber dot + "Noisy"
 *   poor (<=0.4)   — red dot + "Too noisy — relax face muscles"
 *
 * Not rendered when `isStreaming` is false.
 */
export function SignalQualityBadge({ quality, isStreaming }: SignalQualityBadgeProps) {
  if (!isStreaming) return null;

  const { label, dotColor, badgeColor } = getQualityState(quality);

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-xs font-medium ${badgeColor}`}
      data-testid="signal-quality-badge"
      title={`Signal quality: ${Math.round(quality * 100)}%`}
    >
      <span className={`h-2 w-2 rounded-full shrink-0 ${dotColor}`} />
      {label}
    </span>
  );
}

function getQualityState(quality: number): {
  label: string;
  dotColor: string;
  badgeColor: string;
} {
  if (quality > 0.7) {
    return {
      label: "Good signal",
      dotColor: "bg-emerald-500",
      badgeColor: "text-emerald-500 border-emerald-500/30 bg-emerald-500/10",
    };
  }
  if (quality > 0.4) {
    return {
      label: "Noisy",
      dotColor: "bg-amber-500",
      badgeColor: "text-amber-500 border-amber-500/30 bg-amber-500/10",
    };
  }
  return {
    label: "Too noisy \u2014 relax face muscles",
    dotColor: "bg-red-500",
    badgeColor: "text-red-500 border-red-500/30 bg-red-500/10",
  };
}

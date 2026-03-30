/**
 * ConfidenceMeter — Visual confidence indicator for emotion readings.
 *
 * Displays a colored bar/badge with confidence level:
 *   Green  (>70%):   "High confidence"
 *   Amber  (40-70%): "Moderate"
 *   Red    (<40%):   "Low — take with grain of salt"
 *
 * Supports two variants:
 *   - "bar" (default): horizontal progress bar
 *   - "badge": compact inline badge with dot + percentage
 *
 * @see Issue #523
 */

import { calculateEmotionConfidence, type ConfidenceFactors } from "@/lib/confidence-calculator";

export interface ConfidenceMeterProps {
  /** Overall confidence 0-1 (or provide `factors` to compute it) */
  confidence: number;
  /** Optional: compute from multiple factors instead of raw confidence */
  factors?: ConfidenceFactors;
  /** Show text label beside the bar */
  showLabel?: boolean;
  /** Bar size: "sm" = 4px, "md" = 8px, "lg" = 12px */
  size?: "sm" | "md" | "lg";
  /** Display variant: "bar" for progress bar, "badge" for inline chip */
  variant?: "bar" | "badge";
}

function getBarColor(confidence: number): string {
  if (confidence > 0.7) return "bg-emerald-500";
  if (confidence >= 0.4) return "bg-amber-500";
  return "bg-red-500";
}

function getDotColor(confidence: number): string {
  if (confidence > 0.7) return "bg-emerald-500";
  if (confidence >= 0.4) return "bg-amber-500";
  return "bg-red-500";
}

function getBadgeClass(confidence: number): string {
  if (confidence > 0.7) return "bg-emerald-500/15 text-emerald-400 border-emerald-500/30";
  if (confidence >= 0.4) return "bg-amber-500/15 text-amber-400 border-amber-500/30";
  return "bg-red-500/15 text-red-400 border-red-500/30";
}

export function getConfidenceLabel(confidence: number): string {
  if (confidence > 0.7) return "High confidence";
  if (confidence >= 0.4) return "Moderate";
  return "Low — take with grain of salt";
}

function getLabelColor(confidence: number): string {
  if (confidence > 0.7) return "text-emerald-400";
  if (confidence >= 0.4) return "text-amber-400";
  return "text-red-400";
}

/** Returns a warning message for low confidence, or null if confidence is adequate. */
export function getConfidenceWarning(confidence: number): string | null {
  if (confidence < 0.4) return "Low — take with grain of salt";
  return null;
}

export function ConfidenceMeter({
  confidence: rawConfidence,
  factors,
  showLabel = false,
  size = "sm",
  variant = "bar",
}: ConfidenceMeterProps) {
  // If factors are provided, compute the confidence from them
  const computed = factors ? calculateEmotionConfidence(factors) : null;
  const confidence = Math.max(0, Math.min(1, computed ? computed.confidence : rawConfidence));
  const percent = Math.round(confidence * 100);
  const sizeClass = size === "sm" ? "h-1" : size === "lg" ? "h-3" : "h-2";

  if (variant === "badge") {
    return (
      <span
        data-testid="confidence-meter"
        className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[10px] font-medium ${sizeClass} ${getBadgeClass(confidence)}`}
        title={`${percent}% confidence`}
      >
        <span className={`h-1.5 w-1.5 rounded-full ${getDotColor(confidence)} shrink-0`} />
        {percent}%
        {showLabel && <span className="ml-0.5">{getConfidenceLabel(confidence)}</span>}
      </span>
    );
  }

  // Bar variant (default)
  return (
    <div className="flex flex-col gap-1">
      <div
        data-testid="confidence-meter"
        className={`w-full ${sizeClass} rounded-full overflow-hidden bg-border`}
      >
        <div
          data-testid="confidence-bar-fill"
          className={`h-full rounded-full transition-all duration-300 ${getBarColor(confidence)}`}
          style={{ width: `${percent}%` }}
        />
      </div>
      {showLabel && (
        <div className="flex items-center justify-between">
          <span
            data-testid="confidence-label"
            className={`text-[10px] font-medium ${getLabelColor(confidence)}`}
          >
            {getConfidenceLabel(confidence)}
          </span>
          <span className={`text-[10px] font-mono ${getLabelColor(confidence)}`}>
            {percent}%
          </span>
        </div>
      )}
      {showLabel && confidence < 0.4 && (
        <p className="text-[10px] text-amber-400/80 leading-tight" data-testid="confidence-warning">
          {getConfidenceWarning(confidence)}
        </p>
      )}
    </div>
  );
}

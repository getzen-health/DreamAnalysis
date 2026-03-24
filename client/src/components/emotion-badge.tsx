/**
 * EmotionBadge -- compact badge showing current emotion state.
 *
 * Designed for page headers and cards. Shows emoji + label, color-coded
 * by emotion. Pulses when data is stale to encourage a re-check.
 *
 * Supports both base emotions (happy, sad, angry) and nuanced compound
 * emotions (content, excited, anxious, melancholy) via the unified
 * NUANCED_EMOJI and NUANCED_COLORS maps from nuanced-emotion.ts.
 */

import { useCurrentEmotion } from "@/hooks/use-current-emotion";
import { cn } from "@/lib/utils";
import { NUANCED_EMOJI, NUANCED_COLORS } from "@/lib/nuanced-emotion";
import {
  mapToNuancedEmotion,
  type NuancedEmotionResult,
} from "@/lib/nuanced-emotion";

interface EmotionBadgeProps {
  /** sm for headers, md for cards */
  size?: "sm" | "md";
  /** Show text label alongside emoji */
  showLabel?: boolean;
  /** Callback when tapped (e.g. open voice analysis) */
  onClick?: () => void;
  /** Optional pre-computed nuanced emotion (from multimodal fusion). */
  nuanced?: NuancedEmotionResult;
}

const DEFAULT_EMOJI = "\u{1F9E0}";
const DEFAULT_COLOR = "bg-muted text-muted-foreground border-border";

export function EmotionBadge({
  size = "sm",
  showLabel = true,
  onClick,
  nuanced: externalNuanced,
}: EmotionBadgeProps) {
  const { emotion } = useCurrentEmotion();

  const isSm = size === "sm";

  // No data or stale -- show "Check in" prompt
  if (!emotion || emotion.isStale) {
    return (
      <button
        onClick={onClick}
        className={cn(
          "inline-flex items-center gap-1 rounded-full border transition-colors",
          "border-primary/30 bg-primary/10 text-primary hover:bg-primary/20",
          isSm ? "px-2 py-0.5 text-[11px]" : "px-3 py-1 text-xs",
          emotion?.isStale && "animate-pulse",
        )}
      >
        <span className={isSm ? "text-xs" : "text-sm"}>
          {emotion ? NUANCED_EMOJI[emotion.emotion] ?? DEFAULT_EMOJI : "\u{1F3A4}"}
        </span>
        {showLabel && <span className="font-medium">No data yet</span>}
      </button>
    );
  }

  // Derive nuanced label: use pre-computed if passed, otherwise compute from raw data
  const nuanced = externalNuanced ?? mapToNuancedEmotion({
    emotion: emotion.emotion,
    valence: emotion.valence,
    arousal: emotion.arousal,
    stress: emotion.stress,
    confidence: emotion.confidence,
  });

  const emoji = NUANCED_EMOJI[nuanced.label] ?? NUANCED_EMOJI[emotion.emotion] ?? DEFAULT_EMOJI;
  const colorClasses = NUANCED_COLORS[nuanced.label] ?? NUANCED_COLORS[emotion.emotion] ?? DEFAULT_COLOR;

  return (
    <button
      onClick={onClick}
      className={cn(
        "inline-flex items-center gap-1 rounded-full border transition-colors",
        colorClasses,
        isSm ? "px-2 py-0.5 text-[11px]" : "px-3 py-1 text-xs",
        onClick && "cursor-pointer hover:opacity-80",
      )}
    >
      <span className={isSm ? "text-xs" : "text-sm"}>{emoji}</span>
      {showLabel && <span className="font-semibold">{nuanced.displayLabel}</span>}
    </button>
  );
}

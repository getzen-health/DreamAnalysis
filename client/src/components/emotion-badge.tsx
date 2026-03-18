/**
 * EmotionBadge -- compact badge showing current emotion state.
 *
 * Designed for page headers and cards. Shows emoji + label, color-coded
 * by emotion. Pulses when data is stale to encourage a re-check.
 */

import { useCurrentEmotion } from "@/hooks/use-current-emotion";
import { cn } from "@/lib/utils";

const EMOTION_EMOJI: Record<string, string> = {
  happy: "\u{1F60A}",
  sad: "\u{1F614}",
  angry: "\u{1F620}",
  fear: "\u{1F630}",
  surprise: "\u{1F62E}",
  neutral: "\u{1F610}",
};

const EMOTION_COLORS: Record<string, string> = {
  happy: "bg-cyan-500/15 text-cyan-500 border-cyan-500/30",
  sad: "bg-indigo-500/15 text-indigo-400 border-indigo-500/30",
  angry: "bg-rose-500/15 text-rose-400 border-rose-500/30",
  fear: "bg-purple-500/15 text-purple-500 border-purple-500/30",
  surprise: "bg-amber-500/15 text-amber-500 border-amber-500/30",
  neutral: "bg-muted text-muted-foreground border-border",
};

interface EmotionBadgeProps {
  /** sm for headers, md for cards */
  size?: "sm" | "md";
  /** Show text label alongside emoji */
  showLabel?: boolean;
  /** Callback when tapped (e.g. open voice analysis) */
  onClick?: () => void;
}

export function EmotionBadge({
  size = "sm",
  showLabel = true,
  onClick,
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
          {emotion ? EMOTION_EMOJI[emotion.emotion] ?? "\u{1F9E0}" : "\u{1F3A4}"}
        </span>
        {showLabel && <span className="font-medium">No data yet</span>}
      </button>
    );
  }

  const emoji = EMOTION_EMOJI[emotion.emotion] ?? "\u{1F9E0}";
  const colorClasses = EMOTION_COLORS[emotion.emotion] ?? EMOTION_COLORS.neutral;
  const label = emotion.emotion.charAt(0).toUpperCase() + emotion.emotion.slice(1);

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
      {showLabel && <span className="font-semibold">{label}</span>}
    </button>
  );
}

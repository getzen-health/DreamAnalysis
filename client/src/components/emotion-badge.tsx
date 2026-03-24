/**
 * EmotionBadge — premium compact badge showing current emotion state.
 *
 * Features:
 * - Gradient background matching emotion color
 * - Soft glow effect behind badge
 * - Spring animation on tap
 * - Smooth color transitions between emotion states
 * - Pulse animation when data is stale
 */

import { motion, AnimatePresence } from "framer-motion";
import { useCurrentEmotion } from "@/hooks/use-current-emotion";
import { cn } from "@/lib/utils";
import { NUANCED_EMOJI } from "@/lib/nuanced-emotion";
import {
  mapToNuancedEmotion,
  type NuancedEmotionResult,
} from "@/lib/nuanced-emotion";
import { springs } from "@/lib/animations";

interface EmotionBadgeProps {
  size?: "sm" | "md";
  showLabel?: boolean;
  onClick?: () => void;
  nuanced?: NuancedEmotionResult;
}

const DEFAULT_EMOJI = "\u{1F9E0}";

/** Gradient pairs for each emotion — from/to colors */
const EMOTION_GRADIENTS: Record<string, { from: string; to: string; glow: string }> = {
  excited:       { from: "#FBBF24", to: "#F59E0B", glow: "rgba(251, 191, 36, 0.3)" },
  energized:     { from: "#FB923C", to: "#F97316", glow: "rgba(249, 115, 22, 0.3)" },
  content:       { from: "#34D399", to: "#10B981", glow: "rgba(52, 211, 153, 0.3)" },
  serene:        { from: "#2DD4BF", to: "#14B8A6", glow: "rgba(45, 212, 191, 0.3)" },
  anxious:       { from: "#A78BFA", to: "#8B5CF6", glow: "rgba(167, 139, 250, 0.3)" },
  overwhelmed:   { from: "#FB7185", to: "#F43F5E", glow: "rgba(251, 113, 133, 0.3)" },
  frustrated:    { from: "#F87171", to: "#EF4444", glow: "rgba(248, 113, 113, 0.3)" },
  melancholy:    { from: "#818CF8", to: "#6366F1", glow: "rgba(129, 140, 248, 0.3)" },
  drained:       { from: "#94A3B8", to: "#64748B", glow: "rgba(148, 163, 184, 0.2)" },
  focused:       { from: "#60A5FA", to: "#3B82F6", glow: "rgba(96, 165, 250, 0.3)" },
  contemplative: { from: "#A1A1AA", to: "#71717A", glow: "rgba(161, 161, 170, 0.2)" },
  happy:         { from: "#4ADE80", to: "#22C55E", glow: "rgba(74, 222, 128, 0.3)" },
  sad:           { from: "#818CF8", to: "#6366F1", glow: "rgba(129, 140, 248, 0.3)" },
  angry:         { from: "#FB7185", to: "#EF4444", glow: "rgba(251, 113, 133, 0.3)" },
  fear:          { from: "#C084FC", to: "#A855F7", glow: "rgba(192, 132, 252, 0.3)" },
  fearful:       { from: "#C084FC", to: "#A855F7", glow: "rgba(192, 132, 252, 0.3)" },
  surprise:      { from: "#FBBF24", to: "#F59E0B", glow: "rgba(251, 191, 36, 0.3)" },
  neutral:       { from: "#A1A1AA", to: "#71717A", glow: "rgba(161, 161, 170, 0.15)" },
  calm:          { from: "#2DD4BF", to: "#14B8A6", glow: "rgba(45, 212, 191, 0.3)" },
  relaxed:       { from: "#2DD4BF", to: "#14B8A6", glow: "rgba(45, 212, 191, 0.3)" },
};

const DEFAULT_GRADIENT = { from: "#A1A1AA", to: "#71717A", glow: "rgba(161, 161, 170, 0.15)" };

export function EmotionBadge({
  size = "sm",
  showLabel = true,
  onClick,
  nuanced: externalNuanced,
}: EmotionBadgeProps) {
  const { emotion } = useCurrentEmotion();

  const isSm = size === "sm";

  // No data or stale — show "Check in" prompt with breathing glow
  if (!emotion || emotion.isStale) {
    return (
      <motion.button
        onClick={onClick}
        whileTap={{ scale: 0.92 }}
        whileHover={{ scale: 1.05 }}
        transition={springs.snappy}
        className={cn(
          "inline-flex items-center gap-1.5 rounded-full border relative overflow-hidden",
          "border-primary/20 bg-primary/8 text-primary",
          isSm ? "px-3 py-1 text-xs" : "px-4 py-1.5 text-sm",
        )}
      >
        {/* Breathing glow */}
        {emotion?.isStale && (
          <motion.div
            className="absolute inset-0 rounded-full bg-primary/10"
            animate={{ opacity: [0.3, 0.7, 0.3], scale: [1, 1.05, 1] }}
            transition={{ duration: 2.5, repeat: Infinity, ease: "easeInOut" }}
          />
        )}
        <span className={cn("relative z-10", isSm ? "text-sm" : "text-base")}>
          {emotion ? NUANCED_EMOJI[emotion.emotion] ?? DEFAULT_EMOJI : "\u{1F3A4}"}
        </span>
        {showLabel && (
          <span className="relative z-10 font-medium">
            {emotion?.isStale ? "Check in" : "No data yet"}
          </span>
        )}
      </motion.button>
    );
  }

  // Derive nuanced label
  const nuanced = externalNuanced ?? mapToNuancedEmotion({
    emotion: emotion.emotion,
    valence: emotion.valence,
    arousal: emotion.arousal,
    stress: emotion.stress,
    confidence: emotion.confidence,
  });

  const emoji = NUANCED_EMOJI[nuanced.label] ?? NUANCED_EMOJI[emotion.emotion] ?? DEFAULT_EMOJI;
  const gradient = EMOTION_GRADIENTS[nuanced.label] ?? EMOTION_GRADIENTS[emotion.emotion] ?? DEFAULT_GRADIENT;

  return (
    <AnimatePresence mode="wait">
      <motion.button
        key={nuanced.label}
        onClick={onClick}
        initial={{ opacity: 0, scale: 0.8, filter: "blur(4px)" }}
        animate={{ opacity: 1, scale: 1, filter: "blur(0px)" }}
        exit={{ opacity: 0, scale: 0.8 }}
        whileTap={{ scale: 0.92 }}
        whileHover={{ scale: 1.05 }}
        transition={springs.snappy}
        className={cn(
          "inline-flex items-center gap-1.5 rounded-full relative overflow-hidden",
          isSm ? "px-3 py-1" : "px-4 py-1.5",
          onClick && "cursor-pointer",
        )}
        style={{
          background: `linear-gradient(135deg, ${gradient.from}20, ${gradient.to}15)`,
          border: `1px solid ${gradient.from}30`,
          boxShadow: `0 0 20px ${gradient.glow}`,
        }}
      >
        {/* Animated shimmer overlay */}
        <motion.div
          className="absolute inset-0 rounded-full"
          style={{
            background: `linear-gradient(90deg, transparent, ${gradient.from}15, transparent)`,
            backgroundSize: "200% 100%",
          }}
          animate={{ backgroundPosition: ["-200% 0", "200% 0"] }}
          transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
        />

        <motion.span
          className={cn("relative z-10", isSm ? "text-sm" : "text-lg")}
          animate={{ rotate: [0, -5, 5, 0] }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          {emoji}
        </motion.span>
        {showLabel && (
          <span
            className={cn(
              "relative z-10 font-semibold",
              isSm ? "text-xs" : "text-sm",
            )}
            style={{ color: gradient.from }}
          >
            {nuanced.displayLabel}
          </span>
        )}
      </motion.button>
    </AnimatePresence>
  );
}

/**
 * EmotionStateCard — premium glassmorphism card showing emotion state
 * with animated gradient progress bars, glow accents, and smooth transitions.
 *
 * Inspired by: Apple Health metrics, Oura readiness card, Calm daily check-in.
 */

import { motion } from "framer-motion";
import { Badge } from "@/components/ui/badge";
import {
  Smile, CloudSun, AlertTriangle, Battery, Brain, TreePine, Zap, Minus, Lightbulb,
  type LucideIcon,
} from "lucide-react";
import { springs, easings, cardVariants } from "@/lib/animations";

export interface EmotionStateCardProps {
  emotion: string;
  valence: number;
  arousal: number;
  stressIndex?: number;
  focusIndex?: number;
  confidence?: number;
  source?: string;
}

interface StateInfo {
  icon: LucideIcon;
  gradient: string;
  glowColor: string;
  label: string;
}

function getStateInfo(valence: number, arousal: number): StateInfo {
  if (valence > 0.3 && arousal > 0.6) return {
    icon: Smile, gradient: "from-emerald-400 to-cyan-400",
    glowColor: "rgba(52, 211, 153, 0.3)", label: "You're energized"
  };
  if (valence > 0.3 && arousal <= 0.6) return {
    icon: CloudSun, gradient: "from-teal-400 to-cyan-400",
    glowColor: "rgba(45, 212, 191, 0.3)", label: "You're in a calm state"
  };
  if (valence <= -0.3 && arousal > 0.6) return {
    icon: AlertTriangle, gradient: "from-rose-400 to-pink-400",
    glowColor: "rgba(251, 113, 133, 0.3)", label: "You're feeling stressed"
  };
  if (valence <= -0.3 && arousal <= 0.6) return {
    icon: Battery, gradient: "from-indigo-400 to-violet-400",
    glowColor: "rgba(129, 140, 248, 0.3)", label: "You're low on energy"
  };
  if (valence > 0 && arousal > 0.5) return {
    icon: Brain, gradient: "from-blue-400 to-indigo-400",
    glowColor: "rgba(96, 165, 250, 0.3)", label: "You're alert and focused"
  };
  if (valence > 0) return {
    icon: TreePine, gradient: "from-emerald-400 to-green-400",
    glowColor: "rgba(52, 211, 153, 0.3)", label: "You're relaxed"
  };
  if (arousal > 0.6) return {
    icon: Zap, gradient: "from-amber-400 to-orange-400",
    glowColor: "rgba(251, 191, 36, 0.3)", label: "You're tense"
  };
  return {
    icon: Minus, gradient: "from-slate-400 to-zinc-400",
    glowColor: "rgba(148, 163, 184, 0.2)", label: "You're in neutral mode"
  };
}

function getSuggestion(valence: number, arousal: number, stressIndex?: number): string {
  const stress = stressIndex ?? (arousal > 0.6 && valence < 0 ? 0.7 : 0);
  const energy = arousal;
  const calm = 1 - (stressIndex ?? arousal * 0.5);
  const focus = arousal * 0.6 + valence * 0.4;

  if (stress > 0.55) return "Try 4-7-8 breathing to reset";
  if (energy > 0.6 && focus > 0.5) return "Good time for deep work";
  if (energy > 0.6 && focus <= 0.5) return "Try a short walk first";
  if (energy <= 0.4 && calm > 0.5) return "Good time for creative or reflective work";
  if (energy <= 0.4 && calm <= 0.5) return "Rest or light activity recommended";
  return "Check in again in an hour";
}

const SOURCE_LABELS: Record<string, string> = {
  voice: "via voice",
  eeg: "via EEG",
  health: "via health",
};

interface GradientBarProps {
  label: string;
  value: number;
  gradient: string;
  delay?: number;
}

function GradientBar({ label, value, gradient, delay = 0 }: GradientBarProps) {
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between items-center">
        <span className="text-xs font-medium text-foreground/50">{label}</span>
        <motion.span
          className="text-xs font-mono font-semibold text-foreground/70"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: delay + 0.3 }}
        >
          {value}%
        </motion.span>
      </div>
      <div className="progress-gradient">
        <motion.div
          className={`progress-gradient-fill bg-gradient-to-r ${gradient}`}
          initial={{ width: "0%" }}
          animate={{ width: `${value}%` }}
          transition={{ duration: 1.2, ease: easings.premium, delay }}
        />
      </div>
    </div>
  );
}

export default function EmotionStateCard({
  valence,
  arousal,
  stressIndex,
  focusIndex,
  confidence,
  source,
}: EmotionStateCardProps) {
  const state = getStateInfo(valence, arousal);
  const StateIcon = state.icon;
  const suggestion = getSuggestion(valence, arousal, stressIndex);

  const energy = Math.round(Math.max(0, Math.min(1, arousal)) * 100);
  const calm = Math.round(Math.max(0, Math.min(1, stressIndex !== undefined ? 1 - stressIndex : 1 - arousal)) * 100);
  const focus = focusIndex !== undefined
    ? Math.round(Math.max(0, Math.min(1, focusIndex)) * 100)
    : Math.round(Math.max(0, Math.min(1, arousal * 0.6 + valence * 0.4)) * 100);

  return (
    <motion.div
      variants={cardVariants}
      initial="hidden"
      animate="visible"
      custom={0}
      className="glass-card p-5 space-y-4 relative overflow-hidden"
    >
      {/* Ambient glow behind icon */}
      <div
        className="absolute top-3 left-3 w-16 h-16 rounded-full blur-2xl opacity-40"
        style={{ background: state.glowColor }}
      />

      {/* State header */}
      <div className="flex items-center justify-between relative z-10">
        <div className="flex items-center gap-3">
          <motion.div
            initial={{ scale: 0, rotate: -180 }}
            animate={{ scale: 1, rotate: 0 }}
            transition={springs.bouncy}
          >
            <div className={`p-2 rounded-xl bg-gradient-to-br ${state.gradient} bg-opacity-20`}>
              <StateIcon className="w-6 h-6 text-white" />
            </div>
          </motion.div>
          <motion.span
            className="text-base font-semibold text-foreground"
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.15, duration: 0.3, ease: easings.premium }}
          >
            {state.label}
          </motion.span>
        </div>
        <div className="flex items-center gap-2">
          {confidence !== undefined && (
            <span className="text-xs font-mono text-foreground/40">
              {Math.round(confidence * 100)}%
            </span>
          )}
          {source && SOURCE_LABELS[source] && (
            <Badge
              variant="outline"
              className="text-[10px] border-foreground/10 text-foreground/40 bg-foreground/5"
            >
              {SOURCE_LABELS[source]}
            </Badge>
          )}
        </div>
      </div>

      {/* Animated gradient progress bars */}
      <div className="space-y-3 relative z-10">
        <GradientBar label="Energy" value={energy} gradient="from-amber-400 to-orange-400" delay={0.1} />
        <GradientBar label="Calm" value={calm} gradient="from-teal-400 to-cyan-400" delay={0.2} />
        <GradientBar label="Focus" value={focus} gradient="from-blue-400 to-indigo-400" delay={0.3} />
      </div>

      {/* Suggestion with icon */}
      <motion.div
        className="flex items-center gap-2 px-3 py-2 rounded-xl bg-foreground/[0.03] relative z-10"
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5, duration: 0.3, ease: easings.premium }}
      >
        <Lightbulb className="w-3.5 h-3.5 text-amber-400 shrink-0" />
        <p className="text-xs text-foreground/50 italic">{suggestion}</p>
      </motion.div>
    </motion.div>
  );
}

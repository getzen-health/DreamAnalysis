/**
 * BrainCoachCard — Daily AI-synthesized brief fusing EEG brain data + health scores.
 *
 * Differentiates NDW: recovery coaching enhanced with EEG brain coaching.
 * Generates 1-3 personalized recommendations from a rule engine that combines
 * physiological health scores with EEG-derived focus, valence, and stress.
 */

import { motion } from "framer-motion";
import { Zap, Brain, Moon, Heart } from "lucide-react";
import { cardVariants } from "@/lib/animations";
import type { LucideIcon } from "lucide-react";

// ── Types ──────────────────────────────────────────────────────────────────

export interface BrainCoachProps {
  recoveryScore: number | null;
  sleepScore: number | null;
  stressScore: number | null;
  strainScore: number | null;
  avgFocus: number | null;   // 0-1 scale from EEG
  avgValence: number | null; // -1 to 1 from EEG
}

interface Recommendation {
  icon: LucideIcon;
  title: string;
  body: string;
  actionLabel: string;
  actionHref: string;
  /** Tailwind gradient classes for the card background */
  gradientClass: string;
}

// ── Rule engine ────────────────────────────────────────────────────────────

export function buildRecommendations(props: BrainCoachProps): Recommendation[] {
  const { recoveryScore, sleepScore, stressScore, avgFocus, avgValence } = props;

  // No data at all — return empty (caller renders empty state)
  const hasAnyData =
    recoveryScore !== null ||
    sleepScore !== null ||
    stressScore !== null ||
    avgFocus !== null ||
    avgValence !== null;

  if (!hasAnyData) return [];

  const recs: Recommendation[] = [];

  // Rule 1: Both brain fog and poor recovery
  if (avgFocus !== null && recoveryScore !== null && avgFocus < 0.4 && recoveryScore < 50) {
    recs.push({
      icon: Moon,
      title: "Rest day — brain and body both depleted",
      body: "Your brain and body both need rest. Skip complex tasks today.",
      actionLabel: "Log a rest intention",
      actionHref: "/today",
      gradientClass: "from-amber-500/10 to-amber-600/5",
    });
  }
  // Rule 2: Brain fog but physically recovered
  else if (avgFocus !== null && recoveryScore !== null && avgFocus < 0.4 && recoveryScore > 60) {
    recs.push({
      icon: Brain,
      title: "Brain fog despite good recovery",
      body: "Brain fog despite good recovery — try 10min walk, then deep work.",
      actionLabel: "Start a focus session",
      actionHref: "/neurofeedback",
      gradientClass: "from-violet-500/10 to-violet-600/5",
    });
  }

  // Rule 3: Low mood from EEG valence
  if (avgValence !== null && avgValence < -0.1) {
    recs.push({
      icon: Heart,
      title: "Mood is running low",
      body: "Your mood is low. Social interaction or exercise can shift it within 30min.",
      actionLabel: "Try a mood-lift exercise",
      actionHref: "/biofeedback",
      gradientClass: "from-pink-500/10 to-rose-600/5",
    });
  }

  // Rule 4: High stress from health scores
  if (stressScore !== null && stressScore > 65) {
    recs.push({
      icon: Zap,
      title: "Elevated stress — time to regulate",
      body: "Physiological stress is elevated. A 4-4-4-4 breathing cycle activates the parasympathetic nervous system within minutes.",
      actionLabel: "Try box breathing",
      actionHref: "/biofeedback",
      gradientClass: "from-amber-500/10 to-orange-600/5",
    });
  }

  // Rule 5: Peak performance window
  if (
    avgFocus !== null &&
    recoveryScore !== null &&
    avgFocus > 0.65 &&
    recoveryScore > 60
  ) {
    recs.push({
      icon: Zap,
      title: "Peak performance window",
      body: "EEG focus is elevated and recovery is strong. Schedule your hardest cognitive tasks now.",
      actionLabel: "Block focus time",
      actionHref: "/neurofeedback",
      gradientClass: "from-cyan-500/10 to-sky-600/5",
    });
  }

  // Rule 6: Sleep debt
  if (sleepScore !== null && sleepScore < 50) {
    recs.push({
      icon: Moon,
      title: "Sleep debt affecting cognition",
      body: "Sleep debt is affecting cognitive performance. Avoid major decisions, prioritize nap if possible.",
      actionLabel: "View sleep insights",
      actionHref: "/sleep-session",
      gradientClass: "from-indigo-500/10 to-violet-600/5",
    });
  }

  return recs.slice(0, 3);
}

// ── Sub-components ─────────────────────────────────────────────────────────

function RecommendationCard({
  rec,
  index,
}: {
  rec: Recommendation;
  index: number;
}) {
  const Icon = rec.icon;
  return (
    <motion.div
      custom={index}
      variants={cardVariants}
      initial="hidden"
      animate="visible"
      className={`rounded-xl p-4 bg-gradient-to-br ${rec.gradientClass} border border-white/5`}
    >
      <div className="flex items-start gap-3">
        <div className="mt-0.5 shrink-0 rounded-lg p-1.5 bg-white/10">
          <Icon className="w-4 h-4 text-foreground/80" />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold text-foreground leading-snug">{rec.title}</p>
          <p className="text-xs text-muted-foreground mt-1 leading-relaxed">{rec.body}</p>
          <a
            href={rec.actionHref}
            className="inline-block mt-2 text-xs font-medium text-primary underline-offset-2 hover:underline"
          >
            {rec.actionLabel} →
          </a>
        </div>
      </div>
    </motion.div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────

export function BrainCoachCard(props: BrainCoachProps) {
  const recs = buildRecommendations(props);

  return (
    <div data-testid="brain-coach-card" className="glass-card p-5 rounded-2xl space-y-4">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="rounded-xl p-2 bg-gradient-to-br from-violet-500/20 to-purple-600/20 shrink-0">
          <Brain className="w-5 h-5 text-violet-400" />
        </div>
        <div>
          <h2 className="text-sm font-semibold text-foreground">Brain Coach</h2>
          <p className="text-xs text-muted-foreground">EEG + health fusion</p>
        </div>
      </div>

      {/* Recommendations or empty state */}
      {recs.length > 0 ? (
        <div className="space-y-3">
          {recs.map((rec, i) => (
            <RecommendationCard key={rec.title} rec={rec} index={i} />
          ))}
        </div>
      ) : (
        <div
          data-testid="brain-coach-empty"
          className="rounded-xl p-4 bg-violet-500/5 border border-white/5 text-center"
        >
          <Brain className="w-8 h-8 text-violet-400/40 mx-auto mb-2" />
          <p className="text-sm text-muted-foreground">
            Connect EEG or sync health data for brain coaching.
          </p>
        </div>
      )}
    </div>
  );
}

/**
 * Intervention Engine — always pair mood data with actionable suggestions.
 *
 * Research: NO robust evidence mood monitoring alone improves symptoms
 * (JMIR 2026 meta-analysis). 19% of studies report adverse events.
 * BUT mood + mindfulness intervention = significant improvement.
 *
 * Rule: EVERY screen showing mood/emotion data MUST include a
 * "What can I do?" intervention via this engine.
 *
 * @see Issue #524
 */

export type InterventionTier = "breathing" | "reappraisal" | "neurofeedback";

export interface Intervention {
  tier: InterventionTier;
  title: string;
  description: string;
  duration: string;
  route: string;
  icon: string;
  requiresHeadband: boolean;
}

// ── Intervention catalog ────────────────────────────────────────────────────

const BREATHING_BOX: Intervention = {
  tier: "breathing",
  title: "Box breathing (4-4-4-4)",
  description:
    "Inhale for 4 seconds, hold 4, exhale 4, hold 4. Activates your parasympathetic nervous system to reduce stress.",
  duration: "2 min",
  route: "/biofeedback",
  icon: "wind",
  requiresHeadband: false,
};

const BREATHING_CALM: Intervention = {
  tier: "breathing",
  title: "Try a calming breathing exercise",
  description:
    "Slow, deep breaths lower cortisol and bring your heart rate down. Even 2 minutes can make a difference.",
  duration: "2 min",
  route: "/biofeedback",
  icon: "wind",
  requiresHeadband: false,
};

const GROUNDING_54321: Intervention = {
  tier: "reappraisal",
  title: "Grounding exercise (5-4-3-2-1 senses)",
  description:
    "Name 5 things you see, 4 you hear, 3 you touch, 2 you smell, 1 you taste. This anchors you in the present moment.",
  duration: "3 min",
  route: "/ai-companion",
  icon: "hand",
  requiresHeadband: false,
};

const REAPPRAISAL_REFRAME: Intervention = {
  tier: "reappraisal",
  title: "Try reframing this feeling",
  description:
    "Ask yourself: what triggered this? Is there another way to see the situation? Naming the feeling reduces its intensity.",
  duration: "5 min",
  route: "/ai-companion",
  icon: "lightbulb",
  requiresHeadband: false,
};

const PERSPECTIVE_TAKING: Intervention = {
  tier: "reappraisal",
  title: "Perspective-taking exercise",
  description:
    "Consider the situation from another person's viewpoint. This cognitive shift often reduces anger and frustration.",
  duration: "5 min",
  route: "/ai-companion",
  icon: "eye",
  requiresHeadband: false,
};

const PROBABILITY_ESTIMATION: Intervention = {
  tier: "reappraisal",
  title: "Reality-check your worry",
  description:
    "How likely is the worst case, really? Write down the actual probability and what you would do if it happened.",
  duration: "5 min",
  route: "/ai-companion",
  icon: "bar-chart",
  requiresHeadband: false,
};

const GRATITUDE_JOURNAL: Intervention = {
  tier: "reappraisal",
  title: "Gratitude journaling",
  description:
    "Write down three things you are grateful for today. Gratitude practice shifts attention toward positive experiences.",
  duration: "5 min",
  route: "/ai-companion",
  icon: "pen-line",
  requiresHeadband: false,
};

const CALM_MUSIC: Intervention = {
  tier: "reappraisal",
  title: "Listen to calm music",
  description:
    "Music at 60-80 BPM can slow your heart rate and ease sadness. Try nature sounds or ambient music.",
  duration: "5 min",
  route: "/biofeedback",
  icon: "music",
  requiresHeadband: false,
};

const NEUROFEEDBACK_SESSION: Intervention = {
  tier: "neurofeedback",
  title: "Deepen with a neurofeedback session",
  description:
    "Train your brainwaves in real time. Neurofeedback helps strengthen the calm, focused state you are in.",
  duration: "20 min",
  route: "/neurofeedback",
  icon: "brain",
  requiresHeadband: true,
};

const MEDITATION_DEEPEN: Intervention = {
  tier: "breathing",
  title: "Try a guided meditation",
  description:
    "Your mind is calm — a great starting point. A short meditation can deepen this state and build resilience.",
  duration: "5 min",
  route: "/biofeedback",
  icon: "sparkles",
  requiresHeadband: false,
};

const MAINTAIN_STATE: Intervention = {
  tier: "breathing",
  title: "Maintain this balanced state",
  description:
    "You are in a good place right now. A few minutes of mindful breathing can help you stay here longer.",
  duration: "2 min",
  route: "/biofeedback",
  icon: "heart",
  requiresHeadband: false,
};

// ── Engine logic ────────────────────────────────────────────────────────────

/**
 * Returns 2-3 interventions appropriate to the current emotional state.
 *
 * @param emotion - Primary emotion label (e.g. "happy", "sad", "angry", "fear", "anxious", "neutral")
 * @param stress  - Stress index 0-1
 * @param hasHeadband - Whether a Muse headband is connected
 */
export function suggestInterventions(
  emotion: string,
  stress: number,
  hasHeadband: boolean,
): Intervention[] {
  const emo = (emotion || "neutral").toLowerCase();
  const candidates: Intervention[] = [];

  // High stress (>0.6): breathing exercise (tier 1) + reappraisal (tier 2)
  if (stress > 0.6) {
    candidates.push(BREATHING_BOX, REAPPRAISAL_REFRAME);
    if (hasHeadband) {
      candidates.push(NEUROFEEDBACK_SESSION);
    }
    return filterForHeadband(candidates, hasHeadband);
  }

  // Sad / low mood
  if (emo === "sad" || emo === "lonely") {
    candidates.push(GRATITUDE_JOURNAL, CALM_MUSIC);
    if (hasHeadband) {
      candidates.push(NEUROFEEDBACK_SESSION);
    }
    return filterForHeadband(candidates, hasHeadband);
  }

  // Angry
  if (emo === "angry" || emo === "frustrated") {
    candidates.push(BREATHING_BOX, PERSPECTIVE_TAKING);
    if (hasHeadband) {
      candidates.push(NEUROFEEDBACK_SESSION);
    }
    return filterForHeadband(candidates, hasHeadband);
  }

  // Anxious / fearful
  if (emo === "anxious" || emo === "fear" || emo === "fearful" || emo === "nervous" || emo === "overwhelmed") {
    candidates.push(GROUNDING_54321, PROBABILITY_ESTIMATION);
    if (hasHeadband) {
      candidates.push(NEUROFEEDBACK_SESSION);
    }
    return filterForHeadband(candidates, hasHeadband);
  }

  // Neutral / calm / balanced (low stress)
  if (emo === "neutral" || emo === "peaceful" || emo === "calm" || emo === "grateful" || emo === "hopeful") {
    candidates.push(MAINTAIN_STATE, MEDITATION_DEEPEN);
    if (hasHeadband) {
      candidates.push(NEUROFEEDBACK_SESSION);
    }
    return filterForHeadband(candidates, hasHeadband);
  }

  // Happy / excited / surprise — encourage maintaining
  if (emo === "happy" || emo === "excited" || emo === "surprise" || emo === "proud") {
    candidates.push(MAINTAIN_STATE, MEDITATION_DEEPEN);
    if (hasHeadband) {
      candidates.push(NEUROFEEDBACK_SESSION);
    }
    return filterForHeadband(candidates, hasHeadband);
  }

  // Fallback — at least one no-headband option
  candidates.push(BREATHING_CALM);
  if (hasHeadband) {
    candidates.push(NEUROFEEDBACK_SESSION);
  }
  return filterForHeadband(candidates, hasHeadband);
}

/** Remove headband-required interventions when no headband is connected. */
function filterForHeadband(
  interventions: Intervention[],
  hasHeadband: boolean,
): Intervention[] {
  if (hasHeadband) return interventions;
  return interventions.filter((i) => !i.requiresHeadband);
}

// ── Recovery-aware interventions ──────────────────────────────────────────

export interface RecoveryContext {
  recovery?: number;       // 0-100
  sleepHours?: number;     // total hours slept
  strain?: number;         // 0-100
  stress?: number;         // 0-100
  hrvTrend?: "up" | "down" | "stable";
}

export interface RecoveryIntervention {
  title: string;
  description: string;
  icon: string;
  priority: "high" | "medium" | "low";
  action?: { label: string; route: string };
}

/**
 * Returns recovery-specific interventions based on health scores.
 * Actionable suggestions when the body needs attention.
 */
export function suggestRecoveryInterventions(ctx: RecoveryContext): RecoveryIntervention[] {
  const interventions: RecoveryIntervention[] = [];
  const recovery = ctx.recovery ?? 50;
  const sleep = ctx.sleepHours ?? 7;
  const strain = ctx.strain ?? 50;
  const stress = ctx.stress ?? 50;

  // Overtraining: high strain + low recovery
  if (strain > 75 && recovery < 45) {
    interventions.push({
      title: "Active recovery only today",
      description: "Your strain is high and recovery is low. Skip intense workouts — try a walk, yoga, or stretching instead.",
      icon: "heart-pulse",
      priority: "high",
      action: { label: "Log a walk", route: "/workout" },
    });
  }

  // Low recovery
  if (recovery < 40) {
    interventions.push({
      title: "Rest day recommended",
      description: "Your recovery score is low. Prioritize hydration, nutrition, and an earlier bedtime tonight.",
      icon: "battery-low",
      priority: "high",
    });
  }

  // Sleep deficit
  if (sleep < 6) {
    interventions.push({
      title: "Go to bed 30 min earlier tonight",
      description: `You got ${sleep.toFixed(1)}h of sleep. Even 30 extra minutes can significantly improve recovery.`,
      icon: "moon",
      priority: "high",
      action: { label: "Set reminder", route: "/sleep" },
    });
  }

  // HRV declining
  if (ctx.hrvTrend === "down") {
    interventions.push({
      title: "HRV trending down",
      description: "Your heart rate variability has been declining. This can signal accumulated stress or insufficient recovery.",
      icon: "trending-down",
      priority: "medium",
      action: { label: "View HRV", route: "/health" },
    });
  }

  // High stress
  if (stress > 70) {
    interventions.push({
      title: "Try 5-min box breathing",
      description: "Your stress level is elevated. A short breathing exercise can activate your parasympathetic nervous system.",
      icon: "wind",
      priority: "medium",
      action: { label: "Start breathing", route: "/biofeedback" },
    });
  }

  // Low focus (derived from stress — moderate stress = moderate focus)
  if (stress > 50 && recovery < 60) {
    interventions.push({
      title: "Take a 10-min walk outside",
      description: "Fresh air and movement can reset your focus and reduce cortisol levels.",
      icon: "tree-pine",
      priority: "low",
    });
  }

  // Good recovery — encourage maintaining
  if (recovery >= 80 && interventions.length === 0) {
    interventions.push({
      title: "Great recovery — go for it",
      description: "Your body is well-recovered. Today is a good day for a challenging workout or deep work.",
      icon: "zap",
      priority: "low",
    });
  }

  return interventions;
}

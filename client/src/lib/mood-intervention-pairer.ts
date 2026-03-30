/**
 * Mood-Intervention Pairer — never show mood/emotion data without a paired
 * actionable suggestion.
 *
 * Research: tracking without intervention may be harmful (JMIR 2026 meta-analysis).
 * Rule: every mood reading MUST be paired with a concrete action.
 *
 * @see Issue #524
 */

export interface MoodIntervention {
  trigger: string;   // "high_stress" | "low_mood" | "low_focus" | "anxiety" | "anger" | "sadness" | "neutral"
  suggestion: string;
  action: string;
  href: string;
  duration: string;
}

/**
 * Given current mood/emotion state, return the most relevant intervention.
 * ALWAYS returns an intervention — even neutral states get a positive action.
 */
export function pairIntervention(state: {
  stress?: number;   // 0-1
  valence?: number;  // -1 to 1
  focus?: number;    // 0-1
  emotion?: string;  // dominant emotion label
}): MoodIntervention {
  const stress = state.stress ?? 0;
  const valence = state.valence ?? 0;
  const focus = state.focus ?? 0.5;
  const emotion = (state.emotion ?? "").toLowerCase();

  // High stress — breathing is the fastest physiological reset
  if (stress > 0.7) {
    return {
      trigger: "high_stress",
      suggestion: "Try 4-7-8 breathing for 2 minutes",
      action: "Start breathing",
      href: "/biofeedback",
      duration: "2 min",
    };
  }

  // Angry — grounding before anything else
  if (emotion === "angry" || emotion === "frustrated") {
    return {
      trigger: "anger",
      suggestion: "Ground yourself — 3 deep breaths",
      action: "Start breathing",
      href: "/biofeedback",
      duration: "1 min",
    };
  }

  // Fear / anxiety — box breathing
  if (emotion === "fear" || emotion === "anxiety" || emotion === "anxious" || emotion === "fearful") {
    return {
      trigger: "anxiety",
      suggestion: "Box breathing helps in moments like this",
      action: "Start breathing",
      href: "/biofeedback",
      duration: "2 min",
    };
  }

  // Sad — AI companion for processing
  if (emotion === "sad" || emotion === "lonely") {
    return {
      trigger: "sadness",
      suggestion: "Talk it through — your AI companion is here",
      action: "Open companion",
      href: "/ai-companion",
      duration: "5 min",
    };
  }

  // Low mood (negative valence, but not a specific sad/angry emotion)
  if (valence < -0.3) {
    return {
      trigger: "low_mood",
      suggestion: "Check in with AI companion",
      action: "Open companion",
      href: "/ai-companion",
      duration: "5 min",
    };
  }

  // Low focus — neurofeedback session
  if (focus < 0.3) {
    return {
      trigger: "low_focus",
      suggestion: "Try a 5-min focus session",
      action: "Start session",
      href: "/neurofeedback",
      duration: "5 min",
    };
  }

  // Default: neutral or positive — encourage maintaining the state
  return {
    trigger: "neutral",
    suggestion: "You're doing well — try a gratitude check-in",
    action: "Open companion",
    href: "/ai-companion",
    duration: "3 min",
  };
}

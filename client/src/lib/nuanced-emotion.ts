/**
 * Nuanced emotion vocabulary — maps 6-class probability distributions
 * and fused valence/arousal coordinates to 12 human-meaningful compound
 * emotion labels.
 *
 * The base models output coarse labels (happy, sad, angry, neutral, calm,
 * fear, surprise, focused, relaxed). But probability distributions encode
 * richer information:
 *
 *   P(happy)=0.4 + P(calm)=0.3  --> "content"
 *   P(angry)=0.3 + P(sad)=0.3   --> "frustrated"
 *   P(happy)=0.5 + arousal>0.7   --> "excited"
 *
 * This module defines compound emotion rules and a mapper function that
 * takes the fused result (probabilities, valence, arousal, stress) and
 * returns a nuanced label with metadata.
 *
 * The 12 nuanced emotions follow Russell's circumplex model quadrants:
 *
 *   High arousal + positive valence: excited, energized
 *   Low arousal  + positive valence: content, serene
 *   High arousal + negative valence: anxious, overwhelmed, frustrated
 *   Low arousal  + negative valence: melancholy, drained
 *   Moderate / ambiguous:            focused, contemplative, neutral
 */

// ── Types ──────────────────────────────────────────────────────────────────

export interface NuancedEmotionResult {
  /** The nuanced label (e.g. "content", "anxious", "energized"). */
  label: string;
  /** The base emotion it derives from (for backward compat). */
  baseEmotion: string;
  /** Display-friendly label with first letter capitalized. */
  displayLabel: string;
  /** Confidence that this nuanced label is appropriate (0-1). */
  confidence: number;
  /** Which quadrant of Russell's circumplex this falls in. */
  quadrant: "high-pos" | "low-pos" | "high-neg" | "low-neg" | "center";
}

export interface EmotionProbabilities {
  [emotion: string]: number;
}

export interface NuancedInput {
  /** Base emotion label from the model. */
  emotion: string;
  /** Probability distribution over base classes (if available). */
  probabilities?: EmotionProbabilities;
  /** Fused valence (-1 to 1). */
  valence: number;
  /** Fused arousal (0 to 1). */
  arousal: number;
  /** Stress index (0 to 1). */
  stress: number;
  /** Confidence of the base prediction (0 to 1). */
  confidence: number;
}

// ── Compound Emotion Rules ────────────────────────────────────────────────
//
// Each rule defines:
//   - label: the nuanced emotion name
//   - match: a function that scores how well the current state matches (0-1)
//   - quadrant: Russell's circumplex quadrant
//   - baseEmotion: what base emotion this maps to for backward compat
//
// Rules are evaluated in order. The highest-scoring match wins.
// A minimum score of 0.25 is required; below that, fall through to base.

interface CompoundRule {
  label: string;
  baseEmotion: string;
  quadrant: NuancedEmotionResult["quadrant"];
  match: (input: NuancedInput) => number;
}

function prob(probs: EmotionProbabilities | undefined, key: string): number {
  if (!probs) return 0;
  return probs[key] ?? 0;
}

const COMPOUND_RULES: CompoundRule[] = [
  // ── High arousal + positive valence ────────────────────────────────────

  {
    label: "excited",
    baseEmotion: "happy",
    quadrant: "high-pos",
    match: (input) => {
      const happy = prob(input.probabilities, "happy");
      const surprise = prob(input.probabilities, "surprise");
      // High arousal + positive + high happy/surprise probability
      if (input.valence < 0.1 || input.arousal < 0.6) return 0;
      // Give strong weight to happy probability -- this distinguishes
      // excited (happy-driven) from energized (non-specific energy).
      const score = 0.40 * happy + 0.15 * surprise + 0.25 * input.arousal + 0.20 * input.valence;
      return score;
    },
  },
  {
    label: "energized",
    baseEmotion: "happy",
    quadrant: "high-pos",
    match: (input) => {
      // Moderate-high arousal + positive valence + low stress.
      // This is the "generic positive energy" state when happy prob isn't dominant.
      // When happy prob is high (>0.4), "excited" is the better label.
      if (input.valence < 0.1 || input.arousal < 0.5) return 0;
      const happy = prob(input.probabilities, "happy");
      if (happy > 0.4 && input.arousal > 0.65) return 0; // defer to "excited"
      const lowStress = 1 - input.stress;
      return 0.30 * input.arousal + 0.30 * input.valence + 0.30 * lowStress + 0.10 * (1 - happy);
    },
  },

  // ── Low arousal + positive valence ─────────────────────────────────────

  {
    label: "content",
    baseEmotion: "happy",
    quadrant: "low-pos",
    match: (input) => {
      const happy = prob(input.probabilities, "happy");
      const calm = prob(input.probabilities, "calm") + prob(input.probabilities, "relaxed");
      // Moderate happy + calm/relaxed + low arousal.
      // Requires SOME happy probability -- pure calm goes to serene.
      if (input.valence < 0.05 || input.arousal > 0.55) return 0;
      if (happy < 0.15) return 0; // not enough happiness for "content"
      return 0.35 * happy + 0.25 * calm + 0.20 * (1 - input.arousal) + 0.20 * input.valence;
    },
  },
  {
    label: "serene",
    baseEmotion: "calm",
    quadrant: "low-pos",
    match: (input) => {
      const calm = prob(input.probabilities, "calm") + prob(input.probabilities, "relaxed");
      // Very calm + positive + very low arousal + very low stress.
      // Requires calm to be the dominant signal (>0.35 combined).
      if (input.arousal > 0.35 || input.valence < 0) return 0;
      if (calm < 0.35) return 0; // not calm-dominant enough for serene
      const lowStress = 1 - input.stress;
      return 0.35 * calm + 0.25 * (1 - input.arousal) + 0.20 * lowStress + 0.20 * input.valence;
    },
  },

  // ── High arousal + negative valence ────────────────────────────────────

  {
    label: "anxious",
    baseEmotion: "fear",
    quadrant: "high-neg",
    match: (input) => {
      const fear = prob(input.probabilities, "fear") + prob(input.probabilities, "fearful");
      // Fear + high arousal + negative valence + high stress
      if (input.valence > 0 || input.arousal < 0.45) return 0;
      return 0.30 * fear + 0.25 * input.arousal + 0.25 * input.stress + 0.20 * Math.abs(input.valence);
    },
  },
  {
    label: "overwhelmed",
    baseEmotion: "angry",
    quadrant: "high-neg",
    match: (input) => {
      const angry = prob(input.probabilities, "angry");
      const fear = prob(input.probabilities, "fear") + prob(input.probabilities, "fearful");
      // Mixed angry + fear + very high stress + high arousal.
      // Overwhelmed requires BOTH fear and high stress -- it's the
      // "too much at once" state, not just anger or frustration.
      if (input.stress < 0.65 || input.arousal < 0.6) return 0;
      if (fear < 0.15) return 0; // needs a fear component, not just anger
      return 0.20 * angry + 0.25 * fear + 0.25 * input.stress + 0.30 * input.arousal;
    },
  },
  {
    label: "frustrated",
    baseEmotion: "angry",
    quadrant: "high-neg",
    match: (input) => {
      const angry = prob(input.probabilities, "angry");
      const sad = prob(input.probabilities, "sad");
      // Angry + sad mix = frustration (not getting what you want).
      // The angry+sad combination is the key differentiator.
      if (input.valence > 0.1) return 0;
      const angrySadBlend = angry + sad; // combined weight of both
      return 0.35 * angrySadBlend + 0.25 * input.arousal + 0.20 * Math.abs(input.valence) + 0.20 * input.stress;
    },
  },

  // ── Low arousal + negative valence ─────────────────────────────────────

  {
    label: "melancholy",
    baseEmotion: "sad",
    quadrant: "low-neg",
    match: (input) => {
      const sad = prob(input.probabilities, "sad");
      // Sad + low arousal + negative valence + moderate (not high) stress
      if (input.valence > -0.05 || input.arousal > 0.45) return 0;
      return 0.35 * sad + 0.25 * (1 - input.arousal) + 0.25 * Math.abs(input.valence) + 0.15 * (1 - input.stress);
    },
  },
  {
    label: "drained",
    baseEmotion: "sad",
    quadrant: "low-neg",
    match: (input) => {
      const sad = prob(input.probabilities, "sad");
      const neutral = prob(input.probabilities, "neutral");
      // Sad/neutral + very low arousal + high stress (emotional exhaustion)
      if (input.arousal > 0.3) return 0;
      return 0.25 * sad + 0.15 * neutral + 0.30 * (1 - input.arousal) + 0.30 * input.stress;
    },
  },

  // ── Center / ambiguous ─────────────────────────────────────────────────

  {
    label: "focused",
    baseEmotion: "focused",
    quadrant: "center",
    match: (input) => {
      const neutral = prob(input.probabilities, "neutral");
      const focused = prob(input.probabilities, "focused");
      // Moderate arousal + near-zero valence + low stress
      if (Math.abs(input.valence) > 0.3 || input.arousal < 0.35 || input.arousal > 0.75) return 0;
      const lowStress = 1 - input.stress;
      return 0.25 * (neutral + focused) + 0.30 * (1 - Math.abs(input.valence) * 2) + 0.25 * lowStress + 0.20 * input.arousal;
    },
  },
  {
    label: "contemplative",
    baseEmotion: "neutral",
    quadrant: "center",
    match: (input) => {
      const neutral = prob(input.probabilities, "neutral");
      const calm = prob(input.probabilities, "calm") + prob(input.probabilities, "relaxed");
      // Neutral/calm + low-moderate arousal + near-zero valence
      if (Math.abs(input.valence) > 0.25 || input.arousal > 0.5) return 0;
      return 0.30 * neutral + 0.20 * calm + 0.25 * (1 - Math.abs(input.valence) * 3) + 0.25 * (1 - input.arousal);
    },
  },
];

// Minimum match score to use a compound label instead of the base emotion.
const MIN_MATCH_SCORE = 0.25;

// ── Core Mapper ───────────────────────────────────────────────────────────

/**
 * Map a fused emotion result to a nuanced compound emotion label.
 *
 * Evaluates all compound rules, picks the highest-scoring match.
 * Falls through to the base emotion if no rule scores above MIN_MATCH_SCORE.
 *
 * @param input  Fused emotion data (emotion, probabilities, valence, arousal, stress, confidence)
 * @returns      Nuanced emotion result with label, quadrant, and confidence
 */
export function mapToNuancedEmotion(input: NuancedInput): NuancedEmotionResult {
  let bestLabel = input.emotion;
  let bestScore = 0;
  let bestRule: CompoundRule | null = null;

  for (const rule of COMPOUND_RULES) {
    const score = rule.match(input);
    if (score > bestScore) {
      bestScore = score;
      bestLabel = rule.label;
      bestRule = rule;
    }
  }

  // If no compound rule scored high enough, use the base emotion directly
  if (bestScore < MIN_MATCH_SCORE || !bestRule) {
    return {
      label: input.emotion,
      baseEmotion: input.emotion,
      displayLabel: capitalize(input.emotion),
      confidence: input.confidence,
      quadrant: inferQuadrant(input.valence, input.arousal),
    };
  }

  // Confidence for the nuanced label: blend the match score with the base confidence
  // A high match score + high base confidence = high nuanced confidence
  const nuancedConfidence = Math.min(1, input.confidence * (0.5 + 0.5 * bestScore));

  return {
    label: bestLabel,
    baseEmotion: bestRule.baseEmotion,
    displayLabel: capitalize(bestLabel),
    confidence: nuancedConfidence,
    quadrant: bestRule.quadrant,
  };
}

// ── Emoji + Color Maps for Nuanced Emotions ──────────────────────────────

/**
 * Emoji for each nuanced emotion. Falls back to base emotion emoji
 * or brain icon for unknown labels.
 */
export const NUANCED_EMOJI: Record<string, string> = {
  // Nuanced labels
  excited: "\u{1F929}",       // star-struck
  energized: "\u{26A1}",      // lightning
  content: "\u{1F60C}",       // relieved / content face
  serene: "\u{1F338}",        // cherry blossom
  anxious: "\u{1F616}",       // confounded
  overwhelmed: "\u{1F635}",   // dizzy face
  frustrated: "\u{1F624}",    // huffing face
  melancholy: "\u{1F33F}",    // wilting/herb
  drained: "\u{1F6CF}\uFE0F", // bed
  focused: "\u{1F3AF}",       // target
  contemplative: "\u{1F4AD}", // thought bubble
  // Base fallbacks
  happy: "\u{1F60A}",
  sad: "\u{1F614}",
  angry: "\u{1F620}",
  fear: "\u{1F630}",
  fearful: "\u{1F630}",
  surprise: "\u{1F62E}",
  neutral: "\u{1F610}",
  calm: "\u{1F54A}\uFE0F",    // dove
  relaxed: "\u{1F60C}",
};

/**
 * Tailwind CSS color classes for each nuanced emotion.
 * Designed for dark mode (bg-X/15 + text-X + border-X/30).
 */
export const NUANCED_COLORS: Record<string, string> = {
  // Nuanced labels
  excited: "bg-amber-500/15 text-amber-400 border-amber-500/30",
  energized: "bg-orange-500/15 text-orange-400 border-orange-500/30",
  content: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
  serene: "bg-teal-500/15 text-teal-400 border-teal-500/30",
  anxious: "bg-violet-500/15 text-violet-400 border-violet-500/30",
  overwhelmed: "bg-rose-500/15 text-rose-400 border-rose-500/30",
  frustrated: "bg-red-500/15 text-red-400 border-red-500/30",
  melancholy: "bg-indigo-500/15 text-indigo-400 border-indigo-500/30",
  drained: "bg-slate-500/15 text-slate-400 border-slate-500/30",
  focused: "bg-blue-500/15 text-blue-400 border-blue-500/30",
  contemplative: "bg-gray-500/15 text-gray-400 border-gray-500/30",
  // Base fallbacks
  happy: "bg-cyan-500/15 text-cyan-500 border-cyan-500/30",
  sad: "bg-indigo-500/15 text-indigo-400 border-indigo-500/30",
  angry: "bg-rose-500/15 text-rose-400 border-rose-500/30",
  fear: "bg-purple-500/15 text-purple-500 border-purple-500/30",
  fearful: "bg-purple-500/15 text-purple-500 border-purple-500/30",
  surprise: "bg-amber-500/15 text-amber-500 border-amber-500/30",
  neutral: "bg-muted text-muted-foreground border-border",
  calm: "bg-teal-500/15 text-teal-400 border-teal-500/30",
  relaxed: "bg-teal-500/15 text-teal-400 border-teal-500/30",
};

/**
 * Extended valence map that includes nuanced labels.
 * Used by the feedback system to score how close a modality prediction
 * was to the user's correction.
 */
export const NUANCED_VALENCE_MAP: Record<string, number> = {
  // Nuanced
  excited: 0.65,
  energized: 0.55,
  content: 0.45,
  serene: 0.35,
  anxious: -0.45,
  overwhelmed: -0.55,
  frustrated: -0.50,
  melancholy: -0.55,
  drained: -0.40,
  focused: 0.15,
  contemplative: 0.05,
  // Base (same as existing EMOTION_VALENCE_MAP)
  happy: 0.7,
  calm: 0.4,
  relaxed: 0.3,
  neutral: 0.0,
  sad: -0.6,
  angry: -0.5,
  fear: -0.7,
  fearful: -0.7,
  surprise: 0.1,
};

// ── Helpers ───────────────────────────────────────────────────────────────

function capitalize(s: string): string {
  if (!s) return s;
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function inferQuadrant(valence: number, arousal: number): NuancedEmotionResult["quadrant"] {
  if (Math.abs(valence) < 0.15 && arousal > 0.3 && arousal < 0.7) return "center";
  if (valence >= 0 && arousal >= 0.5) return "high-pos";
  if (valence >= 0 && arousal < 0.5) return "low-pos";
  if (valence < 0 && arousal >= 0.5) return "high-neg";
  return "low-neg";
}

/**
 * Inner Score — adaptive composite wellness score (0-100).
 *
 * Three tiers based on available data:
 *   Tier 1 (voice):           stress + valence + energy
 *   Tier 2 (health + voice):  sleep + HRV + stress + valence + activity
 *   Tier 3 (EEG + all):       sleep + brain health + HRV + stress + valence + activity
 *
 * Auto-detects the highest available tier and redistributes weights
 * when optional factors are missing.
 */

// ─── Types ───────────────────────────────────────────────────────────────────

export type Tier = "voice" | "health_voice" | "eeg_health_voice";

export interface ScoreInputs {
  stress?: number | null;
  valence?: number | null;
  arousal?: number | null;
  moodEnergy?: number | null;
  moodScale?: number | null;
  sleepQuality?: number | null;
  hrvTrend?: number | null;
  activity?: number | null;
  brainHealth?: number | null;
}

export interface ScoreResult {
  score: number | null;
  tier: Tier;
  factors: Record<string, number>;
  narrative: string;
  label: string;
  color: string;
}

// ─── Normalization ───────────────────────────────────────────────────────────

/** Invert stress index (0-1) to 0-100 where 100 = no stress */
export function normalizeStress(stress: number): number {
  return Math.round((1 - Math.max(0, Math.min(1, stress))) * 100);
}

/** Map valence (-1 to +1) to 0-100 where 100 = most positive */
export function normalizeValence(valence: number): number {
  return Math.round(((Math.max(-1, Math.min(1, valence)) + 1) / 2) * 100);
}

/** Normalize energy from arousal (0-1) or mood log. Arousal preferred. */
export function normalizeEnergy(input: {
  arousal?: number;
  moodEnergy?: number;
  moodScale?: number;
}): number {
  if (input.arousal != null) {
    return Math.round(Math.max(0, Math.min(1, input.arousal)) * 100);
  }
  if (input.moodEnergy != null && input.moodScale != null && input.moodScale > 0) {
    return Math.round((input.moodEnergy / input.moodScale) * 100);
  }
  return 50; // neutral default
}

// ─── Tier Detection ──────────────────────────────────────────────────────────

/** Detect the highest available tier from input data */
export function detectTier(inputs: ScoreInputs): Tier | null {
  const hasStressOrValence = inputs.stress != null || inputs.valence != null;
  const hasSleep = inputs.sleepQuality != null;
  const hasBrain = inputs.brainHealth != null;

  if (hasBrain && hasSleep) return "eeg_health_voice";
  if (hasSleep && hasStressOrValence) return "health_voice";
  if (hasStressOrValence) return "voice";
  return null;
}

// ─── Weights ─────────────────────────────────────────────────────────────────

const TIER_WEIGHTS: Record<Tier, Record<string, number>> = {
  voice: { stress_inverse: 0.4, valence: 0.4, energy: 0.2 },
  health_voice: {
    sleep_quality: 0.35,
    hrv_trend: 0.2,
    stress_inverse: 0.2,
    valence: 0.15,
    activity: 0.1,
  },
  eeg_health_voice: {
    sleep_quality: 0.3,
    brain_health: 0.25,
    hrv_trend: 0.15,
    stress_inverse: 0.15,
    valence: 0.1,
    activity: 0.05,
  },
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

export function getScoreLabel(score: number): string {
  if (score >= 80) return "Thriving";
  if (score >= 60) return "Good";
  if (score >= 40) return "Steady";
  return "Low";
}

export function getScoreColor(score: number): string {
  if (score >= 80) return "var(--success)";
  if (score >= 60) return "var(--primary)";
  if (score >= 40) return "var(--warning)";
  return "var(--destructive)";
}

function redistributeWeights(
  weights: Record<string, number>,
  presentKeys: string[],
): Record<string, number> {
  const present = Object.entries(weights).filter(([k]) => presentKeys.includes(k));
  const totalPresent = present.reduce((s, [, w]) => s + w, 0);
  if (totalPresent <= 0) return {};
  const result: Record<string, number> = {};
  for (const [k, w] of present) {
    result[k] = w / totalPresent;
  }
  return result;
}

// ─── Score Computation ───────────────────────────────────────────────────────

export function computeScore(inputs: ScoreInputs): ScoreResult {
  const tier = detectTier(inputs);
  if (!tier) {
    return {
      score: null,
      tier: "voice",
      factors: {},
      narrative: "",
      label: "Building",
      color: "var(--muted)",
    };
  }

  // Extract normalized factor values
  const allFactors: Record<string, number> = {};
  if (inputs.stress != null)
    allFactors.stress_inverse = normalizeStress(inputs.stress);
  if (inputs.valence != null)
    allFactors.valence = normalizeValence(inputs.valence);
  allFactors.energy = normalizeEnergy({
    arousal: inputs.arousal ?? undefined,
    moodEnergy: inputs.moodEnergy ?? undefined,
    moodScale: inputs.moodScale ?? undefined,
  });
  if (inputs.sleepQuality != null)
    allFactors.sleep_quality = Math.round(
      Math.max(0, Math.min(100, inputs.sleepQuality)),
    );
  if (inputs.hrvTrend != null)
    allFactors.hrv_trend = Math.round(
      Math.max(0, Math.min(100, inputs.hrvTrend)),
    );
  if (inputs.activity != null)
    allFactors.activity = Math.round(
      Math.max(0, Math.min(100, inputs.activity)),
    );
  if (inputs.brainHealth != null)
    allFactors.brain_health = Math.round(
      Math.max(0, Math.min(100, inputs.brainHealth)),
    );

  // Get tier weights and redistribute for missing optional factors
  const baseWeights = TIER_WEIGHTS[tier];
  const presentKeys = Object.keys(allFactors).filter((k) => k in baseWeights);
  const weights = redistributeWeights(baseWeights, presentKeys);

  // Weighted sum
  let score = 0;
  const usedFactors: Record<string, number> = {};
  for (const [key, weight] of Object.entries(weights)) {
    const val = allFactors[key] ?? 50;
    score += val * weight;
    usedFactors[key] = allFactors[key] ?? 50;
  }
  score = Math.round(Math.max(0, Math.min(100, score)));

  return {
    score,
    tier,
    factors: usedFactors,
    narrative: "",
    label: getScoreLabel(score),
    color: getScoreColor(score),
  };
}

// ─── Narrative ───────────────────────────────────────────────────────────────

const FACTOR_LABELS: Record<string, string> = {
  sleep_quality: "sleep",
  stress_inverse: "stress levels",
  valence: "mood",
  energy: "energy",
  hrv_trend: "heart rate variability",
  activity: "activity",
  brain_health: "brain health",
};

export function computeNarrative(
  factors: Record<string, number>,
  score: number,
  delta: number | null,
): string {
  const entries = Object.entries(factors);
  if (entries.length === 0) return "";

  if (delta != null && delta > 10) return "Strong improvement from yesterday.";
  if (delta != null && delta < -10)
    return "Dip from yesterday \u2014 check what changed.";

  const sorted = [...entries].sort((a, b) => b[1] - a[1]);
  const highest = sorted[0];
  const lowest = sorted[sorted.length - 1];

  if (highest[1] - lowest[1] <= 10) {
    return "You're well-balanced across the board today.";
  }

  const highLabel = FACTOR_LABELS[highest[0]] ?? highest[0];
  const lowLabel = FACTOR_LABELS[lowest[0]] ?? lowest[0];
  return `Good ${highLabel} is carrying you today, but ${lowLabel} could use attention.`;
}

// ─── Tier Confidence Labels ──────────────────────────────────────────────────

export function getTierConfidence(tier: Tier): string {
  switch (tier) {
    case "eeg_health_voice":
      return "Based on your brain, body, and mood";
    case "health_voice":
      return "Based on your sleep, body, and mood";
    case "voice":
      return "Based on how you sound today";
  }
}

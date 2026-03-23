/**
 * age-stratification.ts — Age-stratified emotion baseline adjustments.
 *
 * Different age groups have measurably different emotional baselines:
 *   - Children (6-12): Higher baseline arousal, more volatile emotions
 *   - Teens (13-17): Elevated arousal, higher stress reactivity
 *   - Adults (18-64): Reference baseline (no adjustment)
 *   - Seniors (65+): Lower baseline arousal, more stable emotions
 *
 * Research basis:
 *   - Carstensen et al. (1999): Socioemotional selectivity theory —
 *     emotional regulation improves with age
 *   - Larson et al. (2002): Adolescents show greater emotional variability
 *   - Tottenham et al. (2009): Children's amygdala response is heightened
 *
 * These adjustments are applied in the data-fusion layer to normalize
 * raw emotion readings relative to age-appropriate baselines.
 */

// ── Types ──────────────────────────────────────────────────────────────────

export type AgeGroup = "child" | "teen" | "adult" | "senior";

export interface AgeGroupInfo {
  id: AgeGroup;
  label: string;
  ageRange: string;
  description: string;
}

export interface AgeAdjustments {
  /** Multiplier applied to raw arousal (>1 = lower the effective arousal). */
  arousalMultiplier: number;
  /** Multiplier applied to stress readings. */
  stressMultiplier: number;
  /** Multiplier applied to emotional volatility (variance dampening). */
  volatilityDampening: number;
  /** Offset added to focus baseline (positive = higher expected focus). */
  focusBaselineOffset: number;
}

// ── Constants ──────────────────────────────────────────────────────────────

export const AGE_GROUPS: AgeGroupInfo[] = [
  {
    id: "child",
    label: "Child",
    ageRange: "6-12",
    description: "Higher baseline arousal, more volatile emotions",
  },
  {
    id: "teen",
    label: "Teen",
    ageRange: "13-17",
    description: "Elevated arousal, higher stress reactivity",
  },
  {
    id: "adult",
    label: "Adult",
    ageRange: "18-64",
    description: "Reference baseline — no adjustment applied",
  },
  {
    id: "senior",
    label: "Senior",
    ageRange: "65+",
    description: "Lower baseline arousal, more emotionally stable",
  },
];

/**
 * Per-group adjustment parameters.
 *
 * Adults are the reference group (all multipliers = 1.0, offsets = 0).
 * Other groups are adjusted relative to adults.
 */
const GROUP_ADJUSTMENTS: Record<AgeGroup, AgeAdjustments> = {
  child: {
    // Children have naturally higher arousal — scale down raw readings
    arousalMultiplier: 0.80,
    // Children's stress baseline is higher; normalize down
    stressMultiplier: 0.85,
    // Higher emotional volatility is normal — dampen to avoid false alarms
    volatilityDampening: 0.70,
    // Children have shorter attention spans; lower expected focus
    focusBaselineOffset: -0.10,
  },
  teen: {
    // Teens have elevated but not extreme arousal
    arousalMultiplier: 0.90,
    // Teens are more stress-reactive
    stressMultiplier: 0.90,
    // Moderate volatility dampening
    volatilityDampening: 0.85,
    // Slightly lower focus expectations
    focusBaselineOffset: -0.05,
  },
  adult: {
    // Reference baseline — no adjustment
    arousalMultiplier: 1.0,
    stressMultiplier: 1.0,
    volatilityDampening: 1.0,
    focusBaselineOffset: 0.0,
  },
  senior: {
    // Seniors have lower baseline arousal — scale up raw readings
    arousalMultiplier: 1.15,
    // Seniors generally report lower stress
    stressMultiplier: 1.10,
    // Emotions are more stable — less dampening needed
    volatilityDampening: 1.10,
    // Generally higher sustained attention
    focusBaselineOffset: 0.05,
  },
};

// ── localStorage persistence ───────────────────────────────────────────────

const AGE_KEY = "ndw_user_age";

/**
 * Get the user's stored age, or null if not set.
 */
export function getUserAge(): number | null {
  try {
    const raw = localStorage.getItem(AGE_KEY);
    if (!raw) return null;
    const age = parseInt(raw, 10);
    return isNaN(age) || age < 1 || age > 120 ? null : age;
  } catch {
    return null;
  }
}

/**
 * Store the user's age.
 */
export function setUserAge(age: number): void {
  try {
    localStorage.setItem(AGE_KEY, String(Math.round(age)));
  } catch {
    // localStorage unavailable
  }
}

// ── Classification ─────────────────────────────────────────────────────────

/**
 * Classify an age into an age group.
 */
export function classifyAgeGroup(age: number): AgeGroup {
  if (age >= 6 && age <= 12) return "child";
  if (age >= 13 && age <= 17) return "teen";
  if (age >= 18 && age <= 64) return "adult";
  if (age >= 65) return "senior";
  // Below 6 or invalid — treat as child for safety
  return "child";
}

/**
 * Get the AgeGroupInfo for an age.
 */
export function getAgeGroupInfo(age: number): AgeGroupInfo {
  const group = classifyAgeGroup(age);
  return AGE_GROUPS.find((g) => g.id === group) ?? AGE_GROUPS[2]; // default: adult
}

/**
 * Get adjustment parameters for a given age.
 * Returns adult (no-op) adjustments if age is null.
 */
export function getAgeAdjustments(age: number | null): AgeAdjustments {
  if (age === null) return GROUP_ADJUSTMENTS.adult;
  const group = classifyAgeGroup(age);
  return GROUP_ADJUSTMENTS[group];
}

// ── Application ────────────────────────────────────────────────────────────

function clip(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/**
 * Apply age-based adjustments to raw emotion readings.
 *
 * Call this in the data-fusion layer before presenting values to the UI.
 *
 * @param raw Raw emotion readings (stress, focus, arousal are 0-1)
 * @param age User's age, or null to skip adjustments
 * @returns Adjusted readings
 */
export function applyAgeAdjustments(
  raw: {
    stress: number;
    focus: number;
    arousal: number;
  },
  age: number | null,
): { stress: number; focus: number; arousal: number } {
  const adj = getAgeAdjustments(age);

  return {
    stress: clip(raw.stress * adj.stressMultiplier, 0, 1),
    focus: clip(raw.focus + adj.focusBaselineOffset, 0, 1),
    arousal: clip(raw.arousal * adj.arousalMultiplier, 0, 1),
  };
}

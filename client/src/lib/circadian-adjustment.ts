/**
 * circadian-adjustment.ts — Circadian baseline normalization for emotion/stress/focus readings.
 *
 * Enhances the existing chronotype module with physiological time-of-day adjustments:
 *   - Morning cortisol surge (6-9 AM): stress baseline is naturally higher
 *   - Post-lunch dip (1-3 PM): focus baseline naturally lower
 *   - Evening wind-down (8-11 PM): alpha naturally increases
 *
 * These adjustments are applied in the data fusion layer so that readings
 * are compared against the expected baseline for that time of day, not a
 * flat 24-hour average.
 */

// ── Types ──────────────────────────────────────────────────────────────────

export interface CircadianAdjustment {
  stressBaselineOffset: number;   // positive = stress is naturally higher at this time
  focusBaselineOffset: number;    // negative = focus is naturally lower at this time
  alphaBaselineOffset: number;    // positive = alpha (relaxation) is naturally higher
  label: string;                  // human-readable description
}

export interface RawReadings {
  stress: number;       // 0-1
  focus: number;        // 0-1
  relaxation: number;   // 0-1
  hour: number;         // 0-23
}

export interface NormalizedReadings {
  stress: number;
  focus: number;
  relaxation: number;
}

// ── Circadian adjustment lookup ────────────────────────────────────────────

/**
 * Get the circadian baseline adjustment for a given hour (0-23).
 *
 * Offsets describe how much HIGHER or LOWER the baseline naturally is:
 *   - stressBaselineOffset > 0 means stress is naturally elevated (cortisol)
 *   - focusBaselineOffset < 0 means focus is naturally lower (post-lunch)
 *   - alphaBaselineOffset > 0 means alpha/relaxation naturally higher (evening)
 */
export function getCircadianAdjustment(hour: number): CircadianAdjustment {
  const h = ((hour % 24) + 24) % 24;

  // Sleep zone: 0-5 AM
  if (h >= 0 && h < 6) {
    return {
      stressBaselineOffset: -0.05,
      focusBaselineOffset: -0.15,
      alphaBaselineOffset: 0.1,
      label: "Deep sleep zone — low cortisol, low alertness",
    };
  }

  // Morning cortisol surge: 6-9 AM
  if (h >= 6 && h < 9) {
    return {
      stressBaselineOffset: 0.15,
      focusBaselineOffset: 0.05,
      alphaBaselineOffset: -0.05,
      label: "Morning cortisol surge — stress naturally elevated",
    };
  }

  // Late morning: 9 AM - 12 PM — optimal zone, neutral
  if (h >= 9 && h < 12) {
    return {
      stressBaselineOffset: 0,
      focusBaselineOffset: 0,
      alphaBaselineOffset: 0,
      label: "Late morning — optimal cognitive window",
    };
  }

  // Early afternoon: 12-1 PM — neutral-ish
  if (h >= 12 && h < 13) {
    return {
      stressBaselineOffset: 0,
      focusBaselineOffset: 0,
      alphaBaselineOffset: 0,
      label: "Early afternoon — stable baseline",
    };
  }

  // Post-lunch dip: 1-3 PM
  if (h >= 13 && h < 15) {
    return {
      stressBaselineOffset: -0.05,
      focusBaselineOffset: -0.12,
      alphaBaselineOffset: 0.08,
      label: "post-lunch dip — focus naturally lower, drowsiness normal",
    };
  }

  // Mid-afternoon: 3-5 PM — recovery, neutral
  if (h >= 15 && h < 17) {
    return {
      stressBaselineOffset: 0,
      focusBaselineOffset: 0,
      alphaBaselineOffset: 0,
      label: "Mid-afternoon — second wind period",
    };
  }

  // Early evening: 5-8 PM — neutral
  if (h >= 17 && h < 20) {
    return {
      stressBaselineOffset: 0,
      focusBaselineOffset: 0,
      alphaBaselineOffset: 0,
      label: "Early evening — stable baseline",
    };
  }

  // Evening wind-down: 8-11 PM
  if (h >= 20 && h < 23) {
    return {
      stressBaselineOffset: -0.08,
      focusBaselineOffset: -0.08,
      alphaBaselineOffset: 0.12,
      label: "Evening wind-down — alpha naturally increases, melatonin rising",
    };
  }

  // Late night: 11 PM - midnight
  return {
    stressBaselineOffset: -0.05,
    focusBaselineOffset: -0.12,
    alphaBaselineOffset: 0.1,
    label: "Late night sleep zone — low cortisol, high alpha",
  };
}

// ── Normalization ──────────────────────────────────────────────────────────

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

/**
 * Apply circadian normalization to raw readings.
 *
 * The idea: if stress is naturally higher at 7 AM (cortisol surge),
 * then a stress reading of 0.7 at 7 AM is less alarming than 0.7 at 11 AM.
 * We subtract the natural baseline offset to get a "circadian-adjusted" value.
 */
export function applyCircadianNormalization(raw: RawReadings): NormalizedReadings {
  const adj = getCircadianAdjustment(raw.hour);

  return {
    // If stress baseline is naturally higher (positive offset), subtract it
    // to show the "above-normal" stress level
    stress: clamp(raw.stress - adj.stressBaselineOffset, 0, 1),
    // If focus baseline is naturally lower (negative offset), subtract it
    // (subtracting a negative = adding) to show that low focus is expected
    focus: clamp(raw.focus - adj.focusBaselineOffset, 0, 1),
    // If alpha/relaxation is naturally higher, subtract the offset
    relaxation: clamp(raw.relaxation - adj.alphaBaselineOffset, 0, 1),
  };
}

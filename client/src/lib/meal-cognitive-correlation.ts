/**
 * meal-cognitive-correlation.ts — Correlate meal timing with cognitive state.
 *
 * Reads food logs (timestamps) and emotion/focus readings (timestamps)
 * from Supabase/localStorage. Computes correlation: focus/stress levels
 * 1-2 hours after meals vs baseline.
 *
 * Output examples:
 *   "Your focus tends to dip 1.5 hours after lunch"
 *   "Post-breakfast focus boost detected"
 */

// ── Types ──────────────────────────────────────────────────────────────────

export interface TimestampedReading {
  timestamp: string;  // ISO string
  focus: number;      // 0-1
  stress: number;     // 0-1
}

export interface MealTimestamp {
  timestamp: string;  // ISO string
  label: string;      // "breakfast" | "lunch" | "dinner" | "snack"
}

export type InsightType = "focus_dip" | "focus_boost" | "no_change";

export interface MealCognitiveInsight {
  type: InsightType;
  focusChange: number;      // negative = dip, positive = boost
  stressChange: number;     // positive = stress increased
  peakEffectHours: number;  // hours after meal when effect is strongest
  mealLabel: string;        // which meal type
  sampleCount: number;      // how many readings contributed
}

// ── Constants ──────────────────────────────────────────────────────────────

/** Time window before meal to compute baseline (hours) */
const BASELINE_WINDOW_BEFORE = 3;

/** Earliest post-meal readings to consider (hours after meal) */
const POST_MEAL_START = 0.5;

/** Latest post-meal readings to consider (hours after meal) */
const POST_MEAL_END = 2.5;

/** Minimum change in focus to classify as dip or boost */
const CHANGE_THRESHOLD = 0.1;

/** Minimum readings needed for a correlation */
const MIN_READINGS = 2;

// ── Core computation ───────────────────────────────────────────────────────

function toMs(iso: string): number {
  return new Date(iso).getTime();
}

function hoursBetween(t1: string, t2: string): number {
  return Math.abs(toMs(t1) - toMs(t2)) / (60 * 60 * 1000);
}

/**
 * Compute meal-cognitive correlation.
 * Returns null if insufficient data.
 */
export function computeMealCognitiveCorrelation(
  meals: MealTimestamp[],
  readings: TimestampedReading[],
): MealCognitiveInsight | null {
  if (meals.length === 0 || readings.length === 0) return null;

  // Aggregate across all meals
  const baselineReadings: TimestampedReading[] = [];
  const postMealReadings: Array<{ reading: TimestampedReading; hoursAfter: number }> = [];
  let dominantLabel = "meal";

  // Count labels to find most common
  const labelCounts: Record<string, number> = {};
  for (const meal of meals) {
    labelCounts[meal.label] = (labelCounts[meal.label] ?? 0) + 1;
  }
  dominantLabel = Object.entries(labelCounts).sort((a, b) => b[1] - a[1])[0]?.[0] ?? "meal";

  for (const meal of meals) {
    const mealMs = toMs(meal.timestamp);

    for (const reading of readings) {
      const readingMs = toMs(reading.timestamp);
      const diffHours = (mealMs - readingMs) / (60 * 60 * 1000);

      // Baseline: reading is 1-BASELINE_WINDOW_BEFORE hours BEFORE the meal
      if (diffHours > 0 && diffHours <= BASELINE_WINDOW_BEFORE) {
        baselineReadings.push(reading);
      }

      // Post-meal: reading is POST_MEAL_START to POST_MEAL_END hours AFTER the meal
      const hoursAfter = -diffHours; // positive when reading is after meal
      if (hoursAfter >= POST_MEAL_START && hoursAfter <= POST_MEAL_END) {
        postMealReadings.push({ reading, hoursAfter });
      }
    }
  }

  if (baselineReadings.length < MIN_READINGS || postMealReadings.length < MIN_READINGS) {
    return null;
  }

  // Compute averages
  const avgBaseline = {
    focus: baselineReadings.reduce((s, r) => s + r.focus, 0) / baselineReadings.length,
    stress: baselineReadings.reduce((s, r) => s + r.stress, 0) / baselineReadings.length,
  };

  const avgPostMeal = {
    focus: postMealReadings.reduce((s, r) => s + r.reading.focus, 0) / postMealReadings.length,
    stress: postMealReadings.reduce((s, r) => s + r.reading.stress, 0) / postMealReadings.length,
  };

  const focusChange = avgPostMeal.focus - avgBaseline.focus;
  const stressChange = avgPostMeal.stress - avgBaseline.stress;

  // Find peak effect time (hour with lowest focus, if dip; highest if boost)
  const peakEffectHours = focusChange < 0
    ? postMealReadings.reduce((best, r) =>
        r.reading.focus < best.reading.focus ? r : best
      ).hoursAfter
    : postMealReadings.reduce((best, r) =>
        r.reading.focus > best.reading.focus ? r : best
      ).hoursAfter;

  // Classify
  let type: InsightType;
  if (focusChange < -CHANGE_THRESHOLD) {
    type = "focus_dip";
  } else if (focusChange > CHANGE_THRESHOLD) {
    type = "focus_boost";
  } else {
    type = "no_change";
  }

  return {
    type,
    focusChange: Math.round(focusChange * 100) / 100,
    stressChange: Math.round(stressChange * 100) / 100,
    peakEffectHours: Math.round(peakEffectHours * 10) / 10,
    mealLabel: dominantLabel,
    sampleCount: postMealReadings.length,
  };
}

// ── Insight text generation ────────────────────────────────────────────────

export function generateInsight(insight: MealCognitiveInsight): string {
  const { type, focusChange, peakEffectHours, mealLabel } = insight;

  if (type === "focus_dip") {
    const pct = Math.abs(Math.round(focusChange * 100));
    return `Your focus tends to dip ${peakEffectHours}h after ${mealLabel} (${pct}% drop). Try lighter meals or a short walk.`;
  }

  if (type === "focus_boost") {
    const pct = Math.round(focusChange * 100);
    return `Post-${mealLabel} focus boost detected (+${pct}%). ${mealLabel.charAt(0).toUpperCase() + mealLabel.slice(1)} seems to energize you.`;
  }

  return `Your focus stays stable after ${mealLabel}. Current meal timing works well.`;
}

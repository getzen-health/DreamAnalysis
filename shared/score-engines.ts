/**
 * Score Engines — 6 health score computations for NeuralDreamWorkshop.
 *
 * Each function takes daily_aggregates and user_baselines rows and returns
 * a score (0-100) or null when data is insufficient.
 *
 * Consumed by the compute-scores Edge Function.
 */

// ── Types ─────────────────────────────────────────────────────────────────────

export interface Aggregate {
  user_id: string;
  date: string;
  metric: string;
  avg_value: string | number | null;
  min_value: string | number | null;
  max_value: string | number | null;
  sum_value: string | number | null;
  sample_count: number | null;
}

export interface Baseline {
  user_id: string;
  metric: string;
  baseline_avg: string | number | null;
  baseline_stddev: string | number | null;
  sample_count: number | null;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Safely coerce a string | number | null to a number, returning null on NaN. */
function num(v: string | number | null | undefined): number | null {
  if (v === null || v === undefined) return null;
  const n = typeof v === 'string' ? parseFloat(v) : v;
  return isNaN(n) ? null : n;
}

/** Get the most recent aggregate value for a metric (by avg_value). */
function getLatest(aggregates: Aggregate[], metric: string): number | null {
  const sorted = aggregates
    .filter(a => a.metric === metric)
    .sort((a, b) => b.date.localeCompare(a.date));
  return sorted.length > 0 ? num(sorted[0].avg_value) : null;
}

/** Get the most recent aggregate's sum_value for a metric. */
function getLatestSum(aggregates: Aggregate[], metric: string): number | null {
  const sorted = aggregates
    .filter(a => a.metric === metric)
    .sort((a, b) => b.date.localeCompare(a.date));
  return sorted.length > 0 ? num(sorted[0].sum_value) : null;
}

/** Get avg_value for the N most recent days, sorted newest-first. */
function getRecentValues(aggregates: Aggregate[], metric: string, days: number): number[] {
  return aggregates
    .filter(a => a.metric === metric && num(a.avg_value) !== null)
    .sort((a, b) => b.date.localeCompare(a.date))
    .slice(0, days)
    .map(a => num(a.avg_value)!);
}

/** Look up baseline avg/std for a metric. Returns null if no baseline. */
function getBaseline(baselines: Baseline[], metric: string): { avg: number; std: number } | null {
  const b = baselines.find(bl => bl.metric === metric);
  if (!b || num(b.baseline_avg) === null) return null;
  return { avg: num(b.baseline_avg)!, std: num(b.baseline_stddev) || 0 };
}

/**
 * Score a metric relative to its baseline using z-score mapping.
 *
 * Maps z-score to 0-100:
 *   z = 0  -> 50 (at baseline)
 *   z = +2 -> 100 or 0 depending on direction
 *   z = -2 -> 0 or 100 depending on direction
 *
 * @param higherIsBetter - true if higher values mean a better score (e.g. HRV)
 */
function scoreVsBaseline(
  current: number | null,
  baseline: { avg: number; std: number } | null,
  higherIsBetter: boolean,
): number | null {
  if (current === null || baseline === null) return null;
  const std = baseline.std || 1; // avoid division by zero; use 1 as fallback
  const zScore = (current - baseline.avg) / std;
  const raw = higherIsBetter
    ? 50 + zScore * 25
    : 50 - zScore * 25;
  return Math.max(0, Math.min(100, Math.round(raw)));
}

/** Clamp a number to [min, max]. */
function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

/**
 * Compute weighted average of sub-scores, ignoring null entries and
 * redistributing their weights proportionally to available entries.
 * Returns null if fewer than `minRequired` entries have values.
 */
function weightedAvg(
  entries: { score: number | null; weight: number }[],
  minRequired: number,
): number | null {
  const available = entries.filter(e => e.score !== null) as { score: number; weight: number }[];
  if (available.length < minRequired) return null;
  const totalWeight = available.reduce((s, e) => s + e.weight, 0);
  if (totalWeight === 0) return null;
  const weighted = available.reduce((s, e) => s + e.score * (e.weight / totalWeight), 0);
  return Math.max(0, Math.min(100, Math.round(weighted)));
}

// ── 1. Recovery Score ─────────────────────────────────────────────────────────

/**
 * Recovery Score (0-100)
 *
 * Measures readiness for physical/mental load based on overnight biometrics.
 *
 * Inputs & weights:
 *   - HRV (hrv_rmssd) vs baseline:       25%  (higher = better)
 *   - Resting HR (resting_hr) vs baseline: 20%  (lower = better)
 *   - Sleep score (sleep_score):           25%  (direct 0-100)
 *   - Skin temp deviation (skin_temp):     10%  (closer to baseline = better)
 *   - Respiratory rate (respiratory_rate):  10%  (lower = better)
 *   - SpO2 (spo2) vs baseline:            10%  (higher = better)
 *
 * Calibration gate: returns null if any baseline has sample_count < 7.
 * Minimum data: at least 2 of the 6 sub-scores must be computable.
 */
export function computeRecoveryScore(
  aggregates: Aggregate[],
  baselines: Baseline[],
): number | null {
  // Calibration gate: need at least 7 days of data for the key metrics
  const keyMetrics = ['hrv_rmssd', 'resting_hr'];
  for (const m of keyMetrics) {
    const b = baselines.find(bl => bl.metric === m);
    if (b && (b.sample_count ?? 0) < 7) return null;
  }

  const hrvBaseline = getBaseline(baselines, 'hrv_rmssd');
  const rhrBaseline = getBaseline(baselines, 'resting_hr');
  const skinBaseline = getBaseline(baselines, 'skin_temp');
  const rrBaseline = getBaseline(baselines, 'respiratory_rate');
  const spo2Baseline = getBaseline(baselines, 'spo2');

  const hrvScore = scoreVsBaseline(getLatest(aggregates, 'hrv_rmssd'), hrvBaseline, true);
  const rhrScore = scoreVsBaseline(getLatest(aggregates, 'resting_hr'), rhrBaseline, false);
  const rrScore = scoreVsBaseline(getLatest(aggregates, 'respiratory_rate'), rrBaseline, false);
  const spo2Score = scoreVsBaseline(getLatest(aggregates, 'spo2'), spo2Baseline, true);

  // Skin temp: deviation from baseline — closer = better
  let skinScore: number | null = null;
  const skinCurrent = getLatest(aggregates, 'skin_temp');
  if (skinCurrent !== null && skinBaseline !== null) {
    const std = skinBaseline.std || 0.5;
    const deviation = Math.abs(skinCurrent - skinBaseline.avg) / std;
    // 0 deviation = 100, 2+ SD deviation = 0
    skinScore = Math.max(0, Math.min(100, Math.round(100 - deviation * 50)));
  }

  // Sleep score: use the latest sleep_score aggregate if available
  const sleepVal = getLatest(aggregates, 'sleep_score');
  const sleepScore = sleepVal !== null ? clamp(Math.round(sleepVal), 0, 100) : null;

  return weightedAvg([
    { score: hrvScore, weight: 0.25 },
    { score: rhrScore, weight: 0.20 },
    { score: sleepScore, weight: 0.25 },
    { score: skinScore, weight: 0.10 },
    { score: rrScore, weight: 0.10 },
    { score: spo2Score, weight: 0.10 },
  ], 2);
}

// ── 2. Sleep Score ────────────────────────────────────────────────────────────

/**
 * Sleep Score (0-100)
 *
 * Evaluates sleep quality from duration, stage balance, HR dip, and efficiency.
 *
 * Inputs & weights:
 *   - Time asleep vs 7-9 hr goal:   30%
 *   - Sleep stage balance:           25%
 *   - HR dip during sleep:           20%
 *   - Sleep efficiency:              25%
 *
 * Returns null if no sleep duration data is available.
 */
export function computeSleepScore(
  aggregates: Aggregate[],
  baselines: Baseline[],
): number | null {
  // ── Time asleep (30%) ──────────────────────────────────────────────────
  const totalMinRaw = getLatest(aggregates, 'sleep_total_min');
  let durationScore: number | null = null;

  if (totalMinRaw !== null) {
    const totalMin = totalMinRaw;
    // Goal: 420-540 min (7-9 hrs). Below 420 or above 540 = penalty.
    if (totalMin >= 420 && totalMin <= 540) {
      durationScore = 100;
    } else if (totalMin < 420) {
      // Linear scale: 0 min = 0, 420 min = 100
      durationScore = Math.round((totalMin / 420) * 100);
    } else {
      // Oversleeping penalty: 540 = 100, 660 (11 hrs) = 60
      durationScore = Math.max(0, Math.round(100 - ((totalMin - 540) / 120) * 40));
    }
    durationScore = clamp(durationScore, 0, 100);
  }

  // If no duration data at all, cannot compute sleep score
  if (durationScore === null) return null;

  // ── Sleep stage balance (25%) ──────────────────────────────────────────
  let stageScore: number | null = null;
  const deepMin = getLatest(aggregates, 'sleep_deep_min');
  const remMin = getLatest(aggregates, 'sleep_rem_min');
  const lightMin = getLatest(aggregates, 'sleep_light_min');

  if (deepMin !== null && remMin !== null && lightMin !== null && totalMinRaw !== null && totalMinRaw > 0) {
    const total = totalMinRaw;
    const deepPct = deepMin / total;
    const remPct = remMin / total;
    const lightPct = lightMin / total;

    // Ideal: deep 20-25%, REM 20-25%, light 50-60%
    // Score each stage: how close to ideal midpoint?
    const deepDev = Math.abs(deepPct - 0.225) / 0.225;   // deviation from 22.5% midpoint
    const remDev = Math.abs(remPct - 0.225) / 0.225;
    const lightDev = Math.abs(lightPct - 0.55) / 0.55;   // deviation from 55% midpoint

    // Average deviation, map to 0-100 (0 dev = 100, 1.0 dev = 0)
    const avgDev = (deepDev + remDev + lightDev) / 3;
    stageScore = clamp(Math.round((1 - avgDev) * 100), 0, 100);
  }

  // ── HR dip during sleep (20%) ──────────────────────────────────────────
  let hrDipScore: number | null = null;
  const restingHr = getLatest(aggregates, 'resting_hr');
  const sleepHr = getLatest(aggregates, 'sleep_avg_hr');

  if (restingHr !== null && sleepHr !== null && restingHr > 0) {
    // Dip percentage: typical is 10-20%
    const dipPct = ((restingHr - sleepHr) / restingHr) * 100;
    if (dipPct >= 15) {
      hrDipScore = 100;
    } else if (dipPct >= 10) {
      hrDipScore = 70 + ((dipPct - 10) / 5) * 30; // 10% = 70, 15% = 100
    } else if (dipPct >= 0) {
      hrDipScore = (dipPct / 10) * 70; // 0% = 0, 10% = 70
    } else {
      hrDipScore = 0; // HR increased during sleep — bad
    }
    hrDipScore = clamp(Math.round(hrDipScore), 0, 100);
  }

  // ── Sleep efficiency (25%) ─────────────────────────────────────────────
  let efficiencyScore: number | null = null;
  const efficiency = getLatest(aggregates, 'sleep_efficiency');

  if (efficiency !== null) {
    // Efficiency is 0-100%. >90% = excellent.
    if (efficiency >= 95) {
      efficiencyScore = 100;
    } else if (efficiency >= 85) {
      efficiencyScore = 60 + ((efficiency - 85) / 10) * 40; // 85%=60, 95%=100
    } else {
      efficiencyScore = (efficiency / 85) * 60; // 0%=0, 85%=60
    }
    efficiencyScore = clamp(Math.round(efficiencyScore), 0, 100);
  }

  return weightedAvg([
    { score: durationScore, weight: 0.30 },
    { score: stageScore, weight: 0.25 },
    { score: hrDipScore, weight: 0.20 },
    { score: efficiencyScore, weight: 0.25 },
  ], 1); // Need at least duration to return a score
}

/**
 * Calculate the sleep needed tonight based on strain and sleep debt.
 *
 * @param yesterdayStrain - Strain score from the previous day (0-100+)
 * @param pastWeekSleepMin - Array of actual sleep minutes for the past 7 days (newest first)
 * @param goalMin - Sleep goal in minutes (default 480 = 8 hours)
 * @returns Minutes of sleep needed tonight
 */
export function computeSleepNeeded(
  yesterdayStrain: number,
  pastWeekSleepMin: number[],
  goalMin: number = 480,
): number {
  // Extra minutes needed if high strain yesterday
  const strainAdjustment = Math.max(0, (yesterdayStrain - 10) * 5);

  // Sleep debt: sum of (goal - actual) over past 7 days, weighted (recent = heavier)
  // Weights: day1 (most recent) = 7, day2 = 6, ..., day7 = 1
  let weightedDebt = 0;
  let totalWeight = 0;
  for (let i = 0; i < Math.min(pastWeekSleepMin.length, 7); i++) {
    const weight = 7 - i;
    const deficit = goalMin - pastWeekSleepMin[i];
    weightedDebt += Math.max(0, deficit) * weight;
    totalWeight += weight;
  }
  const avgWeightedDebt = totalWeight > 0 ? weightedDebt / totalWeight : 0;

  // Spread debt recovery over 4 nights
  const debtRecovery = avgWeightedDebt / 4;

  return Math.round(goalMin + strainAdjustment + debtRecovery);
}

// ── 3. Strain Score ───────────────────────────────────────────────────────────

/**
 * Strain Score (0-100+, logarithmic scale)
 *
 * Measures cumulative physiological load for the day.
 *
 * Algorithm:
 *   - Sum all workout_strain values for today
 *   - Add passive strain from steps: (steps / 10000) * 3
 *   - Strain = 14.3 * ln(1 + totalTrimp)
 *   - Returns 0 (not null) if no workout/step data — zero strain is valid
 */
export function computeStrainScore(
  aggregates: Aggregate[],
  _baselines: Baseline[],
): number | null {
  // Sum today's workout strain values
  const workoutStrain = getLatestSum(aggregates, 'workout_strain');
  const activeStrain = workoutStrain ?? 0;

  // Passive strain from steps
  const steps = getLatest(aggregates, 'steps');
  const passiveStrain = steps !== null ? (steps / 10000) * 3 : 0;

  const totalTrimp = activeStrain + passiveStrain;

  // Logarithmic strain formula (same as cardio.ts computeStrain)
  const strain = 14.3 * Math.log(1 + totalTrimp);

  return Math.round(strain * 10) / 10; // one decimal place
}

// ── 4. Stress Score ───────────────────────────────────────────────────────────

/**
 * Stress Score (0-100)
 *
 * Higher score = MORE stressed (0 = relaxed, 100 = extremely stressed).
 *
 * Inputs & weights:
 *   - HRV trend (declining = stress):    30%
 *   - HR elevation above resting:        25%
 *   - Respiratory rate elevation:         15%
 *   - Skin temp deviation:               10%
 *   - EEG stress index (if available):   20%
 *
 * If no EEG data, the 20% is redistributed across other metrics.
 */
export function computeStressScore(
  aggregates: Aggregate[],
  baselines: Baseline[],
): number | null {
  // ── HRV trend (30%): compare today vs 3-day avg ───────────────────────
  let hrvTrendScore: number | null = null;
  const hrvRecent = getRecentValues(aggregates, 'hrv_rmssd', 4);
  if (hrvRecent.length >= 2) {
    const today = hrvRecent[0];
    const prevAvg = hrvRecent.slice(1).reduce((s, v) => s + v, 0) / (hrvRecent.length - 1);
    if (prevAvg > 0) {
      const pctChange = ((today - prevAvg) / prevAvg) * 100;
      // Declining HRV = stress. -20% change = high stress (100), +10% = low stress (0)
      if (pctChange <= -20) {
        hrvTrendScore = 100;
      } else if (pctChange >= 10) {
        hrvTrendScore = 0;
      } else {
        // Linear interpolation: +10% -> 0, -20% -> 100
        hrvTrendScore = Math.round(((10 - pctChange) / 30) * 100);
      }
      hrvTrendScore = clamp(hrvTrendScore, 0, 100);
    }
  }

  // ── HR elevation above resting (25%) ──────────────────────────────────
  let hrElevationScore: number | null = null;
  const currentHr = getLatest(aggregates, 'heart_rate');
  const restingHr = getLatest(aggregates, 'resting_hr');
  if (currentHr !== null && restingHr !== null && restingHr > 0) {
    const elevation = (currentHr - restingHr) / restingHr;
    // 0% elevation = 0, 30%+ elevation = 100
    hrElevationScore = clamp(Math.round((elevation / 0.30) * 100), 0, 100);
  }

  // ── Respiratory rate elevation (15%) ──────────────────────────────────
  let rrScore: number | null = null;
  const rrBaseline = getBaseline(baselines, 'respiratory_rate');
  const rrCurrent = getLatest(aggregates, 'respiratory_rate');
  if (rrCurrent !== null && rrBaseline !== null) {
    // Higher RR = more stress. Flip scoreVsBaseline (higher = worse for stress)
    const raw = scoreVsBaseline(rrCurrent, rrBaseline, true);
    if (raw !== null) {
      // raw: higher RR = higher score. We want higher RR = more stress (higher).
      rrScore = raw;
    }
  }

  // ── Skin temp deviation (10%) ─────────────────────────────────────────
  let skinScore: number | null = null;
  const skinBaseline = getBaseline(baselines, 'skin_temp');
  const skinCurrent = getLatest(aggregates, 'skin_temp');
  if (skinCurrent !== null && skinBaseline !== null) {
    const std = skinBaseline.std || 0.5;
    const deviation = Math.abs(skinCurrent - skinBaseline.avg) / std;
    // Any deviation = stress. 0 dev = 0, 2+ SD = 100
    skinScore = clamp(Math.round(deviation * 50), 0, 100);
  }

  // ── EEG stress index (20%) ────────────────────────────────────────────
  let eegScore: number | null = null;
  const eegStress = getLatest(aggregates, 'eeg_stress_index');
  if (eegStress !== null) {
    // Direct 0-1 value mapped to 0-100
    eegScore = clamp(Math.round(eegStress * 100), 0, 100);
  }

  return weightedAvg([
    { score: hrvTrendScore, weight: 0.30 },
    { score: hrElevationScore, weight: 0.25 },
    { score: rrScore, weight: 0.15 },
    { score: skinScore, weight: 0.10 },
    { score: eegScore, weight: 0.20 },
  ], 1);
}

// ── 5. Nutrition Score ────────────────────────────────────────────────────────

/**
 * Nutrition Score (1-100)
 *
 * Simplified AHEI-style scoring based on macro balance.
 *
 * Optimal ranges:
 *   - Protein: 25-35% of calories
 *   - Carbs:   40-55% of calories
 *   - Fat:     20-35% of calories
 *
 * Returns null if no food data logged today.
 * Minimum score is 1 (never 0).
 */
export function computeNutritionScore(
  aggregates: Aggregate[],
  _baselines: Baseline[],
): number | null {
  const totalCalories = getLatest(aggregates, 'total_calories');
  if (totalCalories === null || totalCalories <= 0) return null;

  // Try to get macro grams; derive percentages from calories
  // protein: 4 cal/g, carbs: 4 cal/g, fat: 9 cal/g
  const proteinG = getLatest(aggregates, 'total_protein_g');
  const carbsG = getLatest(aggregates, 'total_carbs_g');
  const fatG = getLatest(aggregates, 'total_fat_g');

  if (proteinG === null && carbsG === null && fatG === null) {
    // No macro breakdown — can only score on calorie presence
    // Return a neutral 50 (data exists but no macro detail)
    return 50;
  }

  let totalScore = 0;
  let componentCount = 0;

  // Score each macro on how close to optimal range
  if (proteinG !== null) {
    const proteinPct = (proteinG * 4 / totalCalories) * 100;
    totalScore += scoreMacro(proteinPct, 25, 35);
    componentCount++;
  }

  if (carbsG !== null) {
    const carbsPct = (carbsG * 4 / totalCalories) * 100;
    totalScore += scoreMacro(carbsPct, 40, 55);
    componentCount++;
  }

  if (fatG !== null) {
    const fatPct = (fatG * 9 / totalCalories) * 100;
    totalScore += scoreMacro(fatPct, 20, 35);
    componentCount++;
  }

  if (componentCount === 0) return 50;

  const avgScore = totalScore / componentCount;
  return Math.max(1, Math.round(avgScore));
}

/**
 * Score a single macronutrient percentage against its optimal range.
 * Returns 0-100: in-range = 100, linearly penalized outside.
 */
function scoreMacro(actual: number, optMin: number, optMax: number): number {
  if (actual >= optMin && actual <= optMax) return 100;
  if (actual < optMin) {
    // How far below minimum? Every 10% below = -50 points
    const deficit = optMin - actual;
    return Math.max(0, Math.round(100 - (deficit / 10) * 50));
  }
  // Above maximum
  const excess = actual - optMax;
  return Math.max(0, Math.round(100 - (excess / 10) * 50));
}

// ── 6. Energy Bank ────────────────────────────────────────────────────────────

/**
 * Energy Bank (0-100)
 *
 * Composite score: starts at recovery level, depleted by strain,
 * penalized by stress, boosted by good nutrition.
 *
 * Formula:
 *   energy = recovery * strainFactor * sleepFactor * stressPenalty * nutritionBoost
 *
 * Returns null if recovery is null (recovery is the required anchor).
 */
export function computeEnergyBank(scores: {
  recovery: number | null;
  sleep: number | null;
  strain: number | null;
  stress: number | null;
  nutrition: number | null;
}): number | null {
  if (scores.recovery === null) return null;

  const maxStrain = 100; // normalized ceiling

  // strainFactor: 0 strain = 1.0, max strain = 0.0
  const strainFactor = 1 - ((scores.strain ?? 0) / maxStrain);

  // sleepFactor: default 0.5 if no sleep data
  const sleepFactor = (scores.sleep ?? 50) / 100;

  // stressPenalty: stress of 0 = 1.0, stress of 100 = 0.5
  const stressPenalty = 1 - ((scores.stress ?? 0) / 200);

  // nutritionBoost: neutral at 50, slight boost above, slight penalty below
  // nutrition of 100 = 1.25, nutrition of 0 = 0.75, nutrition of 50 = 1.0
  const nutritionBoost = 1 + (((scores.nutrition ?? 50) - 50) / 200);

  const energy = scores.recovery * strainFactor * sleepFactor * stressPenalty * nutritionBoost;

  return clamp(Math.round(energy), 0, 100);
}

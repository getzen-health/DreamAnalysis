/**
 * predictive-alerts.ts — Pure function prediction engine that analyzes recent
 * health data to forecast tomorrow's likely state.
 *
 * No API calls, no side effects — just computation over today's metrics
 * and the last 7 days of history.
 */

// ── Types ──────────────────────────────────────────────────────────────────

export interface PredictionInput {
  // Today's data
  todayStress: number | null;        // 0-1
  todayValence: number | null;       // -1 to 1
  todayFocus: number | null;         // 0-1
  todaySleepHours: number | null;
  todaySleepQuality: number | null;  // 0-100
  todaySteps: number | null;
  todayInnerScore: number | null;    // 0-100
  lateEating: boolean;               // ate after 9pm

  // Historical context (last 7 days)
  recentScores: (number | null)[];   // last 7 inner scores
  recentStress: number[];            // last 7 days avg stress
  recentSleep: number[];             // last 7 days sleep hours
}

export interface PredictiveAlert {
  id: string;
  type: "warning" | "positive" | "neutral";
  headline: string;
  body: string;
  confidence: number;     // 0-100
  factors: string[];
  action?: string;
}

// ── Helpers ────────────────────────────────────────────────────────────────

function avg(nums: number[]): number {
  if (nums.length === 0) return 0;
  return nums.reduce((s, n) => s + n, 0) / nums.length;
}

/** Confidence based on how much data we have. More history = higher confidence. */
function computeConfidence(dataPoints: number, maxExpected: number): number {
  const ratio = Math.min(dataPoints / maxExpected, 1);
  return Math.round(50 + ratio * 35); // 50-85 range
}

// ── Prediction Rules ───────────────────────────────────────────────────────

function checkSleepDebt(input: PredictionInput): PredictiveAlert | null {
  if (input.todaySleepHours == null || input.todaySleepHours >= 6) return null;
  if (input.recentSleep.length === 0) return null;

  const avgSleep = avg(input.recentSleep);
  if (avgSleep >= 6.5) return null;

  const factors: string[] = ["Today: less than 6 hours sleep"];
  if (input.recentSleep.length > 1) {
    factors.push(`${input.recentSleep.length}-day avg: ${avgSleep.toFixed(1)}h`);
  } else if (input.recentSleep.length === 1) {
    factors.push(`Today's sleep: ${avgSleep.toFixed(1)}h`);
  }

  return {
    id: "sleep-debt-warning",
    type: "warning",
    headline: "Sleep debt building — tomorrow may be tough",
    body: `You've averaged ${avgSleep.toFixed(1)} hours this week. Cumulative sleep debt compounds — cognitive performance drops measurably after 3+ days under 6.5 hours.`,
    confidence: computeConfidence(input.recentSleep.length + 1, 7),
    factors,
    action: "Try to get 8+ hours tonight",
  };
}

function checkStressSpiral(input: PredictionInput): PredictiveAlert | null {
  if (input.todayStress == null || input.todayStress <= 0.7) return null;
  if (input.recentStress.length < 3) return null;

  const last3 = input.recentStress.slice(-3);
  const highDays = last3.filter((s) => s > 0.6).length;
  if (highDays < 2) return null;

  const factors: string[] = [
    `Today's stress: ${Math.round(input.todayStress * 100)}%`,
    `${highDays + 1} of last 4 days elevated`,
  ];

  return {
    id: "stress-spiral",
    type: "warning",
    headline: "Stress has been elevated for 3 days",
    body: "Your body needs a reset. Sustained high-beta activity suppresses prefrontal function and impairs recovery sleep.",
    confidence: computeConfidence(input.recentStress.length + 1, 7),
    factors,
    action: "Try a breathing exercise before bed",
  };
}

function checkScoreDeclining(input: PredictionInput): PredictiveAlert | null {
  const scores = input.recentScores.filter((s): s is number => s != null);
  if (scores.length < 3) return null;

  // Check for 3+ consecutive drops
  let consecutiveDrops = 0;
  let maxConsecutiveDrops = 0;
  for (let i = 1; i < scores.length; i++) {
    if (scores[i] < scores[i - 1]) {
      consecutiveDrops++;
      maxConsecutiveDrops = Math.max(maxConsecutiveDrops, consecutiveDrops);
    } else {
      consecutiveDrops = 0;
    }
  }

  if (maxConsecutiveDrops < 2) return null; // Need 3+ consecutive (2+ drops = 3+ values declining)

  const factors: string[] = [
    `${maxConsecutiveDrops + 1} consecutive days declining`,
    `Latest: ${scores[scores.length - 1]}`,
  ];

  return {
    id: "score-declining",
    type: "warning",
    headline: "Your Inner Score has been declining",
    body: "Check sleep and stress — a multi-day decline usually traces back to one or both of these factors.",
    confidence: computeConfidence(scores.length, 7),
    factors,
    action: "Prioritize rest tonight",
  };
}

function checkLateEating(input: PredictionInput): PredictiveAlert | null {
  if (!input.lateEating) return null;

  const avgSleep = input.recentSleep.length > 0 ? avg(input.recentSleep) : null;
  if (avgSleep == null || avgSleep >= 7) return null;

  const factors: string[] = [
    "Late meal detected (after 9pm)",
    `Recent sleep avg: ${avgSleep.toFixed(1)}h`,
  ];

  return {
    id: "late-eating-sleep",
    type: "warning",
    headline: "Late meal detected — this tends to hurt your sleep",
    body: "Eating close to bedtime raises core body temperature and disrupts circadian sleep pressure. Consider an earlier dinner.",
    confidence: computeConfidence(input.recentSleep.length + 1, 7),
    factors,
    action: "No food 2 hours before bed",
  };
}

function checkGreatDayAhead(input: PredictionInput): PredictiveAlert | null {
  if (input.todaySleepQuality == null || input.todaySleepQuality <= 75) return null;
  if (input.todayStress == null || input.todayStress >= 0.3) return null;
  if (input.todayInnerScore == null || input.todayInnerScore <= 75) return null;

  const factors: string[] = [
    `Sleep quality: ${Math.round(input.todaySleepQuality)}%`,
    `Stress: ${Math.round(input.todayStress * 100)}%`,
    `Inner Score: ${input.todayInnerScore}`,
  ];

  let dataPoints = 3; // today's 3 data points
  if (input.recentSleep.length > 0) dataPoints += input.recentSleep.length;

  return {
    id: "great-day-ahead",
    type: "positive",
    headline: "Tomorrow looks strong — you're well-rested and relaxed",
    body: "Good sleep quality combined with low stress is the best predictor of a strong next day. Your recovery systems are firing on all cylinders.",
    confidence: computeConfidence(dataPoints, 10),
    factors,
  };
}

function checkActiveDayBoost(input: PredictionInput): PredictiveAlert | null {
  if (input.todaySteps == null || input.todaySteps <= 8000) return null;
  if (input.todaySleepHours == null || input.todaySleepHours < 7) return null;

  const factors: string[] = [
    `Steps: ${input.todaySteps.toLocaleString()}`,
    `Sleep: ${input.todaySleepHours.toFixed(1)}h`,
  ];

  let dataPoints = 2;
  if (input.recentSleep.length > 0) dataPoints += input.recentSleep.length;

  return {
    id: "active-day-boost",
    type: "positive",
    headline: "Active day + good sleep = strong tomorrow",
    body: "Physical activity enhances slow-wave sleep quality and next-day cognitive performance. Keep it up.",
    confidence: computeConfidence(dataPoints, 9),
    factors,
  };
}

// ── Main Export ─────────────────────────────────────────────────────────────

/**
 * Analyzes today's data and recent history to predict tomorrow's likely state.
 * Returns the first matching alert (priority order), or null if no alert applies.
 *
 * Pure function — no side effects, no API calls.
 */
export function predictNextDay(input: PredictionInput): PredictiveAlert | null {
  const rules = [
    checkSleepDebt,
    checkStressSpiral,
    checkScoreDeclining,
    checkLateEating,
    checkGreatDayAhead,
    checkActiveDayBoost,
  ];

  for (const rule of rules) {
    const alert = rule(input);
    if (alert) return alert;
  }

  return null;
}

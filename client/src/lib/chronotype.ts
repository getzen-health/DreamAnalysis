/**
 * Chronotype assessment and time-of-day baseline adjustment.
 *
 * Based on the reduced Morningness-Eveningness Questionnaire (rMEQ),
 * a validated 5-question instrument (Adan & Almirall, 1991).
 *
 * Evening chronotypes have 2x mood disorder risk (Nature 2025).
 * Alpha/beta power varies significantly by circadian phase.
 * This module adjusts EEG emotion baselines by time-of-day.
 */

// ── Types ─────────────────────────────────────────────────────────────────

export type ChronotypeCategory = "morning" | "intermediate" | "evening";

export interface ChronotypeQuestion {
  id: string;
  text: string;
  options: { label: string; score: number }[];
}

export interface ChronotypeResult {
  score: number;
  category: ChronotypeCategory;
}

export interface BaselineAdjustment {
  alphaMultiplier: number; // adjust expected alpha power
  arousalOffset: number;   // adjust arousal baseline
  valenceOffset: number;   // adjust valence baseline
}

// ── rMEQ Questionnaire ────────────────────────────────────────────────────

export const CHRONOTYPE_QUESTIONS: ChronotypeQuestion[] = [
  {
    id: "wake_time",
    text: "What time would you get up if you were entirely free to plan your day?",
    options: [
      { label: "5:00-6:30 AM", score: 5 },
      { label: "6:30-7:45 AM", score: 4 },
      { label: "7:45-9:45 AM", score: 3 },
      { label: "9:45-11:00 AM", score: 2 },
      { label: "11:00 AM-12:00 PM", score: 1 },
    ],
  },
  {
    id: "morning_feeling",
    text: "During the first half hour after waking, how do you feel?",
    options: [
      { label: "Very refreshed", score: 4 },
      { label: "Fairly refreshed", score: 3 },
      { label: "Fairly tired", score: 2 },
      { label: "Very tired", score: 1 },
    ],
  },
  {
    id: "sleep_time",
    text: "If you had no commitments the next day, what time would you go to bed?",
    options: [
      { label: "8:00-9:00 PM", score: 5 },
      { label: "9:00-10:15 PM", score: 4 },
      { label: "10:15 PM-12:30 AM", score: 3 },
      { label: "12:30-1:45 AM", score: 2 },
      { label: "1:45-3:00 AM", score: 1 },
    ],
  },
  {
    id: "peak_time",
    text: "At what time of day do you feel your best?",
    options: [
      { label: "5:00-8:00 AM", score: 5 },
      { label: "8:00-10:00 AM", score: 4 },
      { label: "10:00 AM-5:00 PM", score: 3 },
      { label: "5:00-10:00 PM", score: 2 },
      { label: "10:00 PM-5:00 AM", score: 1 },
    ],
  },
  {
    id: "self_assessment",
    text: "One hears about 'morning' and 'evening' types. Which do you consider yourself?",
    options: [
      { label: "Definitely morning", score: 6 },
      { label: "More morning than evening", score: 4 },
      { label: "More evening than morning", score: 2 },
      { label: "Definitely evening", score: 0 },
    ],
  },
];

// ── Scoring ───────────────────────────────────────────────────────────────

/**
 * Score chronotype from rMEQ answers.
 * rMEQ scoring: 4-11 = evening, 12-17 = intermediate, 18-25 = morning.
 */
export function scoreChronotype(answers: number[]): ChronotypeResult {
  const score = answers.reduce((a, b) => a + b, 0);
  const category: ChronotypeCategory =
    score >= 18 ? "morning" : score >= 12 ? "intermediate" : "evening";
  return { score, category };
}

// ── Baseline Adjustment ───────────────────────────────────────────────────

/**
 * Compute time-of-day baseline adjustment for EEG emotion readings.
 *
 * Morning types: optimal 6-10 AM (alpha higher, valence positive).
 *   Low energy 8-11 PM.
 * Evening types: optimal 4-10 PM (alpha higher, valence positive).
 *   Low energy 6-9 AM.
 * Intermediate: mild adjustments, optimal 9 AM - 7 PM.
 *
 * Adjustments are small (0.85-1.15 range) -- they shift the baseline,
 * not override the actual EEG signal.
 */
export function getBaselineAdjustment(
  chronotype: ChronotypeCategory,
  hour: number,
): BaselineAdjustment {
  // Clamp hour to 0-23
  const h = ((hour % 24) + 24) % 24;

  if (chronotype === "morning") {
    return getMorningAdjustment(h);
  }
  if (chronotype === "evening") {
    return getEveningAdjustment(h);
  }
  return getIntermediateAdjustment(h);
}

function getMorningAdjustment(h: number): BaselineAdjustment {
  // Peak: 6-10 AM
  if (h >= 6 && h < 10) {
    return { alphaMultiplier: 1.12, arousalOffset: 0.08, valenceOffset: 0.10 };
  }
  // Good: 10 AM - 2 PM
  if (h >= 10 && h < 14) {
    return { alphaMultiplier: 1.05, arousalOffset: 0.04, valenceOffset: 0.05 };
  }
  // Declining: 2 PM - 6 PM
  if (h >= 14 && h < 18) {
    return { alphaMultiplier: 0.97, arousalOffset: -0.02, valenceOffset: 0.0 };
  }
  // Low: 6 PM - 10 PM
  if (h >= 18 && h < 22) {
    return { alphaMultiplier: 0.90, arousalOffset: -0.06, valenceOffset: -0.06 };
  }
  // Very low: 10 PM - 6 AM
  return { alphaMultiplier: 0.88, arousalOffset: -0.08, valenceOffset: -0.08 };
}

function getEveningAdjustment(h: number): BaselineAdjustment {
  // Low energy: 6-9 AM
  if (h >= 6 && h < 9) {
    return { alphaMultiplier: 0.88, arousalOffset: -0.08, valenceOffset: -0.08 };
  }
  // Warming up: 9 AM - 12 PM
  if (h >= 9 && h < 12) {
    return { alphaMultiplier: 0.95, arousalOffset: -0.03, valenceOffset: -0.02 };
  }
  // Good: 12 PM - 4 PM
  if (h >= 12 && h < 16) {
    return { alphaMultiplier: 1.03, arousalOffset: 0.03, valenceOffset: 0.04 };
  }
  // Peak: 4-10 PM
  if (h >= 16 && h < 22) {
    return { alphaMultiplier: 1.12, arousalOffset: 0.08, valenceOffset: 0.10 };
  }
  // Late night: 10 PM - 2 AM (still energized for evening types)
  if (h >= 22 || h < 2) {
    return { alphaMultiplier: 1.05, arousalOffset: 0.04, valenceOffset: 0.04 };
  }
  // Deep night: 2-6 AM
  return { alphaMultiplier: 0.90, arousalOffset: -0.06, valenceOffset: -0.06 };
}

function getIntermediateAdjustment(h: number): BaselineAdjustment {
  // Mild peak: 9 AM - 7 PM
  if (h >= 9 && h < 19) {
    return { alphaMultiplier: 1.04, arousalOffset: 0.03, valenceOffset: 0.03 };
  }
  // Mild trough: 7 PM - 9 AM
  return { alphaMultiplier: 0.97, arousalOffset: -0.02, valenceOffset: -0.02 };
}

// ── localStorage Persistence ──────────────────────────────────────────────

const STORAGE_KEY = "ndw_chronotype";

export function saveChronotype(score: number, category: ChronotypeCategory): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ score, category }));
  } catch {
    // localStorage unavailable or full -- silently fail
  }
}

export function getStoredChronotype(): ChronotypeResult | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (
      typeof parsed === "object" &&
      parsed !== null &&
      typeof parsed.score === "number" &&
      (parsed.category === "morning" || parsed.category === "intermediate" || parsed.category === "evening")
    ) {
      return { score: parsed.score, category: parsed.category };
    }
    return null;
  } catch {
    return null;
  }
}

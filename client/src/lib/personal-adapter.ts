/**
 * Personal EEG Adapter — lightweight on-device personalization layer.
 *
 * Sits AFTER the EEGNet ONNX output and adjusts class probabilities
 * based on user corrections and usage patterns. Inspired by BrainUICL
 * (ICLR 2025): keep the base model frozen, only train a small adapter.
 *
 * No backpropagation or gradient descent — uses a Bayesian-inspired
 * update rule with bias/multiplier adjustments that persist in localStorage.
 *
 * Two adaptation mechanisms:
 *   1. Supervised: user corrects a prediction → boost correct class, reduce wrong class
 *   2. Unsupervised: anti-collapse prevents one class from dominating predictions
 */

import { EEGNET_EMOTIONS } from "./eegnet-utils";
import { sbGetSetting, sbSaveGeneric } from "./supabase-store";

// ── Types ──────────────────────────────────────────────────────────────────

export interface PersonalAdapter {
  /** Per-class bias adjustments (6 classes). Range: [-1, 1], start at 0. */
  biases: number[];
  /** Per-class confidence multipliers (6 classes). Range: [0.5, 1.5], start at 1. */
  multipliers: number[];
  /** Running count of how many times each class was predicted. */
  classCounts: number[];
  /** Running count of how many times each class prediction was corrected. */
  correctionCounts: number[];
  /** Total number of inference sessions processed. */
  totalSessions: number;
  /** Total number of user corrections applied. */
  totalCorrections: number;
  /** ISO timestamp of last update. */
  lastUpdated: string;
}

// ── Constants ──────────────────────────────────────────────────────────────

const STORAGE_KEY = "ndw_personal_adapter";
const NUM_CLASSES = 6;
const DEFAULT_LEARNING_RATE = 0.05;
const BIAS_MIN = -1;
const BIAS_MAX = 1;
const MULTIPLIER_MIN = 0.5;
const MULTIPLIER_MAX = 1.5;
const ANTI_COLLAPSE_DOMINANT_THRESHOLD = 0.4;
const ANTI_COLLAPSE_RARE_THRESHOLD = 0.05;
const ANTI_COLLAPSE_STEP = 0.01;

// ── Helpers ────────────────────────────────────────────────────────────────

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function createFreshAdapter(): PersonalAdapter {
  return {
    biases: new Array(NUM_CLASSES).fill(0),
    multipliers: new Array(NUM_CLASSES).fill(1),
    classCounts: new Array(NUM_CLASSES).fill(0),
    correctionCounts: new Array(NUM_CLASSES).fill(0),
    totalSessions: 0,
    totalCorrections: 0,
    lastUpdated: new Date().toISOString(),
  };
}

// ── Load / Save ────────────────────────────────────────────────────────────

/**
 * Load adapter from localStorage, or create a fresh one if none exists
 * or the stored data is corrupted.
 */
export function loadPersonalAdapter(): PersonalAdapter {
  try {
    const raw = sbGetSetting(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      // Validate structure
      if (
        Array.isArray(parsed.biases) &&
        parsed.biases.length === NUM_CLASSES &&
        Array.isArray(parsed.multipliers) &&
        parsed.multipliers.length === NUM_CLASSES
      ) {
        return {
          biases: parsed.biases,
          multipliers: parsed.multipliers,
          classCounts: parsed.classCounts ?? new Array(NUM_CLASSES).fill(0),
          correctionCounts: parsed.correctionCounts ?? new Array(NUM_CLASSES).fill(0),
          totalSessions: parsed.totalSessions ?? 0,
          totalCorrections: parsed.totalCorrections ?? 0,
          lastUpdated: parsed.lastUpdated ?? new Date().toISOString(),
        };
      }
    }
  } catch {
    /* corrupted — start fresh */
  }
  return createFreshAdapter();
}

/** Save adapter to localStorage. */
export function savePersonalAdapter(adapter: PersonalAdapter): void {
  try {
    sbSaveGeneric(STORAGE_KEY, adapter);
  } catch {
    /* localStorage full or unavailable */
  }
}

// ── Apply adapter to raw probabilities ─────────────────────────────────────

/**
 * Apply the personal adapter to raw EEGNet output probabilities.
 *
 * Transformation: softmax(log(p[i]) + bias[i]) * multiplier[i], then re-normalize.
 *
 * With zero biases and unit multipliers, this is an identity transform:
 * softmax(log(p)) = p (since softmax inverts log in the same space).
 */
export function applyAdapter(
  rawProbabilities: number[],
  adapter: PersonalAdapter,
): number[] {
  const n = rawProbabilities.length;

  // Step 1: Compute log-adjusted logits
  const logits: number[] = new Array(n);
  for (let i = 0; i < n; i++) {
    // Guard against log(0)
    const p = Math.max(rawProbabilities[i], 1e-10);
    logits[i] = Math.log(p) + adapter.biases[i];
  }

  // Step 2: Softmax over adjusted logits
  const maxLogit = Math.max(...logits);
  const exps: number[] = new Array(n);
  let expSum = 0;
  for (let i = 0; i < n; i++) {
    exps[i] = Math.exp(logits[i] - maxLogit);
    expSum += exps[i];
  }

  const softmaxed: number[] = new Array(n);
  for (let i = 0; i < n; i++) {
    softmaxed[i] = exps[i] / expSum;
  }

  // Step 3: Apply multipliers
  const scaled: number[] = new Array(n);
  let scaledSum = 0;
  for (let i = 0; i < n; i++) {
    scaled[i] = softmaxed[i] * adapter.multipliers[i];
    scaledSum += scaled[i];
  }

  // Step 4: Re-normalize to sum to 1
  const result: number[] = new Array(n);
  for (let i = 0; i < n; i++) {
    result[i] = scaledSum > 0 ? scaled[i] / scaledSum : 1 / n;
  }

  return result;
}

// ── Update from user correction ────────────────────────────────────────────

/**
 * Update the adapter after a user corrects a prediction.
 *
 * Boosts the correct class's bias and multiplier,
 * reduces the wrong class's bias and multiplier.
 * Returns a new adapter (does not mutate input).
 */
export function updateFromCorrection(
  adapter: PersonalAdapter,
  predictedClass: number,
  correctClass: number,
  learningRate: number = DEFAULT_LEARNING_RATE,
): PersonalAdapter {
  const updated: PersonalAdapter = {
    biases: [...adapter.biases],
    multipliers: [...adapter.multipliers],
    classCounts: [...adapter.classCounts],
    correctionCounts: [...adapter.correctionCounts],
    totalSessions: adapter.totalSessions,
    totalCorrections: adapter.totalCorrections + 1,
    lastUpdated: new Date().toISOString(),
  };

  // Boost correct class
  updated.biases[correctClass] = clamp(
    updated.biases[correctClass] + learningRate,
    BIAS_MIN,
    BIAS_MAX,
  );
  updated.multipliers[correctClass] = clamp(
    updated.multipliers[correctClass] + learningRate * 0.4,
    MULTIPLIER_MIN,
    MULTIPLIER_MAX,
  );

  // Reduce wrong class
  if (predictedClass !== correctClass) {
    updated.biases[predictedClass] = clamp(
      updated.biases[predictedClass] - learningRate * 0.5,
      BIAS_MIN,
      BIAS_MAX,
    );
    updated.multipliers[predictedClass] = clamp(
      updated.multipliers[predictedClass] - learningRate * 0.2,
      MULTIPLIER_MIN,
      MULTIPLIER_MAX,
    );
  }

  // Track corrections per class
  updated.correctionCounts[correctClass] += 1;

  return updated;
}

// ── Update after session (unsupervised anti-collapse) ──────────────────────

/**
 * Update adapter after a session with the predicted class.
 *
 * Anti-collapse mechanism: if a class is predicted disproportionately often
 * (>40% of sessions), slightly reduce its bias. If a class is rarely predicted
 * (<5% of sessions), slightly boost its bias. Encourages diversity without
 * requiring labels.
 */
export function updateAfterSession(
  adapter: PersonalAdapter,
  predictedClass: number,
): PersonalAdapter {
  const updated: PersonalAdapter = {
    biases: [...adapter.biases],
    multipliers: [...adapter.multipliers],
    classCounts: [...adapter.classCounts],
    correctionCounts: [...adapter.correctionCounts],
    totalSessions: adapter.totalSessions + 1,
    totalCorrections: adapter.totalCorrections,
    lastUpdated: new Date().toISOString(),
  };

  // Increment class count for predicted class
  updated.classCounts[predictedClass] += 1;

  // Anti-collapse: only apply after enough sessions for meaningful statistics
  if (updated.totalSessions >= 5) {
    for (let i = 0; i < NUM_CLASSES; i++) {
      const freq = updated.classCounts[i] / updated.totalSessions;
      if (freq > ANTI_COLLAPSE_DOMINANT_THRESHOLD) {
        // Too dominant — reduce bias
        updated.biases[i] = clamp(
          updated.biases[i] - ANTI_COLLAPSE_STEP,
          BIAS_MIN,
          BIAS_MAX,
        );
      } else if (freq < ANTI_COLLAPSE_RARE_THRESHOLD) {
        // Too rare — boost bias
        updated.biases[i] = clamp(
          updated.biases[i] + ANTI_COLLAPSE_STEP,
          BIAS_MIN,
          BIAS_MAX,
        );
      }
    }
  }

  return updated;
}

// ── Personalization stats ──────────────────────────────────────────────────

/**
 * Get personalization stats for display in the UI.
 */
export function getPersonalizationStats(adapter: PersonalAdapter): {
  sessionsProcessed: number;
  correctionsApplied: number;
  dominantAdjustment: string;
  confidenceLevel: "learning" | "calibrating" | "personalized";
} {
  // Confidence level based on session count
  let confidenceLevel: "learning" | "calibrating" | "personalized";
  if (adapter.totalSessions < 10) {
    confidenceLevel = "learning";
  } else if (adapter.totalSessions < 50) {
    confidenceLevel = "calibrating";
  } else {
    confidenceLevel = "personalized";
  }

  // Find the most adjusted class for a human-readable description
  let dominantAdjustment = "No significant adjustments yet";
  let maxAbsBias = 0;
  let maxBiasIdx = -1;
  for (let i = 0; i < NUM_CLASSES; i++) {
    const absBias = Math.abs(adapter.biases[i]);
    if (absBias > maxAbsBias) {
      maxAbsBias = absBias;
      maxBiasIdx = i;
    }
  }

  if (maxBiasIdx >= 0 && maxAbsBias > 0.05) {
    const emotion = EEGNET_EMOTIONS[maxBiasIdx] ?? "unknown";
    const direction = adapter.biases[maxBiasIdx] > 0 ? "more often" : "rarely";
    dominantAdjustment = `Your model has learned you ${direction} feel ${emotion}`;
  }

  return {
    sessionsProcessed: adapter.totalSessions,
    correctionsApplied: adapter.totalCorrections,
    dominantAdjustment,
    confidenceLevel,
  };
}

// ── Reset ──────────────────────────────────────────────────────────────────

/**
 * Reset the personal adapter to a fresh state.
 * Clears localStorage and returns a new empty adapter.
 */
export function resetPersonalAdapter(): PersonalAdapter {
  const fresh = createFreshAdapter();
  savePersonalAdapter(fresh);
  return fresh;
}

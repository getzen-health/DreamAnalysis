/**
 * EEGNet ONNX utility functions — pure math, no ONNX dependency.
 *
 * Extracted from ml-local.ts so they can be unit-tested without
 * loading onnxruntime-web or hitting the WASM backend.
 */

// ── Constants ────────────────────────────────────────────────────────────────

export const EEGNET_EMOTIONS = ["happy", "sad", "angry", "fearful", "relaxed", "focused"] as const;
export const EEGNET_EXPECTED_CHANNELS = 4;
export const EEGNET_EXPECTED_SAMPLES = 1024; // 4 seconds @ 256 Hz

// ── Valence/arousal weights (matching Python pipeline) ───────────────────────
// valence = P(happy)*1 + P(relaxed)*0.5 + P(focused)*0.3
//         - P(sad)*1 - P(angry)*0.7 - P(fearful)*0.8
const VALENCE_WEIGHTS = [1.0, -1.0, -0.7, -0.8, 0.5, 0.3];

// arousal = P(angry)*1 + P(fearful)*0.9 + P(happy)*0.7 + P(focused)*0.6
//         - P(relaxed)*0.5 - P(sad)*0.2
const AROUSAL_WEIGHTS = [0.7, -0.2, 1.0, 0.9, -0.5, 0.6];

// ── softmax ──────────────────────────────────────────────────────────────────

/**
 * Numerically stable softmax: subtracts max before exponentiating
 * to prevent overflow with large logit values.
 */
export function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

// ── Channel normalization ────────────────────────────────────────────────────

/**
 * Normalize each channel to zero-mean, unit-variance.
 * Matches the Python training pipeline's per-epoch z-score normalization.
 *
 * If a channel has zero variance (constant signal), returns all zeros
 * for that channel instead of NaN.
 */
export function normalizeChannels(channels: Float32Array[]): Float32Array[] {
  return channels.map((ch) => {
    const n = ch.length;
    // Mean
    let sum = 0;
    for (let i = 0; i < n; i++) sum += ch[i];
    const mean = sum / n;

    // Std dev
    let sumSq = 0;
    for (let i = 0; i < n; i++) {
      const diff = ch[i] - mean;
      sumSq += diff * diff;
    }
    const std = Math.sqrt(sumSq / n);

    // Normalize (guard against zero-variance)
    const out = new Float32Array(n);
    if (std < 1e-10) {
      // Constant channel — return zeros
      return out;
    }
    for (let i = 0; i < n; i++) {
      out[i] = (ch[i] - mean) / std;
    }
    return out;
  });
}

// ── Emotion result from probabilities ────────────────────────────────────────

export interface EEGNetEmotionResult {
  emotion: string;
  confidence: number;
  probabilities: Record<string, number>;
  valence: number;
  arousal: number;
}

/**
 * Convert a 6-element probability array (from softmax of EEGNet logits)
 * into a structured emotion result with derived valence and arousal.
 */
export function eegnetEmotionFromProbabilities(probs: number[]): EEGNetEmotionResult {
  // Find dominant emotion
  let maxIdx = 0;
  let maxVal = probs[0];
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > maxVal) {
      maxVal = probs[i];
      maxIdx = i;
    }
  }

  // Build probability dict
  const probabilities: Record<string, number> = {};
  for (let i = 0; i < EEGNET_EMOTIONS.length; i++) {
    probabilities[EEGNET_EMOTIONS[i]] = probs[i] ?? 0;
  }

  // Derive valence: weighted sum, clamped to [-1, 1]
  let valenceRaw = 0;
  for (let i = 0; i < probs.length; i++) {
    valenceRaw += probs[i] * VALENCE_WEIGHTS[i];
  }
  const valence = Math.max(-1, Math.min(1, valenceRaw));

  // Derive arousal: weighted sum, clamped to [0, 1]
  let arousalRaw = 0;
  for (let i = 0; i < probs.length; i++) {
    arousalRaw += probs[i] * AROUSAL_WEIGHTS[i];
  }
  // Shift to [0, 1] range: raw range is roughly [-0.5, 1.0], normalize
  const arousal = Math.max(0, Math.min(1, (arousalRaw + 0.5) / 1.5));

  return {
    emotion: EEGNET_EMOTIONS[maxIdx] ?? "relaxed",
    confidence: maxVal,
    probabilities,
    valence,
    arousal,
  };
}

// ── Input validation ─────────────────────────────────────────────────────────

/**
 * Validate that the input matches EEGNet's expected shape:
 * exactly 4 channels, each with exactly 1024 samples.
 */
export function validateEEGNetInput(channels: Float32Array[]): boolean {
  if (channels.length !== EEGNET_EXPECTED_CHANNELS) return false;
  for (const ch of channels) {
    if (ch.length !== EEGNET_EXPECTED_SAMPLES) return false;
  }
  return true;
}

/**
 * breathing-detector.ts — Breathing state detection from frontal EEG.
 *
 * Respiration-entrained oscillations are detectable in frontal EEG in the
 * 0.1-0.5 Hz band. This module bandpass-filters raw AF7/AF8 data into the
 * respiratory band and uses peak detection to estimate breathing rate,
 * coherence, and qualitative state.
 *
 * Not BPM-accurate (EEG is not a respiratory sensor), but sufficient for
 * detecting deep/shallow/holding during meditation sessions.
 *
 * Algorithm:
 *   1. Bandpass filter 0.1-0.5 Hz (respiratory band, ~6-30 breaths/min)
 *   2. Find peaks in filtered signal (each peak = one breath)
 *   3. Count peaks per minute -> estimated breathing rate
 *   4. Coherence: regularity of inter-peak intervals (low CV = high coherence)
 *   5. State mapping by rate + coherence thresholds
 *
 * @see Issue #529
 */

// ── Types ───────────────────────────────────────────────────────────────────

export type BreathingState = "deep_slow" | "normal" | "shallow_fast" | "holding" | "unknown";

export interface BreathingAnalysis {
  /** Qualitative breathing state */
  state: BreathingState;
  /** Estimated breaths per minute (approximate) */
  estimatedRate: number;
  /** Regularity of breathing pattern, 0-1 (1 = perfectly regular) */
  coherence: number;
  /** Human-readable description of the current state */
  message: string;
}

// ── Constants ───────────────────────────────────────────────────────────────

/** Lower cutoff of the respiratory bandpass filter (Hz). ~6 breaths/min */
const RESP_LOW_HZ = 0.1;

/** Upper cutoff of the respiratory bandpass filter (Hz). ~30 breaths/min */
const RESP_HIGH_HZ = 0.5;

/** Minimum seconds of data required for analysis */
const MIN_DURATION_SEC = 10;

/** If no peaks detected for this many seconds, classify as "holding" */
const HOLD_THRESHOLD_SEC = 5;

// ── Bandpass filter ─────────────────────────────────────────────────────────

/**
 * Simple second-order IIR bandpass filter using cascaded high-pass + low-pass.
 * Both are first-order IIR filters applied forward then backward (zero-phase).
 */
function bandpassFilter(
  signal: Float32Array,
  fs: number,
  lowHz: number,
  highHz: number,
): Float32Array {
  // High-pass (removes below lowHz)
  const hp = highPassIIR(signal, fs, lowHz);
  // Low-pass (removes above highHz)
  const lp = lowPassIIR(hp, fs, highHz);
  // Apply in reverse for zero-phase
  lp.reverse();
  const hp2 = highPassIIR(lp, fs, lowHz);
  const lp2 = lowPassIIR(hp2, fs, highHz);
  lp2.reverse();
  return lp2;
}

/** First-order IIR high-pass filter */
function highPassIIR(signal: Float32Array, fs: number, cutoffHz: number): Float32Array {
  const dt = 1 / fs;
  const rc = 1 / (2 * Math.PI * cutoffHz);
  const alpha = rc / (rc + dt);
  const out = new Float32Array(signal.length);
  out[0] = signal[0];
  for (let i = 1; i < signal.length; i++) {
    out[i] = alpha * (out[i - 1] + signal[i] - signal[i - 1]);
  }
  return out;
}

/** First-order IIR low-pass filter */
function lowPassIIR(signal: Float32Array, fs: number, cutoffHz: number): Float32Array {
  const dt = 1 / fs;
  const rc = 1 / (2 * Math.PI * cutoffHz);
  const alpha = dt / (rc + dt);
  const out = new Float32Array(signal.length);
  out[0] = signal[0];
  for (let i = 1; i < signal.length; i++) {
    out[i] = alpha * signal[i] + (1 - alpha) * out[i - 1];
  }
  return out;
}

// ── Peak detection ──────────────────────────────────────────────────────────

/**
 * Find local maxima in a signal, with a minimum inter-peak distance.
 * Returns an array of sample indices where peaks occur.
 */
function findPeaks(signal: Float32Array, minDistanceSamples: number): number[] {
  const peaks: number[] = [];

  // Compute a threshold: peaks must exceed some fraction of the signal's range
  // to filter out noise ripples
  let maxVal = -Infinity;
  let minVal = Infinity;
  for (let i = 0; i < signal.length; i++) {
    if (signal[i] > maxVal) maxVal = signal[i];
    if (signal[i] < minVal) minVal = signal[i];
  }
  const range = maxVal - minVal;
  const threshold = minVal + range * 0.3; // peak must be above 30% of range

  for (let i = 1; i < signal.length - 1; i++) {
    if (
      signal[i] > signal[i - 1] &&
      signal[i] > signal[i + 1] &&
      signal[i] >= threshold
    ) {
      // Enforce minimum distance from last peak
      if (peaks.length === 0 || i - peaks[peaks.length - 1] >= minDistanceSamples) {
        peaks.push(i);
      }
    }
  }

  return peaks;
}

// ── State mapping ───────────────────────────────────────────────────────────

const STATE_MESSAGES: Record<BreathingState, string> = {
  deep_slow: "Deep, slow breathing detected -- great for relaxation",
  normal: "Normal breathing rhythm",
  shallow_fast: "Shallow, fast breathing -- try slowing down",
  holding: "Breath hold detected",
  unknown: "Breathing pattern unclear -- keep still for better detection",
};

// ── Main detection function ─────────────────────────────────────────────────

/**
 * Detect breathing state from a raw frontal EEG channel (AF7 or AF8).
 *
 * @param samples - Raw EEG signal in microvolts (Float32Array)
 * @param fs - Sampling rate in Hz (256 for Muse 2)
 * @returns BreathingAnalysis with state, rate, coherence, and message
 */
export function detectBreathingState(
  samples: Float32Array,
  fs: number,
): BreathingAnalysis {
  const durationSec = samples.length / fs;

  // Not enough data for reliable analysis
  if (durationSec < MIN_DURATION_SEC) {
    return {
      state: "unknown",
      estimatedRate: 0,
      coherence: 0,
      message: STATE_MESSAGES.unknown,
    };
  }

  // Step 1: Bandpass filter into respiratory band (0.1-0.5 Hz)
  const filtered = bandpassFilter(samples, fs, RESP_LOW_HZ, RESP_HIGH_HZ);

  // Step 2: Find peaks. Minimum distance between peaks corresponds to
  // the maximum breathing rate (0.5 Hz = 2 seconds between breaths)
  const minPeakDistanceSec = 1 / RESP_HIGH_HZ; // 2 seconds
  const minPeakDistanceSamples = Math.floor(minPeakDistanceSec * fs);
  const peaks = findPeaks(filtered, minPeakDistanceSamples);

  // Step 3: Handle no-peaks case (breath holding or flat signal)
  if (peaks.length < 2) {
    return {
      state: durationSec >= HOLD_THRESHOLD_SEC ? "holding" : "unknown",
      estimatedRate: 0,
      coherence: 0,
      message: durationSec >= HOLD_THRESHOLD_SEC ? STATE_MESSAGES.holding : STATE_MESSAGES.unknown,
    };
  }

  // Step 4: Compute breathing rate from inter-peak intervals
  const ipis: number[] = []; // inter-peak intervals in seconds
  for (let i = 1; i < peaks.length; i++) {
    ipis.push((peaks[i] - peaks[i - 1]) / fs);
  }

  const meanIpi = ipis.reduce((a, b) => a + b, 0) / ipis.length;
  const estimatedRate = meanIpi > 0 ? 60 / meanIpi : 0; // breaths per minute

  // Step 5: Compute coherence (1 - coefficient of variation of IPIs)
  // CV = 0 for perfectly regular, higher for irregular
  let coherence = 0;
  if (ipis.length > 0 && meanIpi > 0) {
    const variance = ipis.reduce((acc, v) => acc + (v - meanIpi) ** 2, 0) / ipis.length;
    const cv = Math.sqrt(variance) / meanIpi;
    coherence = Math.max(0, Math.min(1, 1 - cv));
  }

  // Step 6: Classify state
  let state: BreathingState;
  if (estimatedRate < 8 && coherence >= 0.5) {
    state = "deep_slow";
  } else if (estimatedRate > 16) {
    state = "shallow_fast";
  } else if (estimatedRate >= 8 && estimatedRate <= 16) {
    state = "normal";
  } else {
    // Rate < 8 but low coherence -- unclear pattern
    state = "unknown";
  }

  return {
    state,
    estimatedRate,
    coherence,
    message: STATE_MESSAGES[state],
  };
}

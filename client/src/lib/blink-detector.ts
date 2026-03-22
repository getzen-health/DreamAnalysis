/**
 * blink-detector.ts — Eye blink artifact extraction as attention/alertness signal.
 *
 * Eye blink artifacts in frontal EEG (AF7/AF8) are a validated biomarker for
 * drowsiness and attention. Blink morphology predicts alertness without requiring
 * clean EEG — blinks are the artifact, not the noise.
 *
 * Detection algorithm (MED-inspired):
 *   1. High-pass filter at 1 Hz to remove DC drift
 *   2. Detect peaks exceeding 75 uV with duration 100-400ms
 *   3. Minimum inter-blink interval: 200ms (avoid double-counting)
 *
 * Alertness mapping:
 *   <10 blinks/min + high alpha  → "focused" (deep concentration)
 *   <10 blinks/min + high theta  → "drowsy" (falling asleep)
 *   10-20 blinks/min             → "normal"
 *   >25 blinks/min               → "fatigued" (suggest break)
 *
 * @see Issue #533
 */

// ── Types ───────────────────────────────────────────────────────────────────

export interface BlinkStats {
  /** Blinks detected per minute */
  blinksPerMinute: number;
  /** Average blink duration in milliseconds */
  avgBlinkDuration: number;
  /** Coefficient of variation of inter-blink intervals (0 = perfectly regular) */
  interBlinkVariability: number;
  /** Current alertness state derived from blink rate + alpha power */
  alertnessState: "focused" | "normal" | "fatigued" | "drowsy";
  /** True if the user should be prompted to take a break */
  shouldSuggestBreak: boolean;
}

// ── Constants ───────────────────────────────────────────────────────────────

/** Minimum blink amplitude in microvolts (after high-pass filter) */
const BLINK_THRESHOLD_UV = 75;

/** Minimum blink duration in milliseconds */
const MIN_BLINK_DURATION_MS = 100;

/** Maximum blink duration in milliseconds */
const MAX_BLINK_DURATION_MS = 400;

/** Minimum inter-blink interval in milliseconds (avoid double-counting) */
const MIN_IBI_MS = 200;

/** Alpha power threshold for distinguishing focused vs drowsy at low blink rate */
const ALPHA_FOCUSED_THRESHOLD = 0.15;

// ── High-pass filter ────────────────────────────────────────────────────────

/**
 * Simple first-order IIR high-pass filter at the given cutoff frequency.
 * Removes DC drift and slow baseline wander so blink peaks stand out.
 */
function highPassFilter(signal: Float32Array, fs: number, cutoffHz: number = 1): Float32Array {
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

// ── Blink detection ─────────────────────────────────────────────────────────

/**
 * Measure the width of an artifact at one-third of peak amplitude.
 * Returns the duration in samples. Uses 1/3 rather than 1/2 of peak because
 * the high-pass filter narrows blink morphology in the time domain —
 * measuring at a lower fraction captures the true artifact extent more
 * accurately after filtering.
 */
function measureBlinkWidth(abs: Float32Array, peakIdx: number): number {
  const halfMax = abs[peakIdx] / 3;

  // Search left for half-max crossing
  let left = peakIdx;
  while (left > 0 && abs[left] >= halfMax) left--;

  // Search right for half-max crossing
  let right = peakIdx;
  while (right < abs.length - 1 && abs[right] >= halfMax) right++;

  return right - left;
}

/**
 * Detect eye blink artifacts in a raw EEG signal.
 *
 * Uses a peak-finding approach: finds local maxima above the amplitude threshold
 * and then measures the full-width-at-half-maximum (FWHM) to validate duration.
 * This is more robust than pure threshold crossing because the high-pass filter
 * reshapes blink morphology.
 *
 * Works on a single channel (AF7 or AF8 recommended — frontal electrodes
 * are closest to the eyes and capture blink artifacts most strongly).
 *
 * @param samples - Raw EEG signal in microvolts (Float32Array)
 * @param fs - Sampling rate in Hz (256 for Muse 2)
 * @returns Array of sample indices where blink onsets were detected
 */
export function detectBlinks(samples: Float32Array, fs: number): number[] {
  if (samples.length < fs * 0.5) return []; // need at least 0.5s of data

  // Step 1: High-pass filter to remove DC drift
  const filtered = highPassFilter(samples, fs, 1);

  // Step 2: Take absolute value — blinks can be positive or negative depending on reference
  const rawAbs = new Float32Array(filtered.length);
  for (let i = 0; i < filtered.length; i++) {
    rawAbs[i] = Math.abs(filtered[i]);
  }

  // Step 3: Smooth with a short moving average to suppress oscillations from
  // the underlying EEG that create false local maxima within a blink envelope
  const smoothWindow = Math.max(3, Math.floor(fs * 0.02)); // ~20ms window
  const abs = new Float32Array(filtered.length);
  for (let i = 0; i < filtered.length; i++) {
    let sum = 0;
    let count = 0;
    const start = Math.max(0, i - Math.floor(smoothWindow / 2));
    const end = Math.min(filtered.length, i + Math.floor(smoothWindow / 2) + 1);
    for (let j = start; j < end; j++) {
      sum += rawAbs[j];
      count++;
    }
    abs[i] = sum / count;
  }

  // Step 4: Find local maxima above threshold, validate by FWHM duration
  const minDurationSamples = Math.floor((MIN_BLINK_DURATION_MS / 1000) * fs);
  const maxDurationSamples = Math.floor((MAX_BLINK_DURATION_MS / 1000) * fs);
  const minIbiSamples = Math.floor((MIN_IBI_MS / 1000) * fs);

  const blinkOnsets: number[] = [];
  let lastBlinkPeak = -minIbiSamples * 2;

  for (let i = 1; i < abs.length - 1; i++) {
    // Local maximum above threshold
    if (abs[i] >= BLINK_THRESHOLD_UV && abs[i] >= abs[i - 1] && abs[i] >= abs[i + 1]) {
      // Check inter-blink interval
      if (i - lastBlinkPeak < minIbiSamples) continue;

      // Measure blink width at half-maximum
      const fwhm = measureBlinkWidth(abs, i);

      // Validate duration
      if (fwhm >= minDurationSamples && fwhm <= maxDurationSamples) {
        blinkOnsets.push(i);
        lastBlinkPeak = i;
      }
    }
  }

  return blinkOnsets;
}

// ── Blink stats computation ─────────────────────────────────────────────────

/**
 * Compute blink statistics and alertness state from raw EEG.
 *
 * @param samples - Raw EEG signal in microvolts (Float32Array)
 * @param fs - Sampling rate in Hz
 * @param alphaPower - Relative alpha power (0-1) from band power extraction.
 *                     Used to disambiguate focused vs drowsy at low blink rates.
 * @returns BlinkStats with blink rate, duration, variability, and alertness state
 */
export function computeBlinkStats(
  samples: Float32Array,
  fs: number,
  alphaPower: number,
): BlinkStats {
  const onsets = detectBlinks(samples, fs);
  const durationSec = samples.length / fs;
  const durationMin = durationSec / 60;

  // Blinks per minute
  const blinksPerMinute = durationMin > 0 ? onsets.length / durationMin : 0;

  // Average blink duration: measure FWHM at each detected blink peak
  const filtered = highPassFilter(samples, fs, 1);
  const abs = new Float32Array(filtered.length);
  for (let i = 0; i < filtered.length; i++) {
    abs[i] = Math.abs(filtered[i]);
  }

  let totalDurationMs = 0;
  for (const peakIdx of onsets) {
    const fwhm = measureBlinkWidth(abs, peakIdx);
    totalDurationMs += (fwhm / fs) * 1000;
  }
  const avgBlinkDuration = onsets.length > 0 ? totalDurationMs / onsets.length : 0;

  // Inter-blink interval variability (coefficient of variation)
  let interBlinkVariability = 0;
  if (onsets.length > 1) {
    const ibis: number[] = [];
    for (let j = 1; j < onsets.length; j++) {
      ibis.push((onsets[j] - onsets[j - 1]) / fs);
    }
    const mean = ibis.reduce((a, b) => a + b, 0) / ibis.length;
    if (mean > 0) {
      const variance = ibis.reduce((acc, v) => acc + (v - mean) ** 2, 0) / ibis.length;
      interBlinkVariability = Math.sqrt(variance) / mean;
    }
  }

  // Alertness state classification
  let alertnessState: BlinkStats["alertnessState"];
  let shouldSuggestBreak: boolean;

  if (blinksPerMinute > 25) {
    alertnessState = "fatigued";
    shouldSuggestBreak = true;
  } else if (blinksPerMinute < 10) {
    // Low blink rate: use alpha power to disambiguate focused vs drowsy
    if (alphaPower >= ALPHA_FOCUSED_THRESHOLD) {
      alertnessState = "focused";
      shouldSuggestBreak = false;
    } else {
      alertnessState = "drowsy";
      shouldSuggestBreak = true;
    }
  } else {
    // 10-25 blinks/min
    if (blinksPerMinute <= 20) {
      alertnessState = "normal";
    } else {
      // 20-25: still normal range but on the higher side
      alertnessState = "normal";
    }
    shouldSuggestBreak = false;
  }

  return {
    blinksPerMinute,
    avgBlinkDuration,
    interBlinkVariability,
    alertnessState,
    shouldSuggestBreak,
  };
}

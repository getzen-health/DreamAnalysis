/**
 * Brain Age Estimation — a pure computation module (no UI).
 *
 * Estimates "brain age" from EEG spectral features using a simplified
 * heuristic model based on published literature (Paris-Saclay 5,200-subject
 * Muse S study and related alpha-peak-frequency research).
 *
 * This is NOT a trained ML model. It uses linear regression approximations
 * from published correlations. The disclaimer must always be shown.
 */

// ── Types ────────────────────────────────────────────────────────────────────

export interface BrainAgeResult {
  /** Estimated brain age in years */
  estimatedAge: number;
  /** estimatedAge - actualAge. Negative = younger brain. */
  brainAgeGap: number;
  /** 0-1, based on signal quality heuristic */
  confidence: number;
  /** Individual alpha frequency used for estimation */
  alphaPeakHz: number;
  /** Plain-English explanation of each contributing factor */
  factors: {
    alphaPeak: string;
    thetaPower: string;
    betaRatio: string;
  };
  /** Wellness disclaimer — always display this */
  disclaimer: string;
}

export interface BrainAgeInput {
  alphaPeakHz: number;
  thetaPower: number;   // relative, 0-1
  alphaPower: number;   // relative, 0-1
  betaPower: number;    // relative, 0-1
  spectralEntropy: number; // normalized, 0-1
  actualAge: number;
}

// ── Constants ────────────────────────────────────────────────────────────────

const DISCLAIMER =
  "Wellness estimate only. This is a heuristic approximation based on population-level " +
  "EEG correlations, not a clinical brain age measurement. It is not a medical diagnosis " +
  "and should not be used for clinical decisions.";

const MIN_AGE = 15;
const MAX_AGE = 90;
const BRAIN_AGE_KEY = "ndw_brain_age";
const ACTUAL_AGE_KEY = "ndw_actual_age";

// ── Core Estimation ──────────────────────────────────────────────────────────

/**
 * Estimate brain age from EEG spectral features.
 *
 * Uses a linear regression heuristic based on published correlations:
 * - Alpha peak frequency (IAF): primary predictor (r = -0.45 with age).
 *   Regression: base_age ~ 110 - 8 * IAF
 * - Theta/alpha ratio: increases with age. Adds +2 years per 0.1 above 0.5.
 * - Beta/alpha ratio: increases with cognitive decline. Adds +1 year per 0.1 above 0.8.
 * - Spectral entropy: increases with age. Adds +1 year per 0.05 above 0.7.
 *
 * Result clamped to 15-90.
 */
export function estimateBrainAge(
  alphaPeakHz: number,
  thetaPower: number,
  alphaPower: number,
  betaPower: number,
  spectralEntropy: number,
): number {
  // Base age from alpha peak frequency
  let age = 110 - 8 * alphaPeakHz;

  // Theta/alpha ratio adjustment
  const thetaAlphaRatio = alphaPower > 0 ? thetaPower / alphaPower : 0;
  if (thetaAlphaRatio > 0.5) {
    age += ((thetaAlphaRatio - 0.5) / 0.1) * 2;
  }

  // Beta/alpha ratio adjustment
  const betaAlphaRatio = alphaPower > 0 ? betaPower / alphaPower : 0;
  if (betaAlphaRatio > 0.8) {
    age += ((betaAlphaRatio - 0.8) / 0.1) * 1;
  }

  // Spectral entropy adjustment
  if (spectralEntropy > 0.7) {
    age += ((spectralEntropy - 0.7) / 0.05) * 1;
  }

  return Math.round(Math.max(MIN_AGE, Math.min(MAX_AGE, age)));
}

// ── IAF Computation ──────────────────────────────────────────────────────────

/**
 * Find the Individual Alpha Frequency (IAF) — the peak frequency in the
 * 7-13 Hz alpha range from a power spectrum.
 *
 * @param psd   Power spectral density array (one-sided)
 * @param freqs Corresponding frequency values in Hz
 * @returns Peak frequency in Hz within 7-13 Hz, or 10 Hz default
 */
export function computeIAF(psd: number[], freqs: number[]): number {
  let maxPower = -Infinity;
  let peakFreq = 10; // default fallback

  for (let i = 0; i < freqs.length; i++) {
    if (freqs[i] >= 7 && freqs[i] <= 13) {
      if (psd[i] > maxPower) {
        maxPower = psd[i];
        peakFreq = freqs[i];
      }
    }
  }

  return peakFreq;
}

// ── High-Level Result Builder ────────────────────────────────────────────────

/**
 * Compute a full BrainAgeResult from EEG features and user's actual age.
 */
export function getBrainAgeResult(input: BrainAgeInput): BrainAgeResult {
  const { alphaPeakHz, thetaPower, alphaPower, betaPower, spectralEntropy, actualAge } = input;

  const estimatedAge = estimateBrainAge(alphaPeakHz, thetaPower, alphaPower, betaPower, spectralEntropy);
  const brainAgeGap = estimatedAge - actualAge;

  // Simple confidence heuristic: higher if IAF is in a physiologically normal range
  // and alpha power is reasonable (not noise-dominated)
  let confidence = 0.5;
  if (alphaPeakHz >= 8 && alphaPeakHz <= 12) confidence += 0.2;
  if (alphaPower >= 0.15 && alphaPower <= 0.60) confidence += 0.15;
  if (spectralEntropy >= 0.3 && spectralEntropy <= 0.9) confidence += 0.15;
  confidence = Math.min(1, Math.max(0, confidence));

  // Age range descriptors based on IAF
  const iafLow = Math.max(15, Math.round(110 - 8 * (alphaPeakHz + 0.5)));
  const iafHigh = Math.max(15, Math.round(110 - 8 * (alphaPeakHz - 0.5)));

  const factors = {
    alphaPeak: `Your alpha peak of ${alphaPeakHz.toFixed(1)} Hz is typical for age ${iafLow}-${iafHigh}`,
    thetaPower: thetaPower > 0.25
      ? `Elevated theta power (${(thetaPower * 100).toFixed(0)}%) adds to estimated age`
      : `Theta power (${(thetaPower * 100).toFixed(0)}%) is within normal range`,
    betaRatio: betaPower / Math.max(alphaPower, 0.001) > 0.8
      ? `High beta/alpha ratio suggests increased cognitive effort or age-related changes`
      : `Beta/alpha ratio is within normal range`,
  };

  return {
    estimatedAge,
    brainAgeGap,
    confidence,
    alphaPeakHz,
    factors,
    disclaimer: DISCLAIMER,
  };
}

// ── Session Integration ──────────────────────────────────────────────────────

/**
 * Compute brain age from EEG session band powers and cache to localStorage.
 *
 * Called after an EEG session completes. Requires the user's actual age
 * to be stored in localStorage (ndw_actual_age). If not set, does nothing.
 *
 * @param bandPowers  Relative band powers from EEG analysis (keys: theta, alpha, beta)
 * @param spectralEntropy  Normalized spectral entropy (0-1). Defaults to 0.65 if unavailable.
 * @param iaf  Individual alpha frequency in Hz. Defaults to 10 if unavailable.
 */
export function computeAndCacheBrainAge(
  bandPowers: Record<string, number>,
  spectralEntropy: number = 0.65,
  iaf: number = 10,
): BrainAgeResult | null {
  let actualAge: number | null = null;
  try {
    const raw = localStorage.getItem(ACTUAL_AGE_KEY);
    if (raw) {
      const parsed = parseInt(raw, 10);
      if (!isNaN(parsed)) actualAge = parsed;
    }
  } catch {
    return null;
  }

  if (actualAge === null) return null;

  const thetaPower = bandPowers.theta ?? 0;
  const alphaPower = bandPowers.alpha ?? 0;
  const betaPower = bandPowers.beta ?? 0;

  const result = getBrainAgeResult({
    alphaPeakHz: iaf,
    thetaPower,
    alphaPower,
    betaPower,
    spectralEntropy,
    actualAge,
  });

  try {
    localStorage.setItem(
      BRAIN_AGE_KEY,
      JSON.stringify({ ...result, timestamp: Date.now() }),
    );
  } catch {
    // storage full or unavailable
  }

  return result;
}

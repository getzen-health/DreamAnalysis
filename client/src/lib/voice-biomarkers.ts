/**
 * Voice Biomarker Extraction — acoustic feature analysis from raw PCM audio.
 *
 * Extracts wellness-oriented voice features: jitter, shimmer, HNR,
 * pitch (F0), speech rate, spectral tilt, pause analysis, and a composite
 * vocal wellness score. ALL language is wellness-framed — no clinical
 * terminology.
 *
 * HNR (Harmonic-to-Noise Ratio) measures voice clarity vs breathiness.
 * Part of the eGeMAPS standard (Eyben et al. 2016, IEEE Trans. Affective
 * Computing). Healthy voice ~20 dB; hoarse/fatigued voice significantly
 * lower. Cross-cultural depression study using logHNR achieved 0.934 AUC.
 *
 * This module runs entirely on-device with zero network calls.
 */

// ─── Types ──────────────────────────────────────────────────────────────────

export type VocalEnergyLevel = "low" | "moderate" | "high";
export type VocalVariability = "steady" | "moderate" | "dynamic";

export interface VoiceBiomarkers {
  /** Average absolute jitter (ratio, 0-1). Lower = more stable voice. */
  jitter: number;
  /** Average absolute shimmer (ratio, 0-1). Lower = more stable amplitude. */
  shimmer: number;
  /** Harmonic-to-Noise Ratio in dB. Higher = clearer voice. Healthy ~20 dB. */
  hnr: number;
  /** Mean fundamental frequency in Hz. */
  pitchMean: number;
  /** Standard deviation of F0 across frames. */
  pitchStd: number;
  /** Estimated syllables per second. */
  speechRate: number;
  /** Spectral tilt (dB/octave). Negative = energy concentrated in lower frequencies. */
  spectralTilt: number;
  /** Average pause duration in seconds. 0 if no pauses detected. */
  averagePauseDuration: number;
  /** Number of pauses detected. */
  pauseCount: number;
  /** Vocal energy level category. */
  vocalEnergyLevel: VocalEnergyLevel;
  /** Vocal variability category. */
  vocalVariability: VocalVariability;
  /** Composite vocal wellness score 0-100. */
  wellnessScore: number;
  /** Timestamp of extraction (epoch ms). */
  timestamp: number;
  /** Wellness disclaimer. */
  disclaimer: string;
}

// ─── Constants ──────────────────────────────────────────────────────────────

const MIN_F0_HZ = 80;
const MAX_F0_HZ = 400;
const FRAME_DURATION_SEC = 0.03; // 30ms frames
const FRAME_HOP_SEC = 0.015; // 15ms hop (50% overlap)
const SILENCE_THRESHOLD_RMS = 0.01;
const MIN_PAUSE_DURATION_SEC = 0.2;

const DISCLAIMER =
  "Voice wellness insights are for personal awareness only. They reflect acoustic patterns in your voice and are not a medical or clinical assessment.";

// ─── Helpers ────────────────────────────────────────────────────────────────

/**
 * Autocorrelation-based pitch detection for a single frame.
 * Returns the detected period in samples, or 0 if no pitch found.
 */
function detectPitchPeriod(
  frame: Float32Array,
  sr: number,
): number {
  const minLag = Math.floor(sr / MAX_F0_HZ);
  const maxLag = Math.ceil(sr / MIN_F0_HZ);
  const n = frame.length;

  if (maxLag >= n) return 0;

  // Compute autocorrelation for lags in the valid F0 range
  let bestLag = 0;
  let bestCorr = -Infinity;

  for (let lag = minLag; lag <= maxLag && lag < n; lag++) {
    let sum = 0;
    let norm1 = 0;
    let norm2 = 0;
    const count = n - lag;
    for (let i = 0; i < count; i++) {
      sum += frame[i] * frame[i + lag];
      norm1 += frame[i] * frame[i];
      norm2 += frame[i + lag] * frame[i + lag];
    }
    const denom = Math.sqrt(norm1 * norm2);
    const corr = denom > 0 ? sum / denom : 0;

    if (corr > bestCorr) {
      bestCorr = corr;
      bestLag = lag;
    }
  }

  // Require minimum correlation to accept the pitch
  if (bestCorr < 0.3) return 0;

  return bestLag;
}

/**
 * Compute RMS energy of a signal segment.
 */
function rms(samples: Float32Array, start: number, end: number): number {
  let sum = 0;
  const len = end - start;
  if (len <= 0) return 0;
  for (let i = start; i < end; i++) {
    sum += samples[i] * samples[i];
  }
  return Math.sqrt(sum / len);
}

/**
 * Find the peak absolute amplitude in a signal segment.
 */
function peakAmplitude(samples: Float32Array, start: number, end: number): number {
  let peak = 0;
  for (let i = start; i < end; i++) {
    const abs = Math.abs(samples[i]);
    if (abs > peak) peak = abs;
  }
  return peak;
}

/**
 * Compute Harmonic-to-Noise Ratio for a single voiced frame.
 *
 * Uses the autocorrelation method (Boersma 1993, as used in Praat):
 *   HNR = 10 * log10(r / (1 - r))
 * where r is the normalized autocorrelation at the pitch period lag.
 *
 * A pure tone yields r close to 1 -> HNR >> 0 dB.
 * Equal harmonic and noise energy yields r = 0.5 -> HNR = 0 dB.
 * No harmonic structure yields r close to 0 -> HNR << 0 dB.
 *
 * @param frame - Audio frame samples
 * @param sr - Sample rate
 * @param pitchPeriod - Detected pitch period in samples (from detectPitchPeriod)
 * @returns HNR in dB, or 0 if computation not possible
 */
function computeFrameHNR(
  frame: Float32Array,
  sr: number,
  pitchPeriod: number,
): number {
  if (pitchPeriod <= 0) return 0;

  const n = frame.length;
  if (pitchPeriod >= n) return 0;

  // Normalized autocorrelation at the pitch period lag
  const count = n - pitchPeriod;
  let sum = 0;
  let norm1 = 0;
  let norm2 = 0;
  for (let i = 0; i < count; i++) {
    sum += frame[i] * frame[i + pitchPeriod];
    norm1 += frame[i] * frame[i];
    norm2 += frame[i + pitchPeriod] * frame[i + pitchPeriod];
  }
  const denom = Math.sqrt(norm1 * norm2);
  if (denom < 1e-10) return 0;

  let r = sum / denom;
  // Clamp to avoid log(0) or log(negative)
  r = Math.max(0.0001, Math.min(r, 0.9999));

  return 10 * Math.log10(r / (1 - r));
}

// ─── Main Extraction ────────────────────────────────────────────────────────

/**
 * Extract voice biomarkers from raw PCM audio.
 *
 * @param samples - Mono Float32Array of PCM samples (range -1 to 1)
 * @param sr - Sample rate in Hz
 * @returns VoiceBiomarkers with all acoustic features and wellness score
 */
export function extractVoiceBiomarkers(
  samples: Float32Array,
  sr: number,
): VoiceBiomarkers {
  const n = samples.length;
  const frameSize = Math.floor(sr * FRAME_DURATION_SEC);
  const hopSize = Math.floor(sr * FRAME_HOP_SEC);

  if (n < frameSize * 2 || sr <= 0) {
    // Not enough audio — return zeroed biomarkers
    return {
      jitter: 0,
      shimmer: 0,
      hnr: 0,
      pitchMean: 0,
      pitchStd: 0,
      speechRate: 0,
      spectralTilt: 0,
      averagePauseDuration: 0,
      pauseCount: 0,
      vocalEnergyLevel: "low",
      vocalVariability: "steady",
      wellnessScore: 50,
      timestamp: Date.now(),
      disclaimer: DISCLAIMER,
    };
  }

  // ── Step 1: Frame-by-frame pitch period detection ───────────────────────

  const pitchPeriods: number[] = [];
  const pitchFreqs: number[] = [];
  const framePeaks: number[] = [];
  const frameRms: number[] = [];
  const frameVoiced: boolean[] = [];
  const frameHnrValues: number[] = [];

  for (let start = 0; start + frameSize <= n; start += hopSize) {
    const frame = samples.subarray(start, start + frameSize);
    const frameEnergy = rms(samples, start, start + frameSize);
    frameRms.push(frameEnergy);

    const period = detectPitchPeriod(frame, sr);
    const isVoiced = period > 0 && frameEnergy > SILENCE_THRESHOLD_RMS;
    frameVoiced.push(isVoiced);

    if (isVoiced) {
      pitchPeriods.push(period);
      pitchFreqs.push(sr / period);
      framePeaks.push(peakAmplitude(samples, start, start + frameSize));
      frameHnrValues.push(computeFrameHNR(frame, sr, period));
    }
  }

  // ── Step 2: Jitter — average absolute difference between consecutive periods ──

  let jitter = 0;
  if (pitchPeriods.length >= 2) {
    let sumDiff = 0;
    let sumPeriod = 0;
    for (let i = 1; i < pitchPeriods.length; i++) {
      sumDiff += Math.abs(pitchPeriods[i] - pitchPeriods[i - 1]);
    }
    for (const p of pitchPeriods) sumPeriod += p;
    const meanPeriod = sumPeriod / pitchPeriods.length;
    const meanDiff = sumDiff / (pitchPeriods.length - 1);
    jitter = meanPeriod > 0 ? meanDiff / meanPeriod : 0;
  }

  // ── Step 3: Shimmer — average absolute difference in peak amplitude per period ──

  let shimmer = 0;
  if (framePeaks.length >= 2) {
    let sumDiff = 0;
    let sumPeak = 0;
    for (let i = 1; i < framePeaks.length; i++) {
      sumDiff += Math.abs(framePeaks[i] - framePeaks[i - 1]);
    }
    for (const p of framePeaks) sumPeak += p;
    const meanPeak = sumPeak / framePeaks.length;
    const meanDiff = sumDiff / (framePeaks.length - 1);
    shimmer = meanPeak > 0 ? meanDiff / meanPeak : 0;
  }

  // ── Step 3.5: HNR — mean Harmonic-to-Noise Ratio across voiced frames ────

  let hnr = 0;
  if (frameHnrValues.length > 0) {
    let hnrSum = 0;
    for (const h of frameHnrValues) hnrSum += h;
    hnr = hnrSum / frameHnrValues.length;
  }

  // ── Step 4: F0 statistics ─────────────────────────────────────────────────

  let pitchMean = 0;
  let pitchStd = 0;
  if (pitchFreqs.length > 0) {
    let sum = 0;
    for (const f of pitchFreqs) sum += f;
    pitchMean = sum / pitchFreqs.length;

    if (pitchFreqs.length > 1) {
      let sumSq = 0;
      for (const f of pitchFreqs) sumSq += (f - pitchMean) ** 2;
      pitchStd = Math.sqrt(sumSq / (pitchFreqs.length - 1));
    }
  }

  // ── Step 5: Speech rate — energy-based syllable detection ─────────────────

  let syllables = 0;
  let wasSilent = true;
  const syllableThreshold = SILENCE_THRESHOLD_RMS * 3;

  for (const energy of frameRms) {
    if (energy > syllableThreshold && wasSilent) {
      syllables++;
      wasSilent = false;
    } else if (energy <= syllableThreshold) {
      wasSilent = true;
    }
  }

  const durationSec = n / sr;
  const speechRate = durationSec > 0 ? syllables / durationSec : 0;

  // ── Step 6: Spectral tilt — linear regression on log-freq vs log-power ────

  let spectralTilt = 0;
  {
    // Compute power spectrum via DFT on the full signal (or first 4096 samples)
    const fftSize = Math.min(4096, n);
    const segment = samples.subarray(0, fftSize);

    // Simple power spectrum using correlation approach
    const numBins = Math.floor(fftSize / 2);
    const freqs: number[] = [];
    const powers: number[] = [];

    for (let k = 1; k <= numBins; k++) {
      const freq = (k * sr) / fftSize;
      if (freq < 50 || freq > sr / 2) continue;

      // DFT at bin k
      let real = 0;
      let imag = 0;
      for (let t = 0; t < fftSize; t++) {
        const angle = (2 * Math.PI * k * t) / fftSize;
        real += segment[t] * Math.cos(angle);
        imag -= segment[t] * Math.sin(angle);
      }
      const power = (real * real + imag * imag) / fftSize;

      if (power > 0 && freq > 0) {
        freqs.push(Math.log2(freq));
        powers.push(Math.log2(power));
      }
    }

    // Linear regression: powers = slope * freqs + intercept
    if (freqs.length >= 2) {
      let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
      const count = freqs.length;
      for (let i = 0; i < count; i++) {
        sumX += freqs[i];
        sumY += powers[i];
        sumXY += freqs[i] * powers[i];
        sumXX += freqs[i] * freqs[i];
      }
      const denom = count * sumXX - sumX * sumX;
      spectralTilt = denom !== 0 ? (count * sumXY - sumX * sumY) / denom : 0;
    }
  }

  // ── Step 7: Pause analysis — detect silence segments ──────────────────────

  const minPauseFrames = Math.ceil(MIN_PAUSE_DURATION_SEC / FRAME_HOP_SEC);
  let pauseCount = 0;
  let totalPauseFrames = 0;
  let currentSilentRun = 0;

  for (const energy of frameRms) {
    if (energy <= SILENCE_THRESHOLD_RMS) {
      currentSilentRun++;
    } else {
      if (currentSilentRun >= minPauseFrames) {
        pauseCount++;
        totalPauseFrames += currentSilentRun;
      }
      currentSilentRun = 0;
    }
  }
  // Check trailing silence
  if (currentSilentRun >= minPauseFrames) {
    pauseCount++;
    totalPauseFrames += currentSilentRun;
  }

  const averagePauseDuration =
    pauseCount > 0 ? (totalPauseFrames * FRAME_HOP_SEC) / pauseCount : 0;

  // ── Step 8: Derived categories ────────────────────────────────────────────

  // Overall RMS energy
  let overallRms = 0;
  {
    let sum = 0;
    for (let i = 0; i < n; i++) sum += samples[i] * samples[i];
    overallRms = Math.sqrt(sum / n);
  }

  const vocalEnergyLevel: VocalEnergyLevel =
    overallRms < 0.03 ? "low" : overallRms < 0.12 ? "moderate" : "high";

  const pitchCV = pitchMean > 0 ? pitchStd / pitchMean : 0;
  const vocalVariability: VocalVariability =
    pitchCV < 0.08 ? "steady" : pitchCV < 0.2 ? "moderate" : "dynamic";

  // ── Step 9: Wellness score (0-100) ────────────────────────────────────────
  //
  // Composed of five 20-point components:
  //   1. Low jitter (vocal stability)        → 20 pts
  //   2. Good pitch variability (expression) → 20 pts
  //   3. Normal speech rate                  → 20 pts
  //   4. Balanced spectral tilt              → 20 pts
  //   5. High HNR (voice clarity)            → 20 pts

  // Component 1: Low jitter → higher score (jitter < 0.02 is excellent, > 0.1 is high)
  const jitterScore = Math.max(0, 20 * (1 - Math.min(1, jitter / 0.1)));

  // Component 2: Good variability — penalize both too steady (< 0.05 CV) and too erratic (> 0.35 CV)
  let variabilityScore: number;
  if (pitchCV < 0.05) {
    variabilityScore = 20 * (pitchCV / 0.05); // Too monotone
  } else if (pitchCV <= 0.25) {
    variabilityScore = 20; // Ideal range
  } else if (pitchCV <= 0.4) {
    variabilityScore = 20 * (1 - (pitchCV - 0.25) / 0.15); // Getting erratic
  } else {
    variabilityScore = 0;
  }

  // Component 3: Normal speech rate (2-5 syllables/sec is typical)
  let rateScore: number;
  if (speechRate < 1) {
    rateScore = 20 * speechRate; // Very slow or silent
  } else if (speechRate <= 5) {
    rateScore = 20; // Normal range
  } else if (speechRate <= 8) {
    rateScore = 20 * (1 - (speechRate - 5) / 3); // Fast
  } else {
    rateScore = 0;
  }

  // Component 4: Balanced spectral tilt (moderate negative tilt is normal for speech)
  // Normal speech tilt is roughly -3 to -6 dB/octave in log-log space
  const absTilt = Math.abs(spectralTilt);
  let tiltScore: number;
  if (absTilt < 1) {
    tiltScore = 16; // Too flat
  } else if (absTilt <= 8) {
    tiltScore = 20; // Normal range
  } else if (absTilt <= 15) {
    tiltScore = 20 * (1 - (absTilt - 8) / 7);
  } else {
    tiltScore = 0;
  }

  // Component 5: HNR — voice clarity (Boersma 1993; eGeMAPS standard)
  // Healthy voice: ~20 dB. Hoarse/breathy: < 10 dB. 0 dB = equal harmonic & noise.
  let hnrScore: number;
  if (hnr <= 0) {
    hnrScore = 0; // No harmonic structure or no voiced frames
  } else if (hnr <= 10) {
    hnrScore = 20 * (hnr / 10) * 0.5; // Low clarity, partial credit up to 10 pts
  } else if (hnr <= 25) {
    hnrScore = 10 + 10 * ((hnr - 10) / 15); // Normal range, scales to full 20
  } else {
    hnrScore = 20; // Excellent clarity
  }

  const wellnessScore = Math.round(
    Math.max(0, Math.min(100, jitterScore + variabilityScore + rateScore + tiltScore + hnrScore)),
  );

  // ── Return ────────────────────────────────────────────────────────────────

  return {
    jitter: isFinite(jitter) ? jitter : 0,
    shimmer: isFinite(shimmer) ? shimmer : 0,
    hnr: isFinite(hnr) ? hnr : 0,
    pitchMean: isFinite(pitchMean) ? pitchMean : 0,
    pitchStd: isFinite(pitchStd) ? pitchStd : 0,
    speechRate: isFinite(speechRate) ? speechRate : 0,
    spectralTilt: isFinite(spectralTilt) ? spectralTilt : 0,
    averagePauseDuration: isFinite(averagePauseDuration) ? averagePauseDuration : 0,
    pauseCount,
    vocalEnergyLevel,
    vocalVariability,
    wellnessScore,
    timestamp: Date.now(),
    disclaimer: DISCLAIMER,
  };
}

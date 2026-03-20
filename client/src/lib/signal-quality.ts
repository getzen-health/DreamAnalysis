/**
 * signal-quality.ts — EEG signal quality assessment for Muse 2 / Muse S.
 *
 * Provides per-electrode quality metrics using:
 *   1. RMS amplitude — detects disconnected electrodes and artifact saturation
 *   2. Spectral flatness — distinguishes real neural signal (peaked) from noise (flat)
 *   3. Alpha reactivity test — gold standard for consumer EEG signal verification
 *
 * Research context:
 *   - Muse's built-in green/yellow/red indicators can show green with poor signal
 *   - Spectral flatness (geometric mean / arithmetic mean of power spectrum) is a
 *     standard measure: 0 = pure tone, 1 = white noise
 *   - Alpha reactivity: closing eyes increases alpha power (8-12 Hz) in real EEG
 *     due to posterior alpha rhythm. If alpha does NOT increase, signal is likely noise.
 *   - Per-electrode feedback is more useful than a single composite indicator
 */

// ── Types ───────────────────────────────────────────────────────────────────

export interface ChannelQuality {
  channel: string;
  status: "good" | "fair" | "poor" | "disconnected";
  amplitudeUv: number;
  spectralFlatness: number;
  hasAlphaReactivity: boolean;
}

export interface SignalQualityResult {
  overall: "good" | "fair" | "poor";
  channels: ChannelQuality[];
  recommendation: string;
  readyForAnalysis: boolean;
}

// ── Amplitude thresholds (microvolts RMS) ───────────────────────────────────
// Calibrated for Muse 2 dry-electrode EEG:
//   < 5 µV   = disconnected (flat line / ADC noise only)
//   5-75 µV  = good (normal EEG amplitude range for dry electrodes)
//   75-150   = fair (possible artifact contamination — muscle, movement)
//   > 150    = poor (heavy artifact — jaw clench, electrode pop, saturation)

const AMP_DISCONNECTED = 5;
const AMP_GOOD_MAX = 75;
const AMP_FAIR_MAX = 150;

// ── Spectral flatness thresholds ────────────────────────────────────────────
// Spectral flatness = geometric mean / arithmetic mean of power spectrum
//   < 0.3  = peaked (real neural signal with dominant frequencies)
//   0.3-0.6 = fair (some signal + noise mix)
//   > 0.6  = flat (noise-like, no clear neural oscillations)

const SF_GOOD_MAX = 0.3;
const SF_FAIR_MAX = 0.6;

// ── Alpha reactivity threshold ──────────────────────────────────────────────
// If alpha power increases by > 20% when eyes close, signal is real.

const ALPHA_REACTIVITY_THRESHOLD = 0.20;

// ── FFT utilities ───────────────────────────────────────────────────────────

/**
 * Cooley-Tukey radix-2 FFT (in-place).
 * Shared implementation matching eeg-features.ts.
 */
function fft(real: Float64Array, imag: Float64Array, n: number): void {
  let j = 0;
  for (let i = 0; i < n; i++) {
    if (i < j) {
      [real[i], real[j]] = [real[j], real[i]];
      [imag[i], imag[j]] = [imag[j], imag[i]];
    }
    let m = n >> 1;
    while (m >= 1 && j >= m) {
      j -= m;
      m >>= 1;
    }
    j += m;
  }
  for (let size = 2; size <= n; size *= 2) {
    const halfSize = size / 2;
    const angle = (-2 * Math.PI) / size;
    for (let i = 0; i < n; i += size) {
      for (let k = 0; k < halfSize; k++) {
        const cos = Math.cos(angle * k);
        const sin = Math.sin(angle * k);
        const idx1 = i + k;
        const idx2 = i + k + halfSize;
        const tReal = cos * real[idx2] - sin * imag[idx2];
        const tImag = sin * real[idx2] + cos * imag[idx2];
        real[idx2] = real[idx1] - tReal;
        imag[idx2] = imag[idx1] - tImag;
        real[idx1] += tReal;
        imag[idx1] += tImag;
      }
    }
  }
}

/**
 * Compute one-sided power spectrum (magnitude squared) of a windowed signal.
 * Returns array of power values for frequencies from 0 to fs/2.
 */
function computePowerSpectrum(samples: Float32Array, fs: number): { freqs: number[]; power: number[] } {
  const n = samples.length;
  const nfft = Math.pow(2, Math.ceil(Math.log2(n)));
  const real = new Float64Array(nfft);
  const imag = new Float64Array(nfft);

  // Apply Hanning window
  for (let i = 0; i < n; i++) {
    const window = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1));
    real[i] = samples[i] * window;
  }

  fft(real, imag, nfft);

  const nFreqs = Math.floor(nfft / 2) + 1;
  const freqs: number[] = new Array(nFreqs);
  const power: number[] = new Array(nFreqs);

  for (let i = 0; i < nFreqs; i++) {
    freqs[i] = (i * fs) / nfft;
    power[i] = real[i] * real[i] + imag[i] * imag[i];
  }

  return { freqs, power };
}

// ── Core assessment functions ───────────────────────────────────────────────

/**
 * Compute RMS amplitude of a signal in microvolts.
 */
function computeRms(samples: Float32Array): number {
  if (samples.length === 0) return 0;
  let sumSq = 0;
  for (let i = 0; i < samples.length; i++) {
    sumSq += samples[i] * samples[i];
  }
  return Math.sqrt(sumSq / samples.length);
}

/**
 * Compute spectral flatness: geometric mean / arithmetic mean of power spectrum.
 * Values near 0 = peaked spectrum (real signal), near 1 = flat (noise).
 *
 * We use log-domain computation to avoid floating-point underflow with many bins:
 *   geometricMean = exp(mean(log(power)))
 *   spectralFlatness = geometricMean / arithmeticMean
 *                    = exp(mean(log(power)) - log(mean(power)))
 */
function computeSpectralFlatness(samples: Float32Array, fs: number): number {
  if (samples.length < 32) return 1; // too few samples — treat as noise

  const { power } = computePowerSpectrum(samples, fs);

  // Only use bins in the physiological range (1-50 Hz) to avoid DC and Nyquist artifacts
  const minBin = Math.max(1, Math.floor((1 * power.length * 2) / fs));
  const maxBin = Math.min(power.length - 1, Math.floor((50 * power.length * 2) / fs));

  if (maxBin <= minBin) return 1;

  let logSum = 0;
  let linSum = 0;
  let count = 0;

  for (let i = minBin; i <= maxBin; i++) {
    const p = Math.max(power[i], 1e-20); // floor to avoid log(0)
    logSum += Math.log(p);
    linSum += p;
    count++;
  }

  if (count === 0 || linSum === 0) return 1;

  const logGeometricMean = logSum / count;
  const logArithmeticMean = Math.log(linSum / count);

  // spectralFlatness = exp(logGeometricMean - logArithmeticMean)
  const sf = Math.exp(logGeometricMean - logArithmeticMean);

  return Math.max(0, Math.min(1, sf));
}

/**
 * Compute alpha band (8-12 Hz) power from a signal.
 * Returns the sum of power spectrum bins in the alpha range.
 */
function computeAlphaPower(samples: Float32Array, fs: number): number {
  if (samples.length < 32) return 0;

  const { freqs, power } = computePowerSpectrum(samples, fs);

  let alphaPower = 0;
  for (let i = 0; i < freqs.length; i++) {
    if (freqs[i] >= 8 && freqs[i] <= 12) {
      alphaPower += power[i];
    }
  }

  return alphaPower;
}

/**
 * Determine amplitude status from RMS value.
 */
function amplitudeStatus(rms: number): "good" | "fair" | "poor" | "disconnected" {
  if (rms < AMP_DISCONNECTED) return "disconnected";
  if (rms <= AMP_GOOD_MAX) return "good";
  if (rms <= AMP_FAIR_MAX) return "fair";
  return "poor";
}

/**
 * Determine spectral flatness status.
 */
function flatnessStatus(sf: number): "good" | "fair" | "poor" {
  if (sf < SF_GOOD_MAX) return "good";
  if (sf < SF_FAIR_MAX) return "fair";
  return "poor";
}

/**
 * Combine amplitude and spectral flatness into a single channel status.
 * Both metrics must be "good" for overall "good". The worse status wins.
 */
function combineStatus(
  ampStatus: "good" | "fair" | "poor" | "disconnected",
  sfStatus: "good" | "fair" | "poor",
): "good" | "fair" | "poor" | "disconnected" {
  if (ampStatus === "disconnected") return "disconnected";

  const rank = { good: 0, fair: 1, poor: 2 } as const;
  const ampRank = rank[ampStatus as "good" | "fair" | "poor"];
  const sfRank = rank[sfStatus];
  const worst = Math.max(ampRank, sfRank);

  if (worst === 0) return "good";
  if (worst === 1) return "fair";
  return "poor";
}

// ── Public API ──────────────────────────────────────────────────────────────

/**
 * Assess signal quality for a single EEG channel.
 *
 * @param samples - Raw EEG samples in microvolts (Float32Array)
 * @param fs - Sampling rate in Hz (256 for Muse 2)
 * @param channelName - Electrode name (e.g., "TP9", "AF7", "AF8", "TP10")
 */
export function assessChannelQuality(
  samples: Float32Array,
  fs: number,
  channelName: string,
): ChannelQuality {
  const rms = computeRms(samples);
  const sf = computeSpectralFlatness(samples, fs);
  const ampStat = amplitudeStatus(rms);
  const sfStat = flatnessStatus(sf);
  const status = combineStatus(ampStat, sfStat);

  return {
    channel: channelName,
    status,
    amplitudeUv: rms,
    spectralFlatness: sf,
    hasAlphaReactivity: false, // set by runAlphaReactivityTest separately
  };
}

/**
 * Assess signal quality across all EEG channels.
 *
 * @param channels - Array of Float32Array, one per channel (TP9, AF7, AF8, TP10)
 * @param fs - Sampling rate in Hz
 * @param channelNames - Array of electrode names matching channels order
 */
export function assessSignalQuality(
  channels: Float32Array[],
  fs: number,
  channelNames: string[],
): SignalQualityResult {
  const assessed = channels.map((ch, i) =>
    assessChannelQuality(ch, fs, channelNames[i] ?? `CH${i}`),
  );

  // Count channels by status
  const goodCount = assessed.filter((c) => c.status === "good").length;
  const fairOrBetterCount = assessed.filter(
    (c) => c.status === "good" || c.status === "fair",
  ).length;

  // Overall quality: good if 3+ good, fair if 2+ fair-or-better, poor otherwise
  let overall: "good" | "fair" | "poor";
  if (goodCount >= 3) {
    overall = "good";
  } else if (fairOrBetterCount >= 2) {
    overall = "fair";
  } else {
    overall = "poor";
  }

  // Ready for analysis: AF7 and AF8 must both be at least "fair"
  // AF7 and AF8 are the frontal channels needed for FAA (Frontal Alpha Asymmetry)
  const af7Index = channelNames.indexOf("AF7");
  const af8Index = channelNames.indexOf("AF8");

  const af7Ok =
    af7Index >= 0 &&
    (assessed[af7Index].status === "good" || assessed[af7Index].status === "fair");
  const af8Ok =
    af8Index >= 0 &&
    (assessed[af8Index].status === "good" || assessed[af8Index].status === "fair");

  const readyForAnalysis = af7Ok && af8Ok;

  // Generate human-readable recommendation
  const recommendation = generateRecommendation(assessed, overall, readyForAnalysis);

  return {
    overall,
    channels: assessed,
    recommendation,
    readyForAnalysis,
  };
}

/**
 * Run the alpha reactivity test: compare alpha power between eyes-open and eyes-closed.
 *
 * If alpha power increases by > 20% with eyes closed, the signal is real neural EEG.
 * This is the gold standard for consumer EEG signal verification.
 *
 * @param eyesOpenData - Array of Float32Array per channel, recorded with eyes open
 * @param eyesClosedData - Array of Float32Array per channel, recorded with eyes closed
 * @param fs - Sampling rate in Hz
 * @returns true if alpha reactivity is detected (signal is real)
 */
export function runAlphaReactivityTest(
  eyesOpenData: Float32Array[],
  eyesClosedData: Float32Array[],
  fs: number,
): boolean {
  if (eyesOpenData.length === 0 || eyesClosedData.length === 0) return false;

  // Compute average alpha power across all channels for both conditions
  let openAlphaTotal = 0;
  let closedAlphaTotal = 0;
  let validChannels = 0;

  for (let ch = 0; ch < Math.min(eyesOpenData.length, eyesClosedData.length); ch++) {
    const openAlpha = computeAlphaPower(eyesOpenData[ch], fs);
    const closedAlpha = computeAlphaPower(eyesClosedData[ch], fs);

    if (openAlpha > 0 || closedAlpha > 0) {
      openAlphaTotal += openAlpha;
      closedAlphaTotal += closedAlpha;
      validChannels++;
    }
  }

  if (validChannels === 0 || openAlphaTotal === 0) return false;

  // Alpha should increase when eyes close (posterior alpha rhythm)
  const increase = (closedAlphaTotal - openAlphaTotal) / openAlphaTotal;

  return increase > ALPHA_REACTIVITY_THRESHOLD;
}

// ── Recommendation generator ────────────────────────────────────────────────

function generateRecommendation(
  channels: ChannelQuality[],
  overall: "good" | "fair" | "poor",
  readyForAnalysis: boolean,
): string {
  if (overall === "good" && readyForAnalysis) {
    return "Signal looks good! All channels are receiving clean EEG data.";
  }

  const problems: string[] = [];

  const disconnected = channels.filter((c) => c.status === "disconnected");
  if (disconnected.length > 0) {
    const names = disconnected.map((c) => c.channel).join(", ");
    problems.push(`${names}: no signal detected. Check electrode contact.`);
  }

  const noisy = channels.filter((c) => c.spectralFlatness > SF_FAIR_MAX && c.status !== "disconnected");
  if (noisy.length > 0) {
    const names = noisy.map((c) => c.channel).join(", ");
    problems.push(`${names}: noisy signal. Try wetting the electrode pads.`);
  }

  const artifact = channels.filter((c) => c.amplitudeUv > AMP_FAIR_MAX);
  if (artifact.length > 0) {
    const names = artifact.map((c) => c.channel).join(", ");
    problems.push(`${names}: artifact detected. Minimize jaw tension and movement.`);
  }

  if (!readyForAnalysis) {
    problems.push("Frontal channels (AF7/AF8) need better contact for emotion analysis.");
  }

  if (problems.length === 0) {
    return "Signal is fair. Try adjusting the headband for better electrode contact.";
  }

  return problems.join(" ");
}

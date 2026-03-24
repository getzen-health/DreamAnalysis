/**
 * Browser-side EEG feature extraction.
 *
 * TypeScript port of key functions from Python eeg_processor.py.
 * Used for local (edge) inference when ONNX models are loaded in the browser.
 */

const BANDS: Record<string, [number, number]> = {
  delta: [0.5, 4.0],
  theta: [4.0, 8.0],
  alpha: [8.0, 12.0],
  beta: [12.0, 30.0],
  gamma: [30.0, 50.0],  // cap at 50 Hz — Muse 2 hardware bandwidth limit; above 50 Hz is aliasing
};

/**
 * Spectral band importance weights for emotion-relevant feature extraction.
 *
 * Neuroscience rationale (Muse 2 @ AF7/AF8/TP9/TP10):
 *   - Alpha (8-12 Hz): PRIMARY emotion band. Inversely correlated with cortical
 *     arousal; Frontal Alpha Asymmetry (FAA) is the most validated valence marker
 *     (Davidson 1992, 30+ years of replication). Weight: 1.5x
 *   - Theta (4-8 Hz): Meditation depth, creativity, memory encoding, drowsiness.
 *     Frontal Midline Theta (FMT) complements FAA for valence. Weight: 1.3x
 *   - Beta (12-30 Hz): Active cognition, focus, stress/anxiety when elevated.
 *     Well-validated but noisier than alpha on consumer-grade EEG. Weight: 1.0x
 *   - Delta (0.5-4 Hz): Deep sleep marker, minimal emotion relevance during
 *     waking state. Often contaminated by movement artifacts. Weight: 0.8x
 *   - Gamma (30-50 Hz): At AF7/AF8, gamma is predominantly frontalis EMG noise
 *     (jaw clenching, forehead tensing), NOT neural gamma. Including it at full
 *     weight inflates arousal/stress whenever user makes facial movements.
 *     Weight: 0.3x (strongly suppressed)
 *
 * Applied during band power extraction: raw band power is multiplied by the
 * weight before normalization. This amplifies emotion-carrying frequencies
 * and suppresses noise-dominated bands, improving signal-to-noise ratio.
 *
 * References:
 *   - Davidson (1992): FAA and approach/withdrawal motivation
 *   - Zheng & Lu (2015): DE features, alpha/theta most discriminative for emotion
 *   - Badolato et al. (2024): Muse 2 validation, gamma EMG contamination
 */
export const BAND_IMPORTANCE_WEIGHTS: Record<string, number> = {
  delta: 0.8,
  theta: 1.3,
  alpha: 1.5,
  beta: 1.0,
  gamma: 0.3,
};

/**
 * Simple FIR low-pass filter via convolution with a Hanning-window sinc kernel.
 * Cuts frequencies above cutoffHz to match the Muse 2 hardware bandwidth (~50 Hz)
 * and reduce high-frequency noise inflation in band-power estimates.
 */
function lowPassFilter(signal: number[], fs: number, cutoffHz = 50): number[] {
  const numTaps = 31; // odd number → symmetric, linear phase
  const fc = cutoffHz / fs;
  const kernel: number[] = new Array(numTaps);
  const mid = Math.floor(numTaps / 2);
  let kernelSum = 0;
  for (let i = 0; i < numTaps; i++) {
    const n = i - mid;
    const sinc = n === 0 ? 2 * fc : Math.sin(2 * Math.PI * fc * n) / (Math.PI * n);
    const window = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (numTaps - 1));
    kernel[i] = sinc * window;
    kernelSum += kernel[i];
  }
  // Normalise so DC gain = 1
  for (let i = 0; i < numTaps; i++) kernel[i] /= kernelSum;

  const out: number[] = new Array(signal.length).fill(0);
  for (let i = 0; i < signal.length; i++) {
    let acc = 0;
    for (let k = 0; k < numTaps; k++) {
      const si = i - k + mid;
      if (si >= 0 && si < signal.length) acc += kernel[k] * signal[si];
    }
    out[i] = acc;
  }
  return out;
}

/**
 * Compute the power spectral density using a simplified periodogram (FFT).
 */
function computePSD(signal: number[], fs: number): { freqs: number[]; psd: number[] } {
  const n = signal.length;
  // Apply Hanning window
  const windowed = signal.map((v, i) => v * (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1))));

  // Manual DFT (for small signals) or use the efficient approach
  const nfft = Math.pow(2, Math.ceil(Math.log2(n)));
  const real = new Float64Array(nfft);
  const imag = new Float64Array(nfft);

  for (let i = 0; i < n; i++) {
    real[i] = windowed[i];
  }

  // In-place FFT (Cooley-Tukey)
  fft(real, imag, nfft);

  // Compute one-sided PSD
  const nFreqs = Math.floor(nfft / 2) + 1;
  const freqs: number[] = [];
  const psd: number[] = [];
  const scale = 2.0 / (n * fs);

  for (let i = 0; i < nFreqs; i++) {
    freqs.push((i * fs) / nfft);
    psd.push((real[i] * real[i] + imag[i] * imag[i]) * scale);
  }

  return { freqs, psd };
}

/**
 * Cooley-Tukey FFT (in-place, radix-2).
 */
function fft(real: Float64Array, imag: Float64Array, n: number): void {
  // Bit-reversal permutation
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

  // FFT butterfly
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
 * Extract band powers from the signal's PSD.
 */
export function extractBandPowers(
  signal: number[],
  fs: number = 256
): Record<string, number> {
  // Apply low-pass filter at 50 Hz before PSD to match Muse 2 hardware bandwidth.
  // Without this, high-frequency noise inflates gamma to near-100% even with good contact.
  const filtered = signal.length >= 63 ? lowPassFilter(signal, fs, 50) : signal;
  const { freqs, psd } = computePSD(filtered, fs);

  // Total power (trapezoidal integration)
  let totalPower = 0;
  for (let i = 1; i < freqs.length; i++) {
    totalPower += ((psd[i] + psd[i - 1]) / 2) * (freqs[i] - freqs[i - 1]);
  }
  if (totalPower === 0) totalPower = 1e-10;

  const bandPowers: Record<string, number> = {};

  // Step 1: Compute raw relative power per band
  for (const [name, [low, high]] of Object.entries(BANDS)) {
    let power = 0;
    for (let i = 1; i < freqs.length; i++) {
      if (freqs[i] >= low && freqs[i] <= high) {
        power += ((psd[i] + psd[i - 1]) / 2) * (freqs[i] - freqs[i - 1]);
      }
    }
    bandPowers[name] = power / totalPower;
  }

  // Step 2: Apply spectral band importance weights.
  // Amplifies emotion-carrying bands (alpha 1.5x, theta 1.3x) and suppresses
  // noise-dominated bands (gamma 0.3x). Re-normalize to maintain sum ≈ 1.
  let weightedTotal = 0;
  for (const name of Object.keys(bandPowers)) {
    const w = BAND_IMPORTANCE_WEIGHTS[name] ?? 1.0;
    bandPowers[name] *= w;
    weightedTotal += bandPowers[name];
  }
  if (weightedTotal > 0) {
    for (const name of Object.keys(bandPowers)) {
      bandPowers[name] /= weightedTotal;
    }
  }

  return bandPowers;
}

/**
 * Compute Hjorth parameters: activity, mobility, complexity.
 */
export function computeHjorthParameters(signal: number[]): {
  activity: number;
  mobility: number;
  complexity: number;
} {
  const n = signal.length;
  if (n < 3) return { activity: 0, mobility: 0, complexity: 0 };

  // Variance (activity)
  const mean = signal.reduce((a, b) => a + b, 0) / n;
  let activity = 0;
  for (const v of signal) activity += (v - mean) ** 2;
  activity /= n;
  if (activity === 0) activity = 1e-10;

  // First derivative
  const diff1 = [];
  for (let i = 1; i < n; i++) diff1.push(signal[i] - signal[i - 1]);

  const mean1 = diff1.reduce((a, b) => a + b, 0) / diff1.length;
  let var1 = 0;
  for (const v of diff1) var1 += (v - mean1) ** 2;
  var1 /= diff1.length;

  const mobility = Math.sqrt(var1 / activity);

  // Second derivative
  const diff2 = [];
  for (let i = 1; i < diff1.length; i++) diff2.push(diff1[i] - diff1[i - 1]);

  const mean2 = diff2.reduce((a, b) => a + b, 0) / diff2.length;
  let var2 = 0;
  for (const v of diff2) var2 += (v - mean2) ** 2;
  var2 /= diff2.length;

  const complexity = var1 > 0 && mobility > 0
    ? Math.sqrt(var2 / var1) / mobility
    : 0;

  return { activity, mobility, complexity };
}

/**
 * Compute spectral entropy (normalized).
 */
function spectralEntropy(signal: number[], fs: number = 256): number {
  const { psd } = computePSD(signal, fs);
  const total = psd.reduce((a, b) => a + b, 0) + 1e-10;
  const norm = psd.map((p) => p / total);

  let entropy = 0;
  for (const p of norm) {
    if (p > 0) entropy -= p * Math.log(p);
  }

  const maxEntropy = Math.log(norm.length);
  return maxEntropy > 0 ? entropy / maxEntropy : 0;
}

/**
 * Compute differential entropy per band.
 */
function differentialEntropy(
  signal: number[],
  fs: number = 256
): Record<string, number> {
  const de: Record<string, number> = {};

  for (const [name, [low, high]] of Object.entries(BANDS)) {
    // Simple variance approximation (without full bandpass filter in browser)
    const { freqs, psd } = computePSD(signal, fs);
    let bandVar = 0;
    let count = 0;
    for (let i = 0; i < freqs.length; i++) {
      if (freqs[i] >= low && freqs[i] <= high) {
        bandVar += psd[i];
        count++;
      }
    }
    bandVar = count > 0 ? bandVar / count : 1e-10;
    de[name] = 0.5 * Math.log(2 * Math.PI * Math.E * Math.max(bandVar, 1e-10));
  }

  return de;
}

/**
 * Extract the full 17-feature vector matching the Python pipeline.
 */
export function extractFeatures(
  signal: number[],
  fs: number = 256
): number[] {
  const bands = extractBandPowers(signal, fs);
  const hjorth = computeHjorthParameters(signal);
  const se = spectralEntropy(signal, fs);
  const de = differentialEntropy(signal, fs);

  const alpha = bands.alpha || 0;
  const beta = bands.beta || 0.001;
  const theta = bands.theta || 0.001;

  // Feature order must match Python extract_features():
  // band_power_delta, theta, alpha, beta, gamma,
  // hjorth_activity, mobility, complexity,
  // spectral_entropy,
  // de_delta, theta, alpha, beta, gamma,
  // alpha_beta_ratio, theta_beta_ratio, alpha_theta_ratio
  return [
    bands.delta || 0,
    bands.theta || 0,
    bands.alpha || 0,
    bands.beta || 0,
    bands.gamma || 0,
    hjorth.activity,
    hjorth.mobility,
    hjorth.complexity,
    se,
    de.delta || 0,
    de.theta || 0,
    de.alpha || 0,
    de.beta || 0,
    de.gamma || 0,
    alpha / Math.max(beta, 1e-10),
    theta / Math.max(beta, 1e-10),
    alpha / Math.max(theta, 1e-10),
  ];
}

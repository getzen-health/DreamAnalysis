import * as ort from "onnxruntime-web";

// ─── Types ──────────────────────────────────────────────────────────────────

export interface OnDeviceEmotionResult {
  emotion: string;
  valence: number;
  arousal: number;
  confidence: number;
  stress_index: number;
  focus_index: number;
  probabilities: Record<string, number>;
  model_type: "voice-onnx";
}

// ─── Speaker Baseline F0 (Pitch Normalization) ─────────────────────────────
// The first voice check-in establishes the speaker's baseline fundamental
// frequency (F0).  Subsequent check-ins normalize pitch features as a ratio
// to this baseline (relative pitch).  This makes the model see *emotional*
// pitch shifts rather than absolute voice register differences between
// speakers (male ~120 Hz, female ~220 Hz).
//
// Stored in localStorage so it persists across sessions.

const _BASELINE_F0_KEY = "ndw_voice_baseline_f0";

/** Retrieve the stored speaker baseline F0, or null if not yet established. */
export function getBaselinePitchF0(): number | null {
  try {
    const raw = localStorage.getItem(_BASELINE_F0_KEY);
    if (raw === null) return null;
    const val = parseFloat(raw);
    if (!Number.isFinite(val) || val <= 0) return null;
    return val;
  } catch {
    return null;
  }
}

/** Store the speaker baseline F0.  Silently ignores non-positive / non-finite values. */
export function setBaselinePitchF0(f0: number): void {
  if (!Number.isFinite(f0) || f0 <= 0) return;
  try {
    localStorage.setItem(_BASELINE_F0_KEY, String(f0));
  } catch {
    /* quota exceeded — ignore */
  }
}

/** Clear the stored baseline (e.g. when switching speakers). */
export function clearBaselinePitchF0(): void {
  try {
    localStorage.removeItem(_BASELINE_F0_KEY);
  } catch {
    /* ignore */
  }
}

// ─── Emotion Mapping Config ─────────────────────────────────────────────────

export const EMOTION_REGIONS: Array<{
  emotion: string;
  valence: [number, number];
  arousal: [number, number];
  priority: number;
}> = [
  { emotion: "happy",   valence: [0.2, 1],     arousal: [0.3, 1],   priority: 1 },
  { emotion: "sad",     valence: [-1, -0.2],   arousal: [0, 0.4],   priority: 2 },
  { emotion: "angry",   valence: [-1, -0.1],   arousal: [0.6, 1],   priority: 3 },
  { emotion: "calm",    valence: [0.15, 1],    arousal: [0, 0.25],  priority: 4 },
  { emotion: "neutral", valence: [-0.3, 0.3],  arousal: [0, 0.7],   priority: 10 },
];

/** Map valence/arousal coordinates to the best-matching emotion label. */
export function mapValenceArousalToEmotion(valence: number, arousal: number): string {
  let best: string = "neutral";
  let bestPriority = Infinity;

  for (const region of EMOTION_REGIONS) {
    const [vMin, vMax] = region.valence;
    const [aMin, aMax] = region.arousal;
    if (valence >= vMin && valence <= vMax && arousal >= aMin && arousal <= aMax) {
      if (region.priority < bestPriority) {
        best = region.emotion;
        bestPriority = region.priority;
      }
    }
  }
  return best;
}

// ─── Audio Utility ──────────────────────────────────────────────────────────

/** Resample audio to 16kHz using linear interpolation. */
export function resampleTo16k(samples: Float32Array, sourceSr: number): Float32Array {
  if (sourceSr === 16000) return samples;
  const ratio = sourceSr / 16000;
  const outLen = Math.round(samples.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const srcIdx = i * ratio;
    const lo = Math.floor(srcIdx);
    const hi = Math.min(lo + 1, samples.length - 1);
    const frac = srcIdx - lo;
    out[i] = samples[lo] * (1 - frac) + samples[hi] * frac;
  }
  return out;
}

/** Prepare raw PCM for Wav2Small: resample to 16kHz, normalize, pad to >=1s. */
export function prepareAudioInput(samples: Float32Array, sourceSr: number): Float32Array {
  let audio = resampleTo16k(samples, sourceSr);

  // Normalize to [-1, 1]
  let maxAbs = 0;
  for (let i = 0; i < audio.length; i++) {
    const abs = Math.abs(audio[i]);
    if (abs > maxAbs) maxAbs = abs;
  }
  if (maxAbs > 1.0) {
    const scale = 1.0 / maxAbs;
    audio = audio.map(s => s * scale);
  }

  // Pad to minimum 1 second (16000 samples)
  if (audio.length < 16000) {
    const padded = new Float32Array(16000);
    padded.set(audio);
    return padded;
  }
  return audio;
}

// ─── DSP Primitives ─────────────────────────────────────────────────────────

/** Convert frequency in Hz to the mel scale. */
function hzToMel(hz: number): number {
  return 2595 * Math.log10(1 + hz / 700);
}

/** Convert mel-scale value back to Hz. */
function melToHz(mel: number): number {
  return 700 * (Math.pow(10, mel / 2595) - 1);
}

/**
 * Build a mel filterbank matrix.
 * Returns nFilters rows, each of length fftSize/2+1.
 */
function melFilterbank(
  nFilters: number,
  fftSize: number,
  sr: number,
  fMin: number,
  fMax: number,
): number[][] {
  const nBins = Math.floor(fftSize / 2) + 1;
  const melMin = hzToMel(fMin);
  const melMax = hzToMel(fMax);

  // nFilters + 2 equally spaced mel points
  const melPoints = new Float64Array(nFilters + 2);
  for (let i = 0; i < nFilters + 2; i++) {
    melPoints[i] = melMin + (i / (nFilters + 1)) * (melMax - melMin);
  }

  // Convert mel points to FFT bin indices
  const binIndices = new Float64Array(nFilters + 2);
  for (let i = 0; i < nFilters + 2; i++) {
    binIndices[i] = Math.floor((fftSize + 1) * melToHz(melPoints[i]) / sr);
  }

  const filterbank: number[][] = [];
  for (let m = 0; m < nFilters; m++) {
    const row = new Array<number>(nBins).fill(0);
    const left = binIndices[m];
    const center = binIndices[m + 1];
    const right = binIndices[m + 2];

    for (let k = 0; k < nBins; k++) {
      if (k >= left && k <= center && center > left) {
        row[k] = (k - left) / (center - left);
      } else if (k > center && k <= right && right > center) {
        row[k] = (right - k) / (right - center);
      }
    }
    filterbank.push(row);
  }
  return filterbank;
}

/**
 * Compute real FFT magnitude spectrum using a radix-2 DIT approach.
 * Input: real-valued frame of length N (must be power of 2).
 * Returns: magnitude spectrum of length N/2+1.
 */
function fftMagnitude(frame: Float64Array): Float64Array {
  const N = frame.length;
  const nBins = N / 2 + 1;

  // Bit-reversal permutation
  const real = new Float64Array(N);
  const imag = new Float64Array(N);
  const bits = Math.log2(N);

  for (let i = 0; i < N; i++) {
    let rev = 0;
    let val = i;
    for (let b = 0; b < bits; b++) {
      rev = (rev << 1) | (val & 1);
      val >>= 1;
    }
    real[rev] = frame[i];
  }

  // Cooley-Tukey FFT
  for (let size = 2; size <= N; size *= 2) {
    const halfSize = size / 2;
    const angle = -2 * Math.PI / size;
    for (let i = 0; i < N; i += size) {
      for (let j = 0; j < halfSize; j++) {
        const cos = Math.cos(angle * j);
        const sin = Math.sin(angle * j);
        const tReal = real[i + j + halfSize] * cos - imag[i + j + halfSize] * sin;
        const tImag = real[i + j + halfSize] * sin + imag[i + j + halfSize] * cos;
        real[i + j + halfSize] = real[i + j] - tReal;
        imag[i + j + halfSize] = imag[i + j] - tImag;
        real[i + j] += tReal;
        imag[i + j] += tImag;
      }
    }
  }

  const mag = new Float64Array(nBins);
  for (let k = 0; k < nBins; k++) {
    mag[k] = Math.sqrt(real[k] * real[k] + imag[k] * imag[k]);
  }
  return mag;
}

/** Next power of 2 >= n. */
function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

/**
 * Apply a Hanning window in-place.
 */
function hanningWindow(frame: Float64Array): void {
  const N = frame.length;
  for (let i = 0; i < N; i++) {
    frame[i] *= 0.5 * (1 - Math.cos(2 * Math.PI * i / (N - 1)));
  }
}

/**
 * Type-II DCT (orthonormal) of a vector.
 * Returns first nOut coefficients.
 */
function dctII(input: number[], nOut: number): number[] {
  const N = input.length;
  const out = new Array<number>(nOut);
  for (let k = 0; k < nOut; k++) {
    let sum = 0;
    for (let n = 0; n < N; n++) {
      sum += input[n] * Math.cos(Math.PI * k * (n + 0.5) / N);
    }
    // Orthonormal scaling
    out[k] = sum * Math.sqrt(2 / N);
    if (k === 0) out[k] *= Math.SQRT1_2;
  }
  return out;
}

// ─── MFCC Extraction ────────────────────────────────────────────────────────

/**
 * Extract MFCC coefficients per frame.
 *
 * @param samples   Mono PCM audio samples
 * @param sr        Sample rate in Hz
 * @param nMfcc     Number of MFCC coefficients to return per frame
 * @returns         Array of frames, each with nMfcc coefficients
 */
export function extractMFCC(
  samples: Float32Array,
  sr: number,
  nMfcc: number,
): number[][] {
  const nMelFilters = 40;
  const fMin = 0;
  const fMax = Math.min(sr / 2, 8000);
  const frameLenSec = 0.025;
  const hopLenSec = 0.01;
  const frameLen = Math.floor(sr * frameLenSec);
  const hopLen = Math.floor(sr * hopLenSec);
  const fftSize = nextPow2(frameLen);

  const fb = melFilterbank(nMelFilters, fftSize, sr, fMin, fMax);

  const frames: number[][] = [];
  for (let start = 0; start + frameLen <= samples.length; start += hopLen) {
    // Extract and window the frame
    const frame = new Float64Array(fftSize); // zero-padded
    for (let i = 0; i < frameLen; i++) {
      frame[i] = samples[start + i];
    }
    hanningWindow(frame);

    // Compute power spectrum
    const mag = fftMagnitude(frame);
    const nBins = fftSize / 2 + 1;

    // Apply mel filterbank
    const melEnergies = new Array<number>(nMelFilters);
    for (let m = 0; m < nMelFilters; m++) {
      let sum = 0;
      for (let k = 0; k < nBins; k++) {
        sum += fb[m][k] * mag[k] * mag[k]; // power spectrum
      }
      // Floor to avoid log(0)
      melEnergies[m] = Math.log(Math.max(sum, 1e-10));
    }

    // DCT to get MFCCs
    const mfcc = dctII(melEnergies, nMfcc);
    frames.push(mfcc);
  }

  return frames;
}

// ─── Spectral Feature Helpers ───────────────────────────────────────────────

/**
 * Compute per-frame spectral features from magnitude spectra.
 * Returns: { centroid[], bandwidth[], rolloff[], flatness[], zcr[], rms[] }
 */
function computeSpectralFeatures(
  samples: Float32Array,
  sr: number,
): {
  centroid: number[];
  bandwidth: number[];
  rolloff: number[];
  flatness: number[];
  zcr: number[];
  rms: number[];
} {
  const frameLenSec = 0.025;
  const hopLenSec = 0.01;
  const frameLen = Math.floor(sr * frameLenSec);
  const hopLen = Math.floor(sr * hopLenSec);
  const fftSize = nextPow2(frameLen);
  const nBins = fftSize / 2 + 1;

  // Frequency values for each bin
  const freqs = new Float64Array(nBins);
  for (let k = 0; k < nBins; k++) {
    freqs[k] = (k * sr) / fftSize;
  }

  const centroids: number[] = [];
  const bandwidths: number[] = [];
  const rolloffs: number[] = [];
  const flatnesses: number[] = [];
  const zcrs: number[] = [];
  const rmsList: number[] = [];

  for (let start = 0; start + frameLen <= samples.length; start += hopLen) {
    // Extract, window, FFT
    const frame = new Float64Array(fftSize);
    for (let i = 0; i < frameLen; i++) {
      frame[i] = samples[start + i];
    }
    hanningWindow(frame);
    const mag = fftMagnitude(frame);

    // Power spectrum for spectral features
    const power = new Float64Array(nBins);
    let totalPower = 0;
    for (let k = 0; k < nBins; k++) {
      power[k] = mag[k] * mag[k];
      totalPower += power[k];
    }

    // Spectral centroid: weighted mean of frequencies
    let centroid = 0;
    if (totalPower > 1e-10) {
      for (let k = 0; k < nBins; k++) {
        centroid += freqs[k] * power[k];
      }
      centroid /= totalPower;
    }
    centroids.push(centroid);

    // Spectral bandwidth: weighted std dev around centroid
    let bw = 0;
    if (totalPower > 1e-10) {
      for (let k = 0; k < nBins; k++) {
        const diff = freqs[k] - centroid;
        bw += diff * diff * power[k];
      }
      bw = Math.sqrt(bw / totalPower);
    }
    bandwidths.push(bw);

    // Spectral rolloff: frequency below which 85% of energy is concentrated
    let rolloff = 0;
    if (totalPower > 1e-10) {
      const threshold = 0.85 * totalPower;
      let cumulative = 0;
      for (let k = 0; k < nBins; k++) {
        cumulative += power[k];
        if (cumulative >= threshold) {
          rolloff = freqs[k];
          break;
        }
      }
    }
    rolloffs.push(rolloff);

    // Spectral flatness: geometric mean / arithmetic mean of magnitudes
    let logSum = 0;
    let arithSum = 0;
    let nonZeroCount = 0;
    for (let k = 0; k < nBins; k++) {
      if (mag[k] > 1e-10) {
        logSum += Math.log(mag[k]);
        nonZeroCount++;
      }
      arithSum += mag[k];
    }
    let flat = 0;
    if (nonZeroCount > 0 && arithSum > 1e-10) {
      const geoMean = Math.exp(logSum / nonZeroCount);
      const arithMean = arithSum / nBins;
      flat = geoMean / arithMean;
    }
    flatnesses.push(flat);

    // Zero-crossing rate for this frame (time domain)
    let zc = 0;
    for (let i = start + 1; i < start + frameLen && i < samples.length; i++) {
      if ((samples[i] >= 0) !== (samples[i - 1] >= 0)) zc++;
    }
    zcrs.push(zc / frameLen);

    // RMS energy for this frame
    let sumSq = 0;
    for (let i = start; i < start + frameLen && i < samples.length; i++) {
      sumSq += samples[i] * samples[i];
    }
    rmsList.push(Math.sqrt(sumSq / frameLen));
  }

  return {
    centroid: centroids,
    bandwidth: bandwidths,
    rolloff: rolloffs,
    flatness: flatnesses,
    zcr: zcrs,
    rms: rmsList,
  };
}

/** Compute mean and standard deviation of an array. */
function meanStd(arr: number[]): [number, number] {
  if (arr.length === 0) return [0, 0];
  const n = arr.length;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += arr[i];
  const mean = sum / n;

  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    const d = arr[i] - mean;
    sumSq += d * d;
  }
  const std = Math.sqrt(sumSq / n);
  return [mean, std];
}

// ─── 92-Dim Feature Extraction ──────────────────────────────────────────────

/**
 * Extract the full 92-dimensional feature vector expected by the voice
 * emotion ONNX preprocessing model.
 *
 * Layout:
 *   [0..79]  40 MFCC coefficients: mean + std (80 features)
 *   [80..81] spectral centroid: mean + std
 *   [82..83] spectral bandwidth: mean + std
 *   [84..85] spectral rolloff: mean + std
 *   [86..87] zero-crossing rate: mean + std
 *   [88..89] RMS energy: mean + std
 *   [90..91] spectral flatness: mean + std
 */
export function extractFeatures92(samples: Float32Array, sr: number): Float32Array {
  // Pad short audio to at least 1 frame (25ms)
  const minSamples = Math.floor(sr * 0.05); // at least 2 frames
  let audio = samples;
  if (audio.length < minSamples) {
    const padded = new Float32Array(minSamples);
    padded.set(audio);
    audio = padded;
  }

  const features = new Float32Array(92);

  // 40 MFCCs: mean + std (80 features)
  const mfcc = extractMFCC(audio, sr, 40);
  if (mfcc.length === 0) {
    // Not enough data — return zeros (all finite)
    return features;
  }

  for (let c = 0; c < 40; c++) {
    const vals = mfcc.map(frame => frame[c]);
    const [m, s] = meanStd(vals);
    features[c * 2] = m;
    features[c * 2 + 1] = s;
  }

  // Spectral features
  const spec = computeSpectralFeatures(audio, sr);

  const spectralArrays: [number[], number][] = [
    [spec.centroid, 80],    // centroid: mean+std at 80-81
    [spec.bandwidth, 82],   // bandwidth: mean+std at 82-83
    [spec.rolloff, 84],     // rolloff: mean+std at 84-85
    [spec.zcr, 86],         // ZCR: mean+std at 86-87
    [spec.rms, 88],         // RMS: mean+std at 88-89
    [spec.flatness, 90],    // flatness: mean+std at 90-91
  ];

  for (const [arr, idx] of spectralArrays) {
    const [m, s] = meanStd(arr);
    features[idx] = m;
    features[idx + 1] = s;
  }

  // Safety: replace any NaN/Inf with 0
  for (let i = 0; i < 92; i++) {
    if (!Number.isFinite(features[i])) features[i] = 0;
  }

  return features;
}

// ─── Enhanced 140-Dim Feature Extraction ─────────────────────────────────────

/**
 * Estimate pitch (F0) per frame using autocorrelation-based detection.
 * Returns an array of F0 values in Hz (0 for unvoiced frames).
 */
function estimatePitchPerFrame(
  samples: Float32Array,
  sr: number,
): number[] {
  const minF0 = 80;
  const maxF0 = 400;
  const minLag = Math.floor(sr / maxF0);
  const maxLag = Math.ceil(sr / minF0);
  const frameDuration = 0.03; // 30ms
  const hopDuration = 0.015;  // 15ms
  const frameSize = Math.floor(sr * frameDuration);
  const hopSize = Math.floor(sr * hopDuration);

  const pitches: number[] = [];

  for (let start = 0; start + frameSize <= samples.length; start += hopSize) {
    const frame = samples.subarray(start, start + frameSize);
    let bestLag = 0;
    let bestCorr = -Infinity;

    for (let lag = minLag; lag <= maxLag && lag < frame.length; lag++) {
      let sum = 0;
      let norm1 = 0;
      let norm2 = 0;
      const count = frame.length - lag;
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

    if (bestCorr >= 0.3 && bestLag > 0) {
      pitches.push(sr / bestLag);
    } else {
      pitches.push(0); // unvoiced
    }
  }

  return pitches;
}

/**
 * Compute jitter (period perturbation) from voiced pitch values.
 * Returns a ratio in [0, 1].
 */
function computeJitter(pitchValues: number[]): number {
  const voiced = pitchValues.filter(f => f > 0);
  if (voiced.length < 2) return 0;

  const periods = voiced.map(f => 1 / f);
  let sumDiff = 0;
  let sumPeriod = 0;
  for (let i = 1; i < periods.length; i++) {
    sumDiff += Math.abs(periods[i] - periods[i - 1]);
  }
  for (const p of periods) sumPeriod += p;
  const meanPeriod = sumPeriod / periods.length;
  const meanDiff = sumDiff / (periods.length - 1);
  return meanPeriod > 0 ? Math.min(1, meanDiff / meanPeriod) : 0;
}

/**
 * Compute shimmer (amplitude perturbation) from per-frame peak amplitudes
 * of voiced frames. Returns a ratio in [0, 1].
 */
function computeShimmer(
  samples: Float32Array,
  sr: number,
  pitchValues: number[],
): number {
  const frameDuration = 0.03;
  const hopDuration = 0.015;
  const frameSize = Math.floor(sr * frameDuration);
  const hopSize = Math.floor(sr * hopDuration);

  const peaks: number[] = [];
  let frameIdx = 0;
  for (let start = 0; start + frameSize <= samples.length; start += hopSize) {
    if (frameIdx < pitchValues.length && pitchValues[frameIdx] > 0) {
      let peak = 0;
      for (let i = start; i < start + frameSize; i++) {
        const abs = Math.abs(samples[i]);
        if (abs > peak) peak = abs;
      }
      peaks.push(peak);
    }
    frameIdx++;
  }

  if (peaks.length < 2) return 0;

  let sumDiff = 0;
  let sumPeak = 0;
  for (let i = 1; i < peaks.length; i++) {
    sumDiff += Math.abs(peaks[i] - peaks[i - 1]);
  }
  for (const p of peaks) sumPeak += p;
  const meanPeak = sumPeak / peaks.length;
  const meanDiff = sumDiff / (peaks.length - 1);
  return meanPeak > 0 ? Math.min(1, meanDiff / meanPeak) : 0;
}

/**
 * Estimate speaking rate (syllables per second) using energy-based
 * syllable nucleus detection.
 */
function estimateSpeakingRate(
  samples: Float32Array,
  sr: number,
): number {
  const frameDuration = 0.03;
  const hopDuration = 0.015;
  const frameSize = Math.floor(sr * frameDuration);
  const hopSize = Math.floor(sr * hopDuration);
  const silenceThreshold = 0.01;
  const syllableThreshold = silenceThreshold * 3;

  let syllables = 0;
  let wasSilent = true;

  for (let start = 0; start + frameSize <= samples.length; start += hopSize) {
    let sumSq = 0;
    for (let i = start; i < start + frameSize; i++) {
      sumSq += samples[i] * samples[i];
    }
    const energy = Math.sqrt(sumSq / frameSize);

    if (energy > syllableThreshold && wasSilent) {
      syllables++;
      wasSilent = false;
    } else if (energy <= syllableThreshold) {
      wasSilent = true;
    }
  }

  const durationSec = samples.length / sr;
  return durationSec > 0 ? syllables / durationSec : 0;
}

/**
 * Compute pause ratio: ratio of silence frames to total frames.
 * Returns a value in [0, 1].
 */
function computePauseRatio(
  samples: Float32Array,
  sr: number,
): number {
  const frameDuration = 0.03;
  const hopDuration = 0.015;
  const frameSize = Math.floor(sr * frameDuration);
  const hopSize = Math.floor(sr * hopDuration);
  const silenceThreshold = 0.01;

  let totalFrames = 0;
  let silentFrames = 0;

  for (let start = 0; start + frameSize <= samples.length; start += hopSize) {
    let sumSq = 0;
    for (let i = start; i < start + frameSize; i++) {
      sumSq += samples[i] * samples[i];
    }
    const energy = Math.sqrt(sumSq / frameSize);
    totalFrames++;
    if (energy <= silenceThreshold) silentFrames++;
  }

  return totalFrames > 0 ? silentFrames / totalFrames : 0;
}

/**
 * Extract the enhanced 140-dimensional feature vector for the v2 voice
 * emotion model.
 *
 * Layout:
 *   [0..91]    Original 92 features (40 MFCCs mean+std + 6 spectral mean+std)
 *   [92..131]  40 delta MFCC features (mean absolute frame-to-frame difference)
 *   [132..135] Pitch features: pitchMean, pitchStd, pitchRange, pitchSlope
 *   [136..137] Jitter, shimmer
 *   [138]      Speaking rate (syllables/sec)
 *   [139]      Pause ratio (0-1)
 */
export function extractEnhancedFeatures(samples: Float32Array, sr: number): Float32Array {
  // Pad short audio to at least 2 frames (50ms)
  const minSamples = Math.floor(sr * 0.05);
  let audio = samples;
  if (audio.length < minSamples) {
    const padded = new Float32Array(minSamples);
    padded.set(audio);
    audio = padded;
  }

  const features = new Float32Array(140);

  // ── Base 92 features ────────────────────────────────────────────────────────
  const base = extractFeatures92(audio, sr);
  features.set(base, 0);

  // ── Delta MFCCs (40 features at indices 92-131) ─────────────────────────────
  const mfccFrames = extractMFCC(audio, sr, 40);
  if (mfccFrames.length >= 2) {
    for (let c = 0; c < 40; c++) {
      let sum = 0;
      for (let f = 1; f < mfccFrames.length; f++) {
        sum += Math.abs(mfccFrames[f][c] - mfccFrames[f - 1][c]);
      }
      features[92 + c] = sum / (mfccFrames.length - 1);
    }
  }

  // ── Pitch features (4 features at indices 132-135) ──────────────────────────
  // After computing raw pitch, normalize to speaker baseline F0 (if available).
  // First check-in establishes baseline; subsequent check-ins use ratio to
  // baseline.  This makes the model see relative pitch deviations (emotion
  // signal) rather than absolute voice register (speaker identity signal).
  const pitchValues = estimatePitchPerFrame(audio, sr);
  const voicedPitches = pitchValues.filter(f => f > 0);

  if (voicedPitches.length > 0) {
    // pitchMean (raw Hz)
    let pitchSum = 0;
    for (const f of voicedPitches) pitchSum += f;
    const pitchMean = pitchSum / voicedPitches.length;

    // pitchStd (raw Hz)
    let pitchStd = 0;
    if (voicedPitches.length > 1) {
      let sumSq = 0;
      for (const f of voicedPitches) sumSq += (f - pitchMean) ** 2;
      pitchStd = Math.sqrt(sumSq / (voicedPitches.length - 1));
    }

    // pitchRange (raw Hz)
    let pMin = Infinity;
    let pMax = -Infinity;
    for (const f of voicedPitches) {
      if (f < pMin) pMin = f;
      if (f > pMax) pMax = f;
    }
    const pitchRange = pMax - pMin;

    // pitchSlope (raw Hz/frame — linear regression over voiced frame indices)
    let pitchSlope = 0;
    if (voicedPitches.length >= 2) {
      const indices: number[] = [];
      for (let i = 0; i < pitchValues.length; i++) {
        if (pitchValues[i] > 0) indices.push(i);
      }
      let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
      for (let j = 0; j < indices.length; j++) {
        const x = indices[j];
        const y = voicedPitches[j];
        sumX += x;
        sumY += y;
        sumXY += x * y;
        sumXX += x * x;
      }
      const n = indices.length;
      const denom = n * sumXX - sumX * sumX;
      pitchSlope = denom !== 0 ? (n * sumXY - sumX * sumY) / denom : 0;
    }

    // ── Baseline F0 normalization ──────────────────────────────────────────
    // On first voiced check-in (no baseline stored), store pitchMean as
    // baseline and return raw values.  On subsequent check-ins, normalize
    // all pitch features as ratio to baseline:
    //   pitchMean  → pitchMean / baseline  (1.0 = normal, >1 = higher than usual)
    //   pitchStd   → pitchStd / baseline   (relative variability)
    //   pitchRange → pitchRange / baseline  (relative range)
    //   pitchSlope → pitchSlope / baseline  (relative slope)
    const baselineF0 = getBaselinePitchF0();
    if (baselineF0 !== null && baselineF0 > 0) {
      // Normalize as ratio to baseline
      features[132] = pitchMean / baselineF0;
      features[133] = pitchStd / baselineF0;
      features[134] = pitchRange / baselineF0;
      features[135] = pitchSlope / baselineF0;
    } else {
      // No baseline yet — store this as baseline, return raw values
      setBaselinePitchF0(pitchMean);
      features[132] = pitchMean;
      features[133] = pitchStd;
      features[134] = pitchRange;
      features[135] = pitchSlope;
    }
  }

  // ── Jitter and shimmer (2 features at indices 136-137) ──────────────────────
  features[136] = computeJitter(pitchValues);
  features[137] = computeShimmer(audio, sr, pitchValues);

  // ── Speaking rate (1 feature at index 138) ──────────────────────────────────
  features[138] = estimateSpeakingRate(audio, sr);

  // ── Pause ratio (1 feature at index 139) ────────────────────────────────────
  features[139] = computePauseRatio(audio, sr);

  // Safety: replace any NaN/Inf with 0
  for (let i = 0; i < 140; i++) {
    if (!Number.isFinite(features[i])) features[i] = 0;
  }

  return features;
}

// ─── ONNX Session Cache ─────────────────────────────────────────────────────

const CLASS_MAP: Record<number, string> = {
  0: "happy",
  1: "sad",
  2: "angry",
  3: "neutral",
  4: "calm",
};

// v1 models: 92-dim input, 60-dim PCA
let preprocessSessionV1: ort.InferenceSession | null = null;
let classifierSessionV1: ort.InferenceSession | null = null;

// v2 models: 140-dim input, 80-dim PCA (enhanced features)
let preprocessSessionV2: ort.InferenceSession | null = null;
let classifierSessionV2: ort.InferenceSession | null = null;
let v2Available: boolean | null = null; // null = not yet checked

async function getPreprocessSessionV1(): Promise<ort.InferenceSession> {
  if (!preprocessSessionV1) {
    preprocessSessionV1 = await ort.InferenceSession.create(
      "/models/voice_preprocess.onnx",
      { executionProviders: ["wasm"] },
    );
  }
  return preprocessSessionV1;
}

async function getClassifierSessionV1(): Promise<ort.InferenceSession> {
  if (!classifierSessionV1) {
    classifierSessionV1 = await ort.InferenceSession.create(
      "/models/voice_classifier.onnx",
      { executionProviders: ["wasm"] },
    );
  }
  return classifierSessionV1;
}

async function tryLoadV2(): Promise<boolean> {
  if (v2Available !== null) return v2Available;
  try {
    preprocessSessionV2 = await ort.InferenceSession.create(
      "/models/voice_preprocess_v2.onnx",
      { executionProviders: ["wasm"] },
    );
    classifierSessionV2 = await ort.InferenceSession.create(
      "/models/voice_classifier_v2.onnx",
      { executionProviders: ["wasm"] },
    );
    v2Available = true;
  } catch {
    v2Available = false;
  }
  return v2Available;
}

// ─── Full ONNX Inference Pipeline ───────────────────────────────────────────

/** Shared logic to map classifier output to OnDeviceEmotionResult. */
function buildEmotionResult(
  label: number,
  probsRaw: Float32Array,
): OnDeviceEmotionResult {
  // Build probability map
  const probabilities: Record<string, number> = {};
  for (let i = 0; i < 5; i++) {
    probabilities[CLASS_MAP[i]] = probsRaw[i] ?? 0;
  }

  // Map label to emotion
  const emotion = CLASS_MAP[label] ?? "neutral";

  // Derive valence and arousal from probabilities
  // happy/calm -> positive valence; sad/angry -> negative
  const valence = Math.max(-1, Math.min(1,
    probabilities.happy * 0.8
    + probabilities.calm * 0.4
    - probabilities.sad * 0.7
    - probabilities.angry * 0.6
    + probabilities.neutral * 0.0,
  ));

  // happy/angry -> high arousal; calm/sad -> low arousal
  const arousal = Math.max(0, Math.min(1,
    probabilities.happy * 0.7
    + probabilities.angry * 0.9
    + probabilities.neutral * 0.3
    + probabilities.sad * 0.2
    + probabilities.calm * 0.1,
  ));

  // Confidence is the max probability
  const confidence = Math.max(...Object.values(probabilities));

  // Stress: angry + sad weighted
  const stress_index = Math.max(0, Math.min(1,
    probabilities.angry * 0.6
    + probabilities.sad * 0.3
    + (1 - probabilities.calm) * 0.1,
  ));

  // Focus: inverse of stress, modulated by confidence
  const focus_index = Math.max(0.1, Math.min(0.9,
    confidence * 0.5
    + probabilities.calm * 0.3
    + probabilities.neutral * 0.2
    - probabilities.angry * 0.2,
  ));

  return {
    emotion,
    valence,
    arousal,
    confidence,
    stress_index,
    focus_index,
    probabilities,
    model_type: "voice-onnx",
  };
}

/**
 * Run the full on-device voice emotion ONNX pipeline.
 *
 * Prefers v2 models (140-dim enhanced features, 80-dim PCA) when available.
 * Falls back to v1 models (92-dim features, 60-dim PCA) if v2 not found.
 *
 * Pipeline:
 *   1. Extract audio features (140-dim for v2, 92-dim for v1)
 *   2. Run voice_preprocess ONNX (scaler + PCA)
 *   3. Run voice_classifier ONNX (LightGBM) -> label + probabilities
 *   4. Map to emotion + compute stress/focus from probabilities
 */
export async function runVoiceEmotionONNX(
  samples: Float32Array,
  sr: number,
): Promise<OnDeviceEmotionResult> {
  const useV2 = await tryLoadV2();

  if (useV2 && preprocessSessionV2 && classifierSessionV2) {
    // ── V2 path: 140-dim enhanced features ──────────────────────────────────
    const features140 = extractEnhancedFeatures(samples, sr);
    const inputTensor = new ort.Tensor("float32", features140, [1, 140]);
    const preResult = await preprocessSessionV2.run({ features: inputTensor });
    const pcaFeatures = preResult["variable"].data as Float32Array;

    const pcaTensor = new ort.Tensor("float32", pcaFeatures, [1, 80]);
    const clsResult = await classifierSessionV2.run({ pca_features: pcaTensor });

    const label = Number(clsResult["label"].data[0]);
    const probsRaw = clsResult["probabilities"].data as Float32Array;
    return buildEmotionResult(label, probsRaw);
  }

  // ── V1 fallback: 92-dim features ────────────────────────────────────────
  const features92 = extractFeatures92(samples, sr);
  const preprocess = await getPreprocessSessionV1();
  const inputTensor = new ort.Tensor("float32", features92, [1, 92]);
  const preResult = await preprocess.run({ features: inputTensor });
  const pcaFeatures = preResult["variable"].data as Float32Array;

  const classifier = await getClassifierSessionV1();
  const pcaTensor = new ort.Tensor("float32", pcaFeatures, [1, 60]);
  const clsResult = await classifier.run({ pca_features: pcaTensor });

  const label = Number(clsResult["label"].data[0]);
  const probsRaw = clsResult["probabilities"].data as Float32Array;
  return buildEmotionResult(label, probsRaw);
}

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
  /** Audio signal-to-noise ratio in dB.  When SNR < 15 dB, confidence is
   *  scaled down proportionally (confidence * SNR/15) because background
   *  noise degrades voice emotion accuracy.  Consumers can use this to
   *  weight voice predictions lower in the fusion engine. */
  snr_db: number;
  /** F0 contour shape features — captures the prosodic pitch trajectory.
   *  Useful for the fusion engine to adjust valence/arousal based on
   *  intonation patterns (rising = question/surprise, falling = calm). */
  f0_contour?: F0ContourFeatures;
  /** Voice energy envelope features — captures temporal dynamics of
   *  utterance energy (attack, sustain, decay).  Used as post-processing
   *  modulation: fast attack + high sustain nudges arousal up (+0.05),
   *  slow attack nudges arousal down (-0.05). */
  energy_envelope?: EnergyEnvelopeFeatures;
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

// ─── F0 Contour Shape Features ──────────────────────────────────────────────

/**
 * F0 contour shape features capture the SHAPE of pitch over time, which is
 * critical for emotion recognition (Schuller 2013, eGeMAPS feature set).
 *
 * Three features:
 *   1. contourType: overall pitch trajectory shape
 *      - "rising":      question intonation, surprise, uncertainty
 *      - "falling":     declarative, calm, confident statement
 *      - "flat":        monotone, neutral, or constant-pitch signal
 *      - "U":           pitch dips then rises — hesitation then recovery
 *      - "inverted-U":  pitch rises then falls — emphasis/exclamation
 *
 *   2. contourVariance: how much the pitch direction changes across quarters.
 *      High = dynamic/expressive speech.  Low = monotone/flat.
 *
 *   3. maxExcursionPosition: where in the utterance (0=start, 1=end) the
 *      biggest pitch change happens.
 *      - Near 0 = emphasis at start (strong assertion)
 *      - Near 1 = question/trailing emphasis
 *      - Near 0.5 = mid-utterance peak (surprise/exclamation)
 */
export interface F0ContourFeatures {
  contourType: "rising" | "falling" | "flat" | "U" | "inverted-U";
  contourVariance: number;
  maxExcursionPosition: number;
  /** Per-quarter F0 slope (Hz per frame). 4 elements: Q1, Q2, Q3, Q4. */
  quarterSlopes: [number, number, number, number];
}

/**
 * Extract F0 contour shape features from audio.
 *
 * Splits the voiced pitch track into 4 quarters, computes the linear
 * slope in each, then classifies the overall contour shape.
 *
 * @param samples  Mono PCM audio samples
 * @param sr       Sample rate in Hz
 * @returns        F0ContourFeatures object (safe defaults if no voiced frames)
 */
export function extractF0ContourFeatures(
  samples: Float32Array,
  sr: number,
): F0ContourFeatures {
  const pitchValues = estimatePitchPerFrame(samples, sr);
  const voicedPitches = pitchValues.filter((f) => f > 0);

  // Safe defaults when there are no voiced frames
  if (voicedPitches.length < 4) {
    return {
      contourType: "flat",
      contourVariance: 0,
      maxExcursionPosition: 0.5,
      quarterSlopes: [0, 0, 0, 0],
    };
  }

  // Split voiced pitches into 4 approximately equal quarters
  const qLen = Math.floor(voicedPitches.length / 4);
  const quarters: number[][] = [
    voicedPitches.slice(0, qLen),
    voicedPitches.slice(qLen, qLen * 2),
    voicedPitches.slice(qLen * 2, qLen * 3),
    voicedPitches.slice(qLen * 3),
  ];

  // Compute linear slope per quarter using least-squares regression
  function linearSlope(values: number[]): number {
    if (values.length < 2) return 0;
    const n = values.length;
    let sumX = 0,
      sumY = 0,
      sumXY = 0,
      sumXX = 0;
    for (let i = 0; i < n; i++) {
      sumX += i;
      sumY += values[i];
      sumXY += i * values[i];
      sumXX += i * i;
    }
    const denom = n * sumXX - sumX * sumX;
    return denom !== 0 ? (n * sumXY - sumX * sumY) / denom : 0;
  }

  const quarterSlopes: [number, number, number, number] = [
    linearSlope(quarters[0]),
    linearSlope(quarters[1]),
    linearSlope(quarters[2]),
    linearSlope(quarters[3]),
  ];

  // ── Contour variance: variance of the 4 quarter slopes ──
  // Measures how much the pitch direction changes across the utterance.
  let slopeMean = 0;
  for (const s of quarterSlopes) slopeMean += s;
  slopeMean /= 4;
  let slopeVar = 0;
  for (const s of quarterSlopes) slopeVar += (s - slopeMean) ** 2;
  slopeVar /= 4;
  const contourVariance = Number.isFinite(slopeVar) ? slopeVar : 0;

  // ── Max excursion position: where the biggest pitch change occurs ──
  // Compute cumulative absolute pitch change per frame, find the position
  // of the maximum single-frame jump.
  let maxDelta = 0;
  let maxDeltaIdx = 0;
  for (let i = 1; i < voicedPitches.length; i++) {
    const delta = Math.abs(voicedPitches[i] - voicedPitches[i - 1]);
    if (delta > maxDelta) {
      maxDelta = delta;
      maxDeltaIdx = i;
    }
  }
  // Normalize to [0, 1] where 0=start, 1=end
  const maxExcursionPosition =
    voicedPitches.length > 1 ? maxDeltaIdx / (voicedPitches.length - 1) : 0.5;

  // ── Classify contour type ──
  // Use the overall slope (first half vs second half mean) plus quarter slopes.
  const overallSlope = linearSlope(voicedPitches);
  const firstHalfMean = (quarterSlopes[0] + quarterSlopes[1]) / 2;
  const secondHalfMean = (quarterSlopes[2] + quarterSlopes[3]) / 2;

  // Threshold for "significant" slope: 0.5 Hz per frame.
  // At 15ms hop, that's ~33 Hz/sec change — a noticeable pitch shift.
  const SLOPE_THRESH = 0.5;

  let contourType: F0ContourFeatures["contourType"];

  if (Math.abs(overallSlope) < SLOPE_THRESH && contourVariance < 0.5) {
    // Neither overall change nor directional changes → flat
    contourType = "flat";
  } else if (firstHalfMean < -SLOPE_THRESH && secondHalfMean > SLOPE_THRESH) {
    // Pitch drops then rises → U shape
    contourType = "U";
  } else if (firstHalfMean > SLOPE_THRESH && secondHalfMean < -SLOPE_THRESH) {
    // Pitch rises then falls → inverted-U shape
    contourType = "inverted-U";
  } else if (overallSlope > SLOPE_THRESH) {
    contourType = "rising";
  } else if (overallSlope < -SLOPE_THRESH) {
    contourType = "falling";
  } else {
    // Ambiguous — classify by predominant direction
    const posCount = quarterSlopes.filter((s) => s > 0).length;
    if (posCount >= 3) contourType = "rising";
    else if (posCount <= 1) contourType = "falling";
    else contourType = "flat";
  }

  return {
    contourType,
    contourVariance: Number.isFinite(contourVariance) ? contourVariance : 0,
    maxExcursionPosition: Number.isFinite(maxExcursionPosition)
      ? maxExcursionPosition
      : 0.5,
    quarterSlopes,
  };
}

// ─── Voice Energy Envelope Features ──────────────────────────────────────────

/**
 * Energy envelope features capture the temporal dynamics of utterance energy.
 *
 * Three features:
 *   1. attackTimeMs:  Time from silence to peak energy at utterance start.
 *                     Fast attack (~50ms) = assertive/excited speech.
 *                     Slow attack (~200ms+) = hesitant/calm speech.
 *
 *   2. sustainRatio:  Ratio of sustained energy to peak energy (0-1).
 *                     High sustain (~0.7+) = confident, steady speech.
 *                     Low sustain (~0.3) = breathy, fading speech.
 *
 *   3. decayRateDb:   Energy decay rate at utterance end in dB/second.
 *                     Fast decay (large negative) = abrupt stop.
 *                     Slow decay (near 0) = trailing off gently.
 */
export interface EnergyEnvelopeFeatures {
  attackTimeMs: number;
  sustainRatio: number;
  decayRateDb: number;
}

/**
 * Extract energy envelope features from audio.
 *
 * Algorithm:
 *   1. Compute energy per 25ms frame (10ms hop for finer resolution).
 *   2. Find onset: first frame above 10% of max energy.
 *   3. Find peak energy frame.
 *   4. Attack time = (peak_frame - onset_frame) * hop_ms.
 *   5. Sustain = median energy of middle 50% of frames / peak energy.
 *   6. Decay = linear regression slope of log energy over last 25% of frames,
 *             converted to dB/second.
 *
 * @param samples  Mono PCM audio samples
 * @param sr       Sample rate in Hz
 * @returns        EnergyEnvelopeFeatures (safe defaults for silence/short audio)
 */
export function extractEnergyEnvelope(
  samples: Float32Array,
  sr: number,
): EnergyEnvelopeFeatures {
  const frameLenSec = 0.025; // 25ms frames
  const hopLenSec = 0.01;    // 10ms hop
  const frameLen = Math.floor(sr * frameLenSec);
  const hopLen = Math.floor(sr * hopLenSec);
  const hopMs = hopLenSec * 1000;

  // Safe defaults for silence / degenerate input
  const defaults: EnergyEnvelopeFeatures = {
    attackTimeMs: 0,
    sustainRatio: 0,
    decayRateDb: 0,
  };

  if (frameLen <= 0 || hopLen <= 0 || samples.length < frameLen) {
    return defaults;
  }

  // Step 1: Compute energy per frame (mean squared amplitude)
  const energies: number[] = [];
  for (let start = 0; start + frameLen <= samples.length; start += hopLen) {
    let sumSq = 0;
    for (let i = start; i < start + frameLen; i++) {
      sumSq += samples[i] * samples[i];
    }
    energies.push(sumSq / frameLen);
  }

  if (energies.length === 0) return defaults;

  // Find max energy
  let maxEnergy = 0;
  let peakFrame = 0;
  for (let i = 0; i < energies.length; i++) {
    if (energies[i] > maxEnergy) {
      maxEnergy = energies[i];
      peakFrame = i;
    }
  }

  // All silence — return defaults
  if (maxEnergy <= 1e-14) return defaults;

  // Step 2: Find onset — first frame above 10% of max energy
  const onsetThreshold = maxEnergy * 0.1;
  let onsetFrame = 0;
  for (let i = 0; i < energies.length; i++) {
    if (energies[i] >= onsetThreshold) {
      onsetFrame = i;
      break;
    }
  }

  // Step 3: Attack time = (peak - onset) * hop_ms
  const attackTimeMs = Math.max(0, (peakFrame - onsetFrame) * hopMs);

  // Step 4: Sustain ratio = median of middle 50% / peak
  const n = energies.length;
  const midStart = Math.floor(n * 0.25);
  const midEnd = Math.floor(n * 0.75);
  let sustainRatio = 0;
  if (midEnd > midStart) {
    const middleEnergies = energies.slice(midStart, midEnd);
    middleEnergies.sort((a, b) => a - b);
    const median = middleEnergies[Math.floor(middleEnergies.length / 2)];
    sustainRatio = maxEnergy > 0 ? Math.min(1, Math.max(0, median / maxEnergy)) : 0;
  }

  // Step 5: Decay rate — linear regression of log energy over last 25%
  let decayRateDb = 0;
  const decayStart = Math.floor(n * 0.75);
  const decayFrames = energies.slice(decayStart);
  if (decayFrames.length >= 2) {
    // Convert to dB (10*log10), floor to avoid log(0)
    const logEnergies = decayFrames.map(e => 10 * Math.log10(Math.max(e, 1e-14)));

    // Linear regression: slope of dB over frame index
    const m = logEnergies.length;
    let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
    for (let i = 0; i < m; i++) {
      sumX += i;
      sumY += logEnergies[i];
      sumXY += i * logEnergies[i];
      sumXX += i * i;
    }
    const denom = m * sumXX - sumX * sumX;
    if (denom !== 0) {
      const slopePerFrame = (m * sumXY - sumX * sumY) / denom;
      // Convert slope from dB/frame to dB/second
      // Each frame hop = hopLenSec seconds
      decayRateDb = slopePerFrame / hopLenSec;
    }
  }

  // Ensure all values are finite
  return {
    attackTimeMs: Number.isFinite(attackTimeMs) ? attackTimeMs : 0,
    sustainRatio: Number.isFinite(sustainRatio) ? sustainRatio : 0,
    decayRateDb: Number.isFinite(decayRateDb) ? decayRateDb : 0,
  };
}

// ─── Audio SNR Estimation ────────────────────────────────────────────────────

/**
 * Estimate signal-to-noise ratio (SNR) of an audio recording in dB.
 *
 * Method: segment audio into 30ms frames (15ms hop), compute mean squared
 * energy per frame, classify each frame as "speech" or "silence" using an
 * adaptive threshold (geometric mean of 25th/75th percentile frame energies).
 * SNR = 10 * log10(mean_speech_energy / mean_silence_energy).
 *
 * If energy range across frames is < 5 dB (uniform signal), returns 60 dB
 * (treated as clean).  When background noise is high (cafe, traffic), voice
 * emotion predictions are unreliable.  The caller reduces confidence when
 * SNR < 15 dB.
 *
 * @param samples  Mono PCM audio samples (any sample rate)
 * @param sr       Sample rate in Hz
 * @returns        Estimated SNR in dB, clamped to [0, 60].
 *                 Returns 0 for pure silence or degenerate input.
 */
export function computeAudioSNR(samples: Float32Array, sr: number): number {
  const frameDuration = 0.03; // 30ms frames
  const hopDuration = 0.015;  // 15ms hop (50% overlap)
  const frameSize = Math.floor(sr * frameDuration);
  const hopSize = Math.floor(sr * hopDuration);

  if (frameSize <= 0 || samples.length < frameSize) {
    return 0;
  }

  // Compute per-frame mean squared energy (power)
  const energies: number[] = [];
  for (let start = 0; start + frameSize <= samples.length; start += hopSize) {
    let sumSq = 0;
    for (let i = start; i < start + frameSize; i++) {
      sumSq += samples[i] * samples[i];
    }
    energies.push(sumSq / frameSize);
  }

  if (energies.length === 0) return 0;

  // Sort energies to find adaptive threshold.
  const sorted = [...energies].sort((a, b) => a - b);
  const eMin = sorted[0];
  const eMax = sorted[sorted.length - 1];

  // If energy range is very narrow (max/min < 3, i.e. < 5 dB dynamic range),
  // the recording has uniform energy throughout — no discernible speech/silence
  // boundary.  Treat as clean (continuous speech or steady tone).
  if (eMin > 0 && eMax / eMin < 3) {
    return 60;
  }

  // Use the geometric mean of the 25th and 75th percentile energies as the
  // speech/silence boundary.  This adapts to the dynamic range of the
  // recording rather than assuming a fixed ratio.
  const p25 = sorted[Math.floor(sorted.length * 0.25)];
  const p75 = sorted[Math.floor(sorted.length * 0.75)];
  const threshold = Math.max(Math.sqrt(Math.max(p25, 1e-14) * Math.max(p75, 1e-14)), 1e-10);

  // Classify frames and accumulate energy
  let speechEnergySum = 0;
  let speechCount = 0;
  let silenceEnergySum = 0;
  let silenceCount = 0;

  for (const e of energies) {
    if (e > threshold) {
      speechEnergySum += e;
      speechCount++;
    } else {
      silenceEnergySum += e;
      silenceCount++;
    }
  }

  // Edge cases
  if (speechCount === 0) {
    // No speech detected — pure silence / very low energy
    return 0;
  }

  if (silenceCount === 0) {
    // All frames are speech — no silence frames to estimate noise floor.
    // Without silence frames we cannot reliably estimate background noise.
    // Treat as high-SNR (clean recording with continuous speech).
    return 60;
  }

  const avgSpeechEnergy = speechEnergySum / speechCount;
  const avgSilenceEnergy = silenceEnergySum / silenceCount;

  if (avgSilenceEnergy <= 0 || !Number.isFinite(avgSilenceEnergy)) {
    return 60; // silence frames have zero energy — noiseless
  }

  const snrDb = 10 * Math.log10(avgSpeechEnergy / avgSilenceEnergy);
  return Math.max(0, Math.min(60, Number.isFinite(snrDb) ? snrDb : 0));
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

/** Shared logic to map classifier output to OnDeviceEmotionResult.
 *  @param snrDb     Audio SNR in dB — used to modulate confidence.
 *                   When SNR < 15 dB, confidence is scaled by SNR/15.
 *  @param contour   Optional F0 contour features — used to nudge
 *                   valence/arousal based on prosodic intonation.
 *  @param envelope  Optional energy envelope features — used to nudge
 *                   arousal based on utterance dynamics. */
function buildEmotionResult(
  label: number,
  probsRaw: Float32Array,
  snrDb: number = 60,
  contour?: F0ContourFeatures,
  envelope?: EnergyEnvelopeFeatures,
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
  let valence = Math.max(-1, Math.min(1,
    probabilities.happy * 0.8
    + probabilities.calm * 0.4
    - probabilities.sad * 0.7
    - probabilities.angry * 0.6
    + probabilities.neutral * 0.0,
  ));

  // happy/angry -> high arousal; calm/sad -> low arousal
  let arousal = Math.max(0, Math.min(1,
    probabilities.happy * 0.7
    + probabilities.angry * 0.9
    + probabilities.neutral * 0.3
    + probabilities.sad * 0.2
    + probabilities.calm * 0.1,
  ));

  // ── F0 contour-based modulation ──────────────────────────────────────
  // Prosodic contour carries emotion signal independent of spectral
  // features. Apply small nudges (max +/-0.08) based on contour shape.
  // Research: Schuller 2013, Bänziger & Scherer 2005.
  //
  //   rising   → question/surprise:   arousal +0.06, valence +0.04
  //   falling  → declarative/calm:    arousal -0.05, valence +0.02
  //   U-shape  → hesitation/recovery: arousal +0.03, valence -0.03
  //   inverted-U → exclamation/emphasis: arousal +0.08, valence 0
  //   flat     → monotone (no adjustment)
  //
  // contourVariance modulates the magnitude: high variance = expressive
  // speech where contour matters more. Scale factor = min(1, sqrt(var)).
  if (contour) {
    const expressiveness = Math.min(1, Math.sqrt(Math.max(0, contour.contourVariance)));

    let arousalNudge = 0;
    let valenceNudge = 0;

    switch (contour.contourType) {
      case "rising":
        arousalNudge = 0.06;
        valenceNudge = 0.04;
        break;
      case "falling":
        arousalNudge = -0.05;
        valenceNudge = 0.02;
        break;
      case "U":
        arousalNudge = 0.03;
        valenceNudge = -0.03;
        break;
      case "inverted-U":
        arousalNudge = 0.08;
        valenceNudge = 0;
        break;
      case "flat":
      default:
        break;
    }

    valence = Math.max(-1, Math.min(1, valence + valenceNudge * expressiveness));
    arousal = Math.max(0, Math.min(1, arousal + arousalNudge * expressiveness));
  }

  // ── Energy envelope-based modulation ──────────────────────────────────
  // Fast attack + high sustain = assertive/energetic speech → arousal up.
  // Slow attack = hesitant/calm speech → arousal down.
  // Nudge magnitude: +/-0.05 (same scale as F0 contour nudges).
  if (envelope) {
    let envelopeArousalNudge = 0;

    // Fast attack (< 80ms) + high sustain (> 0.5) → energetic
    if (envelope.attackTimeMs < 80 && envelope.sustainRatio > 0.5) {
      envelopeArousalNudge = 0.05;
    }
    // Slow attack (> 150ms) → hesitant/calm
    else if (envelope.attackTimeMs > 150) {
      envelopeArousalNudge = -0.05;
    }

    arousal = Math.max(0, Math.min(1, arousal + envelopeArousalNudge));
  }

  // Confidence is the max probability, modulated by audio SNR.
  // When background noise is high (SNR < 15 dB), scale confidence down
  // proportionally because noisy audio degrades voice emotion accuracy.
  const SNR_THRESHOLD_DB = 15;
  const rawConfidence = Math.max(...Object.values(probabilities));
  const snrScale = snrDb >= SNR_THRESHOLD_DB ? 1.0 : Math.max(0, snrDb / SNR_THRESHOLD_DB);
  const confidence = rawConfidence * snrScale;

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
    snr_db: snrDb,
    f0_contour: contour,
    energy_envelope: envelope,
  };
}

/**
 * Run the full on-device voice emotion ONNX pipeline.
 *
 * Prefers v2 models (140-dim enhanced features, 80-dim PCA) when available.
 * Falls back to v1 models (92-dim features, 60-dim PCA) if v2 not found.
 *
 * Pipeline:
 *   1. Compute audio SNR (used to modulate output confidence)
 *   2. Extract audio features (140-dim for v2, 92-dim for v1)
 *   3. Run voice_preprocess ONNX (scaler + PCA)
 *   4. Run voice_classifier ONNX (LightGBM) -> label + probabilities
 *   5. Map to emotion + compute stress/focus from probabilities
 *   6. Apply SNR-based confidence modulation
 */
export async function runVoiceEmotionONNX(
  samples: Float32Array,
  sr: number,
): Promise<OnDeviceEmotionResult> {
  // Compute SNR before any feature extraction — it needs the raw audio.
  const snrDb = computeAudioSNR(samples, sr);

  // Extract F0 contour features (runs pitch detection once; ~5ms for 2s audio).
  // These are used to modulate valence/arousal based on prosodic intonation
  // pattern, independent of the ONNX spectral model.
  const contour = extractF0ContourFeatures(samples, sr);

  // Extract energy envelope features (attack, sustain, decay).
  // Used for post-processing arousal modulation — fast attack + high sustain
  // nudges arousal up, slow attack nudges arousal down.
  const envelope = extractEnergyEnvelope(samples, sr);

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
    return buildEmotionResult(label, probsRaw, snrDb, contour, envelope);
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
  return buildEmotionResult(label, probsRaw, snrDb, contour, envelope);
}

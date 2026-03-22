/**
 * emg-detector.ts — Detect muscle artifact (EMG contamination) in EEG signals.
 *
 * EMG contamination from jaw clenching, forehead tensing, etc. is the most
 * common artifact in frontal EEG (AF7/AF8 on Muse 2). It appears as broadband
 * power increase in 20-50 Hz, which contaminates beta and gamma bands.
 *
 * Detection rule:
 *   If broadband power in 20-50 Hz exceeds 2x the running mean, flag as EMG.
 *
 * @see Issue #522
 */

export interface EMGDetectionResult {
  /** True if EMG artifact is currently detected */
  emgDetected: boolean;
  /** Percentage of recent frames flagged as EMG artifacts (0-1) */
  artifactPercent: number;
  /** Current 20-50 Hz power relative to the running mean (>2 = flagged) */
  emgRatio: number;
  /** Human-readable message for the UI */
  message: string | null;
}

/** Rolling buffer for EMG power history */
const EMG_HISTORY_SIZE = 30; // ~30 frames = ~45s at 1.5s/frame

/**
 * Compute broadband power in a frequency range using a simple DFT approximation.
 * For real-time use with short windows, this is adequate — full Welch PSD is overkill
 * for a binary EMG detection flag.
 *
 * @param signal Single-channel EEG data (Float32Array or number[])
 * @param sampleRate Sampling rate in Hz (default 256)
 * @param lowHz Lower bound of target band (default 20 Hz)
 * @param highHz Upper bound of target band (default 50 Hz)
 * @returns Average power in the specified band
 */
export function computeBandPower(
  signal: number[] | Float32Array,
  sampleRate: number = 256,
  lowHz: number = 20,
  highHz: number = 50,
): number {
  const n = signal.length;
  if (n === 0) return 0;

  // Simple band power via summing squared amplitudes of DFT bins in the range
  const freqResolution = sampleRate / n;
  const lowBin = Math.ceil(lowHz / freqResolution);
  const highBin = Math.min(Math.floor(highHz / freqResolution), Math.floor(n / 2));

  if (lowBin >= highBin) return 0;

  let totalPower = 0;
  let count = 0;

  for (let k = lowBin; k <= highBin; k++) {
    // Compute DFT magnitude at bin k
    let realPart = 0;
    let imagPart = 0;
    const freq = (2 * Math.PI * k) / n;
    for (let i = 0; i < n; i++) {
      realPart += signal[i] * Math.cos(freq * i);
      imagPart -= signal[i] * Math.sin(freq * i);
    }
    totalPower += (realPart * realPart + imagPart * imagPart) / (n * n);
    count++;
  }

  return count > 0 ? totalPower / count : 0;
}

/**
 * EMGDetector — Stateful detector that maintains a rolling history of 20-50 Hz
 * power and flags frames where current power exceeds 2x the running mean.
 */
export class EMGDetector {
  private history: number[] = [];
  private flagHistory: boolean[] = [];
  private runningSum = 0;

  /**
   * Process a new frame of EEG data across all channels.
   *
   * @param signals Multi-channel EEG data: signals[channel][sample]
   * @param sampleRate Sampling rate in Hz (default 256)
   * @returns EMG detection result for this frame
   */
  detect(signals: number[][], sampleRate: number = 256): EMGDetectionResult {
    if (!signals || signals.length === 0 || signals[0].length === 0) {
      return { emgDetected: false, artifactPercent: 0, emgRatio: 0, message: null };
    }

    // Compute average 20-50 Hz power across all channels
    let totalPower = 0;
    for (const channel of signals) {
      totalPower += computeBandPower(channel, sampleRate, 20, 50);
    }
    const avgPower = totalPower / signals.length;

    // Update rolling history
    this.history.push(avgPower);
    this.runningSum += avgPower;
    if (this.history.length > EMG_HISTORY_SIZE) {
      this.runningSum -= this.history.shift()!;
    }

    // Compute running mean
    const mean = this.history.length > 0 ? this.runningSum / this.history.length : avgPower;

    // EMG ratio: current power relative to running mean
    const emgRatio = mean > 0 ? avgPower / mean : 0;

    // Flag if current power exceeds 2x the mean (and we have enough history)
    const emgDetected = this.history.length >= 3 && emgRatio > 2.0;

    // Track artifact flags for percentage computation
    this.flagHistory.push(emgDetected);
    if (this.flagHistory.length > EMG_HISTORY_SIZE) {
      this.flagHistory.shift();
    }

    // Compute artifact percentage over recent history
    const flaggedCount = this.flagHistory.filter(Boolean).length;
    const artifactPercent = this.flagHistory.length > 0
      ? flaggedCount / this.flagHistory.length
      : 0;

    // Generate user-facing message
    let message: string | null = null;
    if (emgDetected) {
      message = "Signal too noisy — try relaxing your forehead";
    } else if (artifactPercent > 0.3) {
      message = "Frequent muscle artifacts detected — keep your jaw and forehead relaxed";
    }

    return { emgDetected, artifactPercent, emgRatio, message };
  }

  /** Reset the detector state (e.g., on new session) */
  reset(): void {
    this.history = [];
    this.flagHistory = [];
    this.runningSum = 0;
  }

  /** Get current artifact percentage without processing a new frame */
  getArtifactPercent(): number {
    if (this.flagHistory.length === 0) return 0;
    return this.flagHistory.filter(Boolean).length / this.flagHistory.length;
  }
}

/** Singleton instance for use across the app */
export const emgDetector = new EMGDetector();

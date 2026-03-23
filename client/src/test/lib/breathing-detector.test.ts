import { describe, it, expect } from "vitest";
import { detectBreathingState, type BreathingAnalysis, type BreathingState } from "@/lib/breathing-detector";

// ── Helpers ──────────────────────────────────────────────────────────────────

const FS = 256; // Muse 2 sampling rate

/**
 * Generate a sine wave at a given frequency, simulating a respiratory
 * oscillation embedded in frontal EEG.
 *
 * @param freqHz - Breathing frequency (e.g. 0.2 Hz = 12 breaths/min)
 * @param amplitude - Signal amplitude in microvolts
 * @param durationSec - Duration in seconds
 * @param fs - Sampling rate
 */
function generateBreathingSine(
  freqHz: number,
  amplitude: number,
  durationSec: number,
  fs: number,
): Float32Array {
  const n = Math.floor(fs * durationSec);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = amplitude * Math.sin(2 * Math.PI * freqHz * (i / fs));
  }
  return out;
}

/**
 * Generate a noisy random signal (no periodic breathing pattern).
 */
function generateRandomSignal(durationSec: number, fs: number): Float32Array {
  const n = Math.floor(fs * durationSec);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = (Math.random() - 0.5) * 20;
  }
  return out;
}

// ── Tests ────────────────────────────────────────────────────────────────────

describe("detectBreathingState", () => {
  it('classifies 0.2 Hz sine (12 breaths/min) as "normal"', () => {
    // 0.2 Hz = 12 breaths per minute, well within the normal range (8-16)
    const signal = generateBreathingSine(0.2, 50, 30, FS);
    const result = detectBreathingState(signal, FS);
    expect(result.state).toBe("normal");
    expect(result.estimatedRate).toBeGreaterThanOrEqual(10);
    expect(result.estimatedRate).toBeLessThanOrEqual(14);
  });

  it('classifies 0.1 Hz sine (6 breaths/min) as "deep_slow"', () => {
    // 0.1 Hz = 6 breaths per minute, deep meditation breathing (< 8)
    const signal = generateBreathingSine(0.1, 50, 60, FS);
    const result = detectBreathingState(signal, FS);
    expect(result.state).toBe("deep_slow");
    expect(result.estimatedRate).toBeGreaterThanOrEqual(4);
    expect(result.estimatedRate).toBeLessThanOrEqual(8);
  });

  it('classifies 0.3 Hz sine (18 breaths/min) as "shallow_fast"', () => {
    // 0.3 Hz = 18 breaths per minute, shallow/fast breathing (> 16)
    const signal = generateBreathingSine(0.3, 50, 30, FS);
    const result = detectBreathingState(signal, FS);
    expect(result.state).toBe("shallow_fast");
    expect(result.estimatedRate).toBeGreaterThanOrEqual(16);
    expect(result.estimatedRate).toBeLessThanOrEqual(20);
  });

  it('classifies flat/zero signal as "holding" or "unknown"', () => {
    // No oscillation at all — breath hold or no signal
    const n = Math.floor(FS * 30);
    const signal = new Float32Array(n); // all zeros
    const result = detectBreathingState(signal, FS);
    expect(["holding", "unknown"]).toContain(result.state);
  });

  it("estimates rate within +/- 2 breaths/min of true rate", () => {
    // Test multiple known frequencies
    const testCases: Array<{ freqHz: number; expectedBpm: number }> = [
      { freqHz: 0.1, expectedBpm: 6 },
      { freqHz: 0.15, expectedBpm: 9 },
      { freqHz: 0.2, expectedBpm: 12 },
      { freqHz: 0.25, expectedBpm: 15 },
      { freqHz: 0.35, expectedBpm: 21 },
    ];
    for (const { freqHz, expectedBpm } of testCases) {
      const signal = generateBreathingSine(freqHz, 50, 60, FS);
      const result = detectBreathingState(signal, FS);
      expect(Math.abs(result.estimatedRate - expectedBpm)).toBeLessThanOrEqual(2);
    }
  });

  it("returns high coherence for a regular sine wave", () => {
    const signal = generateBreathingSine(0.2, 50, 60, FS);
    const result = detectBreathingState(signal, FS);
    expect(result.coherence).toBeGreaterThanOrEqual(0.7);
  });

  it("returns lower coherence for a random/noisy signal than for a sine", () => {
    // Random noise bandpass-filtered to a narrow band can still produce
    // some apparent periodicity, so we compare against the regular sine
    // rather than using an absolute threshold.
    const sineSignal = generateBreathingSine(0.2, 50, 60, FS);
    const sineResult = detectBreathingState(sineSignal, FS);

    const noiseSignal = generateRandomSignal(60, FS);
    const noiseResult = detectBreathingState(noiseSignal, FS);

    expect(noiseResult.coherence).toBeLessThan(sineResult.coherence);
  });

  it("returns non-empty message for all states", () => {
    const states: BreathingState[] = ["deep_slow", "normal", "shallow_fast", "holding", "unknown"];
    // Generate signals that would produce each state
    const signals: Float32Array[] = [
      generateBreathingSine(0.1, 50, 60, FS),   // deep_slow
      generateBreathingSine(0.2, 50, 30, FS),    // normal
      generateBreathingSine(0.35, 50, 30, FS),   // shallow_fast
      new Float32Array(Math.floor(FS * 30)),      // holding/unknown (flat)
      new Float32Array(Math.floor(FS * 30)),      // holding/unknown (flat)
    ];
    for (const signal of signals) {
      const result = detectBreathingState(signal, FS);
      expect(result.message).toBeTruthy();
      expect(result.message.length).toBeGreaterThan(0);
    }
  });
});

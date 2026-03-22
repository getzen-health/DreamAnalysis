import { describe, it, expect } from "vitest";
import { detectBlinks, computeBlinkStats, type BlinkStats } from "@/lib/blink-detector";

// ── Helpers ──────────────────────────────────────────────────────────────────

const FS = 256; // Muse 2 sampling rate

/**
 * Generate a pure sine wave with no blink-like spikes.
 * This should produce 0 detected blinks.
 */
function generateCleanSine(freqHz: number, amplitudeUv: number, durationSec: number, fs: number): Float32Array {
  const n = Math.floor(fs * durationSec);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = amplitudeUv * Math.sin(2 * Math.PI * freqHz * (i / fs));
  }
  return out;
}

/**
 * Inject blink-like artifacts into a signal at specified times.
 * Blinks are modeled as Gaussian-shaped peaks with given amplitude and duration.
 *
 * @param base - Base signal to inject into (will be copied, not mutated)
 * @param fs - Sampling rate
 * @param blinkTimes - Array of blink onset times in seconds
 * @param blinkAmplitude - Peak amplitude of each blink in uV
 * @param blinkDurationMs - Duration of each blink in milliseconds
 */
function injectBlinks(
  base: Float32Array,
  fs: number,
  blinkTimes: number[],
  blinkAmplitude: number,
  blinkDurationMs: number,
): Float32Array {
  const out = new Float32Array(base);
  const blinkSamples = Math.floor((blinkDurationMs / 1000) * fs);
  const halfWidth = Math.floor(blinkSamples / 2);

  for (const t of blinkTimes) {
    const centerSample = Math.floor(t * fs);
    for (let i = -halfWidth; i <= halfWidth; i++) {
      const idx = centerSample + i;
      if (idx >= 0 && idx < out.length) {
        // Gaussian-shaped blink artifact
        const sigma = halfWidth / 2.5;
        const gaussian = Math.exp(-(i * i) / (2 * sigma * sigma));
        out[idx] += blinkAmplitude * gaussian;
      }
    }
  }
  return out;
}

// ── detectBlinks tests ───────────────────────────────────────────────────────

describe("detectBlinks", () => {
  it("returns 0 blinks for a clean sine wave with no spikes", () => {
    const signal = generateCleanSine(10, 30, 4, FS);
    const onsets = detectBlinks(signal, FS);
    expect(onsets).toHaveLength(0);
  });

  it("detects correct number of blinks from injected 100uV spikes", () => {
    const base = generateCleanSine(10, 20, 10, FS);
    // Inject 5 blinks at known times, each 200ms duration, 100uV amplitude
    const blinkTimes = [1.0, 3.0, 5.0, 7.0, 9.0];
    const signal = injectBlinks(base, FS, blinkTimes, 200, 200);
    const onsets = detectBlinks(signal, FS);
    expect(onsets.length).toBe(5);
  });

  it("does not count a spike shorter than 100ms as a blink", () => {
    const base = generateCleanSine(10, 20, 4, FS);
    // Inject a spike that is only 50ms — should be rejected
    const signal = injectBlinks(base, FS, [2.0], 200, 50);
    const onsets = detectBlinks(signal, FS);
    expect(onsets.length).toBe(0);
  });

  it("does not count a spike longer than 400ms as a blink", () => {
    const base = generateCleanSine(10, 20, 4, FS);
    // Inject a spike that is 500ms — too long for a blink
    const signal = injectBlinks(base, FS, [2.0], 200, 500);
    const onsets = detectBlinks(signal, FS);
    expect(onsets.length).toBe(0);
  });
});

// ── computeBlinkStats tests ──────────────────────────────────────────────────

describe("computeBlinkStats", () => {
  it('returns "focused" when <10 blinks/min and high alpha power', () => {
    // 60 seconds of signal with 5 blinks = 5 blinks/min
    const base = generateCleanSine(10, 20, 60, FS);
    const blinkTimes = [5, 15, 25, 35, 45];
    const signal = injectBlinks(base, FS, blinkTimes, 200, 200);
    const highAlpha = 0.5; // high alpha power
    const stats = computeBlinkStats(signal, FS, highAlpha);
    expect(stats.blinksPerMinute).toBeCloseTo(5, 0);
    expect(stats.alertnessState).toBe("focused");
    expect(stats.shouldSuggestBreak).toBe(false);
  });

  it('returns "drowsy" when <10 blinks/min and low alpha (high theta implied)', () => {
    const base = generateCleanSine(10, 20, 60, FS);
    const blinkTimes = [10, 30, 50];
    const signal = injectBlinks(base, FS, blinkTimes, 200, 200);
    const lowAlpha = 0.05; // low alpha → high theta implied
    const stats = computeBlinkStats(signal, FS, lowAlpha);
    expect(stats.blinksPerMinute).toBeCloseTo(3, 0);
    expect(stats.alertnessState).toBe("drowsy");
    expect(stats.shouldSuggestBreak).toBe(true);
  });

  it('returns "fatigued" when >25 blinks/min and sets shouldSuggestBreak', () => {
    // 60 seconds of signal with 30 blinks = 30 blinks/min
    const base = generateCleanSine(10, 20, 60, FS);
    const blinkTimes: number[] = [];
    for (let i = 0; i < 30; i++) {
      blinkTimes.push(1 + i * 1.9); // every ~1.9 seconds for 30 blinks
    }
    const signal = injectBlinks(base, FS, blinkTimes, 200, 200);
    const normalAlpha = 0.2;
    const stats = computeBlinkStats(signal, FS, normalAlpha);
    expect(stats.blinksPerMinute).toBeGreaterThan(25);
    expect(stats.alertnessState).toBe("fatigued");
    expect(stats.shouldSuggestBreak).toBe(true);
  });

  it('returns "normal" when blink rate is in 10-20 range', () => {
    // 60 seconds with 15 blinks = 15 blinks/min
    const base = generateCleanSine(10, 20, 60, FS);
    const blinkTimes: number[] = [];
    for (let i = 0; i < 15; i++) {
      blinkTimes.push(2 + i * 3.7); // every ~3.7 seconds for 15 blinks
    }
    const signal = injectBlinks(base, FS, blinkTimes, 200, 200);
    const normalAlpha = 0.2;
    const stats = computeBlinkStats(signal, FS, normalAlpha);
    expect(stats.blinksPerMinute).toBeGreaterThanOrEqual(10);
    expect(stats.blinksPerMinute).toBeLessThanOrEqual(20);
    expect(stats.alertnessState).toBe("normal");
    expect(stats.shouldSuggestBreak).toBe(false);
  });
});

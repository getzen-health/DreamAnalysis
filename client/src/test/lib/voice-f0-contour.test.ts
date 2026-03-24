import { describe, it, expect, beforeEach } from "vitest";
import {
  extractF0ContourFeatures,
  type F0ContourFeatures,
  clearBaselinePitchF0,
} from "@/lib/voice-onnx";

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Generate a sine wave at the given frequency. */
function makeSineWave(sr: number, durationSec: number, freq = 440): Float32Array {
  const n = Math.floor(sr * durationSec);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = Math.sin((2 * Math.PI * freq * i) / sr);
  }
  return out;
}

/** Generate silence (all zeros). */
function makeSilence(sr: number, durationSec: number): Float32Array {
  return new Float32Array(Math.floor(sr * durationSec));
}

/**
 * Generate a chirp signal that sweeps from startFreq to endFreq.
 * Useful for testing contour detection — a rising chirp should produce
 * a "rising" contour type.
 */
function makeChirp(
  sr: number,
  durationSec: number,
  startFreq: number,
  endFreq: number,
): Float32Array {
  const n = Math.floor(sr * durationSec);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / sr;
    // Linear frequency sweep
    const instantFreq = startFreq + ((endFreq - startFreq) * t) / durationSec;
    // Integrate frequency to get phase
    const phase =
      2 * Math.PI * (startFreq * t + ((endFreq - startFreq) * t * t) / (2 * durationSec));
    out[i] = 0.8 * Math.sin(phase);
  }
  return out;
}

/**
 * Generate a signal with alternating tone and silence (speech-like).
 */
function makeSpeechLikeSignal(sr: number, durationSec: number): Float32Array {
  const n = Math.floor(sr * durationSec);
  const out = new Float32Array(n);
  const segmentLen = Math.floor(sr * 0.2);
  for (let i = 0; i < n; i++) {
    const segment = Math.floor(i / segmentLen);
    if (segment % 2 === 0) {
      out[i] = 0.5 * Math.sin((2 * Math.PI * 200 * i) / sr);
    }
  }
  return out;
}

// ── Tests ────────────────────────────────────────────────────────────────────

describe("extractF0ContourFeatures", () => {
  beforeEach(() => clearBaselinePitchF0());

  it("returns a well-typed F0ContourFeatures object", () => {
    const sine = makeSineWave(16000, 1.0, 200);
    const result = extractF0ContourFeatures(sine, 16000);

    expect(result).toHaveProperty("contourType");
    expect(result).toHaveProperty("contourVariance");
    expect(result).toHaveProperty("maxExcursionPosition");
    expect(result).toHaveProperty("quarterSlopes");
  });

  it("contourType is one of the valid types", () => {
    const sine = makeSineWave(16000, 1.0, 200);
    const result = extractF0ContourFeatures(sine, 16000);
    const validTypes = ["rising", "falling", "flat", "U", "inverted-U"];
    expect(validTypes).toContain(result.contourType);
  });

  it("contourVariance is non-negative", () => {
    const sine = makeSineWave(16000, 1.0, 200);
    const result = extractF0ContourFeatures(sine, 16000);
    expect(result.contourVariance).toBeGreaterThanOrEqual(0);
  });

  it("maxExcursionPosition is in [0, 1]", () => {
    const sine = makeSineWave(16000, 1.0, 200);
    const result = extractF0ContourFeatures(sine, 16000);
    expect(result.maxExcursionPosition).toBeGreaterThanOrEqual(0);
    expect(result.maxExcursionPosition).toBeLessThanOrEqual(1);
  });

  it("quarterSlopes has exactly 4 elements", () => {
    const sine = makeSineWave(16000, 1.0, 200);
    const result = extractF0ContourFeatures(sine, 16000);
    expect(result.quarterSlopes).toHaveLength(4);
  });

  it("all quarterSlopes are finite", () => {
    const sine = makeSineWave(16000, 1.0, 200);
    const result = extractF0ContourFeatures(sine, 16000);
    for (const slope of result.quarterSlopes) {
      expect(Number.isFinite(slope)).toBe(true);
    }
  });

  it("returns flat contour for a constant-frequency sine wave", () => {
    const sine = makeSineWave(16000, 1.5, 200);
    const result = extractF0ContourFeatures(sine, 16000);
    // A pure tone at 200 Hz should have nearly zero variance and flat contour
    expect(result.contourType).toBe("flat");
    expect(result.contourVariance).toBeLessThan(0.5);
  });

  it("returns all zeros / flat for silence (no voiced frames)", () => {
    const silence = makeSilence(16000, 1.0);
    const result = extractF0ContourFeatures(silence, 16000);
    // With no voiced frames, should return safe defaults
    expect(result.contourType).toBe("flat");
    expect(result.contourVariance).toBe(0);
    expect(result.maxExcursionPosition).toBe(0.5);
    expect(result.quarterSlopes).toEqual([0, 0, 0, 0]);
  });

  it("detects rising contour for a rising chirp (100 -> 300 Hz)", () => {
    const chirp = makeChirp(16000, 2.0, 100, 300);
    const result = extractF0ContourFeatures(chirp, 16000);
    // Rising chirp should produce positive overall slope
    // The contour type should be "rising"
    expect(result.contourType).toBe("rising");
    // Most quarter slopes should be positive
    const positiveSlopes = result.quarterSlopes.filter((s) => s > 0).length;
    expect(positiveSlopes).toBeGreaterThanOrEqual(2);
  });

  it("detects falling contour for a falling chirp (300 -> 100 Hz)", () => {
    const chirp = makeChirp(16000, 2.0, 300, 100);
    const result = extractF0ContourFeatures(chirp, 16000);
    expect(result.contourType).toBe("falling");
    // Most quarter slopes should be negative
    const negativeSlopes = result.quarterSlopes.filter((s) => s < 0).length;
    expect(negativeSlopes).toBeGreaterThanOrEqual(2);
  });

  it("contourVariance is higher for a chirp than for a flat tone", () => {
    const flat = makeSineWave(16000, 2.0, 200);
    const chirp = makeChirp(16000, 2.0, 100, 300);
    const flatResult = extractF0ContourFeatures(flat, 16000);
    const chirpResult = extractF0ContourFeatures(chirp, 16000);
    expect(chirpResult.contourVariance).toBeGreaterThan(flatResult.contourVariance);
  });

  it("handles very short audio (< 100ms) without crashing", () => {
    const short = makeSineWave(16000, 0.05, 200);
    const result = extractF0ContourFeatures(short, 16000);
    expect(result).toHaveProperty("contourType");
    expect(result).toHaveProperty("contourVariance");
    expect(result).toHaveProperty("maxExcursionPosition");
    expect(result).toHaveProperty("quarterSlopes");
    expect(result.quarterSlopes).toHaveLength(4);
  });

  it("handles speech-like signal with pauses", () => {
    const signal = makeSpeechLikeSignal(16000, 2.0);
    const result = extractF0ContourFeatures(signal, 16000);
    // Should not crash; contour type should be valid
    const validTypes = ["rising", "falling", "flat", "U", "inverted-U"];
    expect(validTypes).toContain(result.contourType);
    expect(Number.isFinite(result.contourVariance)).toBe(true);
    expect(result.quarterSlopes).toHaveLength(4);
  });

  it("all returned values are finite for any input", () => {
    const inputs = [
      makeSineWave(16000, 1.0, 200),
      makeSilence(16000, 1.0),
      makeChirp(16000, 1.5, 100, 300),
      makeSpeechLikeSignal(16000, 1.5),
    ];
    for (const input of inputs) {
      const result = extractF0ContourFeatures(input, 16000);
      expect(Number.isFinite(result.contourVariance)).toBe(true);
      expect(Number.isFinite(result.maxExcursionPosition)).toBe(true);
      for (const s of result.quarterSlopes) {
        expect(Number.isFinite(s)).toBe(true);
      }
    }
  });
});

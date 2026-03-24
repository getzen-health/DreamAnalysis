import { describe, it, expect } from "vitest";
import {
  extractBandPowers,
  BAND_IMPORTANCE_WEIGHTS,
} from "@/lib/eeg-features";

// ── Helpers ──────────────────────────────────────────────────────────────────

const FS = 256;

/** Generate a pure sine wave signal. */
function generateSine(freqHz: number, amplitude: number, durationSec: number): number[] {
  const n = Math.floor(FS * durationSec);
  const out: number[] = new Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = amplitude * Math.sin(2 * Math.PI * freqHz * (i / FS));
  }
  return out;
}

/** Generate a multi-frequency signal with known band composition. */
function generateMultiBand(
  bands: Array<{ freq: number; amplitude: number }>,
  durationSec: number,
): number[] {
  const n = Math.floor(FS * durationSec);
  const out: number[] = new Array(n).fill(0);
  for (const { freq, amplitude } of bands) {
    for (let i = 0; i < n; i++) {
      out[i] += amplitude * Math.sin(2 * Math.PI * freq * (i / FS));
    }
  }
  return out;
}

// ── Tests ────────────────────────────────────────────────────────────────────

describe("BAND_IMPORTANCE_WEIGHTS", () => {
  it("should export the weights constant", () => {
    expect(BAND_IMPORTANCE_WEIGHTS).toBeDefined();
    expect(typeof BAND_IMPORTANCE_WEIGHTS).toBe("object");
  });

  it("should have weights for all 5 standard bands", () => {
    expect(BAND_IMPORTANCE_WEIGHTS.delta).toBeDefined();
    expect(BAND_IMPORTANCE_WEIGHTS.theta).toBeDefined();
    expect(BAND_IMPORTANCE_WEIGHTS.alpha).toBeDefined();
    expect(BAND_IMPORTANCE_WEIGHTS.beta).toBeDefined();
    expect(BAND_IMPORTANCE_WEIGHTS.gamma).toBeDefined();
  });

  it("should weight alpha highest (most emotion-relevant)", () => {
    expect(BAND_IMPORTANCE_WEIGHTS.alpha).toBeGreaterThan(BAND_IMPORTANCE_WEIGHTS.beta);
    expect(BAND_IMPORTANCE_WEIGHTS.alpha).toBeGreaterThan(BAND_IMPORTANCE_WEIGHTS.gamma);
  });

  it("should weight theta second highest", () => {
    expect(BAND_IMPORTANCE_WEIGHTS.theta).toBeGreaterThan(BAND_IMPORTANCE_WEIGHTS.beta);
    expect(BAND_IMPORTANCE_WEIGHTS.theta).toBeGreaterThan(BAND_IMPORTANCE_WEIGHTS.gamma);
  });

  it("should strongly suppress gamma (EMG noise on Muse 2)", () => {
    expect(BAND_IMPORTANCE_WEIGHTS.gamma).toBeLessThan(BAND_IMPORTANCE_WEIGHTS.delta);
    expect(BAND_IMPORTANCE_WEIGHTS.gamma).toBeLessThan(0.5);
  });
});

describe("extractBandPowers with importance weighting", () => {
  it("should still return all 5 bands", () => {
    const signal = generateSine(10, 20, 4); // pure 10 Hz alpha
    const powers = extractBandPowers(signal, FS);
    expect(powers).toHaveProperty("delta");
    expect(powers).toHaveProperty("theta");
    expect(powers).toHaveProperty("alpha");
    expect(powers).toHaveProperty("beta");
    expect(powers).toHaveProperty("gamma");
  });

  it("should still sum band powers to approximately 1 (relative powers)", () => {
    const signal = generateMultiBand(
      [
        { freq: 2, amplitude: 10 },   // delta
        { freq: 6, amplitude: 10 },   // theta
        { freq: 10, amplitude: 10 },  // alpha
        { freq: 20, amplitude: 10 },  // beta
        { freq: 40, amplitude: 10 },  // gamma
      ],
      4,
    );
    const powers = extractBandPowers(signal, FS);
    const sum = Object.values(powers).reduce((a, b) => a + b, 0);
    // Should sum to ~1 since they're normalized after weighting
    expect(sum).toBeGreaterThan(0.9);
    expect(sum).toBeLessThan(1.1);
  });

  it("should amplify alpha relative to gamma for equal-power input", () => {
    // Signal with equal alpha and gamma power
    const signal = generateMultiBand(
      [
        { freq: 10, amplitude: 20 },  // alpha center
        { freq: 40, amplitude: 20 },  // gamma center
      ],
      4,
    );
    const powers = extractBandPowers(signal, FS);

    // After importance weighting, alpha should dominate over gamma
    // because alpha weight (1.5) >> gamma weight (0.3)
    expect(powers.alpha).toBeGreaterThan(powers.gamma);
  });

  it("should amplify theta relative to delta for equal-power input", () => {
    const signal = generateMultiBand(
      [
        { freq: 2, amplitude: 20 },  // delta center
        { freq: 6, amplitude: 20 },  // theta center
      ],
      4,
    );
    const powers = extractBandPowers(signal, FS);

    // Theta weight (1.3) > delta weight (0.8)
    // so theta should be relatively larger than delta
    expect(powers.theta).toBeGreaterThan(powers.delta);
  });

  it("should produce finite values for all bands", () => {
    const signal = generateMultiBand(
      [
        { freq: 2, amplitude: 5 },
        { freq: 6, amplitude: 10 },
        { freq: 10, amplitude: 15 },
        { freq: 20, amplitude: 8 },
        { freq: 40, amplitude: 3 },
      ],
      4,
    );
    const powers = extractBandPowers(signal, FS);
    for (const [band, value] of Object.entries(powers)) {
      expect(Number.isFinite(value)).toBe(true);
      expect(value).toBeGreaterThanOrEqual(0);
    }
  });

  it("should handle zero-length signal gracefully", () => {
    const powers = extractBandPowers([], FS);
    // Should return something finite, not crash
    for (const value of Object.values(powers)) {
      expect(Number.isFinite(value)).toBe(true);
    }
  });
});

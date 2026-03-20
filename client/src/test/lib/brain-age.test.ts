import { describe, it, expect } from "vitest";
import {
  estimateBrainAge,
  computeIAF,
  getBrainAgeResult,
  type BrainAgeResult,
} from "@/lib/brain-age";

// ── estimateBrainAge ─────────────────────────────────────────────────────────

describe("estimateBrainAge", () => {
  it("returns young brain age (25-35) for high IAF and low theta", () => {
    // High alpha peak (~10.5 Hz), low theta, moderate alpha, low beta, low entropy
    const age = estimateBrainAge(10.5, 0.10, 0.45, 0.15, 0.60);
    expect(age).toBeGreaterThanOrEqual(25);
    expect(age).toBeLessThanOrEqual(35);
  });

  it("returns older brain age (55-75) for low IAF and high theta", () => {
    // Low alpha peak (~8.5 Hz), high theta, lower alpha, higher beta, higher entropy
    // The high theta/alpha ratio (1.75) pushes the estimate well above the IAF baseline
    const age = estimateBrainAge(8.5, 0.35, 0.20, 0.25, 0.80);
    expect(age).toBeGreaterThanOrEqual(55);
    expect(age).toBeLessThanOrEqual(75);
  });

  it("returns mid-range brain age (35-50) for average features", () => {
    // Mid alpha peak (~9.5 Hz), moderate theta, moderate alpha, moderate beta
    const age = estimateBrainAge(9.5, 0.20, 0.30, 0.18, 0.70);
    expect(age).toBeGreaterThanOrEqual(35);
    expect(age).toBeLessThanOrEqual(50);
  });

  it("clamps result to minimum 15", () => {
    // Very high IAF should push estimate below 15, but clamped
    const age = estimateBrainAge(12.5, 0.05, 0.60, 0.08, 0.50);
    expect(age).toBeGreaterThanOrEqual(15);
  });

  it("clamps result to maximum 90", () => {
    // Very low IAF plus all aging factors maxed
    const age = estimateBrainAge(6.0, 0.50, 0.10, 0.30, 0.95);
    expect(age).toBeLessThanOrEqual(90);
  });
});

// ── computeIAF ───────────────────────────────────────────────────────────────

describe("computeIAF", () => {
  it("finds correct peak frequency in alpha range", () => {
    // Build a synthetic power spectrum with a clear peak at ~10 Hz
    const fs = 256;
    const nfft = 256;
    const nFreqs = Math.floor(nfft / 2) + 1;
    const bandPowers: Record<string, number> = {};
    const psd: number[] = [];
    const freqs: number[] = [];

    for (let i = 0; i < nFreqs; i++) {
      const f = (i * fs) / nfft;
      freqs.push(f);
      // Gaussian peak centered at 10 Hz
      const power = Math.exp(-0.5 * ((f - 10) / 0.5) ** 2);
      psd.push(power);
    }

    const iaf = computeIAF(psd, freqs);
    // Should be close to 10 Hz
    expect(iaf).toBeGreaterThanOrEqual(9.0);
    expect(iaf).toBeLessThanOrEqual(11.0);
  });

  it("returns default 10 Hz when no clear peak exists (flat spectrum)", () => {
    const fs = 256;
    const nfft = 256;
    const nFreqs = Math.floor(nfft / 2) + 1;
    const psd: number[] = [];
    const freqs: number[] = [];

    for (let i = 0; i < nFreqs; i++) {
      const f = (i * fs) / nfft;
      freqs.push(f);
      psd.push(1.0); // flat
    }

    const iaf = computeIAF(psd, freqs);
    // Should fall back to default ~10 Hz or pick lowest in range
    expect(iaf).toBeGreaterThanOrEqual(7.0);
    expect(iaf).toBeLessThanOrEqual(13.0);
  });
});

// ── getBrainAgeResult ────────────────────────────────────────────────────────

describe("getBrainAgeResult", () => {
  it("computes brain age gap correctly (estimated - actual)", () => {
    const result = getBrainAgeResult({
      alphaPeakHz: 10.5,
      thetaPower: 0.10,
      alphaPower: 0.45,
      betaPower: 0.15,
      spectralEntropy: 0.60,
      actualAge: 30,
    });

    expect(result.brainAgeGap).toBe(result.estimatedAge - 30);
  });

  it("negative gap means younger brain", () => {
    // Young brain features for a 50-year-old
    const result = getBrainAgeResult({
      alphaPeakHz: 10.5,
      thetaPower: 0.10,
      alphaPower: 0.45,
      betaPower: 0.15,
      spectralEntropy: 0.60,
      actualAge: 50,
    });

    expect(result.brainAgeGap).toBeLessThan(0);
  });

  it("factor descriptions are non-empty strings", () => {
    const result = getBrainAgeResult({
      alphaPeakHz: 10.0,
      thetaPower: 0.15,
      alphaPower: 0.35,
      betaPower: 0.18,
      spectralEntropy: 0.68,
      actualAge: 40,
    });

    expect(result.factors.alphaPeak.length).toBeGreaterThan(0);
    expect(result.factors.thetaPower.length).toBeGreaterThan(0);
    expect(result.factors.betaRatio.length).toBeGreaterThan(0);
  });

  it("includes disclaimer string", () => {
    const result = getBrainAgeResult({
      alphaPeakHz: 10.0,
      thetaPower: 0.15,
      alphaPower: 0.35,
      betaPower: 0.18,
      spectralEntropy: 0.68,
      actualAge: 40,
    });

    expect(typeof result.disclaimer).toBe("string");
    expect(result.disclaimer.length).toBeGreaterThan(0);
  });

  it("returns alphaPeakHz in the result", () => {
    const result = getBrainAgeResult({
      alphaPeakHz: 9.8,
      thetaPower: 0.20,
      alphaPower: 0.30,
      betaPower: 0.20,
      spectralEntropy: 0.72,
      actualAge: 45,
    });

    expect(result.alphaPeakHz).toBe(9.8);
  });

  it("confidence is between 0 and 1", () => {
    const result = getBrainAgeResult({
      alphaPeakHz: 10.0,
      thetaPower: 0.15,
      alphaPower: 0.35,
      betaPower: 0.18,
      spectralEntropy: 0.68,
      actualAge: 40,
    });

    expect(result.confidence).toBeGreaterThanOrEqual(0);
    expect(result.confidence).toBeLessThanOrEqual(1);
  });
});

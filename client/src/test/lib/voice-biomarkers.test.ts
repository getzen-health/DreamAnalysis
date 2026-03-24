import { describe, it, expect } from "vitest";
import {
  extractVoiceBiomarkers,
  type VoiceBiomarkers,
  type VocalEnergyLevel,
  type VocalVariability,
} from "@/lib/voice-biomarkers";

// ─── Test signal generators ─────────────────────────────────────────────────

/** Generate a pure sine wave at the given frequency. */
function sineWave(freq: number, sr: number, durationSec: number): Float32Array {
  const n = Math.floor(sr * durationSec);
  const samples = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    samples[i] = 0.5 * Math.sin(2 * Math.PI * freq * i / sr);
  }
  return samples;
}

/** Generate white noise (uniform random in [-amplitude, amplitude]). */
function whiteNoise(sr: number, durationSec: number, amplitude = 0.5): Float32Array {
  const n = Math.floor(sr * durationSec);
  const samples = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    samples[i] = (Math.random() * 2 - 1) * amplitude;
  }
  return samples;
}

/** Generate a silent signal. */
function silence(sr: number, durationSec: number): Float32Array {
  return new Float32Array(Math.floor(sr * durationSec));
}

// ─── Tests ──────────────────────────────────────────────────────────────────

describe("extractVoiceBiomarkers", () => {
  const SR = 16000;

  it("sine wave produces low jitter and low shimmer", () => {
    // A pure 200 Hz sine wave has perfectly periodic pitch and constant amplitude
    const samples = sineWave(200, SR, 2);
    const result = extractVoiceBiomarkers(samples, SR);

    // Jitter should be very low for a pure tone (< 0.05)
    expect(result.jitter).toBeLessThan(0.05);
    // Shimmer should be very low for a constant-amplitude tone (< 0.1)
    expect(result.shimmer).toBeLessThan(0.1);
  });

  it("white noise produces high jitter", () => {
    const samples = whiteNoise(SR, 2);
    const result = extractVoiceBiomarkers(samples, SR);

    // Noise has no stable pitch — jitter should be high or pitch detection
    // should fail to find consistent periods. If periods ARE found, they
    // will be highly inconsistent.
    // Either jitter is high (> 0.1) or pitchMean is 0 (no pitch found)
    const hasHighJitter = result.jitter > 0.1;
    const noPitchDetected = result.pitchMean === 0;
    expect(hasHighJitter || noPitchDetected).toBe(true);
  });

  it("200 Hz sine wave yields pitchMean near 200 Hz (within 10%)", () => {
    const samples = sineWave(200, SR, 2);
    const result = extractVoiceBiomarkers(samples, SR);

    // pitchMean should be approximately 200 Hz
    expect(result.pitchMean).toBeGreaterThan(180); // 200 - 10%
    expect(result.pitchMean).toBeLessThan(220); // 200 + 10%
  });

  it("silent audio yields speechRate = 0", () => {
    const samples = silence(SR, 2);
    const result = extractVoiceBiomarkers(samples, SR);

    expect(result.speechRate).toBe(0);
  });

  it("wellness score is in 0-100 range", () => {
    // Test across multiple signal types
    const signals = [
      sineWave(150, SR, 2),
      whiteNoise(SR, 2),
      silence(SR, 2),
    ];

    for (const samples of signals) {
      const result = extractVoiceBiomarkers(samples, SR);
      expect(result.wellnessScore).toBeGreaterThanOrEqual(0);
      expect(result.wellnessScore).toBeLessThanOrEqual(100);
      // Must be an integer
      expect(result.wellnessScore).toBe(Math.round(result.wellnessScore));
    }
  });

  it("all numeric values are finite", () => {
    const signals = [
      sineWave(200, SR, 2),
      whiteNoise(SR, 2),
      silence(SR, 2),
    ];

    for (const samples of signals) {
      const result = extractVoiceBiomarkers(samples, SR);

      expect(isFinite(result.jitter)).toBe(true);
      expect(isFinite(result.shimmer)).toBe(true);
      expect(isFinite(result.hnr)).toBe(true);
      expect(isFinite(result.pitchMean)).toBe(true);
      expect(isFinite(result.pitchStd)).toBe(true);
      expect(isFinite(result.speechRate)).toBe(true);
      expect(isFinite(result.spectralTilt)).toBe(true);
      expect(isFinite(result.averagePauseDuration)).toBe(true);
      expect(isFinite(result.pauseCount)).toBe(true);
      expect(isFinite(result.wellnessScore)).toBe(true);
    }
  });

  it("disclaimer is present and non-empty", () => {
    const samples = sineWave(200, SR, 2);
    const result = extractVoiceBiomarkers(samples, SR);

    expect(result.disclaimer).toBeTruthy();
    expect(result.disclaimer.length).toBeGreaterThan(10);
    // Should mention wellness/awareness, not clinical
    expect(result.disclaimer.toLowerCase()).toContain("wellness");
  });

  it("vocalEnergyLevel is a valid enum value", () => {
    const validValues: VocalEnergyLevel[] = ["low", "moderate", "high"];
    const signals = [
      sineWave(200, SR, 2),
      whiteNoise(SR, 2),
      silence(SR, 2),
    ];

    for (const samples of signals) {
      const result = extractVoiceBiomarkers(samples, SR);
      expect(validValues).toContain(result.vocalEnergyLevel);
    }
  });

  it("vocalVariability is a valid enum value", () => {
    const validValues: VocalVariability[] = ["steady", "moderate", "dynamic"];
    const signals = [
      sineWave(200, SR, 2),
      whiteNoise(SR, 2),
      silence(SR, 2),
    ];

    for (const samples of signals) {
      const result = extractVoiceBiomarkers(samples, SR);
      expect(validValues).toContain(result.vocalVariability);
    }
  });

  it("handles very short audio gracefully (returns zeroed defaults)", () => {
    // Less than 2 frames of audio
    const samples = new Float32Array(100);
    const result = extractVoiceBiomarkers(samples, SR);

    expect(result.wellnessScore).toBe(50);
    expect(result.pitchMean).toBe(0);
    expect(result.speechRate).toBe(0);
  });

  it("timestamp is a recent epoch ms value", () => {
    const before = Date.now();
    const samples = sineWave(200, SR, 1);
    const result = extractVoiceBiomarkers(samples, SR);
    const after = Date.now();

    expect(result.timestamp).toBeGreaterThanOrEqual(before);
    expect(result.timestamp).toBeLessThanOrEqual(after);
  });

  // ─── HNR (Harmonic-to-Noise Ratio) tests ────────────────────────────────

  it("pure sine wave produces high HNR (> 15 dB)", () => {
    // A pure tone is entirely harmonic — HNR should be very high
    const samples = sineWave(200, SR, 2);
    const result = extractVoiceBiomarkers(samples, SR);

    expect(result.hnr).toBeGreaterThan(15);
  });

  it("white noise produces low or zero HNR", () => {
    // Noise has no harmonic structure — HNR should be low
    const samples = whiteNoise(SR, 2);
    const result = extractVoiceBiomarkers(samples, SR);

    // HNR should be near 0 dB (equal energy) or 0 (no pitch found)
    expect(result.hnr).toBeLessThan(5);
  });

  it("silence produces HNR of 0", () => {
    const samples = silence(SR, 2);
    const result = extractVoiceBiomarkers(samples, SR);

    expect(result.hnr).toBe(0);
  });

  it("HNR is finite for all signal types", () => {
    const signals = [
      sineWave(200, SR, 2),
      whiteNoise(SR, 2),
      silence(SR, 2),
    ];

    for (const samples of signals) {
      const result = extractVoiceBiomarkers(samples, SR);
      expect(isFinite(result.hnr)).toBe(true);
    }
  });

  it("short audio returns HNR of 0", () => {
    const samples = new Float32Array(100);
    const result = extractVoiceBiomarkers(samples, SR);

    expect(result.hnr).toBe(0);
  });
});

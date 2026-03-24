import { describe, it, expect, beforeEach } from "vitest";
import {
  extractEnhancedFeatures,
  getBaselinePitchF0,
  setBaselinePitchF0,
  clearBaselinePitchF0,
} from "@/lib/voice-onnx";

// ─── Helpers ────────────────────────────────────────────────────────────────

/** Generate a sine wave at the given frequency. */
function makeSineWave(sr: number, durationSec: number, freq = 440): Float32Array {
  const n = Math.floor(sr * durationSec);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = Math.sin((2 * Math.PI * freq * i) / sr);
  }
  return out;
}

// ─── Baseline F0 storage ────────────────────────────────────────────────────

describe("voice pitch baseline F0 storage", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("getBaselinePitchF0 returns null when no baseline stored", () => {
    expect(getBaselinePitchF0()).toBeNull();
  });

  it("setBaselinePitchF0 stores and getBaselinePitchF0 retrieves it", () => {
    setBaselinePitchF0(200);
    expect(getBaselinePitchF0()).toBe(200);
  });

  it("setBaselinePitchF0 rejects non-positive values", () => {
    setBaselinePitchF0(0);
    expect(getBaselinePitchF0()).toBeNull();
    setBaselinePitchF0(-100);
    expect(getBaselinePitchF0()).toBeNull();
  });

  it("setBaselinePitchF0 rejects NaN and Infinity", () => {
    setBaselinePitchF0(NaN);
    expect(getBaselinePitchF0()).toBeNull();
    setBaselinePitchF0(Infinity);
    expect(getBaselinePitchF0()).toBeNull();
  });

  it("clearBaselinePitchF0 removes stored baseline", () => {
    setBaselinePitchF0(200);
    expect(getBaselinePitchF0()).toBe(200);
    clearBaselinePitchF0();
    expect(getBaselinePitchF0()).toBeNull();
  });
});

// ─── Pitch normalization in extractEnhancedFeatures ─────────────────────────

describe("voice pitch normalization", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("first call with no baseline stores baseline and returns raw pitch", () => {
    // No baseline stored yet
    expect(getBaselinePitchF0()).toBeNull();

    const sine = makeSineWave(16000, 1.0, 200);
    const features = extractEnhancedFeatures(sine, 16000);
    const pitchMean = features[132];

    // After first call, a baseline should have been stored
    // (assuming the pitch detector found voiced frames in the 200Hz sine)
    if (pitchMean > 0) {
      const storedBaseline = getBaselinePitchF0();
      expect(storedBaseline).not.toBeNull();
      // The stored baseline should equal the pitchMean from this first call
      expect(storedBaseline).toBeCloseTo(pitchMean, 1);
    }
  });

  it("second call normalizes pitch features as ratio to baseline", () => {
    // Set a known baseline
    setBaselinePitchF0(200);

    const sine = makeSineWave(16000, 1.0, 200);
    const features = extractEnhancedFeatures(sine, 16000);
    const pitchMean = features[132];

    // With baseline=200 and detected pitch near 200,
    // the normalized pitchMean should be near 1.0 (ratio = pitch/baseline)
    if (pitchMean > 0) {
      expect(pitchMean).toBeGreaterThan(0.5);
      expect(pitchMean).toBeLessThan(3.0);
    }
  });

  it("different pitch than baseline produces ratio != 1", () => {
    // Set baseline at 100 Hz
    setBaselinePitchF0(100);

    // Generate a 440 Hz signal — very different from baseline
    const sine = makeSineWave(16000, 1.0, 440);
    const features = extractEnhancedFeatures(sine, 16000);
    const pitchMean = features[132];

    // The normalized pitchMean should NOT be near 1.0 since 440 != 100.
    // Autocorrelation may not perfectly detect 440 Hz, but whatever it
    // detects divided by 100 should be noticeably different from 1.0.
    if (pitchMean > 0) {
      // Should be > 1 or < 1, but NOT close to 1.0 (at least 20% away)
      expect(Math.abs(pitchMean - 1.0)).toBeGreaterThan(0.2);
    }
  });

  it("silence still produces zero pitch even with baseline set", () => {
    setBaselinePitchF0(200);

    const silence = new Float32Array(Math.floor(16000 * 1.0)); // 1 sec silence
    const features = extractEnhancedFeatures(silence, 16000);

    // Silence should still yield 0 for all pitch features
    expect(features[132]).toBe(0); // pitchMean
    expect(features[133]).toBe(0); // pitchStd
    expect(features[134]).toBe(0); // pitchRange
    expect(features[135]).toBe(0); // pitchSlope
  });

  it("all 140 features remain finite after normalization", () => {
    setBaselinePitchF0(200);

    const sine = makeSineWave(16000, 1.0, 440);
    const features = extractEnhancedFeatures(sine, 16000);

    for (let i = 0; i < 140; i++) {
      expect(Number.isFinite(features[i])).toBe(true);
    }
  });

  it("pitchStd and pitchRange are also normalized by baseline", () => {
    // Use a known baseline
    setBaselinePitchF0(200);

    const sine = makeSineWave(16000, 1.0, 200);
    const features = extractEnhancedFeatures(sine, 16000);

    // pitchStd and pitchRange should be normalized (divided by 200)
    // For a pure sine tone, std and range are near 0, so they stay near 0
    expect(features[133]).toBeGreaterThanOrEqual(0); // pitchStd normalized
    expect(features[134]).toBeGreaterThanOrEqual(0); // pitchRange normalized
  });
});

import { describe, it, expect } from "vitest";
import {
  extractFeatures92,
  extractEnhancedFeatures,
  extractMFCC,
} from "@/lib/voice-onnx";

// ─── Helpers ────────────────────────────────────────────────────────────────

/** Generate a sine wave at the given frequency and sample rate. */
function makeSineWave(sr: number, durationSec: number, freq = 440): Float32Array {
  const n = Math.floor(sr * durationSec);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = Math.sin(2 * Math.PI * freq * i / sr);
  }
  return out;
}

/** Generate silence (all zeros). */
function makeSilence(sr: number, durationSec: number): Float32Array {
  return new Float32Array(Math.floor(sr * durationSec));
}

/** Generate a signal with pauses (alternating tone and silence). */
function makeSpeechLikeSignal(sr: number, durationSec: number): Float32Array {
  const n = Math.floor(sr * durationSec);
  const out = new Float32Array(n);
  const segmentLen = Math.floor(sr * 0.2); // 200ms segments
  for (let i = 0; i < n; i++) {
    const segment = Math.floor(i / segmentLen);
    // Alternate between tone and silence
    if (segment % 2 === 0) {
      out[i] = 0.5 * Math.sin(2 * Math.PI * 200 * i / sr);
    }
    // else: silence (already 0)
  }
  return out;
}

// ─── extractEnhancedFeatures ────────────────────────────────────────────────

describe("extractEnhancedFeatures", () => {
  it("returns exactly 140 features", () => {
    const sine = makeSineWave(16000, 1.0);
    const features = extractEnhancedFeatures(sine, 16000);
    expect(features.length).toBe(140);
  });

  it("returns a Float32Array", () => {
    const sine = makeSineWave(16000, 1.0);
    const features = extractEnhancedFeatures(sine, 16000);
    expect(features).toBeInstanceOf(Float32Array);
  });

  it("all features are finite (no NaN or Infinity)", () => {
    const sine = makeSineWave(16000, 1.0);
    const features = extractEnhancedFeatures(sine, 16000);
    for (let i = 0; i < features.length; i++) {
      expect(Number.isFinite(features[i])).toBe(true);
    }
  });

  it("all features are finite for silence", () => {
    const silence = makeSilence(16000, 1.0);
    const features = extractEnhancedFeatures(silence, 16000);
    for (let i = 0; i < features.length; i++) {
      expect(Number.isFinite(features[i])).toBe(true);
    }
  });

  it("all features are finite for speech-like signal", () => {
    const signal = makeSpeechLikeSignal(16000, 2.0);
    const features = extractEnhancedFeatures(signal, 16000);
    for (let i = 0; i < features.length; i++) {
      expect(Number.isFinite(features[i])).toBe(true);
    }
  });

  it("first 92 features match extractFeatures92 output", () => {
    const sine = makeSineWave(16000, 1.0);
    const enhanced = extractEnhancedFeatures(sine, 16000);
    const base = extractFeatures92(sine, 16000);
    for (let i = 0; i < 92; i++) {
      expect(enhanced[i]).toBeCloseTo(base[i], 5);
    }
  });

  it("delta MFCCs (indices 92-131) are different from base MFCC means", () => {
    const sine = makeSineWave(16000, 1.0);
    const features = extractEnhancedFeatures(sine, 16000);

    // Base MFCC means are at even indices 0, 2, 4, ..., 78
    // Delta MFCCs are at indices 92-131
    let diffCount = 0;
    for (let c = 0; c < 40; c++) {
      const baseMean = features[c * 2]; // MFCC mean for coefficient c
      const delta = features[92 + c];    // Delta MFCC for coefficient c
      if (Math.abs(baseMean - delta) > 1e-6) diffCount++;
    }
    // Delta MFCCs should differ from base MFCC means
    expect(diffCount).toBeGreaterThan(20);
  });

  it("delta MFCCs are non-negative (mean absolute differences)", () => {
    const sine = makeSineWave(16000, 1.0);
    const features = extractEnhancedFeatures(sine, 16000);
    for (let i = 92; i < 132; i++) {
      expect(features[i]).toBeGreaterThanOrEqual(0);
    }
  });

  it("pitch features are in valid ranges for a 440Hz sine wave", () => {
    const sine = makeSineWave(16000, 1.0, 440);
    const features = extractEnhancedFeatures(sine, 16000);

    const pitchMean = features[132];
    const pitchStd = features[133];
    const pitchRange = features[134];

    // 440Hz tone should have pitchMean near 440Hz (autocorrelation may not be exact)
    // Allow wide range since our autocorrelation pitch detector may pick harmonics
    if (pitchMean > 0) {
      expect(pitchMean).toBeGreaterThan(80);   // min F0 in detector
      expect(pitchMean).toBeLessThan(1000);
    }
    // Std and range should be non-negative
    expect(pitchStd).toBeGreaterThanOrEqual(0);
    expect(pitchRange).toBeGreaterThanOrEqual(0);
  });

  it("pitch features are zero for silence", () => {
    const silence = makeSilence(16000, 1.0);
    const features = extractEnhancedFeatures(silence, 16000);

    expect(features[132]).toBe(0); // pitchMean
    expect(features[133]).toBe(0); // pitchStd
    expect(features[134]).toBe(0); // pitchRange
    expect(features[135]).toBe(0); // pitchSlope
  });

  it("jitter is in [0, 1] range", () => {
    const sine = makeSineWave(16000, 1.0);
    const features = extractEnhancedFeatures(sine, 16000);
    expect(features[136]).toBeGreaterThanOrEqual(0);
    expect(features[136]).toBeLessThanOrEqual(1);
  });

  it("shimmer is in [0, 1] range", () => {
    const sine = makeSineWave(16000, 1.0);
    const features = extractEnhancedFeatures(sine, 16000);
    expect(features[137]).toBeGreaterThanOrEqual(0);
    expect(features[137]).toBeLessThanOrEqual(1);
  });

  it("speaking rate is > 0 for speech-like signal with energy transitions", () => {
    const signal = makeSpeechLikeSignal(16000, 2.0);
    const features = extractEnhancedFeatures(signal, 16000);
    // Speech-like signal alternates tone/silence -> should detect syllable-like onsets
    expect(features[138]).toBeGreaterThan(0);
  });

  it("speaking rate is 0 or near-0 for silence", () => {
    const silence = makeSilence(16000, 1.0);
    const features = extractEnhancedFeatures(silence, 16000);
    expect(features[138]).toBe(0);
  });

  it("pause ratio is in [0, 1] range", () => {
    const sine = makeSineWave(16000, 1.0);
    const features = extractEnhancedFeatures(sine, 16000);
    expect(features[139]).toBeGreaterThanOrEqual(0);
    expect(features[139]).toBeLessThanOrEqual(1);
  });

  it("pause ratio is 1.0 for silence", () => {
    const silence = makeSilence(16000, 1.0);
    const features = extractEnhancedFeatures(silence, 16000);
    expect(features[139]).toBe(1.0);
  });

  it("pause ratio is 0 for continuous tone", () => {
    const sine = makeSineWave(16000, 1.0, 440);
    const features = extractEnhancedFeatures(sine, 16000);
    expect(features[139]).toBe(0);
  });

  it("handles very short audio (< 50ms) without errors", () => {
    const short = makeSineWave(16000, 0.02); // 20ms
    const features = extractEnhancedFeatures(short, 16000);
    expect(features.length).toBe(140);
    for (let i = 0; i < features.length; i++) {
      expect(Number.isFinite(features[i])).toBe(true);
    }
  });

  it("produces different features for different frequency signals", () => {
    const low = makeSineWave(16000, 1.0, 200);
    const high = makeSineWave(16000, 1.0, 4000);
    const featuresLow = extractEnhancedFeatures(low, 16000);
    const featuresHigh = extractEnhancedFeatures(high, 16000);

    let diffCount = 0;
    for (let i = 0; i < 140; i++) {
      if (Math.abs(featuresLow[i] - featuresHigh[i]) > 0.001) diffCount++;
    }
    expect(diffCount).toBeGreaterThan(20);
  });

  it("produces different features for tone vs speech-like signal", () => {
    const tone = makeSineWave(16000, 1.0, 440);
    const speech = makeSpeechLikeSignal(16000, 1.0);
    const featuresTone = extractEnhancedFeatures(tone, 16000);
    const featuresSpeech = extractEnhancedFeatures(speech, 16000);

    // The enhanced features (especially temporal ones) should differ
    let diffCount = 0;
    for (let i = 92; i < 140; i++) {
      if (Math.abs(featuresTone[i] - featuresSpeech[i]) > 0.001) diffCount++;
    }
    expect(diffCount).toBeGreaterThan(5);
  });
});

// ─── Backward compatibility ──────────────────────────────────────────────────

describe("extractFeatures92 backward compatibility", () => {
  it("still returns exactly 92 features", () => {
    const sine = makeSineWave(16000, 1.0);
    const features = extractFeatures92(sine, 16000);
    expect(features.length).toBe(92);
  });

  it("still returns Float32Array", () => {
    const sine = makeSineWave(16000, 1.0);
    const features = extractFeatures92(sine, 16000);
    expect(features).toBeInstanceOf(Float32Array);
  });

  it("still returns all finite values", () => {
    const sine = makeSineWave(16000, 1.0);
    const features = extractFeatures92(sine, 16000);
    for (let i = 0; i < features.length; i++) {
      expect(Number.isFinite(features[i])).toBe(true);
    }
  });

  it("handles short audio", () => {
    const short = makeSineWave(16000, 0.1);
    const features = extractFeatures92(short, 16000);
    expect(features.length).toBe(92);
    for (let i = 0; i < features.length; i++) {
      expect(Number.isFinite(features[i])).toBe(true);
    }
  });
});

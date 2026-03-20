import { describe, it, expect } from "vitest";
import {
  resampleTo16k,
  prepareAudioInput,
  extractMFCC,
  extractFeatures92,
  EMOTION_REGIONS,
  mapValenceArousalToEmotion,
} from "@/lib/voice-onnx";

// ─── Helpers ────────────────────────────────────────────────────────────────

/** Generate a 440Hz sine wave at the given sample rate. */
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

// ─── Existing tests ─────────────────────────────────────────────────────────

describe("resampleTo16k", () => {
  it("returns same array if already 16kHz", () => {
    const input = new Float32Array([0.1, -0.2, 0.3]);
    const result = resampleTo16k(input, 16000);
    expect(result.length).toBe(3);
    expect(result[0]).toBeCloseTo(0.1);
  });

  it("downsamples 48kHz to 16kHz (3:1 ratio)", () => {
    // 48kHz signal with 48 samples = 1ms of audio
    const input = new Float32Array(48);
    for (let i = 0; i < 48; i++) input[i] = Math.sin(2 * Math.PI * i / 48);
    const result = resampleTo16k(input, 48000);
    // 48 samples at 48kHz -> 16 samples at 16kHz
    expect(result.length).toBe(16);
  });

  it("downsamples 44100Hz to 16kHz", () => {
    const input = new Float32Array(44100); // 1 second
    const result = resampleTo16k(input, 44100);
    expect(result.length).toBe(16000);
  });
});

describe("prepareAudioInput", () => {
  it("normalizes audio to [-1, 1] range", () => {
    const input = new Float32Array([0.5, -0.5, 2.0, -3.0]);
    const result = prepareAudioInput(input, 16000);
    expect(Math.max(...result)).toBeLessThanOrEqual(1.0);
    expect(Math.min(...result)).toBeGreaterThanOrEqual(-1.0);
  });

  it("resamples and normalizes in one call", () => {
    const input = new Float32Array(48000); // 1s at 48kHz
    for (let i = 0; i < input.length; i++) input[i] = Math.sin(2 * Math.PI * 440 * i / 48000) * 0.5;
    const result = prepareAudioInput(input, 48000);
    expect(result.length).toBe(16000);
    expect(Math.abs(result[0])).toBeLessThanOrEqual(1.0);
  });

  it("pads short audio to minimum 1 second", () => {
    const input = new Float32Array(8000); // 0.5s at 16kHz
    const result = prepareAudioInput(input, 16000);
    expect(result.length).toBeGreaterThanOrEqual(16000);
  });
});

// ─── MFCC extraction ────────────────────────────────────────────────────────

describe("extractMFCC", () => {
  it("returns correct number of MFCC coefficients per frame", () => {
    const sine = makeSineWave(16000, 1.0);
    const mfcc = extractMFCC(sine, 16000, 40);
    // Each row should have 40 coefficients
    expect(mfcc.length).toBeGreaterThan(0);
    expect(mfcc[0].length).toBe(40);
  });

  it("produces expected number of frames for 1 second at 16kHz", () => {
    // 25ms frame, 10ms hop → (16000 - 400) / 160 + 1 = 98 frames
    const sine = makeSineWave(16000, 1.0);
    const mfcc = extractMFCC(sine, 16000, 40);
    // Allow some tolerance for rounding
    expect(mfcc.length).toBeGreaterThanOrEqual(95);
    expect(mfcc.length).toBeLessThanOrEqual(100);
  });

  it("returns all finite values", () => {
    const sine = makeSineWave(16000, 1.0);
    const mfcc = extractMFCC(sine, 16000, 40);
    for (const frame of mfcc) {
      for (const coeff of frame) {
        expect(Number.isFinite(coeff)).toBe(true);
      }
    }
  });

  it("returns different MFCCs for different frequency signals", () => {
    const low = makeSineWave(16000, 1.0, 200);
    const high = makeSineWave(16000, 1.0, 4000);
    const mfccLow = extractMFCC(low, 16000, 13);
    const mfccHigh = extractMFCC(high, 16000, 13);

    // Mean of first MFCC coefficient should differ between signals
    const meanLow = mfccLow.reduce((s, f) => s + f[1], 0) / mfccLow.length;
    const meanHigh = mfccHigh.reduce((s, f) => s + f[1], 0) / mfccHigh.length;
    expect(meanLow).not.toBeCloseTo(meanHigh, 0);
  });
});

// ─── 92-dim feature extraction ──────────────────────────────────────────────

describe("extractFeatures92", () => {
  it("returns exactly 92 features", () => {
    const sine = makeSineWave(16000, 1.0);
    const features = extractFeatures92(sine, 16000);
    expect(features.length).toBe(92);
  });

  it("returns all finite values", () => {
    const sine = makeSineWave(16000, 1.0);
    const features = extractFeatures92(sine, 16000);
    for (let i = 0; i < features.length; i++) {
      expect(Number.isFinite(features[i])).toBe(true);
    }
  });

  it("returns Float32Array", () => {
    const sine = makeSineWave(16000, 1.0);
    const features = extractFeatures92(sine, 16000);
    expect(features).toBeInstanceOf(Float32Array);
  });

  it("produces different features for sine wave vs silence", () => {
    const sine = makeSineWave(16000, 1.0, 440);
    const silence = makeSilence(16000, 1.0);

    const featuresSine = extractFeatures92(sine, 16000);
    const featuresSilence = extractFeatures92(silence, 16000);

    // At least some features should differ meaningfully
    let diffCount = 0;
    for (let i = 0; i < 92; i++) {
      if (Math.abs(featuresSine[i] - featuresSilence[i]) > 0.001) diffCount++;
    }
    // Most features should differ between a tone and silence
    expect(diffCount).toBeGreaterThan(20);
  });

  it("produces different features for different frequency signals", () => {
    const low = makeSineWave(16000, 1.0, 200);
    const high = makeSineWave(16000, 1.0, 4000);

    const featuresLow = extractFeatures92(low, 16000);
    const featuresHigh = extractFeatures92(high, 16000);

    let diffCount = 0;
    for (let i = 0; i < 92; i++) {
      if (Math.abs(featuresLow[i] - featuresHigh[i]) > 0.001) diffCount++;
    }
    expect(diffCount).toBeGreaterThan(10);
  });

  it("handles very short audio (padded to 1 second)", () => {
    // Only 0.1 seconds of audio
    const short = makeSineWave(16000, 0.1);
    // Should not throw — extractFeatures92 should handle short signals
    const features = extractFeatures92(short, 16000);
    expect(features.length).toBe(92);
    for (let i = 0; i < features.length; i++) {
      expect(Number.isFinite(features[i])).toBe(true);
    }
  });
});

// ─── Emotion mapping ────────────────────────────────────────────────────────

describe("EMOTION_REGIONS", () => {
  it("is a non-empty array", () => {
    expect(Array.isArray(EMOTION_REGIONS)).toBe(true);
    expect(EMOTION_REGIONS.length).toBeGreaterThan(0);
  });

  it("each region has required fields", () => {
    for (const region of EMOTION_REGIONS) {
      expect(typeof region.emotion).toBe("string");
      expect(region.valence).toHaveLength(2);
      expect(region.arousal).toHaveLength(2);
      expect(typeof region.priority).toBe("number");
    }
  });
});

describe("mapValenceArousalToEmotion", () => {
  it("maps high valence + high arousal to happy", () => {
    const result = mapValenceArousalToEmotion(0.6, 0.7);
    expect(result).toBe("happy");
  });

  it("maps low valence + low arousal to sad", () => {
    const result = mapValenceArousalToEmotion(-0.5, 0.2);
    expect(result).toBe("sad");
  });

  it("maps low valence + high arousal to angry", () => {
    const result = mapValenceArousalToEmotion(-0.5, 0.8);
    expect(result).toBe("angry");
  });

  it("maps near-zero valence + low arousal to neutral", () => {
    const result = mapValenceArousalToEmotion(0.0, 0.3);
    expect(result).toBe("neutral");
  });

  it("returns neutral as default for ambiguous region", () => {
    // Right at center — should hit neutral
    const result = mapValenceArousalToEmotion(0.0, 0.5);
    expect(result).toBe("neutral");
  });

  it("all 5 model classes are reachable", () => {
    // happy: high valence + moderate-high arousal
    expect(mapValenceArousalToEmotion(0.6, 0.5)).toBe("happy");
    // sad: negative valence + low arousal
    expect(mapValenceArousalToEmotion(-0.5, 0.2)).toBe("sad");
    // angry: negative valence + high arousal
    expect(mapValenceArousalToEmotion(-0.5, 0.8)).toBe("angry");
    // neutral: center
    expect(mapValenceArousalToEmotion(0.0, 0.3)).toBe("neutral");
    // calm: slightly positive + low arousal
    expect(mapValenceArousalToEmotion(0.4, 0.15)).toBe("calm");
  });
});

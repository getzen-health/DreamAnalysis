import { describe, it, expect } from "vitest";
import {
  softmax,
  normalizeChannels,
  eegnetEmotionFromProbabilities,
  EEGNET_EMOTIONS,
  EEGNET_EXPECTED_CHANNELS,
  EEGNET_EXPECTED_SAMPLES,
  validateEEGNetInput,
} from "@/lib/eegnet-utils";

// ── softmax ──────────────────────────────────────────────────────────────────

describe("softmax", () => {
  it("produces valid probability distribution (sums to 1, all positive)", () => {
    const logits = [2.0, 1.0, 0.1, -1.0, 0.5, 0.3];
    const probs = softmax(logits);

    expect(probs.length).toBe(6);
    const sum = probs.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1.0, 5);
    for (const p of probs) {
      expect(p).toBeGreaterThan(0);
      expect(p).toBeLessThanOrEqual(1);
    }
  });

  it("largest logit maps to largest probability", () => {
    const logits = [5.0, 1.0, 0.0, -2.0, 0.5, 0.1];
    const probs = softmax(logits);
    const maxIdx = probs.indexOf(Math.max(...probs));
    expect(maxIdx).toBe(0);
  });

  it("handles all-zero logits (uniform distribution)", () => {
    const logits = [0, 0, 0, 0, 0, 0];
    const probs = softmax(logits);
    for (const p of probs) {
      expect(p).toBeCloseTo(1 / 6, 5);
    }
  });

  it("handles very large logits without overflow (numerically stable)", () => {
    const logits = [1000, 999, 998, 997, 996, 995];
    const probs = softmax(logits);
    const sum = probs.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1.0, 5);
    expect(Number.isFinite(probs[0])).toBe(true);
  });

  it("handles very negative logits", () => {
    const logits = [-1000, -999, -998, -997, -996, -995];
    const probs = softmax(logits);
    const sum = probs.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1.0, 5);
  });
});

// ── normalizeChannels ────────────────────────────────────────────────────────

describe("normalizeChannels", () => {
  it("produces zero-mean output per channel", () => {
    const ch1 = new Float32Array(1024);
    const ch2 = new Float32Array(1024);
    const ch3 = new Float32Array(1024);
    const ch4 = new Float32Array(1024);

    // Fill with non-zero-mean data
    for (let i = 0; i < 1024; i++) {
      ch1[i] = Math.sin(i * 0.01) + 5;
      ch2[i] = Math.cos(i * 0.02) - 3;
      ch3[i] = Math.sin(i * 0.03) * 2 + 10;
      ch4[i] = Math.cos(i * 0.04) * 0.5 - 7;
    }

    const result = normalizeChannels([ch1, ch2, ch3, ch4]);
    expect(result.length).toBe(4);

    for (let c = 0; c < 4; c++) {
      let sum = 0;
      for (let i = 0; i < 1024; i++) {
        sum += result[c][i];
      }
      const mean = sum / 1024;
      expect(mean).toBeCloseTo(0, 3);
    }
  });

  it("produces unit-variance output per channel", () => {
    const channels: Float32Array[] = [];
    for (let c = 0; c < 4; c++) {
      const ch = new Float32Array(1024);
      for (let i = 0; i < 1024; i++) {
        ch[i] = Math.sin(i * 0.01 * (c + 1)) * (c + 1) * 10 + c * 5;
      }
      channels.push(ch);
    }

    const result = normalizeChannels(channels);

    for (let c = 0; c < 4; c++) {
      let sumSq = 0;
      for (let i = 0; i < 1024; i++) {
        sumSq += result[c][i] * result[c][i];
      }
      const variance = sumSq / 1024;
      expect(variance).toBeCloseTo(1.0, 1);
    }
  });

  it("handles constant channel (zero variance) without NaN", () => {
    const channels = [
      new Float32Array(1024).fill(42),
      new Float32Array(1024).fill(0),
      new Float32Array(1024).fill(-5),
      new Float32Array(1024).fill(100),
    ];

    const result = normalizeChannels(channels);

    for (let c = 0; c < 4; c++) {
      for (let i = 0; i < 1024; i++) {
        expect(Number.isFinite(result[c][i])).toBe(true);
      }
    }
  });
});

// ── Emotion mapping from probabilities ───────────────────────────────────────

describe("eegnetEmotionFromProbabilities", () => {
  it("returns correct emotion for dominated distributions", () => {
    // Each test: make one probability dominant
    for (let idx = 0; idx < EEGNET_EMOTIONS.length; idx++) {
      const probs = new Array(6).fill(0.02);
      probs[idx] = 0.9;
      const result = eegnetEmotionFromProbabilities(probs);
      expect(result.emotion).toBe(EEGNET_EMOTIONS[idx]);
      expect(result.confidence).toBeCloseTo(0.9, 1);
    }
  });

  it("probabilities dict contains all 6 emotions", () => {
    const probs = [0.3, 0.1, 0.1, 0.1, 0.3, 0.1];
    const result = eegnetEmotionFromProbabilities(probs);
    for (const emotion of EEGNET_EMOTIONS) {
      expect(emotion in result.probabilities).toBe(true);
    }
  });

  it("valence is positive for happy-dominant distribution", () => {
    const probs = [0.8, 0.04, 0.04, 0.04, 0.04, 0.04]; // happy dominant
    const result = eegnetEmotionFromProbabilities(probs);
    expect(result.valence).toBeGreaterThan(0);
  });

  it("valence is negative for sad-dominant distribution", () => {
    const probs = [0.04, 0.8, 0.04, 0.04, 0.04, 0.04]; // sad dominant
    const result = eegnetEmotionFromProbabilities(probs);
    expect(result.valence).toBeLessThan(0);
  });

  it("arousal is high for angry-dominant distribution", () => {
    const probs = [0.04, 0.04, 0.8, 0.04, 0.04, 0.04]; // angry dominant
    const result = eegnetEmotionFromProbabilities(probs);
    expect(result.arousal).toBeGreaterThan(0.5);
  });

  it("arousal is low for relaxed-dominant distribution", () => {
    const probs = [0.04, 0.04, 0.04, 0.04, 0.8, 0.04]; // relaxed dominant
    const result = eegnetEmotionFromProbabilities(probs);
    expect(result.arousal).toBeLessThan(0.5);
  });

  it("valence in [-1, 1] range for any distribution", () => {
    const distributions = [
      [1, 0, 0, 0, 0, 0],
      [0, 1, 0, 0, 0, 0],
      [0, 0, 1, 0, 0, 0],
      [0, 0, 0, 1, 0, 0],
      [0, 0, 0, 0, 1, 0],
      [0, 0, 0, 0, 0, 1],
    ];
    for (const probs of distributions) {
      const result = eegnetEmotionFromProbabilities(probs);
      expect(result.valence).toBeGreaterThanOrEqual(-1);
      expect(result.valence).toBeLessThanOrEqual(1);
    }
  });

  it("arousal in [0, 1] range for any distribution", () => {
    const distributions = [
      [1, 0, 0, 0, 0, 0],
      [0, 1, 0, 0, 0, 0],
      [0, 0, 1, 0, 0, 0],
      [0, 0, 0, 1, 0, 0],
      [0, 0, 0, 0, 1, 0],
      [0, 0, 0, 0, 0, 1],
    ];
    for (const probs of distributions) {
      const result = eegnetEmotionFromProbabilities(probs);
      expect(result.arousal).toBeGreaterThanOrEqual(0);
      expect(result.arousal).toBeLessThanOrEqual(1);
    }
  });
});

// ── Input validation ─────────────────────────────────────────────────────────

describe("validateEEGNetInput", () => {
  it("accepts valid 4-channel x 1024-sample input", () => {
    const channels = Array.from({ length: 4 }, () => new Float32Array(1024));
    expect(validateEEGNetInput(channels)).toBe(true);
  });

  it("rejects wrong channel count (3 channels)", () => {
    const channels = Array.from({ length: 3 }, () => new Float32Array(1024));
    expect(validateEEGNetInput(channels)).toBe(false);
  });

  it("rejects wrong channel count (5 channels)", () => {
    const channels = Array.from({ length: 5 }, () => new Float32Array(1024));
    expect(validateEEGNetInput(channels)).toBe(false);
  });

  it("rejects wrong sample count (512 samples)", () => {
    const channels = Array.from({ length: 4 }, () => new Float32Array(512));
    expect(validateEEGNetInput(channels)).toBe(false);
  });

  it("rejects wrong sample count (2048 samples)", () => {
    const channels = Array.from({ length: 4 }, () => new Float32Array(2048));
    expect(validateEEGNetInput(channels)).toBe(false);
  });

  it("rejects empty array", () => {
    expect(validateEEGNetInput([])).toBe(false);
  });
});

// ── Constants ────────────────────────────────────────────────────────────────

describe("EEGNET constants", () => {
  it("EEGNET_EMOTIONS has 6 entries", () => {
    expect(EEGNET_EMOTIONS).toHaveLength(6);
  });

  it("EEGNET_EMOTIONS matches expected classes", () => {
    expect(EEGNET_EMOTIONS).toEqual(["happy", "sad", "angry", "fearful", "relaxed", "focused"]);
  });

  it("EEGNET_EXPECTED_CHANNELS is 4", () => {
    expect(EEGNET_EXPECTED_CHANNELS).toBe(4);
  });

  it("EEGNET_EXPECTED_SAMPLES is 1024", () => {
    expect(EEGNET_EXPECTED_SAMPLES).toBe(1024);
  });
});

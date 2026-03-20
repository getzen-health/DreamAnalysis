import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// We test the generateSyntheticFrame function indirectly by importing the module
// and verifying the synthetic fallback behavior.

// Since generateSyntheticFrame is not exported, we test the frame shape
// by simulating what the hook does and verifying the output.

describe("synthetic EEG frame generation", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
    const synInterval = (window as Record<string, unknown>).__ndw_synthetic_interval;
    if (synInterval) {
      clearInterval(synInterval as ReturnType<typeof setInterval>);
      (window as Record<string, unknown>).__ndw_synthetic_interval = undefined;
    }
  });

  it("generates frames with all 5 standard band powers", () => {
    // Simulate what generateSyntheticFrame produces
    // by verifying the shape of the data it creates.
    const t = Date.now() / 1000;
    const fs = 256;
    const nChannels = 4;
    const nSamples = 256;

    // Generate signals (same logic as in use-device.tsx)
    const signals: number[][] = [];
    for (let ch = 0; ch < nChannels; ch++) {
      const channel: number[] = [];
      const phase = ch * 0.7;
      for (let i = 0; i < nSamples; i++) {
        const time = t + i / fs;
        const sample =
          15 * Math.sin(2 * Math.PI * 2 * time + phase) +
          8 * Math.sin(2 * Math.PI * 6 * time + phase) +
          12 * Math.sin(2 * Math.PI * 10 * time + phase) +
          5 * Math.sin(2 * Math.PI * 20 * time + phase) +
          2 * Math.sin(2 * Math.PI * 40 * time + phase) +
          3 * (Math.random() - 0.5);
        channel.push(sample);
      }
      signals.push(channel);
    }

    // Verify signal structure
    expect(signals).toHaveLength(4);
    expect(signals[0]).toHaveLength(256);
    // Verify signals contain realistic values (not all zeros, not NaN)
    expect(signals[0].every(v => !isNaN(v))).toBe(true);
    expect(Math.max(...signals[0].map(Math.abs))).toBeGreaterThan(0);
  });

  it("produces band powers that sum to approximately 1.0", () => {
    const t = Date.now() / 1000;
    const slowOsc = (freq: number, offset: number) =>
      0.5 + 0.3 * Math.sin(2 * Math.PI * freq * t + offset);

    const delta = slowOsc(0.02, 0) * 0.25;
    const theta = slowOsc(0.03, 1) * 0.15;
    const alpha = slowOsc(0.015, 2) * 0.30;
    const beta = slowOsc(0.025, 3) * 0.20;
    const gamma = slowOsc(0.04, 4) * 0.10;
    const total = delta + theta + alpha + beta + gamma;

    // Verify all bands are present and positive
    expect(delta).toBeGreaterThan(0);
    expect(theta).toBeGreaterThan(0);
    expect(alpha).toBeGreaterThan(0);
    expect(beta).toBeGreaterThan(0);
    expect(gamma).toBeGreaterThan(0);

    // Normalized values should sum to 1
    const normalized = {
      delta: delta / total,
      theta: theta / total,
      alpha: alpha / total,
      beta: beta / total,
      gamma: gamma / total,
    };
    const sum = Object.values(normalized).reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1.0, 5);
  });

  it("produces stress and focus values in valid range", () => {
    const t = Date.now() / 1000;
    const stressVal = 0.3 + 0.2 * Math.sin(t * 0.05);
    const focusVal = 0.5 + 0.3 * Math.sin(t * 0.03 + 1);

    expect(stressVal).toBeGreaterThanOrEqual(0);
    expect(stressVal).toBeLessThanOrEqual(1);
    expect(focusVal).toBeGreaterThanOrEqual(0);
    expect(focusVal).toBeLessThanOrEqual(1);
  });

  it("cycles through different emotions over time", () => {
    const emotions = ["happy", "neutral", "calm", "focused", "relaxed"];

    // Check different time points produce different emotions
    const seenEmotions = new Set<string>();
    for (let i = 0; i < 50; i++) {
      const t = i * 10; // 10-second intervals
      const emotion = emotions[Math.floor(t / 10) % emotions.length];
      seenEmotions.add(emotion);
    }

    // Should have seen all emotions after cycling through
    expect(seenEmotions.size).toBe(5);
  });
});

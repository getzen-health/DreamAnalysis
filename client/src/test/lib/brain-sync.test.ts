import { describe, it, expect } from "vitest";
import { computeBrainSync, type BrainSyncResult } from "@/lib/brain-sync";

describe("computeBrainSync", () => {
  it("returns sync = 1.0 for identical band powers", () => {
    const result = computeBrainSync(0.5, 0.3, 0.2, 0.5, 0.3, 0.2);
    expect(result.overallSync).toBeCloseTo(1.0, 5);
    expect(result.bandSync.alpha).toBeCloseTo(1.0, 5);
    expect(result.bandSync.theta).toBeCloseTo(1.0, 5);
    expect(result.bandSync.beta).toBeCloseTo(1.0, 5);
  });

  it("returns low sync for very different powers", () => {
    // Person 1: high alpha, low theta/beta
    // Person 2: low alpha, high theta/beta
    const result = computeBrainSync(0.9, 0.05, 0.05, 0.05, 0.9, 0.05);
    expect(result.overallSync).toBeLessThan(0.3);
  });

  it("assigns phase labels matching sync score ranges", () => {
    // Deep sync: > 0.7
    const deepSync = computeBrainSync(0.5, 0.3, 0.2, 0.5, 0.3, 0.2);
    expect(deepSync.phase).toBe("deep_sync");

    // In sync: 0.5-0.7
    // alpha: (0.8, 0.4) => sync=0.5, theta: (0.5, 0.2) => sync=0.6, beta: (0.3, 0.1) => sync=0.333
    // overall: 0.5*0.5 + 0.3*0.6 + 0.2*0.333 = 0.25 + 0.18 + 0.067 = 0.497 => just under 0.5
    // Try: alpha: (0.6, 0.3) => 0.5, theta: (0.4, 0.1) => 0.25, beta: (0.3, 0.05) => 0.167
    // overall: 0.25 + 0.075 + 0.033 = 0.358 -- too low
    // alpha: (0.8, 0.5) => 0.625, theta: (0.3, 0.15) => 0.5, beta: (0.2, 0.05) => 0.25
    // overall: 0.3125 + 0.15 + 0.05 = 0.5125
    const inSync = computeBrainSync(0.8, 0.3, 0.2, 0.5, 0.15, 0.05);
    expect(inSync.overallSync).toBeGreaterThanOrEqual(0.5);
    expect(inSync.overallSync).toBeLessThan(0.7);
    expect(inSync.phase).toBe("in_sync");

    // Syncing: 0.3-0.5
    // alpha: (0.8, 0.3) => 0.375, theta: (0.1, 0.4) => 0.25, beta: (0.1, 0.3) => 0.333
    // overall: 0.1875 + 0.075 + 0.0667 = 0.329
    const syncing = computeBrainSync(0.8, 0.1, 0.1, 0.3, 0.4, 0.3);
    expect(syncing.overallSync).toBeGreaterThanOrEqual(0.3);
    expect(syncing.overallSync).toBeLessThan(0.5);
    expect(syncing.phase).toBe("syncing");

    // Connecting: < 0.3
    const connecting = computeBrainSync(0.9, 0.05, 0.05, 0.05, 0.9, 0.05);
    expect(connecting.overallSync).toBeLessThan(0.3);
    expect(connecting.phase).toBe("connecting");
  });

  it("computes overall sync as weighted average of band syncs", () => {
    const result = computeBrainSync(0.6, 0.3, 0.1, 0.4, 0.2, 0.15);
    const expected = 0.5 * result.bandSync.alpha + 0.3 * result.bandSync.theta + 0.2 * result.bandSync.beta;
    expect(result.overallSync).toBeCloseTo(expected, 5);
  });

  it("keeps all sync values in 0-1 range", () => {
    // Test with various extreme inputs
    const cases = [
      [0, 0, 0, 0, 0, 0],
      [1, 1, 1, 0, 0, 0],
      [0, 0, 0, 1, 1, 1],
      [0.001, 0.001, 0.001, 0.999, 0.999, 0.999],
      [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    ] as const;

    for (const [a1, t1, b1, a2, t2, b2] of cases) {
      const result = computeBrainSync(a1, t1, b1, a2, t2, b2);
      expect(result.overallSync).toBeGreaterThanOrEqual(0);
      expect(result.overallSync).toBeLessThanOrEqual(1);
      expect(result.bandSync.alpha).toBeGreaterThanOrEqual(0);
      expect(result.bandSync.alpha).toBeLessThanOrEqual(1);
      expect(result.bandSync.theta).toBeGreaterThanOrEqual(0);
      expect(result.bandSync.theta).toBeLessThanOrEqual(1);
      expect(result.bandSync.beta).toBeGreaterThanOrEqual(0);
      expect(result.bandSync.beta).toBeLessThanOrEqual(1);
    }
  });

  it("returns non-empty messages for all phases", () => {
    // Deep sync
    const deep = computeBrainSync(0.5, 0.3, 0.2, 0.5, 0.3, 0.2);
    expect(deep.message.length).toBeGreaterThan(0);

    // In sync
    const inSync = computeBrainSync(0.8, 0.3, 0.2, 0.5, 0.15, 0.05);
    expect(inSync.message.length).toBeGreaterThan(0);

    // Syncing
    const syncing = computeBrainSync(0.8, 0.1, 0.1, 0.3, 0.4, 0.3);
    expect(syncing.message.length).toBeGreaterThan(0);

    // Connecting
    const connecting = computeBrainSync(0.9, 0.05, 0.05, 0.05, 0.9, 0.05);
    expect(connecting.message.length).toBeGreaterThan(0);
  });

  it("handles zero powers gracefully (no division by zero)", () => {
    // Both zero for all bands
    const result = computeBrainSync(0, 0, 0, 0, 0, 0);
    expect(Number.isFinite(result.overallSync)).toBe(true);
    expect(Number.isNaN(result.overallSync)).toBe(false);
    expect(Number.isFinite(result.bandSync.alpha)).toBe(true);
    expect(Number.isFinite(result.bandSync.theta)).toBe(true);
    expect(Number.isFinite(result.bandSync.beta)).toBe(true);

    // One person zero, other non-zero
    const result2 = computeBrainSync(0, 0, 0, 0.5, 0.3, 0.2);
    expect(Number.isFinite(result2.overallSync)).toBe(true);
    expect(Number.isNaN(result2.overallSync)).toBe(false);

    // One band zero on both sides, others non-zero
    const result3 = computeBrainSync(0.5, 0, 0.2, 0.3, 0, 0.1);
    expect(Number.isFinite(result3.overallSync)).toBe(true);
    expect(Number.isNaN(result3.bandSync.theta)).toBe(false);
  });
});

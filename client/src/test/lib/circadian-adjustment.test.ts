import { describe, it, expect } from "vitest";
import {
  getCircadianAdjustment,
  applyCircadianNormalization,
  type CircadianAdjustment,
  type RawReadings,
} from "@/lib/circadian-adjustment";

// ── Circadian adjustment by time of day ──────────────────────────────────

describe("getCircadianAdjustment", () => {
  it("returns elevated stress baseline for morning cortisol surge (6-9 AM)", () => {
    for (const hour of [6, 7, 8]) {
      const adj = getCircadianAdjustment(hour);
      expect(adj.stressBaselineOffset).toBeGreaterThan(0);
      expect(adj.label).toContain("cortisol");
    }
  });

  it("returns lowered focus baseline for post-lunch dip (1-3 PM)", () => {
    for (const hour of [13, 14]) {
      const adj = getCircadianAdjustment(hour);
      expect(adj.focusBaselineOffset).toBeLessThan(0);
      expect(adj.label).toContain("post-lunch");
    }
  });

  it("returns elevated alpha baseline for evening wind-down (8-11 PM)", () => {
    for (const hour of [20, 21, 22]) {
      const adj = getCircadianAdjustment(hour);
      expect(adj.alphaBaselineOffset).toBeGreaterThan(0);
      expect(adj.label).toContain("wind-down");
    }
  });

  it("returns neutral adjustments for midday (10 AM - 12 PM)", () => {
    const adj = getCircadianAdjustment(11);
    expect(adj.stressBaselineOffset).toBe(0);
    expect(adj.focusBaselineOffset).toBe(0);
    expect(adj.alphaBaselineOffset).toBe(0);
  });

  it("returns neutral adjustments for early afternoon (3-5 PM)", () => {
    const adj = getCircadianAdjustment(16);
    expect(adj.stressBaselineOffset).toBe(0);
    expect(adj.focusBaselineOffset).toBe(0);
    expect(adj.alphaBaselineOffset).toBe(0);
  });

  it("handles midnight hours (0-5 AM) — sleep zone", () => {
    for (const hour of [0, 1, 2, 3, 4, 5]) {
      const adj = getCircadianAdjustment(hour);
      expect(adj.label).toContain("sleep");
    }
  });

  it("all offsets stay within +-0.2 range", () => {
    for (let h = 0; h < 24; h++) {
      const adj = getCircadianAdjustment(h);
      expect(Math.abs(adj.stressBaselineOffset)).toBeLessThanOrEqual(0.2);
      expect(Math.abs(adj.focusBaselineOffset)).toBeLessThanOrEqual(0.2);
      expect(Math.abs(adj.alphaBaselineOffset)).toBeLessThanOrEqual(0.2);
    }
  });
});

// ── Normalization of raw readings ────────────────────────────────────────

describe("applyCircadianNormalization", () => {
  it("lowers reported stress during morning cortisol hours", () => {
    const raw: RawReadings = {
      stress: 0.7,
      focus: 0.6,
      relaxation: 0.3,
      hour: 7, // morning cortisol
    };
    const normalized = applyCircadianNormalization(raw);
    // Stress should be adjusted downward because cortisol is naturally high
    expect(normalized.stress).toBeLessThan(raw.stress);
  });

  it("raises reported focus during post-lunch dip", () => {
    const raw: RawReadings = {
      stress: 0.4,
      focus: 0.35,
      relaxation: 0.5,
      hour: 14, // post-lunch dip
    };
    const normalized = applyCircadianNormalization(raw);
    // Focus should be adjusted upward because focus baseline is naturally lower
    expect(normalized.focus).toBeGreaterThan(raw.focus);
  });

  it("lowers reported relaxation during evening wind-down", () => {
    const raw: RawReadings = {
      stress: 0.3,
      focus: 0.4,
      relaxation: 0.7,
      hour: 21, // evening
    };
    const normalized = applyCircadianNormalization(raw);
    // Alpha is naturally higher, so relaxation reading is less remarkable
    expect(normalized.relaxation).toBeLessThan(raw.relaxation);
  });

  it("leaves values unchanged during neutral hours", () => {
    const raw: RawReadings = {
      stress: 0.5,
      focus: 0.5,
      relaxation: 0.5,
      hour: 11, // midday — neutral
    };
    const normalized = applyCircadianNormalization(raw);
    expect(normalized.stress).toBe(0.5);
    expect(normalized.focus).toBe(0.5);
    expect(normalized.relaxation).toBe(0.5);
  });

  it("clamps normalized values to 0-1", () => {
    // High stress during morning cortisol — should not go below 0
    const raw: RawReadings = {
      stress: 0.05,
      focus: 0.95,
      relaxation: 0.95,
      hour: 7,
    };
    const normalized = applyCircadianNormalization(raw);
    expect(normalized.stress).toBeGreaterThanOrEqual(0);
    expect(normalized.focus).toBeLessThanOrEqual(1);
    expect(normalized.relaxation).toBeLessThanOrEqual(1);
  });
});

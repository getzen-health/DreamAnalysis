import { describe, it, expect, beforeEach, vi } from "vitest";
import {
  getCyclePhase,
  getCyclePhaseAdjustment,
  getCyclePhaseContext,
  type CyclePhase,
  type CyclePhaseAdjustment,
  type CyclePhaseContext,
} from "@/lib/cycle-phase-adjustment";

// ── getCyclePhase ─────────────────────────────────────────────────────────

describe("getCyclePhase", () => {
  it("returns menstrual phase for days 1-5", () => {
    for (const day of [1, 2, 3, 4, 5]) {
      expect(getCyclePhase(day, 28)).toBe("menstrual");
    }
  });

  it("returns follicular phase for days 6-13", () => {
    for (const day of [6, 7, 10, 13]) {
      expect(getCyclePhase(day, 28)).toBe("follicular");
    }
  });

  it("returns ovulatory phase for days 14-15", () => {
    expect(getCyclePhase(14, 28)).toBe("ovulatory");
    expect(getCyclePhase(15, 28)).toBe("ovulatory");
  });

  it("returns luteal phase for days 16-28", () => {
    for (const day of [16, 20, 25, 28]) {
      expect(getCyclePhase(day, 28)).toBe("luteal");
    }
  });

  it("handles short cycles (25 days)", () => {
    // Menstrual: 1-5, Follicular: 6-11, Ovulatory: 12-13, Luteal: 14-25
    expect(getCyclePhase(1, 25)).toBe("menstrual");
    expect(getCyclePhase(6, 25)).toBe("follicular");
    expect(getCyclePhase(14, 25)).toBe("luteal");
    expect(getCyclePhase(25, 25)).toBe("luteal");
  });

  it("handles long cycles (35 days)", () => {
    expect(getCyclePhase(1, 35)).toBe("menstrual");
    expect(getCyclePhase(10, 35)).toBe("follicular");
    expect(getCyclePhase(35, 35)).toBe("luteal");
  });

  it("returns null for day 0 or negative", () => {
    expect(getCyclePhase(0, 28)).toBeNull();
    expect(getCyclePhase(-1, 28)).toBeNull();
  });

  it("returns null for day exceeding cycle length", () => {
    expect(getCyclePhase(30, 28)).toBeNull();
  });
});

// ── getCyclePhaseAdjustment ───────────────────────────────────────────────

describe("getCyclePhaseAdjustment", () => {
  it("menstrual phase: lower energy baseline", () => {
    const adj = getCyclePhaseAdjustment("menstrual");
    expect(adj.energyOffset).toBeLessThan(0);
    expect(adj.moodOffset).toBeLessThanOrEqual(0);
  });

  it("follicular phase: improved mood baseline", () => {
    const adj = getCyclePhaseAdjustment("follicular");
    expect(adj.moodOffset).toBeGreaterThan(0);
    expect(adj.energyOffset).toBeGreaterThanOrEqual(0);
  });

  it("ovulatory phase: highest energy and mood", () => {
    const adj = getCyclePhaseAdjustment("ovulatory");
    expect(adj.moodOffset).toBeGreaterThan(0);
    expect(adj.energyOffset).toBeGreaterThan(0);
  });

  it("luteal phase: lower mood baseline, higher irritability", () => {
    const adj = getCyclePhaseAdjustment("luteal");
    expect(adj.moodOffset).toBeLessThan(0);
    expect(adj.irritabilityOffset).toBeGreaterThan(0);
  });

  it("all offsets stay within +-0.2", () => {
    const phases: CyclePhase[] = ["menstrual", "follicular", "ovulatory", "luteal"];
    for (const phase of phases) {
      const adj = getCyclePhaseAdjustment(phase);
      expect(Math.abs(adj.moodOffset)).toBeLessThanOrEqual(0.2);
      expect(Math.abs(adj.energyOffset)).toBeLessThanOrEqual(0.2);
      expect(Math.abs(adj.irritabilityOffset)).toBeLessThanOrEqual(0.2);
    }
  });
});

// ── getCyclePhaseContext ──────────────────────────────────────────────────

describe("getCyclePhaseContext", () => {
  it("returns context message for luteal phase", () => {
    const ctx = getCyclePhaseContext("luteal");
    expect(ctx.message).toContain("mood may be lower");
    expect(ctx.phase).toBe("luteal");
  });

  it("returns context for menstrual phase", () => {
    const ctx = getCyclePhaseContext("menstrual");
    expect(ctx.message).toContain("energy");
    expect(ctx.phase).toBe("menstrual");
  });

  it("returns context for follicular phase", () => {
    const ctx = getCyclePhaseContext("follicular");
    expect(ctx.message).toContain("mood");
    expect(ctx.phase).toBe("follicular");
  });

  it("returns context for ovulatory phase", () => {
    const ctx = getCyclePhaseContext("ovulatory");
    expect(ctx.message).toContain("peak");
    expect(ctx.phase).toBe("ovulatory");
  });

  it("all contexts have a non-empty message", () => {
    const phases: CyclePhase[] = ["menstrual", "follicular", "ovulatory", "luteal"];
    for (const phase of phases) {
      const ctx = getCyclePhaseContext(phase);
      expect(ctx.message.length).toBeGreaterThan(0);
    }
  });
});

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { InterventionLibrary } from "@/lib/insight-engine/intervention-library";

beforeEach(() => localStorage.clear());
afterEach(() => vi.useRealTimers());

describe("InterventionLibrary.getForDeviation", () => {
  it("returns 2-min reset for high-stress deviation", () => {
    const lib = new InterventionLibrary();
    const interventions = lib.getForDeviation("stress", "high");
    expect(interventions.length).toBeGreaterThan(0);
    expect(interventions[0].durationBucket).toBe("2min");
    expect(interventions[0].deeplink).toBe("/biofeedback");
  });

  it("returns interventions for low-focus deviation", () => {
    const lib = new InterventionLibrary();
    const interventions = lib.getForDeviation("focus", "low");
    expect(interventions.length).toBeGreaterThan(0);
  });
});

describe("InterventionLibrary.recordTap + checkEffectiveness", () => {
  it("marks intervention effective when z-score recovers by >0.5 after 25 min", () => {
    vi.useFakeTimers();
    const now = Date.now();
    vi.setSystemTime(now - 30 * 60 * 1000); // 30 min ago
    const lib = new InterventionLibrary();
    lib.recordTap("box_breathing", "stress", 2.0);

    vi.setSystemTime(now);
    const results = lib.checkEffectiveness("stress", 1.0); // z dropped from 2.0 to 1.0 — recovered
    expect(results.length).toBe(1);
    expect(results[0].interventionId).toBe("box_breathing");
    expect(results[0].effective).toBe(true);
  });

  it("marks intervention ineffective when 2+ hours pass without recovery", () => {
    vi.useFakeTimers();
    const now = Date.now();
    vi.setSystemTime(now - 3 * 60 * 60 * 1000); // 3 hours ago
    const lib = new InterventionLibrary();
    lib.recordTap("box_breathing", "stress", 2.0);

    vi.setSystemTime(now);
    const results = lib.checkEffectiveness("stress", 2.1); // still high
    expect(results[0].effective).toBe(false);
  });
});

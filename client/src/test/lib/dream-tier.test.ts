import { describe, it, expect } from "vitest";
import { getDreamTier, estimateRemWindows } from "@/lib/dream-tier";

describe("getDreamTier", () => {
  it("returns 'eeg' when streaming regardless of health data", () => {
    expect(getDreamTier(true, false)).toBe("eeg");
    expect(getDreamTier(true, true)).toBe("eeg");
  });

  it("returns 'phone' when not streaming but health data available", () => {
    expect(getDreamTier(false, true)).toBe("phone");
  });

  it("returns 'none' when not streaming and no health data (in test env without DeviceMotion)", () => {
    // In Node/jsdom DeviceMotionEvent is not present → 'none'
    if (typeof window === "undefined" || !("DeviceMotionEvent" in window)) {
      expect(getDreamTier(false, false)).toBe("none");
    }
  });
});

describe("estimateRemWindows", () => {
  it("returns empty remWindows for less than one 90-min cycle", () => {
    // sleep 01:00 → wake 02:00 = 60 min, no complete 90-min cycle
    const result = estimateRemWindows("01:00", "02:00");
    expect(result.remWindows).toHaveLength(0);
    expect(result.totalHours).toBeCloseTo(1, 1);
    expect(result.mostLikelyDreamWindow).toBeNull();
  });

  it("returns correct number of REM windows for 7.5h of sleep", () => {
    // 23:00 → 06:30 = 7.5h = 5 complete 90-min cycles
    const result = estimateRemWindows("23:00", "06:30");
    expect(result.remWindows).toHaveLength(5);
    expect(result.remWindows[0].cycleNumber).toBe(1);
    expect(result.remWindows[4].cycleNumber).toBe(5);
  });

  it("last REM window is the most likely dream window", () => {
    const result = estimateRemWindows("23:00", "07:00");
    const last = result.remWindows[result.remWindows.length - 1];
    expect(result.mostLikelyDreamWindow).toEqual(last);
  });

  it("REM windows have increasing duration per cycle", () => {
    const result = estimateRemWindows("22:00", "07:30");
    // 9.5h = 6 complete cycles
    expect(result.remWindows.length).toBeGreaterThanOrEqual(5);
    // Cycle 1 = 10 min, cycle 2 = 20 min, cycle 3+ = 25-35 min
    expect(result.remWindows[0].durationMinutes).toBe(10);
    expect(result.remWindows[1].durationMinutes).toBe(20);
    expect(result.remWindows[2].durationMinutes).toBeGreaterThan(20);
  });

  it("handles overnight wrap (sleep onset PM, wake AM)", () => {
    // 23:30 → 07:00: should compute ~7.5h correctly
    const result = estimateRemWindows("23:30", "07:00");
    expect(result.totalHours).toBeCloseTo(7.5, 0);
    expect(result.remWindows.length).toBeGreaterThan(0);
  });

  it("uses healthSleepHours when provided and sets source to health_api", () => {
    const result = estimateRemWindows("23:00", "07:30", 7.0);
    expect(result.source).toBe("health_api");
    expect(result.totalHours).toBe(7.0);
  });

  it("falls back to clock_only when healthSleepHours is null", () => {
    const result = estimateRemWindows("23:00", "07:30", null);
    expect(result.source).toBe("clock_only");
  });

  it("estimated window times are valid HH:MM format", () => {
    const result = estimateRemWindows("23:00", "07:00");
    const hhmmRegex = /^\d{2}:\d{2}$/;
    for (const w of result.remWindows) {
      expect(w.estimatedStart).toMatch(hhmmRegex);
      expect(w.estimatedEnd).toMatch(hhmmRegex);
    }
  });

  it("data note mentions EEG device for clock_only source", () => {
    const result = estimateRemWindows("23:00", "07:00");
    expect(result.dataNote).toContain("EEG");
  });

  it("data note mentions health app for health_api source", () => {
    const result = estimateRemWindows("23:00", "07:00", 8.0);
    expect(result.dataNote).toContain("health");
  });
});

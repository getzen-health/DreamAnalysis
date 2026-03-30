import { describe, it, expect } from "vitest";
import {
  shouldTriggerAlarm,
  type SmartAlarmConfig,
} from "@/lib/smart-alarm";

/** Helper: create a Date at a given HH:MM on a fixed day */
function timeAt(hours: number, minutes: number): Date {
  return new Date(2026, 2, 30, hours, minutes, 0, 0);
}

function makeConfig(
  targetHour: number,
  targetMin: number,
  windowMinutes: number,
): SmartAlarmConfig {
  return {
    targetWakeTime: timeAt(targetHour, targetMin),
    windowMinutes,
  };
}

describe("shouldTriggerAlarm", () => {
  // ── Past target time ─────────────────────────────────────────────────────

  it("wakes immediately when past target time, regardless of stage", () => {
    const config = makeConfig(7, 0, 30);
    const now = timeAt(7, 5); // 5 min past target
    const result = shouldTriggerAlarm(config, "N3", null, now);
    expect(result.shouldWake).toBe(true);
    expect(result.reason).toBe("Target time reached");
    expect(result.minutesEarly).toBe(0);
  });

  it("wakes when exactly at target time", () => {
    const config = makeConfig(7, 0, 30);
    const now = timeAt(7, 0);
    const result = shouldTriggerAlarm(config, "REM", null, now);
    expect(result.shouldWake).toBe(true);
    expect(result.reason).toBe("Target time reached");
    expect(result.minutesEarly).toBe(0);
  });

  // ── Outside window ───────────────────────────────────────────────────────

  it("does not wake when outside the window even if in N1", () => {
    const config = makeConfig(7, 0, 30);
    const now = timeAt(6, 0); // 60 min before target, window is 30
    const result = shouldTriggerAlarm(config, "N1", null, now);
    expect(result.shouldWake).toBe(false);
    expect(result.reason).toBe("Outside wake window");
    expect(result.minutesEarly).toBe(60);
  });

  it("does not wake when outside the window in N2", () => {
    const config = makeConfig(7, 0, 15);
    const now = timeAt(6, 30); // 30 min before target, window is 15
    const result = shouldTriggerAlarm(config, "N2", null, now);
    expect(result.shouldWake).toBe(false);
    expect(result.reason).toBe("Outside wake window");
  });

  // ── Within window — light sleep ──────────────────────────────────────────

  it("wakes in N1 within the window", () => {
    const config = makeConfig(7, 0, 30);
    const now = timeAt(6, 40); // 20 min before target
    const result = shouldTriggerAlarm(config, "N1", null, now);
    expect(result.shouldWake).toBe(true);
    expect(result.reason).toBe("Light sleep — optimal wake window");
    expect(result.minutesEarly).toBe(20);
  });

  it("wakes in N2 within the window", () => {
    const config = makeConfig(7, 0, 30);
    const now = timeAt(6, 45); // 15 min before target
    const result = shouldTriggerAlarm(config, "N2", "N3", now);
    expect(result.shouldWake).toBe(true);
    expect(result.reason).toBe("Light sleep — optimal wake window");
    expect(result.minutesEarly).toBe(15);
  });

  // ── Within window — REM→N1 transition ────────────────────────────────────

  it("wakes on REM→N1 transition within the window", () => {
    const config = makeConfig(7, 0, 30);
    const now = timeAt(6, 50); // 10 min before target
    const result = shouldTriggerAlarm(config, "N1", "REM", now);
    expect(result.shouldWake).toBe(true);
    expect(result.reason).toBe("REM cycle complete");
    expect(result.minutesEarly).toBe(10);
  });

  // ── Within window — deep sleep or REM (should wait) ─────────────────────

  it("does not wake during N3 within the window", () => {
    const config = makeConfig(7, 0, 30);
    const now = timeAt(6, 40);
    const result = shouldTriggerAlarm(config, "N3", "N2", now);
    expect(result.shouldWake).toBe(false);
    expect(result.reason).toBe("Deep sleep — waiting for lighter stage");
  });

  it("does not wake during REM within the window", () => {
    const config = makeConfig(7, 0, 30);
    const now = timeAt(6, 40);
    const result = shouldTriggerAlarm(config, "REM", "N2", now);
    expect(result.shouldWake).toBe(false);
    expect(result.reason).toBe("REM sleep — waiting for cycle to end");
  });

  it("does not wake during Wake stage within the window", () => {
    const config = makeConfig(7, 0, 30);
    const now = timeAt(6, 50);
    const result = shouldTriggerAlarm(config, "Wake", null, now);
    expect(result.shouldWake).toBe(false);
    expect(result.reason).toBe("Waiting for optimal wake stage");
  });

  // ── Edge cases ───────────────────────────────────────────────────────────

  it("handles exact window boundary — should be inside", () => {
    const config = makeConfig(7, 0, 30);
    const now = timeAt(6, 30); // exactly 30 min before = boundary
    const result = shouldTriggerAlarm(config, "N1", null, now);
    expect(result.shouldWake).toBe(true);
    expect(result.reason).toBe("Light sleep — optimal wake window");
    expect(result.minutesEarly).toBe(30);
  });

  it("works with 15-minute window", () => {
    const config = makeConfig(7, 0, 15);
    const now = timeAt(6, 50); // 10 min before target, within 15 min window
    const result = shouldTriggerAlarm(config, "N2", null, now);
    expect(result.shouldWake).toBe(true);
    expect(result.reason).toBe("Light sleep — optimal wake window");
    expect(result.minutesEarly).toBe(10);
  });

  it("works with 45-minute window", () => {
    const config = makeConfig(7, 0, 45);
    const now = timeAt(6, 20); // 40 min before target, within 45 min window
    const result = shouldTriggerAlarm(config, "N1", "N2", now);
    expect(result.shouldWake).toBe(true);
    expect(result.reason).toBe("Light sleep — optimal wake window");
    expect(result.minutesEarly).toBe(40);
  });

  it("prioritizes REM→N1 transition reason over generic N1 reason", () => {
    const config = makeConfig(7, 0, 30);
    const now = timeAt(6, 45);
    // N1 with previous=REM should give "REM cycle complete" not "Light sleep"
    const result = shouldTriggerAlarm(config, "N1", "REM", now);
    expect(result.reason).toBe("REM cycle complete");
  });
});

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  checkTriggers,
  getDefaultTriggerConfig,
  saveTriggerConfig,
  loadTriggerConfig,
  type TriggerConfig,
  type InterventionTrigger,
  _resetCooldowns,
} from "@/lib/eeg-intervention-trigger";

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Build a state object with sensible defaults, overriding only what's needed. */
function makeState(overrides: Partial<Parameters<typeof checkTriggers>[0]> = {}) {
  return {
    stressIndex: 0.3,
    stressDurationSeconds: 0,
    blinksPerMinute: 15,
    sessionMinutes: 5,
    alphaLevel: 0.2,
    betaLevel: 0.15,
    highBetaDurationSeconds: 0,
    ...overrides,
  };
}

// ── Setup / Teardown ─────────────────────────────────────────────────────────

beforeEach(() => {
  _resetCooldowns();
  localStorage.clear();
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

// ── Tests ────────────────────────────────────────────────────────────────────

describe("eeg-intervention-trigger", () => {
  // 1. No stress -> no trigger
  it("returns null when stress is below threshold", () => {
    const config = getDefaultTriggerConfig();
    const state = makeState({ stressIndex: 0.3, stressDurationSeconds: 0 });
    expect(checkTriggers(state, config)).toBeNull();
  });

  // 2. Stress > 0.7 for 30s -> breathing suggestion
  it("returns breathing suggestion when stress exceeds threshold for duration", () => {
    const config = getDefaultTriggerConfig();
    const state = makeState({
      stressIndex: 0.8,
      stressDurationSeconds: 35,
    });
    const trigger = checkTriggers(state, config);
    expect(trigger).not.toBeNull();
    expect(trigger!.type).toBe("breathing");
    expect(trigger!.priority).toBe("high");
    expect(trigger!.reason).toBeTruthy();
  });

  // 3. High blink rate -> break suggestion
  it("returns break suggestion when blink rate exceeds fatigue threshold", () => {
    const config = getDefaultTriggerConfig();
    const state = makeState({
      blinksPerMinute: 30,
    });
    const trigger = checkTriggers(state, config);
    expect(trigger).not.toBeNull();
    expect(trigger!.type).toBe("break_suggestion");
    expect(trigger!.priority).toBe("medium");
  });

  // 4. Long session -> break suggestion (low priority)
  it("returns low-priority break suggestion after 25+ minutes", () => {
    const config = getDefaultTriggerConfig();
    const state = makeState({
      sessionMinutes: 30,
    });
    const trigger = checkTriggers(state, config);
    expect(trigger).not.toBeNull();
    expect(trigger!.type).toBe("break_suggestion");
    expect(trigger!.priority).toBe("low");
  });

  // 5. Stress trigger has higher priority than session length
  it("returns stress trigger (high) over session-length trigger (low)", () => {
    const config = getDefaultTriggerConfig();
    const state = makeState({
      stressIndex: 0.85,
      stressDurationSeconds: 40,
      sessionMinutes: 30,
    });
    const trigger = checkTriggers(state, config);
    expect(trigger).not.toBeNull();
    expect(trigger!.type).toBe("breathing");
    expect(trigger!.priority).toBe("high");
  });

  // 6. Cooldown: same trigger not repeated within 5 min
  it("does not re-trigger same type within 5-minute cooldown", () => {
    const config = getDefaultTriggerConfig();
    const state = makeState({
      stressIndex: 0.85,
      stressDurationSeconds: 40,
    });

    // First call triggers
    const first = checkTriggers(state, config);
    expect(first).not.toBeNull();
    expect(first!.type).toBe("breathing");

    // Second call within 5 minutes: should not trigger the same type
    vi.advanceTimersByTime(2 * 60 * 1000); // 2 minutes
    const second = checkTriggers(state, config);
    // Should be null because breathing is on cooldown and no other trigger qualifies at higher priority
    expect(second).toBeNull();

    // After 5 minutes: should trigger again
    vi.advanceTimersByTime(4 * 60 * 1000); // now 6 minutes total
    const third = checkTriggers(state, config);
    expect(third).not.toBeNull();
    expect(third!.type).toBe("breathing");
  });

  // 7. Disabled config -> no triggers
  it("returns null when config is disabled", () => {
    const config: TriggerConfig = {
      ...getDefaultTriggerConfig(),
      enabled: false,
    };
    const state = makeState({
      stressIndex: 0.9,
      stressDurationSeconds: 60,
      blinksPerMinute: 40,
      sessionMinutes: 60,
    });
    expect(checkTriggers(state, config)).toBeNull();
  });

  // 8. Individual toggles respected
  it("respects individual toggle: autoBreathing disabled skips breathing trigger", () => {
    const config: TriggerConfig = {
      ...getDefaultTriggerConfig(),
      autoBreathing: false,
    };
    const state = makeState({
      stressIndex: 0.85,
      stressDurationSeconds: 40,
    });
    // Breathing is off, stress doesn't produce a breathing trigger
    const trigger = checkTriggers(state, config);
    // Should be null or a different type (not breathing)
    if (trigger !== null) {
      expect(trigger.type).not.toBe("breathing");
    }
  });

  it("respects individual toggle: autoBreakReminder disabled skips break triggers", () => {
    const config: TriggerConfig = {
      ...getDefaultTriggerConfig(),
      autoBreakReminder: false,
    };
    const state = makeState({
      blinksPerMinute: 30,
      sessionMinutes: 30,
    });
    const trigger = checkTriggers(state, config);
    if (trigger !== null) {
      expect(trigger.type).not.toBe("break_suggestion");
    }
  });

  it("respects individual toggle: autoMusicChange disabled skips music trigger", () => {
    const config: TriggerConfig = {
      ...getDefaultTriggerConfig(),
      autoMusicChange: false,
    };
    const state = makeState({
      alphaLevel: 0.02,
      betaLevel: 0.5,
      highBetaDurationSeconds: 70,
    });
    const trigger = checkTriggers(state, config);
    if (trigger !== null) {
      expect(trigger.type).not.toBe("music_change");
    }
  });

  // 9. Default config has sensible values
  it("returns sensible default config values", () => {
    const config = getDefaultTriggerConfig();
    expect(config.enabled).toBe(true);
    expect(config.stressThreshold).toBeGreaterThanOrEqual(0.5);
    expect(config.stressThreshold).toBeLessThanOrEqual(0.9);
    expect(config.stressDuration).toBeGreaterThanOrEqual(15);
    expect(config.stressDuration).toBeLessThanOrEqual(60);
    expect(config.fatigueThreshold).toBeGreaterThanOrEqual(20);
    expect(config.fatigueThreshold).toBeLessThanOrEqual(35);
    expect(config.autoBreathing).toBe(true);
    expect(config.autoMusicChange).toBe(true);
    expect(config.autoBreakReminder).toBe(true);
  });

  // 10. Save/load config roundtrip
  it("roundtrips config through save/load", () => {
    const config: TriggerConfig = {
      enabled: false,
      stressThreshold: 0.6,
      stressDuration: 45,
      fatigueThreshold: 20,
      autoBreathing: false,
      autoMusicChange: true,
      autoBreakReminder: false,
    };
    saveTriggerConfig(config);
    const loaded = loadTriggerConfig();
    expect(loaded).toEqual(config);
  });

  // Additional: alpha very low + high beta for > 60s -> music change
  it("suggests music change when alpha is very low and beta is high for 60+ seconds", () => {
    const config = getDefaultTriggerConfig();
    const state = makeState({
      alphaLevel: 0.02,
      betaLevel: 0.5,
      highBetaDurationSeconds: 70,
    });
    const trigger = checkTriggers(state, config);
    expect(trigger).not.toBeNull();
    expect(trigger!.type).toBe("music_change");
    expect(trigger!.priority).toBe("medium");
  });

  // Additional: loadTriggerConfig returns defaults when nothing saved
  it("returns default config when nothing is saved in localStorage", () => {
    localStorage.clear();
    const loaded = loadTriggerConfig();
    const defaults = getDefaultTriggerConfig();
    expect(loaded).toEqual(defaults);
  });

  // Additional: trigger action is callable
  it("returns a trigger with a callable action function", () => {
    const config = getDefaultTriggerConfig();
    const state = makeState({
      stressIndex: 0.85,
      stressDurationSeconds: 40,
    });
    const trigger = checkTriggers(state, config);
    expect(trigger).not.toBeNull();
    expect(typeof trigger!.action).toBe("function");
    // Should not throw when called
    expect(() => trigger!.action()).not.toThrow();
  });
});

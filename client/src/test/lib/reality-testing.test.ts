import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  scheduleRealityTests,
  REALITY_TESTS,
  DEFAULT_CONFIG,
  loadRealityTestConfig,
  saveRealityTestConfig,
  type RealityTestConfig,
} from "@/lib/reality-testing";

// ── Mock supabase-store ──────────────────────────────────────────────────────

const mockStore: Record<string, string> = {};

vi.mock("@/lib/supabase-store", () => ({
  sbGetSetting: vi.fn((key: string) => mockStore[key] ?? null),
  sbSaveSetting: vi.fn((key: string, value: string) => {
    mockStore[key] = value;
  }),
}));

beforeEach(() => {
  Object.keys(mockStore).forEach((k) => delete mockStore[k]);
});

// ── scheduleRealityTests ────────────────────────────────────────────────────

describe("scheduleRealityTests", () => {
  it("returns the requested number of tests", () => {
    const config: RealityTestConfig = {
      enabled: true,
      frequency: 5,
      startHour: 9,
      endHour: 22,
      randomize: true,
    };
    const schedule = scheduleRealityTests(config);
    expect(schedule).toHaveLength(5);
  });

  it("returns empty array when frequency is 0", () => {
    const config: RealityTestConfig = {
      enabled: true,
      frequency: 0,
      startHour: 9,
      endHour: 22,
      randomize: true,
    };
    expect(scheduleRealityTests(config)).toHaveLength(0);
  });

  it("returns empty array when startHour >= endHour", () => {
    const config: RealityTestConfig = {
      enabled: true,
      frequency: 5,
      startHour: 22,
      endHour: 9,
      randomize: true,
    };
    expect(scheduleRealityTests(config)).toHaveLength(0);
  });

  it("all scheduled times fall within the active window", () => {
    const config: RealityTestConfig = {
      enabled: true,
      frequency: 10,
      startHour: 8,
      endHour: 20,
      randomize: true,
    };
    const schedule = scheduleRealityTests(config);
    for (const slot of schedule) {
      const totalMin = slot.hour * 60 + slot.minute;
      expect(totalMin).toBeGreaterThanOrEqual(config.startHour * 60);
      expect(totalMin).toBeLessThan(config.endHour * 60);
    }
  });

  it("schedule is sorted by time", () => {
    const config: RealityTestConfig = {
      enabled: true,
      frequency: 8,
      startHour: 7,
      endHour: 23,
      randomize: true,
    };
    const schedule = scheduleRealityTests(config);
    for (let i = 1; i < schedule.length; i++) {
      const prevMin = schedule[i - 1].hour * 60 + schedule[i - 1].minute;
      const currMin = schedule[i].hour * 60 + schedule[i].minute;
      expect(currMin).toBeGreaterThanOrEqual(prevMin);
    }
  });

  it("each entry has a valid test from REALITY_TESTS", () => {
    const config: RealityTestConfig = {
      enabled: true,
      frequency: 6,
      startHour: 9,
      endHour: 21,
      randomize: true,
    };
    const schedule = scheduleRealityTests(config);
    const techniques = new Set(REALITY_TESTS.map((t) => t.technique));
    for (const slot of schedule) {
      expect(techniques.has(slot.test.technique)).toBe(true);
      expect(slot.test.prompt.length).toBeGreaterThan(0);
    }
  });

  it("non-randomized schedule places tests at window midpoints", () => {
    const config: RealityTestConfig = {
      enabled: true,
      frequency: 4,
      startHour: 8,
      endHour: 20,
      randomize: false,
    };
    const schedule = scheduleRealityTests(config);
    // 12 hours = 720 minutes, 4 windows of 180 min each
    // Midpoints: 8*60+90=570, +180=750, +180=930, +180=1110
    expect(schedule).toHaveLength(4);
    const windowSize = 180;
    const midOffset = Math.floor(windowSize / 2);
    for (let i = 0; i < schedule.length; i++) {
      const expectedMin = config.startHour * 60 + Math.floor(i * windowSize) + midOffset;
      const actualMin = schedule[i].hour * 60 + schedule[i].minute;
      expect(actualMin).toBe(expectedMin);
    }
  });

  it("calling scheduleRealityTests twice on the same day returns the same schedule", () => {
    const config: RealityTestConfig = {
      enabled: true,
      frequency: 5,
      startHour: 9,
      endHour: 22,
      randomize: true,
    };
    const a = scheduleRealityTests(config);
    const b = scheduleRealityTests(config);
    expect(a).toEqual(b);
  });

  it("handles minimum frequency of 3", () => {
    const config: RealityTestConfig = {
      enabled: true,
      frequency: 3,
      startHour: 10,
      endHour: 18,
      randomize: true,
    };
    const schedule = scheduleRealityTests(config);
    expect(schedule).toHaveLength(3);
  });

  it("handles maximum frequency of 10", () => {
    const config: RealityTestConfig = {
      enabled: true,
      frequency: 10,
      startHour: 6,
      endHour: 23,
      randomize: true,
    };
    const schedule = scheduleRealityTests(config);
    expect(schedule).toHaveLength(10);
  });
});

// ── Config persistence ──────────────────────────────────────────────────────

describe("loadRealityTestConfig", () => {
  it("returns DEFAULT_CONFIG when nothing is stored", () => {
    const config = loadRealityTestConfig();
    expect(config).toEqual(DEFAULT_CONFIG);
  });

  it("loads saved config from store", () => {
    const custom: RealityTestConfig = {
      enabled: true,
      frequency: 7,
      startHour: 10,
      endHour: 20,
      randomize: false,
    };
    mockStore["ndw_reality_test_config"] = JSON.stringify(custom);
    const loaded = loadRealityTestConfig();
    expect(loaded).toEqual(custom);
  });

  it("clamps out-of-range frequency to valid bounds", () => {
    mockStore["ndw_reality_test_config"] = JSON.stringify({
      enabled: true,
      frequency: 99,
      startHour: 9,
      endHour: 22,
      randomize: true,
    });
    const loaded = loadRealityTestConfig();
    expect(loaded.frequency).toBe(10);
  });

  it("handles corrupted JSON gracefully", () => {
    mockStore["ndw_reality_test_config"] = "not-json!!!";
    const loaded = loadRealityTestConfig();
    expect(loaded).toEqual(DEFAULT_CONFIG);
  });
});

describe("saveRealityTestConfig", () => {
  it("persists config to store", () => {
    const config: RealityTestConfig = {
      enabled: true,
      frequency: 6,
      startHour: 8,
      endHour: 21,
      randomize: true,
    };
    saveRealityTestConfig(config);
    expect(mockStore["ndw_reality_test_config"]).toBe(JSON.stringify(config));
  });
});

// ── REALITY_TESTS constant ──────────────────────────────────────────────────

describe("REALITY_TESTS", () => {
  it("has 6 techniques", () => {
    expect(REALITY_TESTS).toHaveLength(6);
  });

  it("all techniques are unique", () => {
    const techniques = REALITY_TESTS.map((t) => t.technique);
    expect(new Set(techniques).size).toBe(techniques.length);
  });

  it("all prompts are non-empty strings", () => {
    for (const t of REALITY_TESTS) {
      expect(typeof t.prompt).toBe("string");
      expect(t.prompt.length).toBeGreaterThan(0);
    }
  });
});

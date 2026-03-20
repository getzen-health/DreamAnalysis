import { describe, it, expect, beforeEach, vi } from "vitest";
import {
  getDefaultPreferences,
  getPresetForFrequency,
  saveNotificationPreferences,
  getNotificationPreferences,
  generateMorningBrief,
  generateSessionReminder,
  generateWeeklyInsight,
  generateStreakEncouragement,
  isInQuietHours,
  type NotificationFrequency,
  type NotificationPreferences,
  type NotificationContent,
} from "@/lib/notification-strategy";

// ── Mock localStorage ─────────────────────────────────────────────────────

const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: vi.fn((key: string) => store[key] ?? null),
    setItem: vi.fn((key: string, value: string) => { store[key] = value; }),
    removeItem: vi.fn((key: string) => { delete store[key]; }),
    clear: vi.fn(() => { store = {}; }),
  };
})();

Object.defineProperty(globalThis, "localStorage", { value: localStorageMock });

beforeEach(() => {
  localStorageMock.clear();
  vi.clearAllMocks();
});

// ── Guilt language detector ──────────────────────────────────────────────

const GUILT_PATTERNS = [
  /\bmissed\b/i,
  /\bhaven'?t\b/i,
  /\bbreak your streak\b/i,
  /\bdon'?t break\b/i,
  /\byou forgot\b/i,
  /\bfailed\b/i,
  /\bfalling behind\b/i,
  /\byou should\b/i,
];

function containsGuiltLanguage(text: string): boolean {
  return GUILT_PATTERNS.some((pattern) => pattern.test(text));
}

// ── Default preferences ──────────────────────────────────────────────────

describe("getDefaultPreferences", () => {
  it("returns 'minimal' frequency by default", () => {
    const prefs = getDefaultPreferences();
    expect(prefs.frequency).toBe("minimal");
  });

  it("enables only morningBrief for minimal frequency", () => {
    const prefs = getDefaultPreferences();
    expect(prefs.enabledTypes.morningBrief).toBe(true);
    expect(prefs.enabledTypes.sessionReminder).toBe(false);
    expect(prefs.enabledTypes.weeklyInsight).toBe(false);
    expect(prefs.enabledTypes.streakEncouragement).toBe(false);
  });

  it("sets quiet hours to 22-7 by default", () => {
    const prefs = getDefaultPreferences();
    expect(prefs.quietHoursStart).toBe(22);
    expect(prefs.quietHoursEnd).toBe(7);
  });
});

// ── Frequency presets ────────────────────────────────────────────────────

describe("getPresetForFrequency", () => {
  it("balanced enables morningBrief + sessionReminder + weeklyInsight", () => {
    const types = getPresetForFrequency("balanced");
    expect(types.morningBrief).toBe(true);
    expect(types.sessionReminder).toBe(true);
    expect(types.weeklyInsight).toBe(true);
    expect(types.streakEncouragement).toBe(false);
  });

  it("engaged enables all types", () => {
    const types = getPresetForFrequency("engaged");
    expect(types.morningBrief).toBe(true);
    expect(types.sessionReminder).toBe(true);
    expect(types.weeklyInsight).toBe(true);
    expect(types.streakEncouragement).toBe(true);
  });

  it("minimal enables only morningBrief", () => {
    const types = getPresetForFrequency("minimal");
    expect(types.morningBrief).toBe(true);
    expect(types.sessionReminder).toBe(false);
    expect(types.weeklyInsight).toBe(false);
    expect(types.streakEncouragement).toBe(false);
  });
});

// ── Persistence roundtrip ────────────────────────────────────────────────

describe("saveNotificationPreferences / getNotificationPreferences", () => {
  it("roundtrips save and load correctly", () => {
    const prefs: NotificationPreferences = {
      frequency: "balanced",
      quietHoursStart: 23,
      quietHoursEnd: 8,
      enabledTypes: {
        morningBrief: true,
        sessionReminder: true,
        weeklyInsight: true,
        streakEncouragement: false,
      },
    };
    saveNotificationPreferences(prefs);
    const loaded = getNotificationPreferences();
    expect(loaded).toEqual(prefs);
  });

  it("returns default preferences when nothing is stored", () => {
    const loaded = getNotificationPreferences();
    expect(loaded).toEqual(getDefaultPreferences());
  });

  it("returns default preferences when stored data is corrupted", () => {
    localStorage.setItem("ndw_notification_prefs", "not-json{{{");
    const loaded = getNotificationPreferences();
    expect(loaded).toEqual(getDefaultPreferences());
  });
});

// ── Morning brief generation ─────────────────────────────────────────────

describe("generateMorningBrief", () => {
  it("generates non-empty title and body", () => {
    const content = generateMorningBrief({});
    expect(content.title.length).toBeGreaterThan(0);
    expect(content.body.length).toBeGreaterThan(0);
  });

  it("sets type to morning_brief", () => {
    const content = generateMorningBrief({});
    expect(content.type).toBe("morning_brief");
  });

  it("never contains guilt language", () => {
    // Test with various input combinations
    const cases = [
      {},
      { lastEmotion: "happy" },
      { lastEmotion: "sad" },
      { streakDays: 0 },
      { streakDays: 5 },
      { streakDays: 0, lastEmotion: "angry" },
      { chronotype: "morning" },
      { chronotype: "evening", lastEmotion: "neutral", streakDays: 3 },
    ];

    for (const input of cases) {
      const content = generateMorningBrief(input);
      expect(containsGuiltLanguage(content.title)).toBe(false);
      expect(containsGuiltLanguage(content.body)).toBe(false);
    }
  });

  it("includes last emotion when provided", () => {
    const content = generateMorningBrief({ lastEmotion: "happy" });
    expect(content.body.toLowerCase()).toContain("happy");
  });

  it("includes streak count when provided and > 0", () => {
    const content = generateMorningBrief({ streakDays: 5 });
    expect(content.body).toContain("5");
  });

  it("title is under 60 characters", () => {
    const cases = [
      {},
      { lastEmotion: "happy", streakDays: 100, chronotype: "evening" },
    ];
    for (const input of cases) {
      const content = generateMorningBrief(input);
      expect(content.title.length).toBeLessThanOrEqual(60);
    }
  });

  it("body is under 120 characters", () => {
    const cases = [
      {},
      { lastEmotion: "happy", streakDays: 100, chronotype: "evening" },
      { lastEmotion: "sad", streakDays: 0 },
    ];
    for (const input of cases) {
      const content = generateMorningBrief(input);
      expect(content.body.length).toBeLessThanOrEqual(120);
    }
  });
});

// ── Session reminder generation ──────────────────────────────────────────

describe("generateSessionReminder", () => {
  it("generates content with session_reminder type", () => {
    const content = generateSessionReminder();
    expect(content.type).toBe("session_reminder");
  });

  it("generates non-empty title and body", () => {
    const content = generateSessionReminder();
    expect(content.title.length).toBeGreaterThan(0);
    expect(content.body.length).toBeGreaterThan(0);
  });

  it("never contains guilt language", () => {
    const content = generateSessionReminder();
    expect(containsGuiltLanguage(content.title)).toBe(false);
    expect(containsGuiltLanguage(content.body)).toBe(false);
  });

  it("title is under 60 characters", () => {
    const content = generateSessionReminder();
    expect(content.title.length).toBeLessThanOrEqual(60);
  });

  it("body is under 120 characters", () => {
    const content = generateSessionReminder();
    expect(content.body.length).toBeLessThanOrEqual(120);
  });
});

// ── Weekly insight generation ────────────────────────────────────────────

describe("generateWeeklyInsight", () => {
  it("generates content with weekly_insight type", () => {
    const content = generateWeeklyInsight({
      voiceCheckins: 4,
      neurofeedbackSessions: 2,
      stressTrend: "improving",
    });
    expect(content.type).toBe("weekly_insight");
  });

  it("includes check-in and session counts in body", () => {
    const content = generateWeeklyInsight({
      voiceCheckins: 4,
      neurofeedbackSessions: 2,
      stressTrend: "improving",
    });
    expect(content.body).toContain("4");
    expect(content.body).toContain("2");
  });

  it("never contains guilt language", () => {
    const cases = [
      { voiceCheckins: 0, neurofeedbackSessions: 0, stressTrend: "stable" as const },
      { voiceCheckins: 7, neurofeedbackSessions: 5, stressTrend: "improving" as const },
      { voiceCheckins: 1, neurofeedbackSessions: 0, stressTrend: "worsening" as const },
    ];
    for (const input of cases) {
      const content = generateWeeklyInsight(input);
      expect(containsGuiltLanguage(content.title)).toBe(false);
      expect(containsGuiltLanguage(content.body)).toBe(false);
    }
  });

  it("title is under 60 characters", () => {
    const content = generateWeeklyInsight({
      voiceCheckins: 4,
      neurofeedbackSessions: 2,
      stressTrend: "improving",
    });
    expect(content.title.length).toBeLessThanOrEqual(60);
  });

  it("body is under 120 characters", () => {
    const content = generateWeeklyInsight({
      voiceCheckins: 4,
      neurofeedbackSessions: 2,
      stressTrend: "improving",
    });
    expect(content.body.length).toBeLessThanOrEqual(120);
  });
});

// ── Streak encouragement ─────────────────────────────────────────────────

describe("generateStreakEncouragement", () => {
  it("generates content with streak type", () => {
    const content = generateStreakEncouragement(7);
    expect(content.type).toBe("streak");
  });

  it("includes streak count in body", () => {
    const content = generateStreakEncouragement(7);
    expect(content.body).toContain("7");
  });

  it("never contains guilt language", () => {
    for (const days of [0, 1, 3, 7, 14, 30, 100]) {
      const content = generateStreakEncouragement(days);
      expect(containsGuiltLanguage(content.title)).toBe(false);
      expect(containsGuiltLanguage(content.body)).toBe(false);
    }
  });

  it("title is under 60 characters", () => {
    for (const days of [0, 1, 7, 100, 365]) {
      const content = generateStreakEncouragement(days);
      expect(content.title.length).toBeLessThanOrEqual(60);
    }
  });

  it("body is under 120 characters", () => {
    for (const days of [0, 1, 7, 100, 365]) {
      const content = generateStreakEncouragement(days);
      expect(content.body.length).toBeLessThanOrEqual(120);
    }
  });
});

// ── Quiet hours ──────────────────────────────────────────────────────────

describe("isInQuietHours", () => {
  it("returns true during quiet hours (22-7, hour=23)", () => {
    expect(isInQuietHours(23, 22, 7)).toBe(true);
  });

  it("returns true during quiet hours (22-7, hour=0)", () => {
    expect(isInQuietHours(0, 22, 7)).toBe(true);
  });

  it("returns true during quiet hours (22-7, hour=6)", () => {
    expect(isInQuietHours(6, 22, 7)).toBe(true);
  });

  it("returns false outside quiet hours (22-7, hour=7)", () => {
    expect(isInQuietHours(7, 22, 7)).toBe(false);
  });

  it("returns false outside quiet hours (22-7, hour=12)", () => {
    expect(isInQuietHours(12, 22, 7)).toBe(false);
  });

  it("returns false outside quiet hours (22-7, hour=21)", () => {
    expect(isInQuietHours(21, 22, 7)).toBe(false);
  });

  it("returns true at the start of quiet hours (22-7, hour=22)", () => {
    expect(isInQuietHours(22, 22, 7)).toBe(true);
  });

  it("handles non-wrapping quiet hours (1-6)", () => {
    expect(isInQuietHours(3, 1, 6)).toBe(true);
    expect(isInQuietHours(0, 1, 6)).toBe(false);
    expect(isInQuietHours(7, 1, 6)).toBe(false);
  });
});

import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import {
  getCompanionUsage,
  recordCompanionSession,
  getUsageNudge,
  type CompanionUsageStats,
} from "@/lib/companion-safeguards";

const STORAGE_KEY = "ndw_companion_usage";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function setStorage(data: unknown) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
}

function clearStorage() {
  localStorage.removeItem(STORAGE_KEY);
}

/** Returns today's date string in YYYY-MM-DD format */
function todayStr(): string {
  return new Date().toISOString().slice(0, 10);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("companion-safeguards", () => {
  beforeEach(() => {
    clearStorage();
    vi.restoreAllMocks();
  });

  afterEach(() => {
    clearStorage();
  });

  // ── getCompanionUsage ───────────────────────────────────────────────────

  describe("getCompanionUsage", () => {
    it("returns zero minutes and zero sessions when no data exists", () => {
      const stats = getCompanionUsage();
      expect(stats.todayMinutes).toBe(0);
      expect(stats.todaySessions).toBe(0);
      expect(stats.consecutiveDaysHeavyUse).toBe(0);
      expect(stats.lastSessionEnd).toBeNull();
    });
  });

  // ── recordCompanionSession ──────────────────────────────────────────────

  describe("recordCompanionSession", () => {
    it("updates minutes and session count", () => {
      recordCompanionSession(10);
      const stats = getCompanionUsage();
      expect(stats.todayMinutes).toBe(10);
      expect(stats.todaySessions).toBe(1);
    });

    it("accumulates across multiple recordings in the same day", () => {
      recordCompanionSession(5);
      recordCompanionSession(8);
      const stats = getCompanionUsage();
      expect(stats.todayMinutes).toBe(13);
      expect(stats.todaySessions).toBe(2);
    });

    it("sets lastSessionEnd to current time", () => {
      const before = new Date().toISOString();
      recordCompanionSession(3);
      const stats = getCompanionUsage();
      expect(stats.lastSessionEnd).not.toBeNull();
      // lastSessionEnd should be >= before
      expect(new Date(stats.lastSessionEnd!).getTime()).toBeGreaterThanOrEqual(
        new Date(before).getTime() - 1000 // 1s tolerance
      );
    });
  });

  // ── getUsageNudge ───────────────────────────────────────────────────────

  describe("getUsageNudge", () => {
    it("returns null under 15 minutes in a single session", () => {
      const stats: CompanionUsageStats = {
        todayMinutes: 10,
        todaySessions: 1,
        consecutiveDaysHeavyUse: 0,
        lastSessionEnd: null,
      };
      expect(getUsageNudge(stats)).toBeNull();
    });

    it("returns a nudge at 15+ minutes in a single session", () => {
      const stats: CompanionUsageStats = {
        todayMinutes: 16,
        todaySessions: 1,
        consecutiveDaysHeavyUse: 0,
        lastSessionEnd: null,
      };
      const nudge = getUsageNudge(stats);
      expect(nudge).not.toBeNull();
      expect(nudge).toContain("break");
    });

    it("returns a nudge at 3+ consecutive heavy-use days", () => {
      const stats: CompanionUsageStats = {
        todayMinutes: 5,
        todaySessions: 1,
        consecutiveDaysHeavyUse: 3,
        lastSessionEnd: null,
      };
      const nudge = getUsageNudge(stats);
      expect(nudge).not.toBeNull();
      expect(nudge).toContain("human connection");
    });

    it("returns daily limit nudge at 30+ minutes total in one day", () => {
      const stats: CompanionUsageStats = {
        todayMinutes: 31,
        todaySessions: 3,
        consecutiveDaysHeavyUse: 0,
        lastSessionEnd: null,
      };
      const nudge = getUsageNudge(stats);
      expect(nudge).not.toBeNull();
      expect(nudge).toContain("Daily limit");
    });

    it("prioritizes daily limit nudge over session nudge", () => {
      const stats: CompanionUsageStats = {
        todayMinutes: 35,
        todaySessions: 2,
        consecutiveDaysHeavyUse: 4,
        lastSessionEnd: null,
      };
      const nudge = getUsageNudge(stats);
      // Daily limit should take precedence (most restrictive)
      expect(nudge).toContain("Daily limit");
    });
  });

  // ── Nudge message quality ───────────────────────────────────────────────

  describe("nudge message quality", () => {
    const allNudgeScenarios: CompanionUsageStats[] = [
      // Session nudge
      { todayMinutes: 16, todaySessions: 1, consecutiveDaysHeavyUse: 0, lastSessionEnd: null },
      // Consecutive days nudge
      { todayMinutes: 5, todaySessions: 1, consecutiveDaysHeavyUse: 3, lastSessionEnd: null },
      // Daily limit nudge
      { todayMinutes: 31, todaySessions: 3, consecutiveDaysHeavyUse: 0, lastSessionEnd: null },
    ];

    it("nudge messages never contain guilt language", () => {
      const guiltPatterns = /\bshould\b|\bshame\b|\bguilty\b|\bwasting\b|\bwaste\b|\baddicted\b|\baddiction\b|\bpathetic\b|\bstop\b/i;
      for (const stats of allNudgeScenarios) {
        const nudge = getUsageNudge(stats);
        if (nudge) {
          expect(nudge).not.toMatch(guiltPatterns);
        }
      }
    });

    it("nudge messages always offer an alternative activity", () => {
      const alternativePatterns = /friend|meditation|meditat|breath|break|call|connect|exercise/i;
      for (const stats of allNudgeScenarios) {
        const nudge = getUsageNudge(stats);
        if (nudge) {
          expect(nudge).toMatch(alternativePatterns);
        }
      }
    });
  });

  // ── Usage resets at midnight ────────────────────────────────────────────

  describe("usage resets at midnight", () => {
    it("resets todayMinutes and todaySessions for a new day", () => {
      // Simulate data from yesterday
      const yesterday = new Date();
      yesterday.setDate(yesterday.getDate() - 1);
      const yesterdayStr = yesterday.toISOString().slice(0, 10);

      setStorage({
        date: yesterdayStr,
        todayMinutes: 25,
        todaySessions: 4,
        consecutiveDaysHeavyUse: 2,
        heavyUseDates: [yesterdayStr],
        lastSessionEnd: yesterday.toISOString(),
      });

      const stats = getCompanionUsage();
      expect(stats.todayMinutes).toBe(0);
      expect(stats.todaySessions).toBe(0);
    });

    it("increments consecutiveDaysHeavyUse when yesterday was heavy use", () => {
      const yesterday = new Date();
      yesterday.setDate(yesterday.getDate() - 1);
      const yesterdayStr = yesterday.toISOString().slice(0, 10);

      setStorage({
        date: yesterdayStr,
        todayMinutes: 20, // > 15 min = heavy use
        todaySessions: 2,
        consecutiveDaysHeavyUse: 2,
        heavyUseDates: [yesterdayStr],
        lastSessionEnd: yesterday.toISOString(),
      });

      const stats = getCompanionUsage();
      // Yesterday had heavy use, previous consecutive was 2, so now 3
      expect(stats.consecutiveDaysHeavyUse).toBe(3);
    });

    it("resets consecutiveDaysHeavyUse when yesterday was light use", () => {
      const yesterday = new Date();
      yesterday.setDate(yesterday.getDate() - 1);
      const yesterdayStr = yesterday.toISOString().slice(0, 10);

      setStorage({
        date: yesterdayStr,
        todayMinutes: 10, // <= 15 min = light use
        todaySessions: 1,
        consecutiveDaysHeavyUse: 5,
        heavyUseDates: [],
        lastSessionEnd: yesterday.toISOString(),
      });

      const stats = getCompanionUsage();
      expect(stats.consecutiveDaysHeavyUse).toBe(0);
    });
  });
});

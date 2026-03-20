import { describe, it, expect, beforeEach, vi } from "vitest";
import {
  getSessionSchedule,
  recordNeurofeedbackSession,
  getSessionHistory,
  type SessionSchedule,
} from "@/lib/neurofeedback-schedule";

// ── Helpers ──────────────────────────────────────────────────────────────────

function daysAgo(n: number): Date {
  const d = new Date();
  d.setDate(d.getDate() - n);
  d.setHours(12, 0, 0, 0); // noon to avoid DST edge cases
  return d;
}

function makeSessions(count: number, startDaysAgo: number, spacingDays: number): Date[] {
  const sessions: Date[] = [];
  for (let i = 0; i < count; i++) {
    sessions.push(daysAgo(startDaysAgo - i * spacingDays));
  }
  return sessions.sort((a, b) => a.getTime() - b.getTime());
}

// ── Tests ────────────────────────────────────────────────────────────────────

describe("getSessionSchedule", () => {
  it("returns sensible defaults for empty history", () => {
    const result = getSessionSchedule([]);
    expect(result.totalSessions).toBe(0);
    expect(result.currentPhase).toBe("beginner");
    expect(result.isTooSoon).toBe(false);
    expect(result.isOptimalWindow).toBe(false);
    expect(result.isTooLate).toBe(false);
    expect(result.progressPercent).toBe(0);
    expect(result.nextSessionDate).toBeNull();
    expect(result.message).toBeTruthy();
  });

  it("flags isTooSoon when session was yesterday (< 2 days)", () => {
    const result = getSessionSchedule([daysAgo(1)]);
    expect(result.isTooSoon).toBe(true);
    expect(result.isOptimalWindow).toBe(false);
    expect(result.isTooLate).toBe(false);
    expect(result.message).toMatch(/consolidat|rest/i);
  });

  it("flags isOptimalWindow when session was 2 days ago", () => {
    const result = getSessionSchedule([daysAgo(2)]);
    expect(result.isOptimalWindow).toBe(true);
    expect(result.isTooSoon).toBe(false);
    expect(result.isTooLate).toBe(false);
    expect(result.message).toMatch(/ready|great/i);
  });

  it("flags isOptimalWindow when session was 3 days ago", () => {
    const result = getSessionSchedule([daysAgo(3)]);
    expect(result.isOptimalWindow).toBe(true);
    expect(result.isTooSoon).toBe(false);
    expect(result.isTooLate).toBe(false);
  });

  it("flags isTooLate when session was > 5 days ago", () => {
    const result = getSessionSchedule([daysAgo(6)]);
    expect(result.isTooLate).toBe(true);
    expect(result.isTooSoon).toBe(false);
    expect(result.isOptimalWindow).toBe(false);
    expect(result.message).toMatch(/while|pick up/i);
  });

  it("shows getting-late message at 4-5 days", () => {
    const result4 = getSessionSchedule([daysAgo(4)]);
    expect(result4.isTooLate).toBe(false);
    expect(result4.isOptimalWindow).toBe(false);
    expect(result4.isTooSoon).toBe(false);
    expect(result4.message).toMatch(/time|consistency/i);

    const result5 = getSessionSchedule([daysAgo(5)]);
    expect(result5.message).toMatch(/time|consistency/i);
  });

  it("sets phase to beginner at 8 sessions (1-8 = beginner)", () => {
    const sessions = makeSessions(8, 24, 3);
    const result = getSessionSchedule(sessions);
    expect(result.currentPhase).toBe("beginner");
    expect(result.totalSessions).toBe(8);
  });

  it("sets phase to building at 9 sessions (9-20 = building)", () => {
    const sessions = makeSessions(9, 27, 3);
    const result = getSessionSchedule(sessions);
    expect(result.currentPhase).toBe("building");
  });

  it("sets phase to maintaining after 21+ sessions", () => {
    const sessions = makeSessions(21, 63, 3);
    const result = getSessionSchedule(sessions);
    expect(result.currentPhase).toBe("maintaining");
  });

  it("caps progressPercent at 100 for 20 sessions", () => {
    const sessions = makeSessions(20, 60, 3);
    const result = getSessionSchedule(sessions);
    expect(result.progressPercent).toBe(100);
  });

  it("calculates progress correctly (10 sessions = 50%)", () => {
    const sessions = makeSessions(10, 30, 3);
    const result = getSessionSchedule(sessions);
    expect(result.progressPercent).toBe(50);
  });

  it("calculates daysSinceLastSession correctly", () => {
    const result = getSessionSchedule([daysAgo(3)]);
    expect(result.daysSinceLastSession).toBeGreaterThanOrEqual(2);
    expect(result.daysSinceLastSession).toBeLessThanOrEqual(4);
  });

  it("recommends next session date 2-3 days from last session", () => {
    const lastSession = daysAgo(1);
    const result = getSessionSchedule([lastSession]);
    expect(result.nextSessionDate).not.toBeNull();
    if (result.nextSessionDate) {
      const diffMs = result.nextSessionDate.getTime() - lastSession.getTime();
      const diffDays = diffMs / (1000 * 60 * 60 * 24);
      expect(diffDays).toBeGreaterThanOrEqual(1.5);
      expect(diffDays).toBeLessThanOrEqual(3.5);
    }
  });
});

// ── localStorage persistence ─────────────────────────────────────────────────

describe("session history persistence", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("returns empty array when no sessions recorded", () => {
    const history = getSessionHistory();
    expect(history).toEqual([]);
  });

  it("records a session and retrieves it", () => {
    recordNeurofeedbackSession();
    const history = getSessionHistory();
    expect(history).toHaveLength(1);
    expect(history[0]).toBeInstanceOf(Date);
    // Should be within the last minute
    const diff = Date.now() - history[0].getTime();
    expect(diff).toBeLessThan(60_000);
  });

  it("accumulates multiple sessions", () => {
    recordNeurofeedbackSession();
    recordNeurofeedbackSession();
    recordNeurofeedbackSession();
    const history = getSessionHistory();
    expect(history).toHaveLength(3);
  });

  it("save/load roundtrip preserves Date objects", () => {
    recordNeurofeedbackSession();
    const history = getSessionHistory();
    expect(history[0]).toBeInstanceOf(Date);
    expect(Number.isNaN(history[0].getTime())).toBe(false);
  });
});

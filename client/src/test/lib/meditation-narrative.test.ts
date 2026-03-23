import { describe, it, expect, beforeEach } from "vitest";
import {
  NARRATIVE_CHAPTERS,
  getCurrentChapter,
  getNarrativeProgress,
  getStoredSessionCount,
  incrementSessionCount,
  type NarrativeChapter,
} from "@/lib/meditation-narrative";

beforeEach(() => {
  localStorage.clear();
});

// ── Chapter definitions ──────────────────────────────────────────────────

describe("NARRATIVE_CHAPTERS", () => {
  it("has exactly 10 chapters", () => {
    expect(NARRATIVE_CHAPTERS).toHaveLength(10);
  });

  it("chapters are ordered by unlockSessionCount", () => {
    for (let i = 1; i < NARRATIVE_CHAPTERS.length; i++) {
      expect(NARRATIVE_CHAPTERS[i].unlockSessionCount).toBeGreaterThan(
        NARRATIVE_CHAPTERS[i - 1].unlockSessionCount,
      );
    }
  });

  it("first chapter unlocks at 0 sessions", () => {
    expect(NARRATIVE_CHAPTERS[0].unlockSessionCount).toBe(0);
  });

  it("each chapter has required fields", () => {
    for (const ch of NARRATIVE_CHAPTERS) {
      expect(ch.chapter).toBeGreaterThanOrEqual(1);
      expect(ch.title.length).toBeGreaterThan(0);
      expect(ch.theme.length).toBeGreaterThan(0);
      expect(ch.description.length).toBeGreaterThan(0);
      expect(typeof ch.unlockSessionCount).toBe("number");
    }
  });
});

// ── getCurrentChapter ────────────────────────────────────────────────────

describe("getCurrentChapter", () => {
  it("returns chapter 1 for 0 sessions", () => {
    const ch = getCurrentChapter(0);
    expect(ch.chapter).toBe(1);
    expect(ch.title).toBe("The Quiet Garden");
  });

  it("returns chapter 1 for negative sessions", () => {
    const ch = getCurrentChapter(-5);
    expect(ch.chapter).toBe(1);
  });

  it("returns chapter 2 after 5 sessions", () => {
    const ch = getCurrentChapter(5);
    expect(ch.chapter).toBe(2);
  });

  it("returns chapter 10 at 130 sessions", () => {
    const ch = getCurrentChapter(130);
    expect(ch.chapter).toBe(10);
  });

  it("returns chapter 10 for very high session counts", () => {
    const ch = getCurrentChapter(999);
    expect(ch.chapter).toBe(10);
  });

  it("returns correct chapter at each boundary", () => {
    // Check that at exactly the unlock count, we get that chapter
    for (const ch of NARRATIVE_CHAPTERS) {
      const result = getCurrentChapter(ch.unlockSessionCount);
      expect(result.chapter).toBe(ch.chapter);
    }
  });
});

// ── getNarrativeProgress ─────────────────────────────────────────────────

describe("getNarrativeProgress", () => {
  it("returns correct structure at 0 sessions", () => {
    const progress = getNarrativeProgress(0);
    expect(progress.currentChapter.chapter).toBe(1);
    expect(progress.totalSessions).toBe(0);
    expect(progress.sessionsToNextChapter).toBe(5); // ch2 unlocks at 5
    expect(progress.chapterProgress).toBe(0);
    expect(progress.journeyComplete).toBe(false);
  });

  it("shows progress within a chapter", () => {
    const progress = getNarrativeProgress(3);
    expect(progress.currentChapter.chapter).toBe(1);
    expect(progress.chapterProgress).toBeCloseTo(3 / 5, 2); // 3 of 5 to next
    expect(progress.sessionsToNextChapter).toBe(2);
  });

  it("journey complete at final chapter", () => {
    const progress = getNarrativeProgress(130);
    expect(progress.journeyComplete).toBe(true);
    expect(progress.sessionsToNextChapter).toBe(0);
    expect(progress.chapterProgress).toBe(1);
  });

  it("journey not complete at second-to-last", () => {
    const progress = getNarrativeProgress(100);
    expect(progress.currentChapter.chapter).toBe(9);
    expect(progress.journeyComplete).toBe(false);
  });
});

// ── Persistence ──────────────────────────────────────────────────────────

describe("getStoredSessionCount", () => {
  it("returns 0 when no data stored", () => {
    expect(getStoredSessionCount()).toBe(0);
  });

  it("reads stored value", () => {
    localStorage.setItem("ndw_meditation_session_count", "42");
    expect(getStoredSessionCount()).toBe(42);
  });

  it("returns 0 for invalid stored value", () => {
    localStorage.setItem("ndw_meditation_session_count", "not-a-number");
    expect(getStoredSessionCount()).toBe(0);
  });

  it("returns 0 for negative stored value", () => {
    localStorage.setItem("ndw_meditation_session_count", "-5");
    expect(getStoredSessionCount()).toBe(0);
  });
});

describe("incrementSessionCount", () => {
  it("increments from 0", () => {
    const result = incrementSessionCount();
    expect(result).toBe(1);
    expect(getStoredSessionCount()).toBe(1);
  });

  it("increments existing value", () => {
    localStorage.setItem("ndw_meditation_session_count", "10");
    const result = incrementSessionCount();
    expect(result).toBe(11);
    expect(getStoredSessionCount()).toBe(11);
  });
});

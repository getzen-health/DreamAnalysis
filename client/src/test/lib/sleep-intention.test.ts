import { describe, it, expect, beforeEach, vi } from "vitest";
import {
  saveIntention,
  getTodayIntention,
  getIntentionForDate,
  getIntentionHistory,
  persistAlignmentScore,
  scoreAlignment,
  alignmentLabel,
  alignmentDescription,
  computeIntentionStats,
  ALIGNMENT_COLOR,
  ALIGNMENT_BG,
  type IntentionEntry,
  type AlignmentLabel,
} from "@/lib/sleep-intention";

// ── localStorage mock ─────────────────────────────────────────────────────────

const store: Record<string, string> = {};
const localStorageMock = {
  getItem: (k: string) => store[k] ?? null,
  setItem: (k: string, v: string) => { store[k] = v; },
  removeItem: (k: string) => { delete store[k]; },
  clear: () => { for (const k in store) delete store[k]; },
  get length() { return Object.keys(store).length; },
  key: (i: number) => Object.keys(store)[i] ?? null,
};

beforeEach(() => {
  localStorageMock.clear();
  vi.stubGlobal("localStorage", localStorageMock);
});

// ── saveIntention ─────────────────────────────────────────────────────────────

describe("saveIntention", () => {
  it("returns an IntentionEntry", () => {
    const entry = saveIntention("I want to fly");
    expect(entry.text).toBe("I want to fly");
    expect(entry.alignmentScore).toBeNull();
    expect(entry.date).toMatch(/^\d{4}-\d{2}-\d{2}$/);
  });

  it("trims whitespace", () => {
    expect(saveIntention("  fly  ").text).toBe("fly");
  });

  it("caps text at 500 characters", () => {
    const long = "a".repeat(600);
    expect(saveIntention(long).text.length).toBe(500);
  });

  it("persists to localStorage", () => {
    saveIntention("ocean dreaming");
    const retrieved = getTodayIntention();
    expect(retrieved).not.toBeNull();
    expect(retrieved!.text).toBe("ocean dreaming");
  });

  it("overwrites an existing today entry", () => {
    saveIntention("first");
    saveIntention("second");
    expect(getTodayIntention()!.text).toBe("second");
  });
});

// ── getTodayIntention ─────────────────────────────────────────────────────────

describe("getTodayIntention", () => {
  it("returns null when nothing saved", () => {
    expect(getTodayIntention()).toBeNull();
  });

  it("returns today's entry after save", () => {
    saveIntention("clarity");
    expect(getTodayIntention()!.text).toBe("clarity");
  });
});

// ── getIntentionForDate ───────────────────────────────────────────────────────

describe("getIntentionForDate", () => {
  it("returns null for a date with no entry", () => {
    expect(getIntentionForDate("2020-01-01")).toBeNull();
  });

  it("returns entry for today's date", () => {
    const today = new Date().toISOString().slice(0, 10);
    saveIntention("test");
    expect(getIntentionForDate(today)).not.toBeNull();
  });
});

// ── getIntentionHistory ───────────────────────────────────────────────────────

describe("getIntentionHistory", () => {
  it("returns empty array when nothing saved", () => {
    expect(getIntentionHistory()).toHaveLength(0);
  });

  it("returns today's intention", () => {
    saveIntention("fly high");
    expect(getIntentionHistory()).toHaveLength(1);
    expect(getIntentionHistory()[0].text).toBe("fly high");
  });
});

// ── persistAlignmentScore ─────────────────────────────────────────────────────

describe("persistAlignmentScore", () => {
  it("returns null when no intention exists for date", () => {
    expect(persistAlignmentScore("2020-01-01", 0.7, "d1")).toBeNull();
  });

  it("updates alignment score on an existing entry", () => {
    const today = new Date().toISOString().slice(0, 10);
    saveIntention("ocean");
    const updated = persistAlignmentScore(today, 0.75, "dream-123");
    expect(updated).not.toBeNull();
    expect(updated!.alignmentScore).toBeCloseTo(0.75);
    expect(updated!.alignedDreamId).toBe("dream-123");
  });

  it("clamps score to [0, 1]", () => {
    const today = new Date().toISOString().slice(0, 10);
    saveIntention("test");
    expect(persistAlignmentScore(today, 1.5, "d")!.alignmentScore).toBe(1);
    expect(persistAlignmentScore(today, -0.5, "d")!.alignmentScore).toBe(0);
  });

  it("persisted score is readable via getTodayIntention", () => {
    const today = new Date().toISOString().slice(0, 10);
    saveIntention("fly");
    persistAlignmentScore(today, 0.6, "d1");
    expect(getTodayIntention()!.alignmentScore).toBeCloseTo(0.6);
  });
});

// ── scoreAlignment ────────────────────────────────────────────────────────────

describe("scoreAlignment", () => {
  it("returns 0 for empty intention", () => {
    expect(scoreAlignment("", "I dreamed of the ocean", null)).toBe(0);
  });

  it("returns 0 for empty dream text", () => {
    expect(scoreAlignment("fly over ocean", "", null)).toBe(0);
  });

  it("returns 1 for perfect word match", () => {
    const score = scoreAlignment("resolve stress", "I tried to resolve stress today", null);
    expect(score).toBe(1);
  });

  it("returns 0 when no intention words appear in dream", () => {
    const score = scoreAlignment("unicorn magic castle", "I was running through trees", null);
    expect(score).toBe(0);
  });

  it("partial match returns value between 0 and 1", () => {
    const score = scoreAlignment("ocean flying freedom", "I was near the ocean", null);
    expect(score).toBeGreaterThan(0);
    expect(score).toBeLessThan(1);
  });

  it("includes themes in corpus", () => {
    const score = scoreAlignment(
      "transformation",
      "nothing useful here",
      ["self-exploration", "transformation"]
    );
    expect(score).toBeGreaterThan(0);
  });

  it("is case-insensitive", () => {
    const a = scoreAlignment("OCEAN", "ocean was calm", null);
    const b = scoreAlignment("ocean", "OCEAN WAS CALM", null);
    expect(a).toBeCloseTo(b);
  });

  it("ignores stop words", () => {
    // "and", "the", "for" etc. should not inflate score
    const score = scoreAlignment("the and for", "I was the one", null);
    // All tokens are stop words, so intentTokens will be empty → score 0
    expect(score).toBe(0);
  });

  it("stem partial match: intention 'resolv' matches 'resolving' in dream", () => {
    const score = scoreAlignment("resolving", "the dream was resolving all tension", null);
    expect(score).toBe(1);
  });

  it("score is always in [0, 1]", () => {
    const pairs = [
      ["fly ocean stars", "the sky was full of stars"],
      ["conflict anger fear", "peaceful walk in the garden"],
      ["bridge water flowing calm", "water bridge flowing over the river calm"],
    ] as [string, string][];
    for (const [intent, dream] of pairs) {
      const s = scoreAlignment(intent, dream, null);
      expect(s).toBeGreaterThanOrEqual(0);
      expect(s).toBeLessThanOrEqual(1);
    }
  });
});

// ── alignmentLabel ────────────────────────────────────────────────────────────

describe("alignmentLabel", () => {
  it("strong for score >= 0.5", () => {
    expect(alignmentLabel(0.5)).toBe("strong");
    expect(alignmentLabel(0.9)).toBe("strong");
  });

  it("partial for 0.25 <= score < 0.5", () => {
    expect(alignmentLabel(0.25)).toBe("partial");
    expect(alignmentLabel(0.4)).toBe("partial");
  });

  it("weak for 0 < score < 0.25", () => {
    expect(alignmentLabel(0.1)).toBe("weak");
  });

  it("none for score = 0", () => {
    expect(alignmentLabel(0)).toBe("none");
  });

  it("ALIGNMENT_COLOR and ALIGNMENT_BG have entries for all labels", () => {
    const labels: AlignmentLabel[] = ["strong", "partial", "weak", "none"];
    for (const l of labels) {
      expect(typeof ALIGNMENT_COLOR[l]).toBe("string");
      expect(typeof ALIGNMENT_BG[l]).toBe("string");
    }
  });

  it("alignmentDescription returns a non-empty string for each label", () => {
    const labels: AlignmentLabel[] = ["strong", "partial", "weak", "none"];
    for (const l of labels) {
      expect(alignmentDescription(l).length).toBeGreaterThan(0);
    }
  });
});

// ── computeIntentionStats ─────────────────────────────────────────────────────

describe("computeIntentionStats", () => {
  const makeEntry = (score: number | null): IntentionEntry => ({
    date: "2026-03-28",
    text: "test",
    savedAt: new Date().toISOString(),
    alignmentScore: score,
    alignedDreamId: score !== null ? "d1" : null,
  });

  it("returns zero stats for empty history", () => {
    const s = computeIntentionStats([]);
    expect(s.totalSet).toBe(0);
    expect(s.totalScored).toBe(0);
    expect(s.avgAlignment).toBeNull();
    expect(s.strongRate).toBe(0);
  });

  it("totalSet counts all entries, totalScored counts non-null", () => {
    const history = [makeEntry(0.8), makeEntry(null), makeEntry(0.3)];
    const s = computeIntentionStats(history);
    expect(s.totalSet).toBe(3);
    expect(s.totalScored).toBe(2);
  });

  it("avgAlignment is mean of scored entries", () => {
    const history = [makeEntry(0.6), makeEntry(0.4)];
    const s = computeIntentionStats(history);
    expect(s.avgAlignment).toBeCloseTo(0.5);
  });

  it("strongRate is fraction of scored entries with score >= 0.5", () => {
    const history = [makeEntry(0.8), makeEntry(0.2), makeEntry(0.6)];
    const s = computeIntentionStats(history);
    expect(s.strongRate).toBeCloseTo(2 / 3);
  });
});

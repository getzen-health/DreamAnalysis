import { describe, it, expect } from "vitest";
import {
  analyzeDreamPatterns,
  type DreamEntry,
} from "@/lib/dream-theme-tracker";

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Create a dream entry with the given params and a timestamp N days ago. */
function makeDream(
  overrides: Partial<DreamEntry> & { daysAgo?: number } = {},
): DreamEntry {
  const { daysAgo = 0, ...rest } = overrides;
  const ts = new Date(Date.now() - daysAgo * 86_400_000).toISOString();
  return {
    dreamText: "a dream",
    emotions: [],
    symbols: [],
    timestamp: ts,
    ...rest,
  };
}

// ── analyzeDreamPatterns ─────────────────────────────────────────────────────

describe("analyzeDreamPatterns", () => {
  it("returns empty summary for no dreams", () => {
    const result = analyzeDreamPatterns([], 30);
    expect(result.totalDreams).toBe(0);
    expect(result.topThemes).toHaveLength(0);
    expect(result.emotionDistribution).toEqual({});
    expect(result.lucidDreamCount).toBe(0);
    expect(result.periodDays).toBe(30);
  });

  it("returns empty summary when all dreams are outside the period", () => {
    const dreams = [makeDream({ symbols: ["water"], daysAgo: 100 })];
    const result = analyzeDreamPatterns(dreams, 30);
    expect(result.totalDreams).toBe(0);
    expect(result.topThemes).toHaveLength(0);
  });

  it("counts symbols correctly and returns top themes", () => {
    const dreams = [
      makeDream({ symbols: ["water", "flying"], daysAgo: 1 }),
      makeDream({ symbols: ["water", "forest"], daysAgo: 2 }),
      makeDream({ symbols: ["water", "flying", "snake"], daysAgo: 3 }),
    ];
    const result = analyzeDreamPatterns(dreams, 30);

    expect(result.totalDreams).toBe(3);
    expect(result.topThemes[0].theme).toBe("water");
    expect(result.topThemes[0].count).toBe(3);
    expect(result.topThemes[1].theme).toBe("flying");
    expect(result.topThemes[1].count).toBe(2);
  });

  it("limits to top 5 themes", () => {
    const dreams = [
      makeDream({
        symbols: ["a", "b", "c", "d", "e", "f", "g"],
        daysAgo: 1,
      }),
    ];
    const result = analyzeDreamPatterns(dreams, 30);
    expect(result.topThemes.length).toBeLessThanOrEqual(5);
  });

  it("tracks firstSeen and lastSeen correctly", () => {
    const dreams = [
      makeDream({ symbols: ["water"], daysAgo: 10 }),
      makeDream({ symbols: ["water"], daysAgo: 2 }),
      makeDream({ symbols: ["water"], daysAgo: 5 }),
    ];
    const result = analyzeDreamPatterns(dreams, 30);
    const waterTheme = result.topThemes.find((t) => t.theme === "water");

    expect(waterTheme).toBeDefined();
    // firstSeen should be the oldest (10 days ago)
    expect(new Date(waterTheme!.firstSeen).getTime()).toBeLessThan(
      new Date(waterTheme!.lastSeen).getTime(),
    );
  });

  it("computes emotion distribution across all dreams", () => {
    const dreams = [
      makeDream({ emotions: ["anxiety", "fear"], daysAgo: 1 }),
      makeDream({ emotions: ["anxiety", "joy"], daysAgo: 2 }),
      makeDream({ emotions: ["joy"], daysAgo: 3 }),
    ];
    const result = analyzeDreamPatterns(dreams, 30);

    expect(result.emotionDistribution["anxiety"]).toBe(2);
    expect(result.emotionDistribution["joy"]).toBe(2);
    expect(result.emotionDistribution["fear"]).toBe(1);
  });

  it("counts lucid dreams (lucidityScore > 0.5)", () => {
    const dreams = [
      makeDream({ lucidityScore: 0.8, daysAgo: 1 }),
      makeDream({ lucidityScore: 0.3, daysAgo: 2 }),
      makeDream({ lucidityScore: 0.6, daysAgo: 3 }),
      makeDream({ daysAgo: 4 }), // no score
    ];
    const result = analyzeDreamPatterns(dreams, 30);
    expect(result.lucidDreamCount).toBe(2);
  });

  it("normalizes symbols to lowercase", () => {
    const dreams = [
      makeDream({ symbols: ["Water"], daysAgo: 1 }),
      makeDream({ symbols: ["water"], daysAgo: 2 }),
      makeDream({ symbols: ["WATER"], daysAgo: 3 }),
    ];
    const result = analyzeDreamPatterns(dreams, 30);
    expect(result.topThemes[0].theme).toBe("water");
    expect(result.topThemes[0].count).toBe(3);
  });

  it("normalizes emotions to lowercase", () => {
    const dreams = [
      makeDream({ emotions: ["Anxiety", "FEAR"], daysAgo: 1 }),
      makeDream({ emotions: ["anxiety"], daysAgo: 2 }),
    ];
    const result = analyzeDreamPatterns(dreams, 30);
    expect(result.emotionDistribution["anxiety"]).toBe(2);
    expect(result.emotionDistribution["fear"]).toBe(1);
  });
});

// ── Trend computation ────────────────────────────────────────────────────────

describe("trend computation", () => {
  it("marks increasing trend when more recent occurrences", () => {
    // All occurrences in the recent half of a 30-day window
    const dreams = [
      makeDream({ symbols: ["water"], daysAgo: 1 }),
      makeDream({ symbols: ["water"], daysAgo: 3 }),
      makeDream({ symbols: ["water"], daysAgo: 5 }),
    ];
    const result = analyzeDreamPatterns(dreams, 30);
    const waterTheme = result.topThemes.find((t) => t.theme === "water");
    expect(waterTheme?.trend).toBe("increasing");
  });

  it("marks decreasing trend when more older occurrences", () => {
    // All occurrences in the older half of a 30-day window
    const dreams = [
      makeDream({ symbols: ["water"], daysAgo: 20 }),
      makeDream({ symbols: ["water"], daysAgo: 22 }),
      makeDream({ symbols: ["water"], daysAgo: 25 }),
    ];
    const result = analyzeDreamPatterns(dreams, 30);
    const waterTheme = result.topThemes.find((t) => t.theme === "water");
    expect(waterTheme?.trend).toBe("decreasing");
  });

  it("marks stable trend for single occurrence", () => {
    const dreams = [makeDream({ symbols: ["water"], daysAgo: 5 })];
    const result = analyzeDreamPatterns(dreams, 30);
    const waterTheme = result.topThemes.find((t) => t.theme === "water");
    expect(waterTheme?.trend).toBe("stable");
  });
});

// ── Associated emotions ──────────────────────────────────────────────────────

describe("associated emotions", () => {
  it("returns top emotions co-occurring with a theme", () => {
    const dreams = [
      makeDream({ symbols: ["water"], emotions: ["anxiety", "fear"], daysAgo: 1 }),
      makeDream({ symbols: ["water"], emotions: ["anxiety", "joy"], daysAgo: 2 }),
      makeDream({ symbols: ["water"], emotions: ["calm"], daysAgo: 3 }),
    ];
    const result = analyzeDreamPatterns(dreams, 30);
    const waterTheme = result.topThemes.find((t) => t.theme === "water");

    expect(waterTheme?.associatedEmotions).toBeDefined();
    // anxiety appears twice, should be first
    expect(waterTheme!.associatedEmotions[0]).toBe("anxiety");
    // max 3 associated emotions
    expect(waterTheme!.associatedEmotions.length).toBeLessThanOrEqual(3);
  });

  it("returns empty associated emotions when theme has no co-occurring emotions", () => {
    const dreams = [
      makeDream({ symbols: ["water"], emotions: [], daysAgo: 1 }),
    ];
    const result = analyzeDreamPatterns(dreams, 30);
    const waterTheme = result.topThemes.find((t) => t.theme === "water");
    expect(waterTheme?.associatedEmotions).toEqual([]);
  });
});

// ── Period filtering ─────────────────────────────────────────────────────────

describe("period filtering", () => {
  it("7-day period excludes dreams older than 7 days", () => {
    const dreams = [
      makeDream({ symbols: ["water"], daysAgo: 3 }),
      makeDream({ symbols: ["water"], daysAgo: 10 }),
    ];
    const result = analyzeDreamPatterns(dreams, 7);
    expect(result.totalDreams).toBe(1);
    expect(result.topThemes[0].count).toBe(1);
  });

  it("90-day period includes dreams up to 90 days old", () => {
    const dreams = [
      makeDream({ symbols: ["water"], daysAgo: 5 }),
      makeDream({ symbols: ["water"], daysAgo: 45 }),
      makeDream({ symbols: ["water"], daysAgo: 85 }),
      makeDream({ symbols: ["water"], daysAgo: 100 }),
    ];
    const result = analyzeDreamPatterns(dreams, 90);
    expect(result.totalDreams).toBe(3);
    expect(result.topThemes[0].count).toBe(3);
  });
});

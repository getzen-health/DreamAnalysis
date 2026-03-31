import { describe, it, expect } from "vitest";
import {
  isNightmare,
  isLucid,
  hasInsight,
  entryQualityBand,
  bandColorClass,
  formatEntryDate,
  formatEntryDatetime,
  truncateDreamText,
  topThemes,
  applyFilter,
  searchDreams,
  sortNewest,
  computeStats,
  type DreamEntry,
  type DreamFilter,
} from "@/lib/dream-history";

// ── fixtures ─────────────────────────────────────────────────────────────────

const makeEntry = (overrides: Partial<DreamEntry> = {}): DreamEntry => ({
  id: "1",
  userId: "u1",
  dreamText: "I was flying over a vast ocean.",
  symbols: ["water", "sky"],
  aiAnalysis: null,
  lucidityScore: 50,
  sleepQuality: 70,
  tags: null,
  sleepDuration: 7.5,
  themes: ["self-exploration", "transition"],
  emotionalArc: "wonder then peace",
  keyInsight: "Freedom is within reach.",
  threatSimulationIndex: 0.2,
  timestamp: "2026-03-28T08:00:00Z",
  ...overrides,
});

const nightmare = makeEntry({
  id: "2",
  dreamText: "Something chased me through dark corridors.",
  threatSimulationIndex: 0.85,
  keyInsight: null,
  lucidityScore: 10,
  timestamp: "2026-03-27T06:00:00Z",
});

const lucidDream = makeEntry({
  id: "3",
  lucidityScore: 80,
  timestamp: "2026-03-26T07:00:00Z",
});

// ── isNightmare ──────────────────────────────────────────────────────────────

describe("isNightmare", () => {
  it("returns true when tsi > 0.5", () => {
    expect(isNightmare(nightmare)).toBe(true);
  });

  it("returns false when tsi = 0.5", () => {
    expect(isNightmare(makeEntry({ threatSimulationIndex: 0.5 }))).toBe(false);
  });

  it("returns false when tsi < 0.5", () => {
    expect(isNightmare(makeEntry({ threatSimulationIndex: 0.2 }))).toBe(false);
  });

  it("returns false when tsi is null", () => {
    expect(isNightmare(makeEntry({ threatSimulationIndex: null }))).toBe(false);
  });
});

// ── isLucid ──────────────────────────────────────────────────────────────────

describe("isLucid", () => {
  it("returns true when lucidityScore >= 70", () => {
    expect(isLucid(lucidDream)).toBe(true);
  });

  it("boundary: exactly 70 is lucid", () => {
    expect(isLucid(makeEntry({ lucidityScore: 70 }))).toBe(true);
  });

  it("returns false below 70", () => {
    expect(isLucid(makeEntry({ lucidityScore: 69 }))).toBe(false);
  });

  it("returns false when null", () => {
    expect(isLucid(makeEntry({ lucidityScore: null }))).toBe(false);
  });
});

// ── hasInsight ───────────────────────────────────────────────────────────────

describe("hasInsight", () => {
  it("true when keyInsight is non-empty", () => {
    expect(hasInsight(makeEntry())).toBe(true);
  });

  it("false when keyInsight is null", () => {
    expect(hasInsight(makeEntry({ keyInsight: null }))).toBe(false);
  });

  it("false when keyInsight is whitespace-only", () => {
    expect(hasInsight(makeEntry({ keyInsight: "   " }))).toBe(false);
  });
});

// ── entryQualityBand ─────────────────────────────────────────────────────────

describe("entryQualityBand", () => {
  it("returns null when no quality signals", () => {
    const entry = makeEntry({ sleepQuality: null, threatSimulationIndex: null, lucidityScore: 0 });
    expect(entryQualityBand(entry)).toBeNull();
  });

  it("nightmare entry with no dream text gets poor band (tsi penalty, no recall bonus)", () => {
    // base 50 − tsi(0.85)×25 = 50−21.25 = 28.75 → poor (< 35)
    const band = entryQualityBand(makeEntry({ dreamText: "", threatSimulationIndex: 0.85, sleepQuality: null, lucidityScore: 0 }));
    expect(band).toBe("poor");
  });

  it("nightmare entry with dream text gets fair band (recall bonus lifts it above 35)", () => {
    // base 50 − 21.25 + recall+10 = 38.75 → fair (35-55)
    const band = entryQualityBand(makeEntry({ threatSimulationIndex: 0.85, sleepQuality: null, lucidityScore: 0 }));
    expect(band).toBe("fair");
  });

  it("high sleep quality and low tsi gets good/great band", () => {
    const band = entryQualityBand(makeEntry({ sleepQuality: 80, threatSimulationIndex: 0.1, lucidityScore: 80 }));
    expect(["good", "great"]).toContain(band);
  });

  it("bandColorClass returns a string for all bands", () => {
    for (const band of ["poor", "fair", "good", "great"] as const) {
      expect(typeof bandColorClass(band)).toBe("string");
      expect(bandColorClass(band).length).toBeGreaterThan(0);
    }
  });
});

// ── formatEntryDate / formatEntryDatetime ────────────────────────────────────

describe("formatEntryDate", () => {
  it("returns a short date string", () => {
    const s = formatEntryDate("2026-03-28T08:00:00Z");
    expect(s).toContain("28");
    expect(s).toContain("Mar");
  });

  it("does not crash on invalid date", () => {
    expect(() => formatEntryDate("not-a-date")).not.toThrow();
  });
});

describe("formatEntryDatetime", () => {
  it("includes time (AM/PM)", () => {
    const s = formatEntryDatetime("2026-03-28T08:00:00Z");
    expect(s.toLowerCase()).toMatch(/am|pm/);
  });
});

// ── truncateDreamText ────────────────────────────────────────────────────────

describe("truncateDreamText", () => {
  it("returns text unchanged when shorter than maxLen", () => {
    expect(truncateDreamText("short text", 200)).toBe("short text");
  });

  it("appends ellipsis when truncated", () => {
    const long = "word ".repeat(30);
    const result = truncateDreamText(long, 50);
    expect(result.endsWith("…")).toBe(true);
    expect(result.length).toBeLessThanOrEqual(51); // maxLen + ellipsis char
  });

  it("breaks on word boundary", () => {
    const result = truncateDreamText("one two three four five", 12);
    expect(result).not.toMatch(/ …/); // no trailing space before ellipsis
    expect(result.endsWith("…")).toBe(true);
  });
});

// ── topThemes ────────────────────────────────────────────────────────────────

describe("topThemes", () => {
  it("returns up to 3 by default", () => {
    const entry = makeEntry({ themes: ["a", "b", "c", "d", "e"] });
    expect(topThemes(entry)).toHaveLength(3);
  });

  it("respects custom n", () => {
    const entry = makeEntry({ themes: ["a", "b", "c", "d"] });
    expect(topThemes(entry, 2)).toHaveLength(2);
  });

  it("returns empty array for null themes", () => {
    expect(topThemes(makeEntry({ themes: null }))).toHaveLength(0);
  });
});

// ── applyFilter ──────────────────────────────────────────────────────────────

describe("applyFilter", () => {
  const entries = [makeEntry(), nightmare, lucidDream];

  it("all returns all entries", () => {
    expect(applyFilter(entries, "all")).toHaveLength(3);
  });

  it("nightmares returns only nightmares", () => {
    const result = applyFilter(entries, "nightmares");
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("2");
  });

  it("lucid returns only lucid dreams", () => {
    const result = applyFilter(entries, "lucid");
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("3");
  });

  it("with-insight returns entries having keyInsight", () => {
    const result = applyFilter(entries, "with-insight");
    result.forEach((e) => expect(e.keyInsight).toBeTruthy());
  });
});

// ── searchDreams ─────────────────────────────────────────────────────────────

describe("searchDreams", () => {
  const entries = [
    makeEntry({ dreamText: "Flying over ocean", themes: ["freedom"] }),
    makeEntry({ dreamText: "Dark corridors", keyInsight: "Fear of change.", themes: ["threat-simulation"] }),
  ];

  it("returns all entries for empty query", () => {
    expect(searchDreams(entries, "")).toHaveLength(2);
  });

  it("matches on dreamText (case-insensitive)", () => {
    expect(searchDreams(entries, "OCEAN")).toHaveLength(1);
  });

  it("matches on themes", () => {
    expect(searchDreams(entries, "freedom")).toHaveLength(1);
  });

  it("matches on keyInsight", () => {
    expect(searchDreams(entries, "fear of change")).toHaveLength(1);
  });

  it("returns empty array when no match", () => {
    expect(searchDreams(entries, "xyzzy")).toHaveLength(0);
  });
});

// ── sortNewest ───────────────────────────────────────────────────────────────

describe("sortNewest", () => {
  it("sorts entries newest-first", () => {
    const entries = [nightmare, makeEntry(), lucidDream];
    const sorted = sortNewest(entries);
    expect(sorted[0].timestamp >= sorted[1].timestamp).toBe(true);
    expect(sorted[1].timestamp >= sorted[2].timestamp).toBe(true);
  });

  it("does not mutate the original array", () => {
    const entries = [nightmare, makeEntry()];
    const original = [...entries];
    sortNewest(entries);
    expect(entries[0].id).toBe(original[0].id);
  });
});

// ── computeStats ─────────────────────────────────────────────────────────────

describe("computeStats", () => {
  it("returns zeros for empty list", () => {
    const s = computeStats([]);
    expect(s.total).toBe(0);
    expect(s.nightmares).toBe(0);
    expect(s.lucid).toBe(0);
    expect(s.withInsight).toBe(0);
  });

  it("counts each category independently", () => {
    const entries = [makeEntry(), nightmare, lucidDream];
    const s = computeStats(entries);
    expect(s.total).toBe(3);
    expect(s.nightmares).toBe(1);
    expect(s.lucid).toBe(1);
    // makeEntry and lucidDream have keyInsight; nightmare has null
    expect(s.withInsight).toBe(2);
  });
});

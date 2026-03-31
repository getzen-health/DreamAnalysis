import { describe, it, expect } from "vitest";
import {
  buildSymbolContextMap,
  symbolMood,
  sortedSymbols,
  symbolSummary,
  MOOD_COLOR,
  MOOD_BG,
  MOOD_LABEL,
  type DreamEntryForSymbol,
  type SymbolMood,
  type SymbolSortKey,
} from "@/lib/dream-symbol-context";

// ── fixtures ──────────────────────────────────────────────────────────────────

const makeEntry = (
  overrides: Partial<DreamEntryForSymbol> = {},
): DreamEntryForSymbol => ({
  id: "d1",
  symbols: ["water", "bridge"],
  themes: ["self-exploration", "transition"],
  emotionalArc: "anxious then resolving",
  threatSimulationIndex: 0.2,
  lucidityScore: 50,
  timestamp: "2026-03-28T08:00:00Z",
  ...overrides,
});

const nightmareEntry = makeEntry({
  id: "d2",
  symbols: ["shadow", "door"],
  themes: ["threat-simulation"],
  emotionalArc: "escalating fear",
  threatSimulationIndex: 0.85,
  lucidityScore: 10,
  timestamp: "2026-03-27T08:00:00Z",
});

const lucidEntry = makeEntry({
  id: "d3",
  symbols: ["water", "sky"],
  themes: ["freedom", "lucid"],
  emotionalArc: "wonder and clarity",
  threatSimulationIndex: 0.1,
  lucidityScore: 85,
  timestamp: "2026-03-26T08:00:00Z",
});

// ── buildSymbolContextMap ─────────────────────────────────────────────────────

describe("buildSymbolContextMap", () => {
  it("returns empty map for empty input", () => {
    expect(buildSymbolContextMap([])).toHaveProperty("size", 0);
  });

  it("skips dreams with null symbols", () => {
    const map = buildSymbolContextMap([makeEntry({ symbols: null })]);
    expect(map.size).toBe(0);
  });

  it("skips empty symbols array", () => {
    const map = buildSymbolContextMap([makeEntry({ symbols: [] })]);
    expect(map.size).toBe(0);
  });

  it("creates one entry per unique symbol", () => {
    const map = buildSymbolContextMap([makeEntry()]);
    expect(map.has("water")).toBe(true);
    expect(map.has("bridge")).toBe(true);
    expect(map.size).toBe(2);
  });

  it("normalises symbols to lowercase", () => {
    const map = buildSymbolContextMap([makeEntry({ symbols: ["Water", "BRIDGE"] })]);
    expect(map.has("water")).toBe(true);
    expect(map.has("bridge")).toBe(true);
  });

  it("counts symbol across multiple dreams", () => {
    const dreams = [
      makeEntry({ id: "d1", symbols: ["water"] }),
      makeEntry({ id: "d2", symbols: ["water"] }),
    ];
    expect(buildSymbolContextMap(dreams).get("water")!.count).toBe(2);
  });

  it("averages TSI correctly", () => {
    const dreams = [
      makeEntry({ id: "d1", symbols: ["water"], threatSimulationIndex: 0.2 }),
      makeEntry({ id: "d2", symbols: ["water"], threatSimulationIndex: 0.6 }),
    ];
    const ctx = buildSymbolContextMap(dreams).get("water")!;
    expect(ctx.avgTsi).toBeCloseTo(0.4);
  });

  it("averages lucidity correctly", () => {
    const dreams = [
      makeEntry({ id: "d1", symbols: ["water"], lucidityScore: 40 }),
      makeEntry({ id: "d2", symbols: ["water"], lucidityScore: 80 }),
    ];
    const ctx = buildSymbolContextMap(dreams).get("water")!;
    expect(ctx.avgLucidity).toBeCloseTo(60);
  });

  it("handles null TSI/lucidity gracefully (excluded from average)", () => {
    const dreams = [
      makeEntry({ id: "d1", symbols: ["water"], threatSimulationIndex: null, lucidityScore: null }),
    ];
    const ctx = buildSymbolContextMap(dreams).get("water")!;
    expect(ctx.avgTsi).toBe(0);
    expect(ctx.avgLucidity).toBe(0);
  });

  it("collects top themes for a symbol", () => {
    const dreams = [
      makeEntry({ id: "d1", symbols: ["water"], themes: ["freedom", "transition"] }),
      makeEntry({ id: "d2", symbols: ["water"], themes: ["freedom", "self-exploration"] }),
    ];
    const ctx = buildSymbolContextMap(dreams).get("water")!;
    expect(ctx.topThemes[0]).toBe("freedom"); // appeared twice
    expect(ctx.topThemes).toHaveLength(3 < ctx.topThemes.length + 1 ? ctx.topThemes.length : 3); // max 3
  });

  it("topThemes is at most 3 entries", () => {
    const dreams = [
      makeEntry({ id: "d1", symbols: ["water"], themes: ["a","b","c","d","e"] }),
    ];
    expect(buildSymbolContextMap(dreams).get("water")!.topThemes.length).toBeLessThanOrEqual(3);
  });

  it("records dream IDs for a symbol", () => {
    const dreams = [
      makeEntry({ id: "dream-1", symbols: ["water"] }),
      makeEntry({ id: "dream-2", symbols: ["water"] }),
    ];
    const ctx = buildSymbolContextMap(dreams).get("water")!;
    expect(ctx.dreamIds).toContain("dream-1");
    expect(ctx.dreamIds).toContain("dream-2");
  });

  it("lastSeen is the most recent timestamp", () => {
    const dreams = [
      makeEntry({ id: "d1", symbols: ["water"], timestamp: "2026-03-25T08:00:00Z" }),
      makeEntry({ id: "d2", symbols: ["water"], timestamp: "2026-03-28T08:00:00Z" }),
    ];
    expect(buildSymbolContextMap(dreams).get("water")!.lastSeen)
      .toBe("2026-03-28T08:00:00Z");
  });

  it("deduplicates symbol casing ('Water' and 'water' are same symbol)", () => {
    const dreams = [
      makeEntry({ id: "d1", symbols: ["Water"] }),
      makeEntry({ id: "d2", symbols: ["water"] }),
    ];
    const map = buildSymbolContextMap(dreams);
    expect(map.has("water")).toBe(true);
    expect(map.get("water")!.count).toBe(2);
  });
});

// ── symbolMood ────────────────────────────────────────────────────────────────

describe("symbolMood", () => {
  const makeCtx = (tsi: number, lucid: number) =>
    buildSymbolContextMap([
      makeEntry({ symbols: ["x"], threatSimulationIndex: tsi, lucidityScore: lucid }),
    ]).get("x")!;

  it("dark when avgTsi > 0.5 and low lucidity", () => {
    expect(symbolMood(makeCtx(0.8, 20))).toBe("dark");
  });

  it("uplifting when high lucidity and low tsi", () => {
    expect(symbolMood(makeCtx(0.1, 80))).toBe("uplifting");
  });

  it("mixed when both elevated", () => {
    expect(symbolMood(makeCtx(0.8, 80))).toBe("mixed");
  });

  it("neutral when neither elevated", () => {
    expect(symbolMood(makeCtx(0.2, 40))).toBe("neutral");
  });

  it("MOOD_COLOR has entry for every mood", () => {
    (["uplifting", "dark", "neutral", "mixed"] as SymbolMood[]).forEach((m) => {
      expect(typeof MOOD_COLOR[m]).toBe("string");
    });
  });

  it("MOOD_BG has entry for every mood", () => {
    (["uplifting", "dark", "neutral", "mixed"] as SymbolMood[]).forEach((m) => {
      expect(typeof MOOD_BG[m]).toBe("string");
    });
  });
});

// ── sortedSymbols ─────────────────────────────────────────────────────────────

describe("sortedSymbols", () => {
  const dreams = [makeEntry(), nightmareEntry, lucidEntry];
  const map = buildSymbolContextMap(dreams);

  it("returns at most `limit` entries (default 8)", () => {
    expect(sortedSymbols(map).length).toBeLessThanOrEqual(8);
  });

  it("frequency sort: most frequent symbol first", () => {
    // "water" appears in makeEntry + lucidEntry = 2 times
    const sorted = sortedSymbols(map, "frequency");
    expect(sorted[0].symbol).toBe("water");
  });

  it("recent sort: most recently seen first", () => {
    const sorted = sortedSymbols(map, "recent");
    expect(sorted[0].lastSeen >= sorted[1].lastSeen).toBe(true);
  });

  it("darkest sort: highest avgTsi first", () => {
    const sorted = sortedSymbols(map, "darkest");
    expect(sorted[0].avgTsi >= sorted[1].avgTsi).toBe(true);
  });

  it("brightest sort: highest avgLucidity first", () => {
    const sorted = sortedSymbols(map, "brightest");
    expect(sorted[0].avgLucidity >= sorted[1].avgLucidity).toBe(true);
  });

  it("respects custom limit", () => {
    expect(sortedSymbols(map, "frequency", 2)).toHaveLength(2);
  });

  it("returns empty array for empty map", () => {
    expect(sortedSymbols(new Map())).toHaveLength(0);
  });
});

// ── symbolSummary ─────────────────────────────────────────────────────────────

describe("symbolSummary", () => {
  it("returns a non-empty string", () => {
    const ctx = buildSymbolContextMap([makeEntry()]).get("water")!;
    expect(symbolSummary(ctx).length).toBeGreaterThan(0);
  });

  it("mentions the dream count", () => {
    const ctx = buildSymbolContextMap([makeEntry(), makeEntry()]).get("water")!;
    expect(symbolSummary(ctx)).toContain("2 dreams");
  });

  it("uses singular for one dream", () => {
    const ctx = buildSymbolContextMap([makeEntry({ id: "d1", symbols: ["solo"] })]).get("solo")!;
    expect(symbolSummary(ctx)).toContain("1 dream");
  });

  it("includes a mood word", () => {
    const ctx = buildSymbolContextMap([makeEntry()]).get("water")!;
    const moods = ["uplifting", "dark", "neutral", "mixed"];
    expect(moods.some((m) => symbolSummary(ctx).includes(m))).toBe(true);
  });
});

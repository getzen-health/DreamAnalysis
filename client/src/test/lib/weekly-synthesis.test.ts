import { describe, it, expect } from "vitest";
import {
  formatDreamRow,
  buildSynthesisContext,
  extractWeekTopThemes,
  countWeekNightmares,
  extractWeekSymbols,
  buildWeeklySynthesisPrompt,
  type SynthesisDream,
} from "@/lib/weekly-synthesis";

// ── fixtures ──────────────────────────────────────────────────────────────────

const makeDream = (overrides: Partial<SynthesisDream> = {}): SynthesisDream => ({
  date: "2026-03-28T08:00:00Z",
  themes: ["self-exploration", "transition"],
  emotionalArc: "anxious then resolving",
  keyInsight: "Fear of change is natural.",
  threatSimulationIndex: 0.2,
  symbols: ["water", "bridge"],
  ...overrides,
});

const nightmare = makeDream({
  date: "2026-03-27T08:00:00Z",
  themes: ["threat-simulation"],
  emotionalArc: "escalating fear",
  keyInsight: "Running from something unresolved.",
  threatSimulationIndex: 0.85,
  symbols: ["shadow", "door"],
});

// ── formatDreamRow ────────────────────────────────────────────────────────────

describe("formatDreamRow", () => {
  it("includes the date and index", () => {
    const row = formatDreamRow(makeDream(), 0);
    expect(row).toContain("Dream 1");
    expect(row).toContain("2026-03-28");
  });

  it("includes themes when present", () => {
    const row = formatDreamRow(makeDream(), 0);
    expect(row).toContain("self-exploration");
    expect(row).toContain("transition");
  });

  it("includes emotional arc when present", () => {
    const row = formatDreamRow(makeDream(), 0);
    expect(row).toContain("anxious then resolving");
  });

  it("includes key insight in quotes", () => {
    const row = formatDreamRow(makeDream(), 0);
    expect(row).toContain('"Fear of change is natural."');
  });

  it("includes up to 3 symbols", () => {
    const row = formatDreamRow(
      makeDream({ symbols: ["water", "bridge", "fire", "sky"] }),
      0,
    );
    expect(row).toContain("water");
    expect(row).toContain("bridge");
    expect(row).toContain("fire");
    expect(row).not.toContain("sky"); // 4th symbol omitted
  });

  it("marks nightmare entries", () => {
    const row = formatDreamRow(nightmare, 1);
    expect(row.toLowerCase()).toContain("nightmare");
  });

  it("does not mark non-nightmares", () => {
    const row = formatDreamRow(makeDream({ threatSimulationIndex: 0.3 }), 0);
    expect(row.toLowerCase()).not.toContain("nightmare");
  });

  it("handles null themes gracefully", () => {
    const row = formatDreamRow(makeDream({ themes: null }), 0);
    expect(row).toContain("Dream 1");
    expect(row).not.toContain("themes:");
  });

  it("handles null emotionalArc gracefully", () => {
    const row = formatDreamRow(makeDream({ emotionalArc: null }), 0);
    expect(row).not.toContain("arc:");
  });

  it("handles index > 0 (Dream 2, Dream 3)", () => {
    const row = formatDreamRow(makeDream(), 2);
    expect(row).toContain("Dream 3");
  });
});

// ── buildSynthesisContext ─────────────────────────────────────────────────────

describe("buildSynthesisContext", () => {
  it("returns fallback for empty input", () => {
    const ctx = buildSynthesisContext([]);
    expect(ctx.toLowerCase()).toContain("no dream data");
  });

  it("returns one line per dream", () => {
    const ctx = buildSynthesisContext([makeDream(), nightmare]);
    const lines = ctx.split("\n").filter(Boolean);
    expect(lines).toHaveLength(2);
  });

  it("preserves order (index 1 then 2)", () => {
    const ctx = buildSynthesisContext([makeDream(), nightmare]);
    const idx1 = ctx.indexOf("Dream 1");
    const idx2 = ctx.indexOf("Dream 2");
    expect(idx1).toBeLessThan(idx2);
  });
});

// ── extractWeekTopThemes ──────────────────────────────────────────────────────

describe("extractWeekTopThemes", () => {
  it("returns empty array for dreams with no themes", () => {
    const dreams = [makeDream({ themes: null }), makeDream({ themes: [] })];
    expect(extractWeekTopThemes(dreams)).toHaveLength(0);
  });

  it("deduplicates and counts themes", () => {
    const dreams = [
      makeDream({ themes: ["self-exploration", "transition"] }),
      makeDream({ themes: ["self-exploration", "achievement"] }),
    ];
    const top = extractWeekTopThemes(dreams);
    expect(top[0]).toBe("self-exploration"); // appeared twice
  });

  it("returns at most topN entries (default 5)", () => {
    const manyThemes = Array.from({ length: 10 }, (_, i) => `theme-${i}`);
    const dreams = [makeDream({ themes: manyThemes })];
    expect(extractWeekTopThemes(dreams)).toHaveLength(5);
  });

  it("respects custom topN", () => {
    const dreams = [makeDream({ themes: ["a", "b", "c", "d", "e", "f"] })];
    expect(extractWeekTopThemes(dreams, 3)).toHaveLength(3);
  });

  it("sorts by frequency descending", () => {
    const dreams = [
      makeDream({ themes: ["rare"] }),
      makeDream({ themes: ["common", "common", "also-common"] }),
      makeDream({ themes: ["common"] }),
    ];
    const top = extractWeekTopThemes(dreams);
    const freqIdx = top.indexOf("common");
    const rareIdx = top.indexOf("rare");
    expect(freqIdx).toBeLessThan(rareIdx);
  });
});

// ── countWeekNightmares ───────────────────────────────────────────────────────

describe("countWeekNightmares", () => {
  it("returns 0 for empty list", () => {
    expect(countWeekNightmares([])).toBe(0);
  });

  it("returns 0 when no nightmares", () => {
    expect(countWeekNightmares([makeDream(), makeDream()])).toBe(0);
  });

  it("counts entries with tsi > 0.5", () => {
    expect(countWeekNightmares([nightmare, makeDream(), nightmare])).toBe(2);
  });

  it("boundary: tsi = 0.5 is NOT a nightmare", () => {
    expect(countWeekNightmares([makeDream({ threatSimulationIndex: 0.5 })])).toBe(0);
  });

  it("boundary: tsi = 0.51 IS a nightmare", () => {
    expect(countWeekNightmares([makeDream({ threatSimulationIndex: 0.51 })])).toBe(1);
  });

  it("null/undefined tsi counts as 0", () => {
    expect(countWeekNightmares([makeDream({ threatSimulationIndex: null })])).toBe(0);
    expect(countWeekNightmares([makeDream({ threatSimulationIndex: undefined })])).toBe(0);
  });
});

// ── extractWeekSymbols ────────────────────────────────────────────────────────

describe("extractWeekSymbols", () => {
  it("returns empty for dreams with no symbols", () => {
    expect(extractWeekSymbols([makeDream({ symbols: null })])).toHaveLength(0);
  });

  it("deduplicates across dreams (case-insensitive)", () => {
    const dreams = [
      makeDream({ symbols: ["Water", "Bridge"] }),
      makeDream({ symbols: ["water", "Fire"] }),
    ];
    const syms = extractWeekSymbols(dreams);
    expect(syms).toContain("water");
    expect(syms.filter((s) => s === "water")).toHaveLength(1); // deduped
  });

  it("lowercases all symbols", () => {
    const syms = extractWeekSymbols([makeDream({ symbols: ["SHADOW", "Door"] })]);
    expect(syms).toContain("shadow");
    expect(syms).toContain("door");
  });
});

// ── buildWeeklySynthesisPrompt ────────────────────────────────────────────────

describe("buildWeeklySynthesisPrompt", () => {
  it("returns system and user strings", () => {
    const { system, user } = buildWeeklySynthesisPrompt([makeDream()]);
    expect(typeof system).toBe("string");
    expect(typeof user).toBe("string");
    expect(system.length).toBeGreaterThan(10);
    expect(user.length).toBeGreaterThan(10);
  });

  it("includes dream count in user message", () => {
    const { user } = buildWeeklySynthesisPrompt([makeDream(), nightmare]);
    expect(user).toContain("2 dream");
  });

  it("includes nightmare count in user message", () => {
    const { user } = buildWeeklySynthesisPrompt([makeDream(), nightmare]);
    expect(user).toContain("1 nightmare");
  });

  it("requests JSON response format", () => {
    const { user } = buildWeeklySynthesisPrompt([makeDream()]);
    expect(user).toContain("JSON");
    expect(user).toContain('"synthesis"');
  });

  it("lists top themes when present", () => {
    const dreams = [
      makeDream({ themes: ["transition", "transition", "achievement"] }),
    ];
    const { user } = buildWeeklySynthesisPrompt(dreams);
    expect(user).toContain("transition");
  });

  it("handles dreams with no themes without crashing", () => {
    const dreams = [makeDream({ themes: null }), makeDream({ themes: [] })];
    expect(() => buildWeeklySynthesisPrompt(dreams)).not.toThrow();
  });
});

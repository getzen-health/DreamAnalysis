import { describe, it, expect } from "vitest";

/**
 * Tests for the dream-patterns aggregation logic.
 *
 * The GET /api/dream-patterns/:userId endpoint was upgraded to:
 * 1. Use stored Hall/Van de Castle themes (from multi-pass analysis)
 *    instead of emotion names as theme proxies.
 * 2. Track nightmare frequency via threatSimulationIndex > 0.5.
 * 3. Enrich topInsights with keyInsight field.
 * 4. Return 7/30/90-day counts sub-view.
 *
 * These tests verify the pure aggregation logic extracted from the endpoint.
 */

// ── Types mirroring the dreamAnalysis row (with new columns) ─────────────────

interface DreamRow {
  id: string;
  timestamp: Date;
  symbols: string[];
  emotions: Array<{ emotion: string; intensity: number }>;
  aiAnalysis: string | null;
  themes?: string[] | null;
  emotionalArc?: string | null;
  keyInsight?: string | null;
  threatSimulationIndex?: number | null;
}

// ── Pure aggregation helpers (mirrors server logic) ───────────────────────────

function aggregateThemes(rows: DreamRow[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const row of rows) {
    if (row.themes && row.themes.length > 0) {
      for (const t of row.themes) {
        if (t) counts[t] = (counts[t] || 0) + 1;
      }
    } else {
      // Legacy fallback: emotion names
      for (const em of row.emotions) {
        if (em.emotion) counts[em.emotion] = (counts[em.emotion] || 0) + 1;
      }
    }
  }
  return counts;
}

function aggregateSymbols(rows: DreamRow[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const row of rows) {
    for (const sym of row.symbols) {
      if (sym) counts[sym] = (counts[sym] || 0) + 1;
    }
  }
  return counts;
}

function buildSentimentTrend(rows: DreamRow[]): Array<{ date: string; valence: number }> {
  return rows
    .map((row) => ({
      date: row.timestamp.toISOString().split("T")[0],
      valence:
        row.emotions.reduce((sum, e) => sum + (e.intensity || 0), 0) /
        Math.max(row.emotions.length, 1),
    }))
    .sort((a, b) => a.date.localeCompare(b.date));
}

function countNightmares(rows: DreamRow[]): { count: number; dates: string[] } {
  const dates = rows
    .filter((r) => r.threatSimulationIndex != null && r.threatSimulationIndex > 0.5)
    .map((r) => r.timestamp.toISOString().split("T")[0]);
  return { count: dates.length, dates };
}

function topN<T extends { count: number }>(items: T[], n: number): T[] {
  return [...items].sort((a, b) => b.count - a.count).slice(0, n);
}

// ── Fixtures ──────────────────────────────────────────────────────────────────

const makeRow = (
  overrides: Partial<DreamRow> & { id: string; timestamp: Date },
): DreamRow => ({
  symbols: [],
  emotions: [],
  aiAnalysis: null,
  themes: null,
  emotionalArc: null,
  keyInsight: null,
  threatSimulationIndex: null,
  ...overrides,
});

const rows: DreamRow[] = [
  makeRow({
    id: "1",
    timestamp: new Date("2026-03-01"),
    symbols: ["water", "house"],
    emotions: [{ emotion: "fear", intensity: 0.8 }],
    themes: ["transition", "self-exploration"],
    threatSimulationIndex: 0.6,
    keyInsight: "Anxiety about change",
  }),
  makeRow({
    id: "2",
    timestamp: new Date("2026-03-02"),
    symbols: ["water", "flying"],
    emotions: [{ emotion: "joy", intensity: 0.9 }],
    themes: ["transition", "achievement"],
    threatSimulationIndex: 0.1,
    keyInsight: "Embracing freedom",
  }),
  makeRow({
    id: "3",
    timestamp: new Date("2026-03-03"),
    symbols: ["house"],
    emotions: [{ emotion: "sadness", intensity: 0.4 }],
    // No themes — should fall back to emotion names
    themes: null,
    threatSimulationIndex: null,
    keyInsight: null,
  }),
  makeRow({
    id: "4",
    timestamp: new Date("2026-03-04"),
    symbols: ["water"],
    emotions: [{ emotion: "fear", intensity: 0.85 }],
    themes: ["threat-simulation"],
    threatSimulationIndex: 0.9,
    keyInsight: "Recurring nightmare",
  }),
];

// ── Tests ─────────────────────────────────────────────────────────────────────

describe("aggregateThemes", () => {
  it("counts Hall/Van de Castle themes when present", () => {
    const counts = aggregateThemes(rows);
    expect(counts["transition"]).toBe(2);
    expect(counts["self-exploration"]).toBe(1);
    expect(counts["achievement"]).toBe(1);
    expect(counts["threat-simulation"]).toBe(1);
  });

  it("falls back to emotion names when themes is null", () => {
    const legacyRow = makeRow({
      id: "x",
      timestamp: new Date(),
      symbols: [],
      emotions: [{ emotion: "joy", intensity: 0.5 }],
      themes: null,
    });
    const counts = aggregateThemes([legacyRow]);
    expect(counts["joy"]).toBe(1);
  });

  it("falls back to emotion names when themes is empty array", () => {
    const emptyThemeRow = makeRow({
      id: "y",
      timestamp: new Date(),
      symbols: [],
      emotions: [{ emotion: "anger", intensity: 0.7 }],
      themes: [],
    });
    const counts = aggregateThemes([emptyThemeRow]);
    expect(counts["anger"]).toBe(1);
  });

  it("skips empty string themes", () => {
    const rowWithEmptyTheme = makeRow({
      id: "z",
      timestamp: new Date(),
      symbols: [],
      emotions: [],
      themes: ["", "valid-theme"],
    });
    const counts = aggregateThemes([rowWithEmptyTheme]);
    expect(counts["valid-theme"]).toBe(1);
    expect(counts[""]).toBeUndefined();
  });

  it("returns empty when no rows", () => {
    expect(aggregateThemes([])).toEqual({});
  });
});

describe("aggregateSymbols", () => {
  it("counts symbol frequencies across rows", () => {
    const counts = aggregateSymbols(rows);
    expect(counts["water"]).toBe(3);
    expect(counts["house"]).toBe(2);
    expect(counts["flying"]).toBe(1);
  });

  it("returns empty for empty rows", () => {
    expect(aggregateSymbols([])).toEqual({});
  });
});

describe("buildSentimentTrend", () => {
  it("returns one entry per row sorted by date", () => {
    const trend = buildSentimentTrend(rows);
    expect(trend).toHaveLength(4);
    expect(trend[0].date).toBe("2026-03-01");
    expect(trend[3].date).toBe("2026-03-04");
  });

  it("computes mean emotion intensity as valence", () => {
    const singleRow = makeRow({
      id: "a",
      timestamp: new Date("2026-03-10"),
      symbols: [],
      emotions: [
        { emotion: "joy", intensity: 0.8 },
        { emotion: "calm", intensity: 0.4 },
      ],
    });
    const trend = buildSentimentTrend([singleRow]);
    expect(trend[0].valence).toBeCloseTo(0.6, 3);
  });

  it("returns 0 valence for rows with no emotions", () => {
    const emptyRow = makeRow({ id: "b", timestamp: new Date("2026-03-05"), symbols: [], emotions: [] });
    const trend = buildSentimentTrend([emptyRow]);
    expect(trend[0].valence).toBe(0);
  });
});

describe("countNightmares", () => {
  it("counts rows where threatSimulationIndex > 0.5", () => {
    const result = countNightmares(rows);
    expect(result.count).toBe(2); // rows 1 (0.6) and 4 (0.9)
  });

  it("includes the nightmare dates", () => {
    const result = countNightmares(rows);
    expect(result.dates).toContain("2026-03-01");
    expect(result.dates).toContain("2026-03-04");
    expect(result.dates).not.toContain("2026-03-02"); // 0.1 is not a nightmare
  });

  it("returns 0 count when no nightmares", () => {
    const cleanRows = rows.map((r) => ({ ...r, threatSimulationIndex: 0.1 }));
    expect(countNightmares(cleanRows).count).toBe(0);
  });

  it("treats null threatSimulationIndex as non-nightmare", () => {
    const nullRow = makeRow({ id: "n", timestamp: new Date(), symbols: [], emotions: [], threatSimulationIndex: null });
    expect(countNightmares([nullRow]).count).toBe(0);
  });
});

describe("topN", () => {
  it("returns top N items by count descending", () => {
    const items = [
      { name: "a", count: 1 },
      { name: "b", count: 5 },
      { name: "c", count: 3 },
    ];
    const top2 = topN(items, 2);
    expect(top2[0].name).toBe("b");
    expect(top2[1].name).toBe("c");
    expect(top2).toHaveLength(2);
  });

  it("returns fewer than N if list is short", () => {
    expect(topN([{ name: "x", count: 1 }], 5)).toHaveLength(1);
  });
});

import { describe, it, expect } from "vitest";

/**
 * Tests for the dream symbol library logic.
 *
 * Covers:
 * 1. persistDreamSymbols — helper that normalises input before upsert
 * 2. GET /api/dream-symbols sort order
 * 3. Edge-cases: empty input, null userId, whitespace symbols, duplicates
 *
 * These are pure-logic mirrors of the server code, so they run without a DB.
 */

// ── Mirror of server persistDreamSymbols helper ───────────────────────────────

interface UpsertCall {
  userId: string;
  symbol: string;
  meaning: string | null;
}

function collectPersistCalls(
  userId: string | null,
  symbols: Array<string | { symbol: string; meaning?: string | null }>,
): UpsertCall[] {
  if (!userId || symbols.length === 0) return [];
  const calls: UpsertCall[] = [];
  for (const s of symbols) {
    const sym = (typeof s === "string" ? s : s.symbol)?.trim().toLowerCase();
    const meaning = typeof s === "string" ? null : (s.meaning ?? null);
    if (!sym) continue;
    calls.push({ userId, symbol: sym, meaning });
  }
  return calls;
}

describe("persistDreamSymbols logic", () => {
  it("returns empty array for null userId", () => {
    expect(collectPersistCalls(null, ["water", "house"])).toHaveLength(0);
  });

  it("returns empty array for empty symbols list", () => {
    expect(collectPersistCalls("u1", [])).toHaveLength(0);
  });

  it("handles plain string symbols (EEG path)", () => {
    const calls = collectPersistCalls("u1", ["Water", "Flying"]);
    expect(calls).toHaveLength(2);
    expect(calls[0]).toEqual({ userId: "u1", symbol: "water", meaning: null });
    expect(calls[1]).toEqual({ userId: "u1", symbol: "flying", meaning: null });
  });

  it("lowercases and trims symbol names", () => {
    const calls = collectPersistCalls("u1", ["  HOUSE  "]);
    expect(calls[0].symbol).toBe("house");
  });

  it("handles object symbols with meaning (multi-pass path)", () => {
    const calls = collectPersistCalls("u1", [
      { symbol: "Snake", meaning: "Transformation or threat" },
    ]);
    expect(calls[0]).toEqual({ userId: "u1", symbol: "snake", meaning: "Transformation or threat" });
  });

  it("sets meaning to null for object symbols with no meaning field", () => {
    const calls = collectPersistCalls("u1", [{ symbol: "river" }]);
    expect(calls[0].meaning).toBeNull();
  });

  it("skips empty-string symbols", () => {
    const calls = collectPersistCalls("u1", ["", "tree"]);
    expect(calls).toHaveLength(1);
    expect(calls[0].symbol).toBe("tree");
  });

  it("skips whitespace-only symbols after trim", () => {
    const calls = collectPersistCalls("u1", ["   ", "mirror"]);
    expect(calls).toHaveLength(1);
  });

  it("preserves null meaning from object when meaning is explicitly null", () => {
    const calls = collectPersistCalls("u1", [{ symbol: "bridge", meaning: null }]);
    expect(calls[0].meaning).toBeNull();
  });
});

// ── Mirror of server GET /api/dream-symbols sort logic ────────────────────────

interface SymbolRow {
  id: string;
  symbol: string;
  meaning: string | null;
  frequency: number;
  firstSeen: string;
  lastSeen: string;
}

function sortSymbols(rows: SymbolRow[]): SymbolRow[] {
  return [...rows].sort(
    (a, b) => (b.frequency ?? 0) - (a.frequency ?? 0) || a.symbol.localeCompare(b.symbol),
  );
}

const makeSymbol = (id: string, symbol: string, frequency: number): SymbolRow => ({
  id, symbol, meaning: null, frequency,
  firstSeen: "2026-03-01T00:00:00.000Z",
  lastSeen: "2026-03-30T00:00:00.000Z",
});

describe("GET /api/dream-symbols sort order", () => {
  it("sorts by frequency descending", () => {
    const rows = [makeSymbol("1", "snake", 2), makeSymbol("2", "water", 5), makeSymbol("3", "house", 1)];
    const sorted = sortSymbols(rows);
    expect(sorted.map(r => r.symbol)).toEqual(["water", "snake", "house"]);
  });

  it("breaks ties alphabetically", () => {
    const rows = [makeSymbol("1", "snake", 3), makeSymbol("2", "bridge", 3), makeSymbol("3", "arrow", 3)];
    const sorted = sortSymbols(rows);
    expect(sorted.map(r => r.symbol)).toEqual(["arrow", "bridge", "snake"]);
  });

  it("returns empty array for empty input", () => {
    expect(sortSymbols([])).toEqual([]);
  });

  it("returns single-item arrays unchanged", () => {
    const rows = [makeSymbol("1", "water", 4)];
    expect(sortSymbols(rows)).toHaveLength(1);
  });

  it("does not mutate the input array", () => {
    const rows = [makeSymbol("1", "z", 1), makeSymbol("2", "a", 3)];
    const original = [...rows];
    sortSymbols(rows);
    expect(rows[0].symbol).toBe(original[0].symbol);
  });
});

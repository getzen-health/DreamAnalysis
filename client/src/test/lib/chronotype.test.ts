import { describe, it, expect, beforeEach, vi } from "vitest";
import {
  CHRONOTYPE_QUESTIONS,
  scoreChronotype,
  getBaselineAdjustment,
  saveChronotype,
  getStoredChronotype,
  type ChronotypeCategory,
} from "@/lib/chronotype";

// ── Mock localStorage ─────────────────────────────────────────────────────

const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: vi.fn((key: string) => store[key] ?? null),
    setItem: vi.fn((key: string, value: string) => { store[key] = value; }),
    removeItem: vi.fn((key: string) => { delete store[key]; }),
    clear: vi.fn(() => { store = {}; }),
  };
})();

Object.defineProperty(globalThis, "localStorage", { value: localStorageMock });

beforeEach(() => {
  localStorageMock.clear();
  vi.clearAllMocks();
});

// ── Questionnaire structure ───────────────────────────────────────────────

describe("CHRONOTYPE_QUESTIONS", () => {
  it("has exactly 5 questions", () => {
    expect(CHRONOTYPE_QUESTIONS).toHaveLength(5);
  });

  it("each question has id, text, and options array", () => {
    for (const q of CHRONOTYPE_QUESTIONS) {
      expect(q.id).toBeDefined();
      expect(typeof q.text).toBe("string");
      expect(Array.isArray(q.options)).toBe(true);
      expect(q.options.length).toBeGreaterThanOrEqual(4);
      for (const opt of q.options) {
        expect(typeof opt.label).toBe("string");
        expect(typeof opt.score).toBe("number");
      }
    }
  });
});

// ── Score boundaries ──────────────────────────────────────────────────────

describe("scoreChronotype", () => {
  it("classifies minimum score (4) as evening", () => {
    // Min possible: 1+1+1+1+0 = 4
    const result = scoreChronotype([1, 1, 1, 1, 0]);
    expect(result.score).toBe(4);
    expect(result.category).toBe("evening");
  });

  it("classifies maximum score (25) as morning", () => {
    // Max possible: 5+4+5+5+6 = 25
    const result = scoreChronotype([5, 4, 5, 5, 6]);
    expect(result.score).toBe(25);
    expect(result.category).toBe("morning");
  });

  it("classifies score 11 as evening (upper boundary)", () => {
    const result = scoreChronotype([3, 2, 3, 1, 2]);
    expect(result.score).toBe(11);
    expect(result.category).toBe("evening");
  });

  it("classifies score 12 as intermediate (lower boundary)", () => {
    const result = scoreChronotype([3, 2, 3, 2, 2]);
    expect(result.score).toBe(12);
    expect(result.category).toBe("intermediate");
  });

  it("classifies score 17 as intermediate (upper boundary)", () => {
    const result = scoreChronotype([4, 3, 4, 2, 4]);
    expect(result.score).toBe(17);
    expect(result.category).toBe("intermediate");
  });

  it("classifies score 18 as morning (lower boundary)", () => {
    const result = scoreChronotype([4, 3, 4, 3, 4]);
    expect(result.score).toBe(18);
    expect(result.category).toBe("morning");
  });
});

// ── Baseline adjustments ──────────────────────────────────────────────────

describe("getBaselineAdjustment", () => {
  it("morning type at 7 AM gets positive valence offset", () => {
    const adj = getBaselineAdjustment("morning", 7);
    expect(adj.valenceOffset).toBeGreaterThan(0);
    expect(adj.alphaMultiplier).toBeGreaterThan(1.0);
  });

  it("evening type at 7 AM gets negative valence offset", () => {
    const adj = getBaselineAdjustment("evening", 7);
    expect(adj.valenceOffset).toBeLessThan(0);
    expect(adj.alphaMultiplier).toBeLessThan(1.0);
  });

  it("morning type at 10 PM gets negative valence offset (off-peak)", () => {
    const adj = getBaselineAdjustment("morning", 22);
    expect(adj.valenceOffset).toBeLessThan(0);
  });

  it("evening type at 8 PM gets positive valence offset (peak)", () => {
    const adj = getBaselineAdjustment("evening", 20);
    expect(adj.valenceOffset).toBeGreaterThan(0);
    expect(adj.alphaMultiplier).toBeGreaterThan(1.0);
  });

  it("intermediate type gets mild adjustments (small absolute values)", () => {
    // Check across several hours
    for (const hour of [7, 12, 18, 22]) {
      const adj = getBaselineAdjustment("intermediate", hour);
      expect(Math.abs(adj.valenceOffset)).toBeLessThanOrEqual(0.08);
      expect(Math.abs(adj.arousalOffset)).toBeLessThanOrEqual(0.08);
      expect(adj.alphaMultiplier).toBeGreaterThanOrEqual(0.92);
      expect(adj.alphaMultiplier).toBeLessThanOrEqual(1.08);
    }
  });

  it("all multipliers stay within 0.85-1.15 range", () => {
    const types: ChronotypeCategory[] = ["morning", "intermediate", "evening"];
    for (const type of types) {
      for (let h = 0; h < 24; h++) {
        const adj = getBaselineAdjustment(type, h);
        expect(adj.alphaMultiplier).toBeGreaterThanOrEqual(0.85);
        expect(adj.alphaMultiplier).toBeLessThanOrEqual(1.15);
        expect(Math.abs(adj.arousalOffset)).toBeLessThanOrEqual(0.15);
        expect(Math.abs(adj.valenceOffset)).toBeLessThanOrEqual(0.15);
      }
    }
  });
});

// ── localStorage persistence ──────────────────────────────────────────────

describe("saveChronotype / getStoredChronotype", () => {
  it("roundtrips save and load correctly", () => {
    saveChronotype(20, "morning");
    const stored = getStoredChronotype();
    expect(stored).not.toBeNull();
    expect(stored!.score).toBe(20);
    expect(stored!.category).toBe("morning");
  });

  it("returns null when no chronotype stored", () => {
    const stored = getStoredChronotype();
    expect(stored).toBeNull();
  });

  it("overwrites previous value on re-save", () => {
    saveChronotype(8, "evening");
    saveChronotype(20, "morning");
    const stored = getStoredChronotype();
    expect(stored!.category).toBe("morning");
  });

  it("handles corrupted localStorage gracefully", () => {
    localStorage.setItem("ndw_chronotype", "not-json{{{");
    const stored = getStoredChronotype();
    expect(stored).toBeNull();
  });
});

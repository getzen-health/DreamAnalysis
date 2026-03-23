import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import {
  saveSubstanceLog,
  getLatestSubstanceLog,
  getBaselineAdjustment,
  hasAnsweredToday,
  type SubstanceLog,
} from "@/lib/substance-context";

beforeEach(() => {
  localStorage.clear();
  vi.useFakeTimers();
});

afterEach(() => {
  localStorage.clear();
  vi.useRealTimers();
});

// ── Test 1: No caffeine → no alpha/beta adjustment ──────────────────────────

describe("getBaselineAdjustment", () => {
  it("returns zero offsets when no substances reported", () => {
    const log: SubstanceLog = {
      timestamp: new Date().toISOString(),
      caffeine: "none",
      alcohol: "none",
      medications: [],
      cannabis: "none",
    };

    const adj = getBaselineAdjustment(log);

    expect(adj.alphaOffset).toBe(0);
    expect(adj.betaOffset).toBe(0);
    expect(adj.thetaOffset).toBe(0);
    expect(adj.note).toBe("");
  });

  // ── Test 2: Recent caffeine → alpha decreases, beta increases ─────────

  it("adjusts alpha down and beta up for recent caffeine (1-2hrs ago)", () => {
    const log: SubstanceLog = {
      timestamp: new Date().toISOString(),
      caffeine: "1-2hrs_ago",
      alcohol: "none",
      medications: [],
      cannabis: "none",
    };

    const adj = getBaselineAdjustment(log);

    expect(adj.alphaOffset).toBe(-0.15);
    expect(adj.betaOffset).toBe(0.20);
    expect(adj.thetaOffset).toBe(0);
  });

  it("adjusts alpha down and beta up less for older caffeine (3-6hrs ago)", () => {
    const log: SubstanceLog = {
      timestamp: new Date().toISOString(),
      caffeine: "3-6hrs_ago",
      alcohol: "none",
      medications: [],
      cannabis: "none",
    };

    const adj = getBaselineAdjustment(log);

    expect(adj.alphaOffset).toBe(-0.08);
    expect(adj.betaOffset).toBe(0.10);
    expect(adj.thetaOffset).toBe(0);
  });

  // ── Test 3: SSRI medication → alpha/theta decrease, beta increase ─────

  it("adjusts for SSRI keyword in medications", () => {
    const log: SubstanceLog = {
      timestamp: new Date().toISOString(),
      caffeine: "none",
      alcohol: "none",
      medications: ["SSRI"],
      cannabis: "none",
    };

    const adj = getBaselineAdjustment(log);

    expect(adj.alphaOffset).toBe(-0.10);
    expect(adj.betaOffset).toBe(0.15);
    expect(adj.thetaOffset).toBe(-0.10);
  });

  it("adjusts for common SSRI name (sertraline)", () => {
    const log: SubstanceLog = {
      timestamp: new Date().toISOString(),
      caffeine: "none",
      alcohol: "none",
      medications: ["sertraline 50mg"],
      cannabis: "none",
    };

    const adj = getBaselineAdjustment(log);

    expect(adj.alphaOffset).toBe(-0.10);
    expect(adj.betaOffset).toBe(0.15);
    expect(adj.thetaOffset).toBe(-0.10);
  });

  it("adjusts for fluoxetine (case-insensitive)", () => {
    const log: SubstanceLog = {
      timestamp: new Date().toISOString(),
      caffeine: "none",
      alcohol: "none",
      medications: ["Fluoxetine"],
      cannabis: "none",
    };

    const adj = getBaselineAdjustment(log);

    expect(adj.alphaOffset).toBe(-0.10);
    expect(adj.betaOffset).toBe(0.15);
    expect(adj.thetaOffset).toBe(-0.10);
  });

  // ── Alcohol last night → hangover effect ──────────────────────────────

  it("adjusts for alcohol consumed last night (hangover effect)", () => {
    const log: SubstanceLog = {
      timestamp: new Date().toISOString(),
      caffeine: "none",
      alcohol: "last_night",
      medications: [],
      cannabis: "none",
    };

    const adj = getBaselineAdjustment(log);

    expect(adj.alphaOffset).toBe(-0.10);
    expect(adj.thetaOffset).toBe(0.10);
    expect(adj.betaOffset).toBe(0);
  });

  // ── Combined substances stack offsets ─────────────────────────────────

  it("stacks offsets from multiple substances", () => {
    const log: SubstanceLog = {
      timestamp: new Date().toISOString(),
      caffeine: "1-2hrs_ago",
      alcohol: "last_night",
      medications: ["sertraline"],
      cannabis: "none",
    };

    const adj = getBaselineAdjustment(log);

    // caffeine: alpha -0.15, beta +0.20
    // alcohol: alpha -0.10, theta +0.10
    // SSRI: alpha -0.10, beta +0.15, theta -0.10
    expect(adj.alphaOffset).toBe(-0.35);
    expect(adj.betaOffset).toBe(0.35);
    expect(adj.thetaOffset).toBe(0); // +0.10 - 0.10 = 0
  });

  // ── Test 6: Note is always non-empty when adjustments are non-zero ────

  it("returns non-empty note when any adjustment is non-zero", () => {
    const log: SubstanceLog = {
      timestamp: new Date().toISOString(),
      caffeine: "1-2hrs_ago",
      alcohol: "none",
      medications: [],
      cannabis: "none",
    };

    const adj = getBaselineAdjustment(log);

    expect(adj.note).not.toBe("");
    expect(adj.note.length).toBeGreaterThan(0);
  });

  it("includes caffeine in note when caffeine reported", () => {
    const log: SubstanceLog = {
      timestamp: new Date().toISOString(),
      caffeine: "3-6hrs_ago",
      alcohol: "none",
      medications: [],
      cannabis: "none",
    };

    const adj = getBaselineAdjustment(log);

    expect(adj.note.toLowerCase()).toContain("caffeine");
  });
});

// ── Test 4: Save/load roundtrip ─────────────────────────────────────────────

describe("saveSubstanceLog / getLatestSubstanceLog", () => {
  it("roundtrips a substance log through localStorage", () => {
    const log: SubstanceLog = {
      timestamp: "2026-03-23T10:00:00.000Z",
      caffeine: "1-2hrs_ago",
      alcohol: "none",
      medications: ["vitamin D"],
      cannabis: "none",
    };

    saveSubstanceLog(log);
    const loaded = getLatestSubstanceLog();

    expect(loaded).not.toBeNull();
    expect(loaded!.caffeine).toBe("1-2hrs_ago");
    expect(loaded!.alcohol).toBe("none");
    expect(loaded!.medications).toEqual(["vitamin D"]);
    expect(loaded!.cannabis).toBe("none");
    expect(loaded!.timestamp).toBe("2026-03-23T10:00:00.000Z");
  });

  it("returns null when nothing saved", () => {
    const loaded = getLatestSubstanceLog();
    expect(loaded).toBeNull();
  });

  it("returns null when localStorage contains invalid JSON", () => {
    localStorage.setItem("ndw_substance_log", "not-valid-json");
    const loaded = getLatestSubstanceLog();
    expect(loaded).toBeNull();
  });
});

// ── Test 5: Skip returns null (no adjustments applied) ──────────────────────

describe("skip behavior", () => {
  it("getBaselineAdjustment returns null for null log (skip)", () => {
    const adj = getBaselineAdjustment(null);
    expect(adj).toBeNull();
  });
});

// ── Test 7: Only asks once per day (check timestamp) ────────────────────────

describe("hasAnsweredToday", () => {
  it("returns false when no log exists", () => {
    expect(hasAnsweredToday()).toBe(false);
  });

  it("returns true when log was saved today", () => {
    vi.setSystemTime(new Date("2026-03-23T14:00:00.000Z"));

    const log: SubstanceLog = {
      timestamp: new Date("2026-03-23T09:00:00.000Z").toISOString(),
      caffeine: "none",
      alcohol: "none",
      medications: [],
      cannabis: "none",
    };
    saveSubstanceLog(log);

    expect(hasAnsweredToday()).toBe(true);
  });

  it("returns false when log was saved yesterday", () => {
    vi.setSystemTime(new Date("2026-03-23T14:00:00.000Z"));

    const log: SubstanceLog = {
      timestamp: new Date("2026-03-22T09:00:00.000Z").toISOString(),
      caffeine: "none",
      alcohol: "none",
      medications: [],
      cannabis: "none",
    };
    saveSubstanceLog(log);

    expect(hasAnsweredToday()).toBe(false);
  });
});

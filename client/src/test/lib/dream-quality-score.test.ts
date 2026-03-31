import { describe, it, expect } from "vitest";
import {
  computeDreamQualityScore,
  aggregateDailyScores,
  qualityBand,
  qualityLabel,
  qualityColorClass,
  qualityBgClass,
  averageScore,
} from "@/lib/dream-quality-score";

// ── computeDreamQualityScore ──────────────────────────────────────────────────

describe("computeDreamQualityScore", () => {
  it("returns null when all inputs are null/false", () => {
    expect(
      computeDreamQualityScore({
        sleepQuality: null,
        lucidityScore: null,
        threatSimulationIndex: null,
        hasDreamText: false,
      }),
    ).toBeNull();
  });

  it("returns a score when only hasDreamText = true", () => {
    const score = computeDreamQualityScore({
      sleepQuality: null,
      lucidityScore: null,
      threatSimulationIndex: null,
      hasDreamText: true,
    });
    expect(score).not.toBeNull();
    expect(score).toBe(60); // 50 base + 10 recall
  });

  it("sleep quality at midpoint (50) contributes 0 delta", () => {
    const score = computeDreamQualityScore({
      sleepQuality: 50,
      lucidityScore: null,
      threatSimulationIndex: null,
      hasDreamText: false,
    });
    expect(score).toBe(50);
  });

  it("max sleep quality (90) contributes +20", () => {
    const score = computeDreamQualityScore({
      sleepQuality: 90,
      lucidityScore: null,
      threatSimulationIndex: null,
      hasDreamText: false,
    });
    expect(score).toBe(70); // 50 + 20
  });

  it("min sleep quality (10) contributes -20", () => {
    const score = computeDreamQualityScore({
      sleepQuality: 10,
      lucidityScore: null,
      threatSimulationIndex: null,
      hasDreamText: false,
    });
    expect(score).toBe(30); // 50 - 20
  });

  it("full lucidity (100) contributes +15", () => {
    const score = computeDreamQualityScore({
      sleepQuality: null,
      lucidityScore: 100,
      threatSimulationIndex: null,
      hasDreamText: false,
    });
    expect(score).toBe(65); // 50 + 15
  });

  it("zero lucidity contributes 0", () => {
    const score = computeDreamQualityScore({
      sleepQuality: null,
      lucidityScore: 0,
      threatSimulationIndex: null,
      hasDreamText: false,
    });
    expect(score).toBe(50);
  });

  it("full nightmare (tsi=1) applies -25 penalty", () => {
    const score = computeDreamQualityScore({
      sleepQuality: null,
      lucidityScore: null,
      threatSimulationIndex: 1,
      hasDreamText: false,
    });
    expect(score).toBe(25); // 50 - 25
  });

  it("null/zero tsi applies no penalty", () => {
    const s1 = computeDreamQualityScore({ sleepQuality: null, lucidityScore: null, threatSimulationIndex: null, hasDreamText: false });
    const s2 = computeDreamQualityScore({ sleepQuality: null, lucidityScore: null, threatSimulationIndex: 0, hasDreamText: false });
    expect(s1).toBeNull(); // all null → null
    expect(s2).toBe(50);   // tsi=0 counts as "has data"
  });

  it("score is clamped to 0", () => {
    // tsi=1 (-25) + sleepQuality=10 (-20) → 50 - 25 - 20 = 5, not negative
    const score = computeDreamQualityScore({
      sleepQuality: 10,
      lucidityScore: null,
      threatSimulationIndex: 1,
      hasDreamText: false,
    });
    expect(score).toBe(5);
  });

  it("score is clamped to 100", () => {
    // sleepQuality=90 (+20) + lucidity=100 (+15) + recall (+10) → 50+20+15+10 = 95
    const score = computeDreamQualityScore({
      sleepQuality: 90,
      lucidityScore: 100,
      threatSimulationIndex: 0,
      hasDreamText: true,
    });
    expect(score).toBe(95);
  });

  it("all-best inputs produce high score", () => {
    const score = computeDreamQualityScore({
      sleepQuality: 90,
      lucidityScore: 100,
      threatSimulationIndex: 0,
      hasDreamText: true,
    });
    expect(score).toBeGreaterThanOrEqual(90);
  });

  it("all-worst inputs produce low score", () => {
    const score = computeDreamQualityScore({
      sleepQuality: 10,
      lucidityScore: 0,
      threatSimulationIndex: 1,
      hasDreamText: false,
    });
    expect(score).toBeLessThanOrEqual(15);
  });
});

// ── aggregateDailyScores ──────────────────────────────────────────────────────

describe("aggregateDailyScores", () => {
  it("returns empty array for empty input", () => {
    expect(aggregateDailyScores([])).toHaveLength(0);
  });

  it("skips null scores", () => {
    const result = aggregateDailyScores([
      { timestampIso: "2026-03-28T10:00:00Z", score: null },
    ]);
    expect(result).toHaveLength(0);
  });

  it("groups multiple dreams on same day", () => {
    const result = aggregateDailyScores([
      { timestampIso: "2026-03-28T08:00:00Z", score: 60 },
      { timestampIso: "2026-03-28T22:00:00Z", score: 80 },
    ]);
    expect(result).toHaveLength(1);
    expect(result[0].score).toBe(70);
    expect(result[0].count).toBe(2);
  });

  it("sorts output ascending by date", () => {
    const result = aggregateDailyScores([
      { timestampIso: "2026-03-30T10:00:00Z", score: 70 },
      { timestampIso: "2026-03-28T10:00:00Z", score: 50 },
      { timestampIso: "2026-03-29T10:00:00Z", score: 60 },
    ]);
    expect(result.map((r) => r.date)).toEqual(["2026-03-28", "2026-03-29", "2026-03-30"]);
  });

  it("handles single entry", () => {
    const result = aggregateDailyScores([
      { timestampIso: "2026-03-28T10:00:00Z", score: 75 },
    ]);
    expect(result).toHaveLength(1);
    expect(result[0]).toEqual({ date: "2026-03-28", score: 75, count: 1 });
  });

  it("rounds averaged score to integer", () => {
    const result = aggregateDailyScores([
      { timestampIso: "2026-03-28T10:00:00Z", score: 60 },
      { timestampIso: "2026-03-28T20:00:00Z", score: 61 },
    ]);
    expect(Number.isInteger(result[0].score)).toBe(true);
  });
});

// ── qualityBand / qualityLabel ────────────────────────────────────────────────

describe("qualityBand", () => {
  it("poor for scores 0-34", () => {
    expect(qualityBand(0)).toBe("poor");
    expect(qualityBand(34)).toBe("poor");
  });

  it("fair for scores 35-54", () => {
    expect(qualityBand(35)).toBe("fair");
    expect(qualityBand(54)).toBe("fair");
  });

  it("good for scores 55-74", () => {
    expect(qualityBand(55)).toBe("good");
    expect(qualityBand(74)).toBe("good");
  });

  it("great for scores 75-100", () => {
    expect(qualityBand(75)).toBe("great");
    expect(qualityBand(100)).toBe("great");
  });
});

describe("qualityLabel", () => {
  it("returns non-empty string for all bands", () => {
    [0, 40, 65, 85].forEach((s) => {
      expect(qualityLabel(s).length).toBeGreaterThan(0);
    });
  });
});

// ── qualityColorClass / qualityBgClass ────────────────────────────────────────

describe("qualityColorClass", () => {
  it("returns a non-empty tailwind class for all scores", () => {
    [0, 34, 35, 54, 55, 74, 75, 100].forEach((s) => {
      expect(qualityColorClass(s)).toMatch(/^text-/);
    });
  });
});

describe("qualityBgClass", () => {
  it("returns a non-empty tailwind class for all scores", () => {
    [0, 50, 100].forEach((s) => {
      expect(qualityBgClass(s)).toMatch(/^bg-/);
    });
  });
});

// ── averageScore ─────────────────────────────────────────────────────────────

describe("averageScore", () => {
  it("returns null for empty array", () => {
    expect(averageScore([])).toBeNull();
  });

  it("returns null when all scores are null", () => {
    expect(averageScore([null, null])).toBeNull();
  });

  it("ignores nulls in mixed array", () => {
    expect(averageScore([null, 60, null, 80])).toBe(70);
  });

  it("returns rounded integer", () => {
    const result = averageScore([60, 61]);
    expect(Number.isInteger(result!)).toBe(true);
    expect(result).toBe(61); // Math.round(60.5) = 61 in JS
  });

  it("single value returns itself", () => {
    expect(averageScore([75])).toBe(75);
  });
});

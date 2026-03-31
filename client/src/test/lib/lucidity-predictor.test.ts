import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  computeLucidityPrediction,
  lucidityBand,
  lucidityTrend,
  BAND_LABEL,
  BAND_COLOR,
  BAND_BG,
  type DreamForLucidity,
  type IntentionForLucidity,
} from "@/lib/lucidity-predictor";

// ── date helpers ──────────────────────────────────────────────────────────────

function daysAgo(n: number): string {
  return new Date(Date.now() - n * 24 * 60 * 60 * 1000).toISOString().slice(0, 10);
}

function makeTs(daysAgoN: number): string {
  return daysAgo(daysAgoN) + "T08:00:00.000Z";
}

// ── fixtures ──────────────────────────────────────────────────────────────────

const highLucidDreams: DreamForLucidity[] = [0, 1, 2, 3, 4, 5, 6].map((d) => ({
  lucidityScore: 80,
  timestamp: makeTs(d),
}));

const lowLucidDreams: DreamForLucidity[] = [0, 1, 2, 3, 4, 5, 6].map((d) => ({
  lucidityScore: 10,
  timestamp: makeTs(d),
}));

const todayIntention: IntentionForLucidity = {
  date: daysAgo(0),
  text: "I will become lucid and fly",
};

const noIntentions: IntentionForLucidity[] = [];

// ── lucidityBand ──────────────────────────────────────────────────────────────

describe("lucidityBand", () => {
  it("returns 'optimal' for score >= 75", () => {
    expect(lucidityBand(75)).toBe("optimal");
    expect(lucidityBand(100)).toBe("optimal");
  });

  it("returns 'high' for 50-74", () => {
    expect(lucidityBand(50)).toBe("high");
    expect(lucidityBand(74)).toBe("high");
  });

  it("returns 'moderate' for 25-49", () => {
    expect(lucidityBand(25)).toBe("moderate");
    expect(lucidityBand(49)).toBe("moderate");
  });

  it("returns 'low' for < 25", () => {
    expect(lucidityBand(0)).toBe("low");
    expect(lucidityBand(24)).toBe("low");
  });
});

// ── computeLucidityPrediction ─────────────────────────────────────────────────

describe("computeLucidityPrediction", () => {
  it("returns score 0 for empty dreams and no intention", () => {
    const p = computeLucidityPrediction([], noIntentions);
    expect(p.likelihood).toBe(0);
  });

  it("likelihood is in [0, 100]", () => {
    const p = computeLucidityPrediction(highLucidDreams, [todayIntention]);
    expect(p.likelihood).toBeGreaterThanOrEqual(0);
    expect(p.likelihood).toBeLessThanOrEqual(100);
  });

  it("high lucidity dreams + streak + intention → high or optimal band", () => {
    const p = computeLucidityPrediction(highLucidDreams, [todayIntention]);
    expect(["high", "optimal"]).toContain(p.band);
  });

  it("low lucidity + no intention → low or moderate band", () => {
    const p = computeLucidityPrediction(lowLucidDreams, noIntentions);
    expect(["low", "moderate"]).toContain(p.band);
  });

  it("setting today's intention increases likelihood vs no intention", () => {
    const withIntent    = computeLucidityPrediction(highLucidDreams, [todayIntention]);
    const withoutIntent = computeLucidityPrediction(highLucidDreams, noIntentions);
    expect(withIntent.likelihood).toBeGreaterThan(withoutIntent.likelihood);
  });

  it("returns 4 factors", () => {
    const p = computeLucidityPrediction(highLucidDreams, noIntentions);
    expect(p.factors).toHaveLength(4);
  });

  it("each factor has value in [0, 100]", () => {
    const p = computeLucidityPrediction(highLucidDreams, [todayIntention]);
    for (const f of p.factors) {
      expect(f.value).toBeGreaterThanOrEqual(0);
      expect(f.value).toBeLessThanOrEqual(100);
    }
  });

  it("each factor has non-empty label, tip", () => {
    const p = computeLucidityPrediction(highLucidDreams, noIntentions);
    for (const f of p.factors) {
      expect(f.label.length).toBeGreaterThan(0);
      expect(f.tip.length).toBeGreaterThan(0);
    }
  });

  it("weights sum to 1.0 across all 4 factors", () => {
    const p = computeLucidityPrediction([], noIntentions);
    const total = p.factors.reduce((s, f) => s + f.weight, 0);
    expect(total).toBeCloseTo(1.0);
  });

  it("band matches likelihood", () => {
    const p = computeLucidityPrediction(highLucidDreams, [todayIntention]);
    expect(p.band).toBe(lucidityBand(p.likelihood));
  });

  it("recommendation and summary are non-empty strings", () => {
    const p = computeLucidityPrediction([], noIntentions);
    expect(p.recommendation.length).toBeGreaterThan(0);
    expect(p.summary.length).toBeGreaterThan(0);
  });

  it("null lucidityScore dreams excluded from average", () => {
    const nullScored: DreamForLucidity[] = [0, 1, 2].map((d) => ({
      lucidityScore: null,
      timestamp: makeTs(d),
    }));
    const p = computeLucidityPrediction(nullScored, noIntentions);
    // avg lucidity = 0, but recall rate/streak still contribute
    expect(p.likelihood).toBeGreaterThanOrEqual(0);
  });

  it("intention from yesterday is not counted as today", () => {
    const yesterdayIntent: IntentionForLucidity = {
      date: daysAgo(1),
      text: "Last night's intention",
    };
    const withYesterday = computeLucidityPrediction(highLucidDreams, [yesterdayIntent]);
    const withToday     = computeLucidityPrediction(highLucidDreams, [todayIntention]);
    expect(withToday.likelihood).toBeGreaterThan(withYesterday.likelihood);
  });

  it("presleep factor shows 100 when intention is set today", () => {
    const p = computeLucidityPrediction([], [todayIntention]);
    const intentFactor = p.factors.find((f) => f.label.toLowerCase().includes("intention"))!;
    expect(intentFactor.value).toBe(100);
  });

  it("presleep factor shows 0 when no intention today", () => {
    const p = computeLucidityPrediction([], noIntentions);
    const intentFactor = p.factors.find((f) => f.label.toLowerCase().includes("intention"))!;
    expect(intentFactor.value).toBe(0);
  });

  it("recall streak factor increases with consecutive days", () => {
    const shortStreak: DreamForLucidity[] = [0, 1, 2].map((d) => ({
      lucidityScore: null,
      timestamp: makeTs(d),
    }));
    const longStreak: DreamForLucidity[] = Array.from({ length: 10 }, (_, i) => ({
      lucidityScore: null as null,
      timestamp: makeTs(i),
    }));
    const pShort = computeLucidityPrediction(shortStreak, noIntentions);
    const pLong  = computeLucidityPrediction(longStreak,  noIntentions);
    const streakFactor = (p: typeof pShort) =>
      p.factors.find((f) => f.label.toLowerCase().includes("streak"))!.value;
    expect(streakFactor(pLong)).toBeGreaterThan(streakFactor(pShort));
  });

  it("dreams older than 7 days don't affect recall rate", () => {
    const oldDreams: DreamForLucidity[] = [10, 11, 12, 13, 14].map((d) => ({
      lucidityScore: 80,
      timestamp: makeTs(d),
    }));
    const p = computeLucidityPrediction(oldDreams, noIntentions);
    const recallFactor = p.factors.find((f) => f.label.toLowerCase().includes("recall rate"))!;
    expect(recallFactor.value).toBe(0);
  });
});

// ── lucidityTrend ─────────────────────────────────────────────────────────────

describe("lucidityTrend", () => {
  it("returns empty array for empty input", () => {
    expect(lucidityTrend([])).toHaveLength(0);
  });

  it("returns one entry per unique day with lucidity scores", () => {
    const dreams: DreamForLucidity[] = [
      { lucidityScore: 60, timestamp: makeTs(0) },
      { lucidityScore: 40, timestamp: makeTs(1) },
    ];
    expect(lucidityTrend(dreams, 14)).toHaveLength(2);
  });

  it("averages multiple dreams on same day", () => {
    const dreams: DreamForLucidity[] = [
      { lucidityScore: 60, timestamp: daysAgo(0) + "T06:00:00Z" },
      { lucidityScore: 80, timestamp: daysAgo(0) + "T10:00:00Z" },
    ];
    const trend = lucidityTrend(dreams, 7);
    expect(trend[0].avgScore).toBe(70);
  });

  it("entries are sorted oldest first", () => {
    const dreams: DreamForLucidity[] = [0, 1, 2, 3].map((d) => ({
      lucidityScore: 50,
      timestamp: makeTs(d),
    }));
    const trend = lucidityTrend(dreams, 7);
    for (let i = 1; i < trend.length; i++) {
      expect(trend[i].date >= trend[i - 1].date).toBe(true);
    }
  });

  it("excludes null lucidityScore entries", () => {
    const dreams: DreamForLucidity[] = [
      { lucidityScore: null, timestamp: makeTs(0) },
      { lucidityScore: 70,   timestamp: makeTs(1) },
    ];
    const trend = lucidityTrend(dreams, 7);
    expect(trend).toHaveLength(1);
    expect(trend[0].avgScore).toBe(70);
  });

  it("excludes dreams outside the window", () => {
    const dreams: DreamForLucidity[] = [
      { lucidityScore: 80, timestamp: makeTs(20) }, // too old
      { lucidityScore: 50, timestamp: makeTs(1) },
    ];
    const trend = lucidityTrend(dreams, 7);
    expect(trend).toHaveLength(1);
  });
});

// ── constants ─────────────────────────────────────────────────────────────────

describe("band constants", () => {
  const bands = ["low", "moderate", "high", "optimal"] as const;

  it("BAND_LABEL has entry for every band", () => {
    for (const b of bands) expect(typeof BAND_LABEL[b]).toBe("string");
  });

  it("BAND_COLOR has entry for every band", () => {
    for (const b of bands) expect(typeof BAND_COLOR[b]).toBe("string");
  });

  it("BAND_BG has entry for every band", () => {
    for (const b of bands) expect(typeof BAND_BG[b]).toBe("string");
  });
});

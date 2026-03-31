import { describe, it, expect } from "vitest";
import {
  scoreArcValence,
  arcLabel,
  ARC_LABEL_COLOR,
  ARC_LABEL_BG,
  aggregateArcTrend,
  extractArcPattern,
  topArcPatterns,
  computeArcSummary,
  type DreamForArc,
  type ArcLabel,
} from "@/lib/emotional-arc-tracker";

// ── fixtures ──────────────────────────────────────────────────────────────────

const makeDream = (emotionalArc: string | null, date = "2026-03-28"): DreamForArc => ({
  emotionalArc,
  timestamp: `${date}T08:00:00Z`,
});

const POSITIVE_ARC = "anxious at first then resolving into peaceful wonder";
const NEGATIVE_ARC = "escalating fear and helpless terror throughout";
const NEUTRAL_ARC  = "neutral progression with no clear change";

// ── scoreArcValence ───────────────────────────────────────────────────────────

describe("scoreArcValence", () => {
  it("returns null for null input", () => {
    expect(scoreArcValence(null)).toBeNull();
  });

  it("returns null for undefined input", () => {
    expect(scoreArcValence(undefined)).toBeNull();
  });

  it("returns null for blank string", () => {
    expect(scoreArcValence("   ")).toBeNull();
  });

  it("returns positive value for clearly positive arc", () => {
    const v = scoreArcValence(POSITIVE_ARC);
    expect(v).not.toBeNull();
    expect(v!).toBeGreaterThan(0);
  });

  it("returns negative value for clearly negative arc", () => {
    const v = scoreArcValence(NEGATIVE_ARC);
    expect(v).not.toBeNull();
    expect(v!).toBeLessThan(0);
  });

  it("returns 0 for text with no keywords", () => {
    expect(scoreArcValence("the dream progressed and then changed")).toBe(0);
  });

  it("result is always in [-1, 1]", () => {
    const arcs = [POSITIVE_ARC, NEGATIVE_ARC, NEUTRAL_ARC, "total bliss and peace joy calm"];
    for (const arc of arcs) {
      const v = scoreArcValence(arc);
      if (v !== null) {
        expect(v).toBeGreaterThanOrEqual(-1);
        expect(v).toBeLessThanOrEqual(1);
      }
    }
  });

  it("is case-insensitive", () => {
    const lower = scoreArcValence("escalating fear");
    const upper = scoreArcValence("ESCALATING FEAR");
    expect(lower).not.toBeNull();
    expect(upper).not.toBeNull();
    expect(lower).toBeCloseTo(upper!);
  });
});

// ── arcLabel ──────────────────────────────────────────────────────────────────

describe("arcLabel", () => {
  it("labels strong negatives as distressing", () => {
    expect(arcLabel(-0.8)).toBe("distressing");
  });

  it("labels mild negatives as tense", () => {
    expect(arcLabel(-0.2)).toBe("tense");
  });

  it("labels near-zero as mixed", () => {
    expect(arcLabel(0)).toBe("mixed");
    expect(arcLabel(0.05)).toBe("mixed");
    expect(arcLabel(-0.05)).toBe("mixed");
  });

  it("labels mild positives as neutral", () => {
    expect(arcLabel(0.25)).toBe("neutral");
  });

  it("labels strong positives as uplifting", () => {
    expect(arcLabel(0.7)).toBe("uplifting");
  });

  it("ARC_LABEL_COLOR has entry for every label", () => {
    const labels: ArcLabel[] = ["distressing", "tense", "mixed", "neutral", "uplifting"];
    for (const l of labels) {
      expect(typeof ARC_LABEL_COLOR[l]).toBe("string");
    }
  });

  it("ARC_LABEL_BG has entry for every label", () => {
    const labels: ArcLabel[] = ["distressing", "tense", "mixed", "neutral", "uplifting"];
    for (const l of labels) {
      expect(typeof ARC_LABEL_BG[l]).toBe("string");
    }
  });
});

// ── aggregateArcTrend ─────────────────────────────────────────────────────────

describe("aggregateArcTrend", () => {
  it("returns empty for no dreams", () => {
    expect(aggregateArcTrend([])).toHaveLength(0);
  });

  it("skips dreams with null arc", () => {
    const dreams = [makeDream(null), makeDream(null)];
    expect(aggregateArcTrend(dreams)).toHaveLength(0);
  });

  it("groups multiple dreams on the same day", () => {
    const dreams = [
      makeDream(POSITIVE_ARC, "2026-03-28"),
      makeDream(NEGATIVE_ARC, "2026-03-28"),
    ];
    const trend = aggregateArcTrend(dreams);
    expect(trend).toHaveLength(1);
    expect(trend[0].dreamCount).toBe(2);
  });

  it("produces separate points for different days", () => {
    const dreams = [
      makeDream(POSITIVE_ARC, "2026-03-28"),
      makeDream(NEGATIVE_ARC, "2026-03-27"),
    ];
    const trend = aggregateArcTrend(dreams);
    expect(trend).toHaveLength(2);
  });

  it("sorts results oldest-first", () => {
    const dreams = [
      makeDream(POSITIVE_ARC, "2026-03-29"),
      makeDream(POSITIVE_ARC, "2026-03-27"),
      makeDream(POSITIVE_ARC, "2026-03-28"),
    ];
    const trend = aggregateArcTrend(dreams);
    expect(trend[0].date).toBe("2026-03-27");
    expect(trend[2].date).toBe("2026-03-29");
  });

  it("avgValence is average of daily scores", () => {
    // Both positive — avg should be positive
    const dreams = [
      makeDream(POSITIVE_ARC, "2026-03-28"),
      makeDream("wonder and resolving peace", "2026-03-28"),
    ];
    const [pt] = aggregateArcTrend(dreams);
    expect(pt.avgValence).toBeGreaterThan(0);
  });

  it("date field is YYYY-MM-DD format", () => {
    const [pt] = aggregateArcTrend([makeDream(POSITIVE_ARC, "2026-03-28")]);
    expect(pt.date).toBe("2026-03-28");
  });
});

// ── extractArcPattern ─────────────────────────────────────────────────────────

describe("extractArcPattern", () => {
  it("lowercases the result", () => {
    expect(extractArcPattern("ESCALATING FEAR")).toBe(extractArcPattern("escalating fear"));
  });

  it("strips leading 'starts'", () => {
    const p = extractArcPattern("starts anxious then resolving");
    expect(p).not.toMatch(/^starts/);
    expect(p).toContain("anxious");
  });

  it("strips leading 'begins'", () => {
    const p = extractArcPattern("begins with fear");
    expect(p).not.toMatch(/^begins/);
  });

  it("limits to 5 words", () => {
    const words = extractArcPattern("one two three four five six seven").split(" ");
    expect(words.length).toBeLessThanOrEqual(5);
  });

  it("returns non-empty string for normal arc", () => {
    expect(extractArcPattern(POSITIVE_ARC).length).toBeGreaterThan(0);
  });
});

// ── topArcPatterns ────────────────────────────────────────────────────────────

describe("topArcPatterns", () => {
  it("returns empty for no dreams", () => {
    expect(topArcPatterns([])).toHaveLength(0);
  });

  it("skips dreams with null arc", () => {
    expect(topArcPatterns([makeDream(null)])).toHaveLength(0);
  });

  it("returns at most topN entries (default 5)", () => {
    const dreams = Array.from({ length: 10 }, (_, i) =>
      makeDream(`unique pattern number ${i} here`),
    );
    expect(topArcPatterns(dreams)).toHaveLength(5);
  });

  it("respects custom topN", () => {
    const dreams = [
      makeDream("escalating fear"),
      makeDream("escalating fear"),
      makeDream("peaceful wonder"),
    ];
    expect(topArcPatterns(dreams, 1)).toHaveLength(1);
    expect(topArcPatterns(dreams, 1)[0].pattern).toContain("escalating fear");
  });

  it("sorts by frequency descending", () => {
    const dreams = [
      makeDream("escalating fear"),
      makeDream("escalating fear"),
      makeDream("resolving peace"),
    ];
    const top = topArcPatterns(dreams);
    expect(top[0].count).toBeGreaterThanOrEqual(top[1].count);
  });
});

// ── computeArcSummary ─────────────────────────────────────────────────────────

describe("computeArcSummary", () => {
  it("returns null meanValence for empty input", () => {
    expect(computeArcSummary([])).toMatchObject({ meanValence: null, arcCount: 0 });
  });

  it("returns null meanValence when all arcs are null", () => {
    expect(computeArcSummary([makeDream(null)])).toMatchObject({ meanValence: null });
  });

  it("positive arc raises positiveRate", () => {
    const s = computeArcSummary([makeDream(POSITIVE_ARC)]);
    expect(s.positiveRate).toBeGreaterThan(0);
    expect(s.arcCount).toBe(1);
  });

  it("negative arc raises negativeRate", () => {
    const s = computeArcSummary([makeDream(NEGATIVE_ARC)]);
    expect(s.negativeRate).toBeGreaterThan(0);
  });

  it("rates are between 0 and 1", () => {
    const dreams = [makeDream(POSITIVE_ARC), makeDream(NEGATIVE_ARC), makeDream(NEUTRAL_ARC)];
    const s = computeArcSummary(dreams);
    expect(s.positiveRate).toBeGreaterThanOrEqual(0);
    expect(s.positiveRate).toBeLessThanOrEqual(1);
    expect(s.negativeRate).toBeGreaterThanOrEqual(0);
    expect(s.negativeRate).toBeLessThanOrEqual(1);
  });
});

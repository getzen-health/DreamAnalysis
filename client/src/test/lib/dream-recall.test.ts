import { describe, it, expect } from "vitest";
import {
  buildRecallCalendar,
  computeRecallStreak,
  computeRecallRate,
  recallWeeklyTrend,
  recallTrendDirection,
  shortDate,
  recallCellClass,
} from "@/lib/dream-recall";

// ── helpers ───────────────────────────────────────────────────────────────────

function daysAgo(n: number): string {
  const d = new Date();
  d.setDate(d.getDate() - n);
  return d.toISOString().slice(0, 10) + "T08:00:00Z";
}

const today    = daysAgo(0);
const yday     = daysAgo(1);
const twoDays  = daysAgo(2);
const threeDays = daysAgo(3);

const make = (iso: string) => ({ timestamp: iso });

// ── buildRecallCalendar ───────────────────────────────────────────────────────

describe("buildRecallCalendar", () => {
  it("returns exactly `days` entries", () => {
    expect(buildRecallCalendar([], 14)).toHaveLength(14);
    expect(buildRecallCalendar([], 7)).toHaveLength(7);
  });

  it("oldest entry is first", () => {
    const cal = buildRecallCalendar([], 7);
    expect(cal[0].date < cal[cal.length - 1].date).toBe(true);
  });

  it("last entry is today", () => {
    const cal = buildRecallCalendar([], 7);
    const todayStr = new Date().toISOString().slice(0, 10);
    expect(cal[cal.length - 1].date).toBe(todayStr);
    expect(cal[cal.length - 1].isToday).toBe(true);
  });

  it("count=0 for days with no dreams", () => {
    const cal = buildRecallCalendar([], 7);
    expect(cal.every((d) => d.count === 0)).toBe(true);
  });

  it("count=1 for day with one dream", () => {
    const cal = buildRecallCalendar([make(today)], 7);
    const todayEntry = cal.find((d) => d.isToday);
    expect(todayEntry?.count).toBe(1);
  });

  it("count=2 when two dreams on same day", () => {
    const cal = buildRecallCalendar([make(today), make(today)], 7);
    const todayEntry = cal.find((d) => d.isToday);
    expect(todayEntry?.count).toBe(2);
  });

  it("no entry is isFuture", () => {
    const cal = buildRecallCalendar([], 14);
    expect(cal.every((d) => !d.isFuture)).toBe(true);
  });
});

// ── computeRecallStreak ───────────────────────────────────────────────────────

describe("computeRecallStreak", () => {
  it("returns 0 for no dreams", () => {
    expect(computeRecallStreak([])).toBe(0);
  });

  it("returns 1 for only today's dream", () => {
    expect(computeRecallStreak([make(today)])).toBe(1);
  });

  it("returns 1 for only yesterday's dream", () => {
    expect(computeRecallStreak([make(yday)])).toBe(1);
  });

  it("returns 2 for today + yesterday", () => {
    expect(computeRecallStreak([make(today), make(yday)])).toBe(2);
  });

  it("streak breaks on gap", () => {
    // today + 3 days ago — gap on day -2 breaks it
    expect(computeRecallStreak([make(today), make(threeDays)])).toBe(1);
  });

  it("returns 3 for three consecutive days", () => {
    const dreams = [make(today), make(yday), make(twoDays)];
    expect(computeRecallStreak(dreams)).toBe(3);
  });

  it("counts multiple dreams on same day as one streak day", () => {
    const dreams = [make(today), make(today), make(yday)];
    expect(computeRecallStreak(dreams)).toBe(2);
  });
});

// ── computeRecallRate ─────────────────────────────────────────────────────────

describe("computeRecallRate", () => {
  it("returns 0 for no dreams", () => {
    expect(computeRecallRate([])).toBe(0);
  });

  it("returns 1 when every day in window has a dream", () => {
    const dreams = [0, 1, 2, 3, 4, 5, 6].map((n) => make(daysAgo(n)));
    expect(computeRecallRate(dreams, 7)).toBe(1);
  });

  it("returns 0.5 for half the days", () => {
    const dreams = [0, 2, 4].map((n) => make(daysAgo(n)));
    const rate = computeRecallRate(dreams, 6);
    // 3 out of 6 days → 0.5
    expect(rate).toBeCloseTo(0.5, 1);
  });

  it("result is always in [0, 1]", () => {
    const r = computeRecallRate([make(today), make(yday)], 7);
    expect(r).toBeGreaterThanOrEqual(0);
    expect(r).toBeLessThanOrEqual(1);
  });
});

// ── recallWeeklyTrend ─────────────────────────────────────────────────────────

describe("recallWeeklyTrend", () => {
  it("returns at most `weeks` entries", () => {
    const trend = recallWeeklyTrend([], 4);
    expect(trend.length).toBeLessThanOrEqual(4);
  });

  it("each point has rate in [0, 1]", () => {
    const dreams = [make(today), make(yday)];
    const trend = recallWeeklyTrend(dreams, 2);
    for (const pt of trend) {
      expect(pt.rate).toBeGreaterThanOrEqual(0);
      expect(pt.rate).toBeLessThanOrEqual(1);
    }
  });

  it("dreamDays <= totalDays for every point", () => {
    const dreams = [make(today)];
    for (const pt of recallWeeklyTrend(dreams, 3)) {
      expect(pt.dreamDays).toBeLessThanOrEqual(pt.totalDays);
    }
  });

  it("weekStart is a Monday (UTCDay = 1)", () => {
    for (const pt of recallWeeklyTrend([], 4)) {
      const d = new Date(pt.weekStart + "T12:00:00Z");
      expect(d.getUTCDay()).toBe(1); // 1 = Monday
    }
  });

  it("oldest week is first", () => {
    const trend = recallWeeklyTrend([], 3);
    if (trend.length >= 2) {
      expect(trend[0].weekStart <= trend[1].weekStart).toBe(true);
    }
  });
});

// ── recallTrendDirection ──────────────────────────────────────────────────────

describe("recallTrendDirection", () => {
  it("returns insufficient when no data at all", () => {
    // Empty dreams → recallWeeklyTrend returns 0 points with totalDays>0
    // that aren't enough to compare two weeks → insufficient
    expect(recallTrendDirection([])).toBe("insufficient");
  });

  it("returns a valid trend label", () => {
    const dreams = Array.from({ length: 14 }, (_, i) => make(daysAgo(i)));
    const dir = recallTrendDirection(dreams);
    expect(["improving", "stable", "declining", "insufficient"]).toContain(dir);
  });
});

// ── shortDate ─────────────────────────────────────────────────────────────────

describe("shortDate", () => {
  it("formats 2026-03-28 as Mar 28", () => {
    expect(shortDate("2026-03-28")).toBe("Mar 28");
  });

  it("formats 2026-01-01 as Jan 1", () => {
    expect(shortDate("2026-01-01")).toBe("Jan 1");
  });
});

// ── recallCellClass ───────────────────────────────────────────────────────────

describe("recallCellClass", () => {
  it("returns a string for count=0", () => {
    expect(typeof recallCellClass(0, false)).toBe("string");
  });

  it("today with no dreams gets a ring class", () => {
    expect(recallCellClass(0, true)).toContain("ring");
  });

  it("higher counts return progressively more opaque classes", () => {
    const c1 = recallCellClass(1, false);
    const c2 = recallCellClass(2, false);
    const c3 = recallCellClass(3, false);
    // Each should be distinct
    expect(c1).not.toBe(c2);
    expect(c2).not.toBe(c3);
  });

  it("returns a class string for all reasonable counts", () => {
    for (const n of [0, 1, 2, 3, 5, 10]) {
      expect(typeof recallCellClass(n, false)).toBe("string");
    }
  });
});

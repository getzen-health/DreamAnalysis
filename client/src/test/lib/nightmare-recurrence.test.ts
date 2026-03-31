import { describe, it, expect } from "vitest";
import {
  computeNightmareTrend,
  trendLabel,
  irtEffectivenessLabel,
  shouldSuggestIrt,
  formatShortDate,
  type NightmareRecurrenceData,
} from "@/lib/nightmare-recurrence";

// ── computeNightmareTrend ─────────────────────────────────────────────────────

describe("computeNightmareTrend", () => {
  it("returns unknown when total < 2", () => {
    expect(computeNightmareTrend(0, 0)).toBe("unknown");
    expect(computeNightmareTrend(1, 0)).toBe("unknown");
    expect(computeNightmareTrend(0, 1)).toBe("unknown");
  });

  it("returns improving when recent < older", () => {
    expect(computeNightmareTrend(1, 3)).toBe("improving");
    expect(computeNightmareTrend(0, 2)).toBe("improving");
  });

  it("returns worsening when recent > older", () => {
    expect(computeNightmareTrend(4, 1)).toBe("worsening");
    expect(computeNightmareTrend(3, 0)).toBe("worsening");
  });

  it("returns stable when recent === older and total >= 2", () => {
    expect(computeNightmareTrend(2, 2)).toBe("stable");
    expect(computeNightmareTrend(1, 1)).toBe("stable");
  });

  it("exactly 2 total splits 1/1 = stable", () => {
    expect(computeNightmareTrend(1, 1)).toBe("stable");
  });
});

// ── trendLabel ────────────────────────────────────────────────────────────────

describe("trendLabel", () => {
  it("returns a non-empty string for every trend value", () => {
    const trends = ["improving", "worsening", "stable", "unknown"] as const;
    for (const t of trends) {
      expect(trendLabel(t).length).toBeGreaterThan(0);
    }
  });

  it("improving label mentions decreasing", () => {
    expect(trendLabel("improving").toLowerCase()).toMatch(/decreas/);
  });

  it("worsening label mentions increasing", () => {
    expect(trendLabel("worsening").toLowerCase()).toMatch(/increas/);
  });
});

// ── irtEffectivenessLabel ─────────────────────────────────────────────────────

describe("irtEffectivenessLabel", () => {
  it("returns empty string when no IRT sessions", () => {
    expect(irtEffectivenessLabel(0, 0)).toBe("");
    expect(irtEffectivenessLabel(5, 0)).toBe("");
  });

  it("returns success message when postIrt = 0 and IRT sessions > 0", () => {
    const msg = irtEffectivenessLabel(0, 2);
    expect(msg.toLowerCase()).toMatch(/no nightmare/);
  });

  it("returns singular for postIrt = 1", () => {
    const msg = irtEffectivenessLabel(1, 1);
    expect(msg).toMatch(/1 nightmare/);
    expect(msg).not.toMatch(/nightmares/);
  });

  it("returns plural for postIrt > 1", () => {
    const msg = irtEffectivenessLabel(3, 1);
    expect(msg).toMatch(/3 nightmares/);
  });

  it("always mentions 'since' for non-zero postIrt", () => {
    expect(irtEffectivenessLabel(2, 1).toLowerCase()).toContain("since");
  });
});

// ── shouldSuggestIrt ──────────────────────────────────────────────────────────

const makeData = (overrides: Partial<NightmareRecurrenceData>): NightmareRecurrenceData => ({
  recentNightmares: 0,
  olderNightmares: 0,
  trend: "unknown",
  lastNightmareDate: null,
  irtSessionCount: 0,
  lastIrtDate: null,
  postIrtNightmares: 0,
  ...overrides,
});

describe("shouldSuggestIrt", () => {
  it("returns false when recentNightmares = 0 (no active nightmares)", () => {
    expect(shouldSuggestIrt(makeData({ recentNightmares: 0, trend: "worsening" }))).toBe(false);
  });

  it("returns true when postIrtNightmares > 0 (nightmare recurred after IRT)", () => {
    expect(shouldSuggestIrt(makeData({ recentNightmares: 1, postIrtNightmares: 1, irtSessionCount: 1 }))).toBe(true);
  });

  it("returns true for worsening trend with recent nightmares", () => {
    expect(shouldSuggestIrt(makeData({ recentNightmares: 2, trend: "worsening" }))).toBe(true);
  });

  it("returns true for stable trend with recent nightmares", () => {
    expect(shouldSuggestIrt(makeData({ recentNightmares: 1, trend: "stable" }))).toBe(true);
  });

  it("returns true for unknown trend with recent nightmares", () => {
    expect(shouldSuggestIrt(makeData({ recentNightmares: 1, trend: "unknown" }))).toBe(true);
  });

  it("returns false for improving trend when recentNightmares > 0 and postIrt = 0", () => {
    expect(shouldSuggestIrt(makeData({ recentNightmares: 1, trend: "improving", postIrtNightmares: 0 }))).toBe(false);
  });
});

// ── formatShortDate ───────────────────────────────────────────────────────────

describe("formatShortDate", () => {
  it("returns empty string for null", () => {
    expect(formatShortDate(null)).toBe("");
  });

  it("returns empty string for invalid date string", () => {
    expect(formatShortDate("not-a-date")).toBe("");
  });

  it("formats a valid ISO date to month + day", () => {
    const result = formatShortDate("2026-03-28T10:00:00.000Z");
    // Should include month abbreviation and day number
    expect(result).toMatch(/Mar/);
    expect(result).toMatch(/28/);
  });

  it("handles midnight UTC boundary", () => {
    const result = formatShortDate("2026-01-01T00:00:00.000Z");
    expect(result.length).toBeGreaterThan(0);
  });
});

import React from "react";
import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import {
  EEGWeekCompareCard,
  computeWeekComparison,
  buildSparkline,
} from "@/components/eeg-week-compare-card";
import type { StoredEmotionReading } from "@/lib/ml-api";

// ─── Helpers ─────────────────────────────────────────────────────────────────

const MS = (daysAgo: number) => new Date(Date.now() - daysAgo * 86_400_000).toISOString();

function makeReading(daysAgo: number, focus = 0.7, stress = 0.3, happiness = 0.6): StoredEmotionReading {
  return {
    id: `r-${daysAgo}`,
    userId: "test",
    sessionId: null,
    stress,
    happiness,
    focus,
    energy: 0.5,
    dominantEmotion: "neutral",
    valence: null,
    arousal: null,
    timestamp: MS(daysAgo),
  };
}

// ─── computeWeekComparison ────────────────────────────────────────────────────

describe("computeWeekComparison", () => {
  it("returns null for both when history is empty", () => {
    const result = computeWeekComparison([], "focus");
    expect(result.thisWeek).toBeNull();
    expect(result.lastWeek).toBeNull();
  });

  it("correctly splits readings into this week and last week", () => {
    const history = [
      makeReading(1, 0.8),  // this week
      makeReading(3, 0.6),  // this week
      makeReading(8, 0.4),  // last week
      makeReading(12, 0.5), // last week
    ];
    const { thisWeek, lastWeek } = computeWeekComparison(history, "focus");
    expect(thisWeek).toBeCloseTo(0.7, 5); // (0.8 + 0.6) / 2
    expect(lastWeek).toBeCloseTo(0.45, 5); // (0.4 + 0.5) / 2
  });

  it("returns null for lastWeek when no readings older than 7 days", () => {
    const history = [makeReading(1, 0.8), makeReading(3, 0.6)];
    const { thisWeek, lastWeek } = computeWeekComparison(history, "focus");
    expect(thisWeek).not.toBeNull();
    expect(lastWeek).toBeNull();
  });
});

// ─── buildSparkline ───────────────────────────────────────────────────────────

describe("buildSparkline", () => {
  it("returns 7 slots", () => {
    const history = [makeReading(1, 0.7), makeReading(3, 0.5)];
    expect(buildSparkline(history, "focus")).toHaveLength(7);
  });

  it("fills correct slots based on day offset", () => {
    const history = [makeReading(0, 0.8)]; // today
    const sparkline = buildSparkline(history, "focus");
    expect(sparkline[6]).toBeCloseTo(0.8, 3); // today = slot 6
  });

  it("returns all nulls for empty history", () => {
    const sparkline = buildSparkline([], "focus");
    expect(sparkline.every((v) => v === null)).toBe(true);
  });
});

// ─── EEGWeekCompareCard ───────────────────────────────────────────────────────

describe("EEGWeekCompareCard", () => {
  it("renders without crashing", () => {
    renderWithProviders(<EEGWeekCompareCard history={[]} />);
    expect(screen.getByTestId("eeg-week-compare-card")).toBeInTheDocument();
  });

  it("shows empty state when history is empty", () => {
    renderWithProviders(<EEGWeekCompareCard history={[]} />);
    expect(screen.getByTestId("eeg-week-compare-empty")).toBeInTheDocument();
    expect(screen.getByText(/complete EEG sessions/i)).toBeInTheDocument();
  });

  it("shows metric labels when data is present", () => {
    const history = [makeReading(1), makeReading(2), makeReading(3)];
    renderWithProviders(<EEGWeekCompareCard history={history} />);
    expect(screen.getByText("Focus")).toBeInTheDocument();
    expect(screen.getByText("Stress")).toBeInTheDocument();
    expect(screen.getByText("Mood")).toBeInTheDocument();
  });

  it("shows Brain Trends heading", () => {
    renderWithProviders(<EEGWeekCompareCard history={[makeReading(1)]} />);
    expect(screen.getByText("Brain Trends")).toBeInTheDocument();
  });

  it("shows 7-day badge", () => {
    renderWithProviders(<EEGWeekCompareCard history={[makeReading(1)]} />);
    expect(screen.getByText("7-day")).toBeInTheDocument();
  });

  it("shows vs comparison when last week data is available", () => {
    const history = [
      makeReading(1, 0.8),  // this week
      makeReading(8, 0.5),  // last week
    ];
    renderWithProviders(<EEGWeekCompareCard history={history} />);
    expect(screen.getAllByText(/vs/i).length).toBeGreaterThanOrEqual(1);
  });
});

import React from "react";
import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import {
  EEGPeakHours,
  computeHourlyFocus,
  topPeakHours,
} from "@/components/eeg-peak-hours";
import type { StoredEmotionReading } from "@/lib/ml-api";

// ─── Helpers ─────────────────────────────────────────────────────────────────

function makeReading(hour: number, focus = 0.7): StoredEmotionReading {
  const d = new Date();
  d.setHours(hour, 0, 0, 0);
  return {
    id: `r-${hour}`,
    userId: "test",
    sessionId: null,
    stress: 0.3,
    happiness: 0.6,
    focus,
    energy: 0.5,
    dominantEmotion: "neutral",
    valence: null,
    arousal: null,
    timestamp: d.toISOString(),
  };
}

// ─── computeHourlyFocus ───────────────────────────────────────────────────────

describe("computeHourlyFocus", () => {
  it("returns 24 buckets", () => {
    expect(computeHourlyFocus([])).toHaveLength(24);
  });

  it("all nulls for empty history", () => {
    const hourly = computeHourlyFocus([]);
    expect(hourly.every((h) => h.avg === null && h.count === 0)).toBe(true);
  });

  it("averages readings in the correct hour bucket", () => {
    const history = [makeReading(10, 0.8), makeReading(10, 0.6)];
    const hourly = computeHourlyFocus(history);
    expect(hourly[10].avg).toBeCloseTo(0.7, 5);
    expect(hourly[10].count).toBe(2);
  });

  it("places a reading in the correct hour", () => {
    const history = [makeReading(9, 0.9)];
    const hourly = computeHourlyFocus(history);
    expect(hourly[9].avg).toBeCloseTo(0.9, 5);
    expect(hourly[11].avg).toBeNull();
  });
});

// ─── topPeakHours ────────────────────────────────────────────────────────────

describe("topPeakHours", () => {
  it("returns empty array when no data", () => {
    const hourly = Array.from({ length: 24 }, () => ({ avg: null }));
    expect(topPeakHours(hourly)).toHaveLength(0);
  });

  it("returns the correct top-N hours", () => {
    const hourly: { avg: number | null }[] = Array.from({ length: 24 }, () => ({
      avg: null,
    }));
    hourly[10] = { avg: 0.9 };
    hourly[14] = { avg: 0.7 };
    hourly[9]  = { avg: 0.8 };
    const peaks = topPeakHours(hourly, 2);
    expect(peaks).toContain(10);
    expect(peaks).toContain(9);
    expect(peaks).toHaveLength(2);
  });

  it("caps at N even when more hours have data", () => {
    const hourly: { avg: number | null }[] = Array.from({ length: 24 }, (_, i) => ({
      avg: i * 0.04,
    }));
    expect(topPeakHours(hourly, 3)).toHaveLength(3);
  });
});

// ─── EEGPeakHours ─────────────────────────────────────────────────────────────

describe("EEGPeakHours", () => {
  it("renders without crashing", () => {
    renderWithProviders(<EEGPeakHours history={[]} />);
    expect(screen.getByTestId("eeg-peak-hours")).toBeInTheDocument();
  });

  it("shows empty state when history is empty", () => {
    renderWithProviders(<EEGPeakHours history={[]} />);
    expect(screen.getByTestId("eeg-peak-hours-empty")).toBeInTheDocument();
    expect(screen.getByText(/EEG session data/i)).toBeInTheDocument();
  });

  it("shows heading when rendered", () => {
    renderWithProviders(<EEGPeakHours history={[makeReading(10, 0.8)]} />);
    expect(screen.getByText("Peak Focus Hours")).toBeInTheDocument();
  });

  it("shows EEG badge", () => {
    renderWithProviders(<EEGPeakHours history={[makeReading(10, 0.8)]} />);
    expect(screen.getByText("EEG")).toBeInTheDocument();
  });

  it("shows best hours label when data is present", () => {
    renderWithProviders(<EEGPeakHours history={[makeReading(10, 0.9), makeReading(14, 0.7)]} />);
    expect(screen.getByText(/best hours/i)).toBeInTheDocument();
  });

  it("does not show empty state when history has data", () => {
    renderWithProviders(<EEGPeakHours history={[makeReading(9, 0.8)]} />);
    expect(screen.queryByTestId("eeg-peak-hours-empty")).not.toBeInTheDocument();
  });
});

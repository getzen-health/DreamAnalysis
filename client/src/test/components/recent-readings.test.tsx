import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen, act, cleanup } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import { RecentReadings, formatTimeAgo } from "@/components/recent-readings";

beforeEach(() => {
  localStorage.clear();
});

afterEach(() => {
  cleanup();
  localStorage.clear();
});

describe("formatTimeAgo", () => {
  it("returns 'just now' for timestamps less than 1 minute ago", () => {
    expect(formatTimeAgo(Date.now())).toBe("just now");
  });

  it("returns minutes for timestamps less than 1 hour ago", () => {
    const fiveMinAgo = Date.now() - 5 * 60_000;
    expect(formatTimeAgo(fiveMinAgo)).toBe("5m ago");
  });

  it("returns hours for timestamps less than 1 day ago", () => {
    const threeHoursAgo = Date.now() - 3 * 3_600_000;
    expect(formatTimeAgo(threeHoursAgo)).toBe("3h ago");
  });

  it("returns 'yesterday' for timestamps 1 day ago", () => {
    const oneDayAgo = Date.now() - 25 * 3_600_000;
    expect(formatTimeAgo(oneDayAgo)).toBe("yesterday");
  });

  it("returns days for timestamps 2-6 days ago", () => {
    const threeDaysAgo = Date.now() - 3 * 86_400_000;
    expect(formatTimeAgo(threeDaysAgo)).toBe("3d ago");
  });

  it("returns empty string for null/undefined", () => {
    expect(formatTimeAgo(null)).toBe("");
    expect(formatTimeAgo(undefined)).toBe("");
  });

  it("handles ISO date strings", () => {
    const recent = new Date(Date.now() - 2 * 60_000).toISOString();
    expect(formatTimeAgo(recent)).toBe("2m ago");
  });

  it("handles seconds-based timestamps (auto-converts to ms)", () => {
    const nowSeconds = Math.floor(Date.now() / 1000);
    expect(formatTimeAgo(nowSeconds)).toBe("just now");
  });
});

describe("RecentReadings component", () => {
  it("renders nothing when localStorage is empty and no emptyMessage", () => {
    const { container } = renderWithProviders(
      <RecentReadings
        storageKey="ndw_test_readings"
        title="Recent Tests"
        renderEntry={(entry) => <span>{entry.label}</span>}
      />,
    );
    expect(container.innerHTML).toBe("");
  });

  it("renders emptyMessage when localStorage is empty and emptyMessage is provided", () => {
    renderWithProviders(
      <RecentReadings
        storageKey="ndw_test_readings"
        title="Recent Tests"
        renderEntry={(entry) => <span>{entry.label}</span>}
        emptyMessage="No readings yet"
      />,
    );
    expect(screen.getByText("No readings yet")).toBeInTheDocument();
    expect(screen.getByTestId("recent-readings-empty")).toBeInTheDocument();
  });

  it("renders entries from localStorage", () => {
    localStorage.setItem(
      "ndw_test_readings",
      JSON.stringify([
        { id: "1", label: "First Reading", loggedAt: new Date().toISOString() },
        { id: "2", label: "Second Reading", loggedAt: new Date().toISOString() },
      ]),
    );

    renderWithProviders(
      <RecentReadings
        storageKey="ndw_test_readings"
        title="Recent Tests"
        renderEntry={(entry) => <span>{entry.label}</span>}
      />,
    );

    expect(screen.getByText("First Reading")).toBeInTheDocument();
    expect(screen.getByText("Second Reading")).toBeInTheDocument();
    expect(screen.getByText("Recent Tests")).toBeInTheDocument();
  });

  it("respects maxEntries limit", () => {
    const entries = Array.from({ length: 10 }, (_, i) => ({
      id: String(i),
      label: `Entry ${i}`,
      loggedAt: new Date().toISOString(),
    }));
    localStorage.setItem("ndw_test_readings", JSON.stringify(entries));

    renderWithProviders(
      <RecentReadings
        storageKey="ndw_test_readings"
        title="Recent Tests"
        maxEntries={3}
        renderEntry={(entry) => <span>{entry.label}</span>}
      />,
    );

    expect(screen.getByText("Entry 0")).toBeInTheDocument();
    expect(screen.getByText("Entry 1")).toBeInTheDocument();
    expect(screen.getByText("Entry 2")).toBeInTheDocument();
    expect(screen.queryByText("Entry 3")).not.toBeInTheDocument();
  });

  it("defaults maxEntries to 5", () => {
    const entries = Array.from({ length: 8 }, (_, i) => ({
      id: String(i),
      label: `Entry ${i}`,
    }));
    localStorage.setItem("ndw_test_readings", JSON.stringify(entries));

    renderWithProviders(
      <RecentReadings
        storageKey="ndw_test_readings"
        title="Recent Tests"
        renderEntry={(entry) => <span>{entry.label}</span>}
      />,
    );

    expect(screen.getByText("Entry 4")).toBeInTheDocument();
    expect(screen.queryByText("Entry 5")).not.toBeInTheDocument();
  });

  it("shows the title", () => {
    localStorage.setItem(
      "ndw_test_readings",
      JSON.stringify([{ id: "1", label: "Test" }]),
    );

    renderWithProviders(
      <RecentReadings
        storageKey="ndw_test_readings"
        title="My Custom Title"
        renderEntry={(entry) => <span>{entry.label}</span>}
      />,
    );

    expect(screen.getByText("My Custom Title")).toBeInTheDocument();
  });

  it("handles singleObject mode (wraps single object in array)", () => {
    localStorage.setItem(
      "ndw_test_single",
      JSON.stringify({ id: "1", label: "Single Object", timestamp: Date.now() }),
    );

    renderWithProviders(
      <RecentReadings
        storageKey="ndw_test_single"
        title="Latest"
        singleObject
        renderEntry={(entry) => <span>{entry.label}</span>}
      />,
    );

    expect(screen.getByText("Single Object")).toBeInTheDocument();
  });

  it("handles corrupted localStorage gracefully", () => {
    localStorage.setItem("ndw_test_readings", "not valid json{{{");

    const { container } = renderWithProviders(
      <RecentReadings
        storageKey="ndw_test_readings"
        title="Recent Tests"
        renderEntry={(entry) => <span>{entry.label}</span>}
      />,
    );

    // Should render nothing (graceful fallback)
    expect(container.innerHTML).toBe("");
  });

  it("has data-testid recent-readings when entries exist", () => {
    localStorage.setItem(
      "ndw_test_readings",
      JSON.stringify([{ id: "1", label: "Test" }]),
    );

    renderWithProviders(
      <RecentReadings
        storageKey="ndw_test_readings"
        title="Recent Tests"
        renderEntry={(entry) => <span>{entry.label}</span>}
      />,
    );

    expect(screen.getByTestId("recent-readings")).toBeInTheDocument();
  });
});

import { describe, it, expect, beforeAll } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import { HabitAnalytics } from "@/components/habit-analytics";

/* ---------- jsdom polyfill for Recharts ---------- */

beforeAll(() => {
  global.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  } as unknown as typeof ResizeObserver;
});

/* ---------- fixtures ---------- */

const HABITS = [
  { id: "h1", name: "Water" },
  { id: "h2", name: "Exercise" },
];

function makeLog(habitId: string, daysAgo: number) {
  const d = new Date();
  d.setDate(d.getDate() - daysAgo);
  return {
    id: `log-${habitId}-${daysAgo}`,
    habitId,
    loggedAt: d.toISOString(),
  };
}

// Logs for the last 7 days — partial completion
const RECENT_LOGS = [
  makeLog("h1", 0),
  makeLog("h2", 0),
  makeLog("h1", 1),
  makeLog("h1", 2),
  makeLog("h2", 3),
  makeLog("h1", 5),
  makeLog("h2", 6),
];

/* ---------- tests ---------- */

describe("HabitAnalytics", () => {
  it("renders the Analytics heading", () => {
    renderWithProviders(<HabitAnalytics habits={HABITS} logs={RECENT_LOGS} />);
    expect(screen.getByText("Analytics")).toBeInTheDocument();
  });

  it("renders the 30-day rate label", () => {
    renderWithProviders(<HabitAnalytics habits={HABITS} logs={RECENT_LOGS} />);
    expect(screen.getByText("30-day rate")).toBeInTheDocument();
  });

  it("shows weekly completion rate label", () => {
    renderWithProviders(<HabitAnalytics habits={HABITS} logs={RECENT_LOGS} />);
    expect(screen.getByText("Weekly completion rate")).toBeInTheDocument();
  });

  it("shows best day label", () => {
    renderWithProviders(<HabitAnalytics habits={HABITS} logs={RECENT_LOGS} />);
    expect(screen.getByText(/Most consistent/)).toBeInTheDocument();
  });

  it("shows worst day label", () => {
    renderWithProviders(<HabitAnalytics habits={HABITS} logs={RECENT_LOGS} />);
    expect(screen.getByText(/Least consistent/)).toBeInTheDocument();
  });

  it("shows On a roll or Needs attention status", () => {
    renderWithProviders(<HabitAnalytics habits={HABITS} logs={RECENT_LOGS} />);
    const onARoll = screen.queryByText("On a roll");
    const needsAttention = screen.queryByText("Needs attention");
    expect(onARoll ?? needsAttention).toBeTruthy();
  });

  it("shows last 7 days percentage", () => {
    renderWithProviders(<HabitAnalytics habits={HABITS} logs={RECENT_LOGS} />);
    expect(screen.getByText(/last 7 days/)).toBeInTheDocument();
  });

  it("renders nothing when no habits", () => {
    const { container } = renderWithProviders(
      <HabitAnalytics habits={[]} logs={[]} />,
    );
    expect(container.innerHTML).toBe("");
  });
});

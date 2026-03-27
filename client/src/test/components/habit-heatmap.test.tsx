import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import { HabitHeatmap } from "@/components/habit-heatmap";

/* ---------- fixtures ---------- */

const HABITS = [
  { id: "h1", name: "Water", icon: "droplets" },
  { id: "h2", name: "Exercise", icon: "dumbbell" },
];

function makeLog(habitId: string, daysAgo: number) {
  const d = new Date();
  d.setDate(d.getDate() - daysAgo);
  return {
    id: `log-${habitId}-${daysAgo}`,
    habitId,
    value: "1",
    loggedAt: d.toISOString(),
  };
}

const LOGS = [
  makeLog("h1", 0),
  makeLog("h2", 0),
  makeLog("h1", 1),
  makeLog("h1", 2),
  makeLog("h2", 5),
];

/* ---------- tests ---------- */

describe("HabitHeatmap", () => {
  it("renders the Activity heading", () => {
    renderWithProviders(<HabitHeatmap habits={HABITS} logs={LOGS} />);
    expect(screen.getByText("Activity")).toBeInTheDocument();
  });

  it("renders the 90-day label", () => {
    renderWithProviders(<HabitHeatmap habits={HABITS} logs={LOGS} />);
    expect(screen.getByText("Last 90 days")).toBeInTheDocument();
  });

  it("renders day labels M, W, F", () => {
    renderWithProviders(<HabitHeatmap habits={HABITS} logs={LOGS} />);
    expect(screen.getByText("M")).toBeInTheDocument();
    expect(screen.getByText("W")).toBeInTheDocument();
    expect(screen.getByText("F")).toBeInTheDocument();
  });

  it("renders legend with Less and More labels", () => {
    renderWithProviders(<HabitHeatmap habits={HABITS} logs={LOGS} />);
    expect(screen.getByText("Less")).toBeInTheDocument();
    expect(screen.getByText("More")).toBeInTheDocument();
  });

  it("renders nothing when no habits", () => {
    const { container } = renderWithProviders(
      <HabitHeatmap habits={[]} logs={[]} />,
    );
    expect(container.innerHTML).toBe("");
  });

  it("renders clickable day cells with aria labels", () => {
    renderWithProviders(<HabitHeatmap habits={HABITS} logs={LOGS} />);
    // Today's cell should exist and have an aria-label
    const todayCell = screen.getAllByRole("button").find((btn) =>
      btn.getAttribute("aria-label")?.includes("habits completed"),
    );
    expect(todayCell).toBeDefined();
  });
});

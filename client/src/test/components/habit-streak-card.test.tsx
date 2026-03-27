import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import { HabitStreakCard } from "@/components/habit-streak-card";

/* ---------- fixtures ---------- */

const HABIT = { id: "h1", name: "Water", icon: "droplets" };

function makeLog(daysAgo: number) {
  const d = new Date();
  d.setDate(d.getDate() - daysAgo);
  return {
    id: `log-${daysAgo}`,
    habitId: "h1",
    loggedAt: d.toISOString(),
  };
}

// Consecutive logs: today, yesterday, day before = 3-day streak
const CONSECUTIVE_LOGS = [makeLog(0), makeLog(1), makeLog(2)];

// Longer log history for longest streak computation
const LONGER_LOGS = [
  makeLog(0),
  makeLog(1),
  makeLog(2),
  // gap at day 3
  makeLog(10),
  makeLog(11),
  makeLog(12),
  makeLog(13),
  makeLog(14),
];

/* ---------- tests ---------- */

describe("HabitStreakCard", () => {
  it("renders the habit name", () => {
    renderWithProviders(
      <HabitStreakCard habit={HABIT} currentStreak={3} logs={CONSECUTIVE_LOGS} index={0} />,
    );
    expect(screen.getByText("Water")).toBeInTheDocument();
  });

  it("shows the days label", () => {
    renderWithProviders(
      <HabitStreakCard habit={HABIT} currentStreak={5} logs={CONSECUTIVE_LOGS} index={0} />,
    );
    expect(screen.getByText("days")).toBeInTheDocument();
  });

  it("shows Best streak stat", () => {
    renderWithProviders(
      <HabitStreakCard habit={HABIT} currentStreak={3} logs={LONGER_LOGS} index={0} />,
    );
    // Longest streak from LONGER_LOGS: 5 consecutive (days 10-14)
    expect(screen.getByText("Best: 5d")).toBeInTheDocument();
  });

  it("shows completion rate percentage", () => {
    renderWithProviders(
      <HabitStreakCard habit={HABIT} currentStreak={3} logs={CONSECUTIVE_LOGS} index={0} />,
    );
    // 3 days out of 30 = 10%
    expect(screen.getByText("10%")).toBeInTheDocument();
  });

  it("shows green styling for active streaks", () => {
    const { container } = renderWithProviders(
      <HabitStreakCard habit={HABIT} currentStreak={5} logs={CONSECUTIVE_LOGS} index={0} />,
    );
    // Active streak (>= 3) should have emerald colors
    const streakNum = container.querySelector(".text-emerald-400");
    expect(streakNum).toBeInTheDocument();
  });

  it("shows gray styling for zero streak", () => {
    const { container } = renderWithProviders(
      <HabitStreakCard habit={HABIT} currentStreak={0} logs={[]} index={0} />,
    );
    const streakNum = container.querySelector(".text-muted-foreground");
    expect(streakNum).toBeInTheDocument();
  });

  it("shows amber styling for at-risk streak (1 day)", () => {
    const { container } = renderWithProviders(
      <HabitStreakCard habit={HABIT} currentStreak={1} logs={[makeLog(0)]} index={0} />,
    );
    const amberEl = container.querySelector(".text-amber-400");
    expect(amberEl).toBeInTheDocument();
  });
});

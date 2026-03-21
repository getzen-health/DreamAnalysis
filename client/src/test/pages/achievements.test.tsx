import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach, beforeAll } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import AchievementsPage from "@/pages/achievements";

beforeAll(() => {
  global.ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
});

vi.mock("framer-motion", () => ({
  motion: {
    main: React.forwardRef(
      (
        { children, ...props }: React.PropsWithChildren<Record<string, unknown>>,
        ref: React.Ref<HTMLElement>,
      ) => React.createElement("main", { ...props, ref }, children),
    ),
  },
}));

vi.mock("@/lib/animations", () => ({
  pageTransition: { initial: {}, animate: {}, transition: {} },
}));

describe("Achievements page", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  afterEach(() => {
    localStorage.clear();
  });

  it("renders the page heading", () => {
    renderWithProviders(<AchievementsPage />);
    // "Achievements" appears both as the H1 and inside the AchievementBadges header
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(
      "Achievements",
    );
  });

  it("renders the subtitle", () => {
    renderWithProviders(<AchievementsPage />);
    expect(
      screen.getByText("Your badges and milestones"),
    ).toBeInTheDocument();
  });

  it("shows achievement badges component", () => {
    renderWithProviders(<AchievementsPage />);
    // The AchievementBadges component renders the overall progress bar
    expect(screen.getByTestId("overall-progress")).toBeInTheDocument();
  });

  it("shows locked badges section", () => {
    renderWithProviders(<AchievementsPage />);
    expect(screen.getByText("Locked")).toBeInTheDocument();
  });

  it("shows earned badge when data exists", () => {
    localStorage.setItem(
      "ndw_last_emotion",
      JSON.stringify({ emotion: "happy" }),
    );
    renderWithProviders(<AchievementsPage />);
    expect(screen.getByTestId("badge-first-checkin")).toBeInTheDocument();
    expect(screen.getByText("First Voice")).toBeInTheDocument();
  });
});

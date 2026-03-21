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

vi.mock("framer-motion", () => {
  const ReactM = require("react");
  return {
    motion: {
      main: ReactM.forwardRef(
        (
          { children, ...props }: React.PropsWithChildren<Record<string, unknown>>,
          ref: React.Ref<HTMLElement>,
        ) => {
          const {
            initial, animate, exit, transition, variants, custom,
            whileHover, whileTap, layout, layoutId,
            onAnimationStart, onAnimationComplete,
            ...domProps
          } = props;
          return ReactM.createElement("main", { ...domProps, ref }, children);
        },
      ),
      div: ReactM.forwardRef(
        (
          { children, ...props }: React.PropsWithChildren<Record<string, unknown>>,
          ref: React.Ref<HTMLDivElement>,
        ) => {
          const {
            initial, animate, exit, transition, variants, custom,
            whileHover, whileTap, layout, layoutId,
            onAnimationStart, onAnimationComplete,
            ...domProps
          } = props;
          return ReactM.createElement("div", { ...domProps, ref }, children);
        },
      ),
    },
    AnimatePresence: ({ children }: { children: React.ReactNode }) =>
      ReactM.createElement(ReactM.Fragment, null, children),
  };
});

vi.mock("@/lib/animations", () => ({
  pageTransition: { initial: {}, animate: {}, transition: {} },
  cardVariants: {
    hidden: { opacity: 0 },
    visible: () => ({ opacity: 1 }),
  },
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

  it("shows hero section with stats", () => {
    renderWithProviders(<AchievementsPage />);
    expect(screen.getByTestId("achievements-hero")).toBeInTheDocument();
    expect(screen.getByTestId("hero-earned-count")).toBeInTheDocument();
    expect(screen.getByTestId("hero-completion-pct")).toBeInTheDocument();
  });

  it("shows overall progress bar", () => {
    renderWithProviders(<AchievementsPage />);
    expect(screen.getByTestId("overall-progress")).toBeInTheDocument();
  });

  it("shows category filter pills", () => {
    renderWithProviders(<AchievementsPage />);
    expect(screen.getByTestId("category-filters")).toBeInTheDocument();
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

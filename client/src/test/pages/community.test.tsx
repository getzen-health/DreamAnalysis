import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach, beforeAll } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import Community from "@/pages/community";

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
}));

describe("Community page", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  afterEach(() => {
    localStorage.clear();
  });

  it("renders the page heading", () => {
    renderWithProviders(<Community />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(
      "Community",
    );
  });

  it("renders the subtitle", () => {
    renderWithProviders(<Community />);
    expect(
      screen.getByText("Anonymous mood sharing and challenges"),
    ).toBeInTheDocument();
  });

  it("shows collective mood section", () => {
    renderWithProviders(<Community />);
    expect(screen.getByTestId("collective-mood")).toBeInTheDocument();
    expect(screen.getByText("Collective Mood Today")).toBeInTheDocument();
  });

  it("shows daily challenge section", () => {
    renderWithProviders(<Community />);
    expect(screen.getByTestId("daily-challenge")).toBeInTheDocument();
    expect(screen.getByText("Daily Challenge")).toBeInTheDocument();
  });

  it("shows streaks leaderboard", () => {
    renderWithProviders(<Community />);
    expect(screen.getByTestId("streaks-leaderboard")).toBeInTheDocument();
    expect(screen.getByText("Streaks Leaderboard")).toBeInTheDocument();
  });

  it("shows your streak", () => {
    renderWithProviders(<Community />);
    expect(screen.getByText("Your Streak")).toBeInTheDocument();
  });

  it("shows mood voting buttons", () => {
    renderWithProviders(<Community />);
    expect(screen.getByText("Positive")).toBeInTheDocument();
    expect(screen.getByText("Neutral")).toBeInTheDocument();
    expect(screen.getByText("Negative")).toBeInTheDocument();
  });

  it("shows mark as done button for challenge", () => {
    renderWithProviders(<Community />);
    expect(screen.getByText("Mark as Done")).toBeInTheDocument();
  });
});

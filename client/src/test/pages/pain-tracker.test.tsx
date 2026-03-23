import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach, beforeAll } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import PainTracker from "@/pages/pain-tracker";

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

describe("Pain Tracker page", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  afterEach(() => {
    localStorage.clear();
  });

  it("renders the page heading", () => {
    renderWithProviders(<PainTracker />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(
      "Pain Tracker",
    );
  });

  it("renders the subtitle", () => {
    renderWithProviders(<PainTracker />);
    expect(
      screen.getByText("Track pain episodes and EEG patterns"),
    ).toBeInTheDocument();
  });

  it("shows summary cards", () => {
    renderWithProviders(<PainTracker />);
    expect(screen.getByTestId("pain-summary")).toBeInTheDocument();
  });

  it("shows no-episodes message when empty", () => {
    renderWithProviders(<PainTracker />);
    expect(screen.getByTestId("no-episodes")).toBeInTheDocument();
    expect(screen.getByText("No pain episodes logged")).toBeInTheDocument();
  });

  it("shows log button", () => {
    renderWithProviders(<PainTracker />);
    expect(screen.getByTestId("log-pain-button")).toBeInTheDocument();
  });

  it("shows form when log button clicked", () => {
    renderWithProviders(<PainTracker />);
    fireEvent.click(screen.getByTestId("log-pain-button"));
    expect(screen.getByTestId("pain-form")).toBeInTheDocument();
  });

  it("shows pain history after logging an episode", () => {
    // Pre-seed localStorage with an episode
    const episode = {
      id: "pain_1",
      severity: 7,
      location: "Head - Frontal",
      durationMinutes: 30,
      timestamp: Date.now(),
    };
    localStorage.setItem("ndw_pain_episodes", JSON.stringify([episode]));

    renderWithProviders(<PainTracker />);
    expect(screen.getByTestId("pain-history")).toBeInTheDocument();
    expect(screen.getByText("Head - Frontal")).toBeInTheDocument();
  });

  it("displays severity number in history", () => {
    const episode = {
      id: "pain_2",
      severity: 8,
      location: "Head - Full",
      durationMinutes: 60,
      timestamp: Date.now(),
    };
    localStorage.setItem("ndw_pain_episodes", JSON.stringify([episode]));

    renderWithProviders(<PainTracker />);
    expect(screen.getByText("8")).toBeInTheDocument();
  });
});

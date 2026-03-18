import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import { ScoreGauge, SCORE_COLORS, type ScoreColor } from "@/components/score-gauge";

vi.mock("framer-motion", () => ({
  motion: {
    circle: (props: any) => <circle {...props} />,
    div: ({ children, ...rest }: any) => <div {...rest}>{children}</div>,
  },
  useMotionValue: (initial: number) => ({
    get: () => initial,
    set: vi.fn(),
  }),
  useTransform: (_mv: any, fn: (v: number) => number) => fn(0),
  animate: () => ({ stop: vi.fn() }),
  AnimatePresence: ({ children }: any) => <>{children}</>,
}));

describe("ScoreGauge", () => {
  it("renders the value when given a number", () => {
    renderWithProviders(
      <ScoreGauge value={72} label="Recovery" color="recovery" />
    );
    expect(screen.getByText("72")).toBeInTheDocument();
  });

  it("renders em-dash when value is null", () => {
    renderWithProviders(
      <ScoreGauge value={null} label="Sleep" color="sleep" />
    );
    // em-dash character U+2014
    expect(screen.getByText("\u2014")).toBeInTheDocument();
  });

  it("renders the label text", () => {
    renderWithProviders(
      <ScoreGauge value={50} label="Strain" color="strain" />
    );
    expect(screen.getByText("Strain")).toBeInTheDocument();
  });

  it("renders the subtitle when provided", () => {
    renderWithProviders(
      <ScoreGauge value={80} label="Recovery" color="recovery" subtitle="67ms HRV" />
    );
    expect(screen.getByText("67ms HRV")).toBeInTheDocument();
  });

  it("has correct aria-label with value", () => {
    renderWithProviders(
      <ScoreGauge value={85} label="Sleep" color="sleep" />
    );
    expect(screen.getByLabelText("Sleep: 85")).toBeInTheDocument();
  });

  it("has correct aria-label when null", () => {
    renderWithProviders(
      <ScoreGauge value={null} label="Stress" color="stress" />
    );
    expect(screen.getByLabelText("Stress: no data")).toBeInTheDocument();
  });

  it("renders SVG with gradient stop colors matching the score color type", () => {
    const { container } = renderWithProviders(
      <ScoreGauge value={60} label="Recovery" color="recovery" />
    );
    const stops = container.querySelectorAll("stop");
    expect(stops.length).toBeGreaterThanOrEqual(2);
    expect(stops[0].getAttribute("stop-color")).toBe(SCORE_COLORS.recovery.from);
    expect(stops[1].getAttribute("stop-color")).toBe(SCORE_COLORS.recovery.to);
  });

  it.each<ScoreColor>(["recovery", "sleep", "strain", "stress", "nutrition", "energy"])(
    "uses correct gradient colors for %s",
    (color) => {
      const { container } = renderWithProviders(
        <ScoreGauge value={50} label="Test" color={color} />
      );
      const stops = container.querySelectorAll("stop");
      expect(stops[0].getAttribute("stop-color")).toBe(SCORE_COLORS[color].from);
      expect(stops[1].getAttribute("stop-color")).toBe(SCORE_COLORS[color].to);
    }
  );

  it("rounds the displayed value", () => {
    renderWithProviders(
      <ScoreGauge value={72.7} label="Score" color="recovery" />
    );
    expect(screen.getByText("73")).toBeInTheDocument();
  });
});

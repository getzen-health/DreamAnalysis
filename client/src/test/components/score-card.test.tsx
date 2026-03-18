import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import { ScoreCard } from "@/components/score-card";
import { Heart, Moon } from "lucide-react";

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

describe("ScoreCard", () => {
  it("renders the title", () => {
    renderWithProviders(
      <ScoreCard
        title="Recovery"
        value={75}
        color="recovery"
        icon={<Heart data-testid="heart-icon" />}
      />
    );
    expect(screen.getByText("Recovery")).toBeInTheDocument();
  });

  it("renders the subtitle when provided", () => {
    renderWithProviders(
      <ScoreCard
        title="Recovery"
        value={75}
        color="recovery"
        icon={<Heart />}
        subtitle="67ms HRV - 58 RHR"
      />
    );
    expect(screen.getByText("67ms HRV - 58 RHR")).toBeInTheDocument();
  });

  it("renders gauge with the correct value", () => {
    renderWithProviders(
      <ScoreCard
        title="Sleep"
        value={82}
        color="sleep"
        icon={<Moon />}
      />
    );
    // The ScoreGauge inside ScoreCard renders the number
    expect(screen.getByText("82")).toBeInTheDocument();
  });

  it("renders null state when value is null", () => {
    renderWithProviders(
      <ScoreCard
        title="Sleep"
        value={null}
        color="sleep"
        icon={<Moon />}
      />
    );
    // em-dash from the ScoreGauge inside
    expect(screen.getByText("\u2014")).toBeInTheDocument();
  });

  it("shows trend up indicator with value", () => {
    renderWithProviders(
      <ScoreCard
        title="Recovery"
        value={80}
        color="recovery"
        icon={<Heart />}
        trend="up"
        trendValue="+5% from yesterday"
      />
    );
    expect(screen.getByText("+5% from yesterday")).toBeInTheDocument();
  });

  it("shows trend down indicator with value", () => {
    renderWithProviders(
      <ScoreCard
        title="Strain"
        value={45}
        color="strain"
        icon={<Heart />}
        trend="down"
        trendValue="-10% from yesterday"
      />
    );
    expect(screen.getByText("-10% from yesterday")).toBeInTheDocument();
  });

  it("shows stable trend indicator with value", () => {
    renderWithProviders(
      <ScoreCard
        title="Stress"
        value={30}
        color="stress"
        icon={<Heart />}
        trend="stable"
        trendValue="No change"
      />
    );
    expect(screen.getByText("No change")).toBeInTheDocument();
  });

  it("does not show trend when not provided", () => {
    renderWithProviders(
      <ScoreCard
        title="Sleep"
        value={90}
        color="sleep"
        icon={<Moon />}
      />
    );
    expect(screen.queryByText(/from yesterday/)).not.toBeInTheDocument();
  });
});

import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import { BrainCoachCard, buildRecommendations } from "@/components/brain-coach-card";

const nullProps = {
  recoveryScore: null,
  sleepScore: null,
  stressScore: null,
  strainScore: null,
  avgFocus: null,
  avgValence: null,
  avgStress: null,
};

describe("BrainCoachCard", () => {
  it("renders without crashing", () => {
    renderWithProviders(<BrainCoachCard {...nullProps} />);
    expect(screen.getByTestId("brain-coach-card")).toBeInTheDocument();
  });

  it("shows empty state when all props are null", () => {
    renderWithProviders(<BrainCoachCard {...nullProps} />);
    expect(screen.getByTestId("brain-coach-empty")).toBeInTheDocument();
    expect(
      screen.getByText(/connect EEG or sync health data/i),
    ).toBeInTheDocument();
  });

  it("shows peak performance message when focus > 0.65 and recovery > 60", () => {
    renderWithProviders(
      <BrainCoachCard
        {...nullProps}
        avgFocus={0.75}
        recoveryScore={70}
      />,
    );
    expect(screen.getByText(/peak performance window/i)).toBeInTheDocument();
  });

  it("shows rest recommendation when focus < 0.4 and recovery < 50", () => {
    renderWithProviders(
      <BrainCoachCard
        {...nullProps}
        avgFocus={0.3}
        recoveryScore={40}
      />,
    );
    expect(screen.getByText(/rest day/i)).toBeInTheDocument();
  });

  it("shows stress recommendation when stressScore > 65", () => {
    renderWithProviders(
      <BrainCoachCard
        {...nullProps}
        stressScore={70}
      />,
    );
    expect(screen.getByText(/elevated stress/i)).toBeInTheDocument();
    expect(screen.getByText(/box breathing/i)).toBeInTheDocument();
  });
});

describe("buildRecommendations (unit)", () => {
  it("returns empty array when all props are null", () => {
    expect(buildRecommendations(nullProps)).toHaveLength(0);
  });

  it("returns peak performance rec for high focus + good recovery", () => {
    const recs = buildRecommendations({
      ...nullProps,
      avgFocus: 0.8,
      recoveryScore: 75,
    });
    expect(recs.some((r) => /peak performance/i.test(r.title))).toBe(true);
  });

  it("returns rest rec for low focus + low recovery", () => {
    const recs = buildRecommendations({
      ...nullProps,
      avgFocus: 0.2,
      recoveryScore: 30,
    });
    expect(recs.some((r) => /rest day/i.test(r.title))).toBe(true);
  });

  it("returns stress rec when stressScore > 65", () => {
    const recs = buildRecommendations({ ...nullProps, stressScore: 80 });
    expect(recs.some((r) => /elevated stress/i.test(r.title))).toBe(true);
  });

  it("caps recommendations at 3", () => {
    const recs = buildRecommendations({
      avgFocus: 0.2,
      recoveryScore: 30,
      sleepScore: 30,
      stressScore: 80,
      strainScore: null,
      avgValence: -0.5,
      avgStress: 0.8,
    });
    expect(recs.length).toBeLessThanOrEqual(3);
  });
});

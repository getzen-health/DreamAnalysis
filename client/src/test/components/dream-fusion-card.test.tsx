import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import { DreamFusionCard } from "@/components/dream-fusion-card";
import type { DreamFusionInsight } from "@/lib/dream-biometric-fusion";

const sampleInsight: DreamFusionInsight = {
  headline: "Anxious dream, restless body",
  body: "Your heart rate averaged 72 bpm \u2014 8 bpm above your usual. HRV was 28ms, suggesting your nervous system was active during this dream.",
  dreamEmotions: ["anxiety", "fear"],
  biometricHighlights: [
    { label: "HR", value: "72 bpm", status: "elevated" },
    { label: "HRV", value: "28ms", status: "low" },
    { label: "Deep", value: "18%", status: "low" },
    { label: "REM", value: "22%", status: "normal" },
  ],
  sleepContext: "6.2h sleep, 18% deep, 22% REM",
};

describe("DreamFusionCard", () => {
  it("renders without crashing", () => {
    renderWithProviders(<DreamFusionCard insight={null} />);
    expect(screen.getByTestId("dream-fusion-card")).toBeInTheDocument();
  });

  it("shows empty state when insight is null", () => {
    renderWithProviders(<DreamFusionCard insight={null} />);
    expect(screen.getByTestId("dream-fusion-empty")).toBeInTheDocument();
    expect(
      screen.getByText(/log a dream to see how your body responded/i),
    ).toBeInTheDocument();
  });

  it("shows headline when insight is provided", () => {
    renderWithProviders(<DreamFusionCard insight={sampleInsight} />);
    expect(screen.getByTestId("dream-fusion-headline")).toBeInTheDocument();
    expect(
      screen.getByText("Anxious dream, restless body"),
    ).toBeInTheDocument();
  });

  it("shows body text with biometric values", () => {
    renderWithProviders(<DreamFusionCard insight={sampleInsight} />);
    const body = screen.getByTestId("dream-fusion-body");
    expect(body).toBeInTheDocument();
    expect(body.textContent).toContain("72 bpm");
    expect(body.textContent).toContain("28ms");
  });

  it("shows biometric highlights", () => {
    renderWithProviders(<DreamFusionCard insight={sampleInsight} />);
    expect(screen.getByTestId("dream-fusion-highlights")).toBeInTheDocument();
    expect(screen.getByText(/HR: 72 bpm/)).toBeInTheDocument();
    expect(screen.getByText(/HRV: 28ms/)).toBeInTheDocument();
    expect(screen.getByText(/Deep: 18%/)).toBeInTheDocument();
    expect(screen.getByText(/REM: 22%/)).toBeInTheDocument();
  });

  it("shows sleep context", () => {
    renderWithProviders(<DreamFusionCard insight={sampleInsight} />);
    const ctx = screen.getByTestId("dream-fusion-sleep-context");
    expect(ctx).toBeInTheDocument();
    expect(ctx.textContent).toContain("6.2h sleep");
  });

  it("does not render highlights section when no highlights", () => {
    const noHighlights: DreamFusionInsight = {
      ...sampleInsight,
      biometricHighlights: [],
    };
    renderWithProviders(<DreamFusionCard insight={noHighlights} />);
    expect(screen.queryByTestId("dream-fusion-highlights")).not.toBeInTheDocument();
  });

  it("does not render empty state when insight is provided", () => {
    renderWithProviders(<DreamFusionCard insight={sampleInsight} />);
    expect(screen.queryByTestId("dream-fusion-empty")).not.toBeInTheDocument();
  });
});

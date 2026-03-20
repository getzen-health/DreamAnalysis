import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import { EFSHeroScore } from "@/components/efs-hero-score";

vi.mock("@/hooks/use-theme", () => ({
  useTheme: () => ({ theme: "dark", setTheme: vi.fn() }),
}));

// ── Fixtures ──────────────────────────────────────────────────────────────────

const BASE_PROPS = {
  score: 82 as number | null,
  color: "green" as const,
  label: "Strong" as string | null,
  confidence: "full" as const,
  trend: { direction: "up" as const, delta: 8, period: "30d" },
};

const BUILDING_PROGRESS = {
  daysTracked: 1,
  daysRequired: 3,
  percentage: 33,
  message: "Keep tracking for 2 more days",
};

// ── Tests ─────────────────────────────────────────────────────────────────────

describe("EFSHeroScore", () => {
  it("renders score number when score is provided", () => {
    renderWithProviders(<EFSHeroScore {...BASE_PROPS} />);
    expect(screen.getByText("82")).toBeInTheDocument();
  });

  it("shows 'Emotional Fitness' label", () => {
    renderWithProviders(<EFSHeroScore {...BASE_PROPS} />);
    expect(screen.getByText("Emotional Fitness")).toBeInTheDocument();
  });

  it("renders score arc with correct aria-label", () => {
    renderWithProviders(<EFSHeroScore {...BASE_PROPS} />);
    expect(screen.getByLabelText("Emotional Fitness Score: 82")).toBeInTheDocument();
  });

  it("shows building progress when score is null with progress data", () => {
    renderWithProviders(
      <EFSHeroScore
        score={null}
        color={null}
        label={null}
        confidence="building"
        trend={null}
        progress={BUILDING_PROGRESS}
      />
    );
    expect(screen.getByText("33%")).toBeInTheDocument();
    expect(screen.getByText("building")).toBeInTheDocument();
    expect(screen.getByText("Keep tracking for 2 more days")).toBeInTheDocument();
    expect(screen.getByText("1 / 3 days tracked")).toBeInTheDocument();
  });

  it("shows building progress arc with correct aria-label", () => {
    renderWithProviders(
      <EFSHeroScore
        score={null}
        color={null}
        label={null}
        confidence="building"
        trend={null}
        progress={BUILDING_PROGRESS}
      />
    );
    expect(screen.getByLabelText("Building score: 33%")).toBeInTheDocument();
  });

  it("shows 'Early estimate' badge when confidence is early_estimate", () => {
    renderWithProviders(
      <EFSHeroScore {...BASE_PROPS} confidence="early_estimate" />
    );
    expect(screen.getByText("Early estimate")).toBeInTheDocument();
  });

  it("does not show 'Early estimate' badge when confidence is full", () => {
    renderWithProviders(<EFSHeroScore {...BASE_PROPS} confidence="full" />);
    expect(screen.queryByText("Early estimate")).not.toBeInTheDocument();
  });

  it("shows trend badge with up direction", () => {
    renderWithProviders(<EFSHeroScore {...BASE_PROPS} />);
    expect(screen.getByText(/\+8 pts 30d/)).toBeInTheDocument();
  });

  it("shows trend badge with down direction", () => {
    renderWithProviders(
      <EFSHeroScore
        {...BASE_PROPS}
        trend={{ direction: "down", delta: 5, period: "7d" }}
      />
    );
    expect(screen.getByText(/-5 pts 7d/)).toBeInTheDocument();
  });

  it("does not render trend badge when trend is null", () => {
    renderWithProviders(<EFSHeroScore {...BASE_PROPS} trend={null} />);
    expect(screen.queryByText(/pts/)).not.toBeInTheDocument();
  });

  it("does not render trend badge when delta is 0", () => {
    renderWithProviders(
      <EFSHeroScore
        {...BASE_PROPS}
        trend={{ direction: "stable", delta: 0, period: "30d" }}
      />
    );
    expect(screen.queryByText(/pts/)).not.toBeInTheDocument();
  });

  it("shows label text when label is provided", () => {
    renderWithProviders(<EFSHeroScore {...BASE_PROPS} label="Strong" />);
    expect(screen.getByText("Strong")).toBeInTheDocument();
  });

  it("does not show label text when label is null", () => {
    renderWithProviders(<EFSHeroScore {...BASE_PROPS} label={null} />);
    expect(screen.queryByText("Strong")).not.toBeInTheDocument();
  });

  it("applies cyan color class for green (score >= 70)", () => {
    renderWithProviders(<EFSHeroScore {...BASE_PROPS} color="green" label="Strong" />);
    const labelEl = screen.getByText("Strong");
    expect(labelEl.className).toContain("text-cyan-400");
  });

  it("applies amber color class for amber (score 40-69)", () => {
    renderWithProviders(
      <EFSHeroScore {...BASE_PROPS} score={55} color="amber" label="Fair" />
    );
    const labelEl = screen.getByText("Fair");
    expect(labelEl.className).toContain("text-amber-400");
  });

  it("applies rose color class for red (score < 40)", () => {
    renderWithProviders(
      <EFSHeroScore {...BASE_PROPS} score={25} color="red" label="Low" />
    );
    const labelEl = screen.getByText("Low");
    expect(labelEl.className).toContain("text-rose-400");
  });
});

import { describe, it, expect, vi, beforeAll } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import { EFSVitalCard } from "@/components/efs-vital-card";
import { Shield } from "lucide-react";
import type { EFSVitalData } from "@/lib/ml-api";

// Recharts uses ResizeObserver
beforeAll(() => {
  global.ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
});

vi.mock("@/hooks/use-theme", () => ({
  useTheme: () => ({ theme: "dark", setTheme: vi.fn() }),
}));

// ── Fixtures ──────────────────────────────────────────────────────────────────

const AVAILABLE_VITAL: EFSVitalData = {
  score: 85,
  status: "available",
  insight: "Good recovery from setbacks",
  history: [
    { date: "2026-03-10", score: 70 },
    { date: "2026-03-11", score: 75 },
    { date: "2026-03-12", score: 80 },
    { date: "2026-03-13", score: 85 },
  ],
};

const UNAVAILABLE_VITAL: EFSVitalData = {
  score: null,
  status: "unavailable",
  insight: "",
  unlockHint: "Track for 3+ days",
  history: [],
};

// ── Tests ─────────────────────────────────────────────────────────────────────

describe("EFSVitalCard", () => {
  it("renders vital name and score when available", () => {
    renderWithProviders(
      <EFSVitalCard name="resilience" icon={Shield} vital={AVAILABLE_VITAL} />
    );
    expect(screen.getByText("resilience")).toBeInTheDocument();
    expect(screen.getByText("85")).toBeInTheDocument();
  });

  it("shows Lock icon and unlock hint when status is unavailable", () => {
    renderWithProviders(
      <EFSVitalCard name="resilience" icon={Shield} vital={UNAVAILABLE_VITAL} />
    );
    expect(screen.getByText("resilience")).toBeInTheDocument();
    expect(screen.getByText("Track for 3+ days")).toBeInTheDocument();
    // Score should not appear
    expect(screen.queryByText("85")).not.toBeInTheDocument();
  });

  it("shows insight text", () => {
    renderWithProviders(
      <EFSVitalCard name="resilience" icon={Shield} vital={AVAILABLE_VITAL} />
    );
    expect(screen.getByText("Good recovery from setbacks")).toBeInTheDocument();
  });

  it("does not show insight text when empty", () => {
    const noInsight: EFSVitalData = { ...AVAILABLE_VITAL, insight: "" };
    renderWithProviders(
      <EFSVitalCard name="resilience" icon={Shield} vital={noInsight} />
    );
    expect(screen.queryByText("Good recovery from setbacks")).not.toBeInTheDocument();
  });

  it("expands on click to show explanation and tips", () => {
    renderWithProviders(
      <EFSVitalCard name="resilience" icon={Shield} vital={AVAILABLE_VITAL} />
    );

    // Explanation should not be visible initially
    expect(
      screen.queryByText("How quickly your emotions return to baseline after a negative event.")
    ).not.toBeInTheDocument();

    // Click to expand
    fireEvent.click(screen.getByRole("button"));

    // Now explanation and tips should be visible
    expect(
      screen.getByText("How quickly your emotions return to baseline after a negative event.")
    ).toBeInTheDocument();
    expect(
      screen.getByText("Practice 4-7-8 breathing. Regular sleep improves recovery.")
    ).toBeInTheDocument();
  });

  it("collapses on second click", () => {
    renderWithProviders(
      <EFSVitalCard name="resilience" icon={Shield} vital={AVAILABLE_VITAL} />
    );

    const button = screen.getByRole("button");

    // Expand
    fireEvent.click(button);
    expect(
      screen.getByText("How quickly your emotions return to baseline after a negative event.")
    ).toBeInTheDocument();

    // Collapse
    fireEvent.click(button);
    expect(
      screen.queryByText("How quickly your emotions return to baseline after a negative event.")
    ).not.toBeInTheDocument();
  });

  it("applies cyan color class for score >= 70", () => {
    renderWithProviders(
      <EFSVitalCard name="resilience" icon={Shield} vital={AVAILABLE_VITAL} />
    );
    const scoreEl = screen.getByText("85");
    expect(scoreEl.className).toContain("text-cyan-400");
  });

  it("applies amber color class for score 40-69", () => {
    const amberVital: EFSVitalData = { ...AVAILABLE_VITAL, score: 55 };
    renderWithProviders(
      <EFSVitalCard name="awareness" icon={Shield} vital={amberVital} />
    );
    const scoreEl = screen.getByText("55");
    expect(scoreEl.className).toContain("text-amber-400");
  });

  it("applies rose color class for score < 40", () => {
    const redVital: EFSVitalData = { ...AVAILABLE_VITAL, score: 25 };
    renderWithProviders(
      <EFSVitalCard name="stability" icon={Shield} vital={redVital} />
    );
    const scoreEl = screen.getByText("25");
    expect(scoreEl.className).toContain("text-rose-400");
  });

  it("does not render expand button when unavailable", () => {
    renderWithProviders(
      <EFSVitalCard name="resilience" icon={Shield} vital={UNAVAILABLE_VITAL} />
    );
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });
});

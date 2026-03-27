// client/src/test/components/morning-briefing-card.test.tsx
import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "@/test/test-utils";
import { MorningBriefingCard } from "@/components/morning-briefing-card";

const mockBriefing = {
  stateSummary: "You slept 6 hours. Your HRV is in your lower range. Expect an afternoon dip.",
  actions: ["Start with creative work before 11AM", "Avoid heavy meals until 2PM", "Plan a 20-min walk at 3PM"] as [string, string, string],
  forecast: { label: "Moderate", probability: 0.72, reason: "Based on your sleep and HRV patterns" },
};

describe("MorningBriefingCard", () => {
  it("renders loading state when loading=true", () => {
    renderWithProviders(<MorningBriefingCard loading={true} briefing={null} onGenerate={vi.fn()} />);
    expect(screen.getByText(/generating/i)).toBeInTheDocument();
  });

  it("renders generate button when no briefing and not loading", () => {
    renderWithProviders(<MorningBriefingCard loading={false} briefing={null} onGenerate={vi.fn()} />);
    expect(screen.getByRole("button", { name: /good morning/i })).toBeInTheDocument();
  });

  it("renders briefing content when provided", () => {
    renderWithProviders(<MorningBriefingCard loading={false} briefing={mockBriefing} onGenerate={vi.fn()} />);
    expect(screen.getByText(/You slept 6 hours/i)).toBeInTheDocument();
    expect(screen.getByText(/creative work before 11AM/i)).toBeInTheDocument();
    expect(screen.getByText(/Moderate/i)).toBeInTheDocument();
  });

  it("renders all 3 action items", () => {
    renderWithProviders(<MorningBriefingCard loading={false} briefing={mockBriefing} onGenerate={vi.fn()} />);
    expect(screen.getByText("Start with creative work before 11AM")).toBeInTheDocument();
    expect(screen.getByText("Avoid heavy meals until 2PM")).toBeInTheDocument();
    expect(screen.getByText("Plan a 20-min walk at 3PM")).toBeInTheDocument();
  });
});

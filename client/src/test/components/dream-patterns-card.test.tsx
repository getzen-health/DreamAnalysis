import { describe, it, expect, vi } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import { DreamPatternsCard } from "@/components/dream-patterns-card";
import type { DreamEntry } from "@/lib/dream-theme-tracker";

// ── Helpers ──────────────────────────────────────────────────────────────────

function makeDream(
  overrides: Partial<DreamEntry> & { daysAgo?: number } = {},
): DreamEntry {
  const { daysAgo = 0, ...rest } = overrides;
  const ts = new Date(Date.now() - daysAgo * 86_400_000).toISOString();
  return {
    dreamText: "a dream",
    emotions: [],
    symbols: [],
    timestamp: ts,
    ...rest,
  };
}

// ── Tests ────────────────────────────────────────────────────────────────────

describe("DreamPatternsCard", () => {
  it("renders with data-testid", () => {
    renderWithProviders(<DreamPatternsCard dreams={[]} />);
    expect(screen.getByTestId("dream-patterns-card")).toBeInTheDocument();
  });

  it("shows Dream Patterns header with Moon icon", () => {
    renderWithProviders(<DreamPatternsCard dreams={[]} />);
    expect(screen.getByText("Dream Patterns")).toBeInTheDocument();
  });

  it("shows empty state when no dreams are provided", () => {
    renderWithProviders(<DreamPatternsCard dreams={[]} />);
    expect(
      screen.getByText("Log more dreams to discover patterns"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("dream-patterns-empty")).toBeInTheDocument();
  });

  it("shows period selector pills (7d, 30d, 90d)", () => {
    renderWithProviders(<DreamPatternsCard dreams={[]} />);
    expect(screen.getByText("7d")).toBeInTheDocument();
    expect(screen.getByText("30d")).toBeInTheDocument();
    expect(screen.getByText("90d")).toBeInTheDocument();
  });

  it("shows top themes when dreams have symbols", () => {
    const dreams = [
      makeDream({ symbols: ["water", "flying"], emotions: ["anxiety"], daysAgo: 1 }),
      makeDream({ symbols: ["water", "forest"], emotions: ["calm"], daysAgo: 2 }),
      makeDream({ symbols: ["water"], emotions: ["fear"], daysAgo: 3 }),
    ];
    renderWithProviders(<DreamPatternsCard dreams={dreams} />);

    expect(screen.getByTestId("theme-list")).toBeInTheDocument();
    expect(screen.getByTestId("theme-row-water")).toBeInTheDocument();
    // water count badge should show "3"
    expect(screen.getByText("3")).toBeInTheDocument();
  });

  it("shows emotion distribution bar when emotions are present", () => {
    const dreams = [
      makeDream({ symbols: ["water"], emotions: ["anxiety", "fear"], daysAgo: 1 }),
      makeDream({ symbols: ["water"], emotions: ["anxiety", "joy"], daysAgo: 2 }),
    ];
    renderWithProviders(<DreamPatternsCard dreams={dreams} />);

    expect(screen.getByTestId("emotion-bar")).toBeInTheDocument();
    expect(screen.getByText("Emotion Distribution")).toBeInTheDocument();
  });

  it("switching period pills changes the analysis", () => {
    // One dream at 5 days ago (within 7d), one at 20 days ago (outside 7d but within 30d)
    const dreams = [
      makeDream({ symbols: ["water"], daysAgo: 5 }),
      makeDream({ symbols: ["water"], daysAgo: 20 }),
    ];
    renderWithProviders(<DreamPatternsCard dreams={dreams} />);

    // Default is 30d, should see count 2
    expect(screen.getByText("2")).toBeInTheDocument();

    // Switch to 7d
    fireEvent.click(screen.getByText("7d"));

    // Now only 1 dream in range
    expect(screen.getByText("1")).toBeInTheDocument();
  });

  it("shows dream count and lucid count in header", () => {
    const dreams = [
      makeDream({ symbols: ["water"], lucidityScore: 0.8, daysAgo: 1 }),
      makeDream({ symbols: ["forest"], lucidityScore: 0.2, daysAgo: 2 }),
    ];
    renderWithProviders(<DreamPatternsCard dreams={dreams} />);

    // "2 dreams" should appear
    expect(screen.getByText(/2 dreams/)).toBeInTheDocument();
    // "1 lucid" should appear
    expect(screen.getByText(/1 lucid/)).toBeInTheDocument();
  });

  it("shows empty state for all dreams outside the selected period", () => {
    const dreams = [makeDream({ symbols: ["water"], daysAgo: 100 })];
    renderWithProviders(<DreamPatternsCard dreams={dreams} />);

    // Default 30d period, dream is at 100 days ago
    expect(screen.getByTestId("dream-patterns-empty")).toBeInTheDocument();
  });
});

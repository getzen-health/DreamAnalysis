import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import DailyBrainReport from "@/pages/daily-brain-report";

vi.mock("wouter", () => ({
  useLocation: () => ["/brain-report", vi.fn()],
  Link: (props: any) => <a href={props.href}>{props.children}</a>,
}));

// Stub fetch: all API calls return empty arrays (no data state)
beforeEach(() => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true,
      json: async () => [],
    })
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("DailyBrainReport page — no data state", () => {
  it("renders without crashing", () => {
    renderWithProviders(<DailyBrainReport />);
  });

  it("shows a time-of-day greeting", async () => {
    renderWithProviders(<DailyBrainReport />);
    expect(
      await screen.findByText(/Good (morning|afternoon|evening)/i, {}, { timeout: 3000 })
    ).toBeInTheDocument();
  });

  it("shows the 'Right now' card section", async () => {
    renderWithProviders(<DailyBrainReport />);
    expect(
      await screen.findByText("Right now", {}, { timeout: 3000 })
    ).toBeInTheDocument();
  });

  it("shows the 'Do this now' card section", async () => {
    renderWithProviders(<DailyBrainReport />);
    expect(
      await screen.findByText("Do this now", {}, { timeout: 3000 })
    ).toBeInTheDocument();
  });

  it("shows default recommended action when no health data", async () => {
    renderWithProviders(<DailyBrainReport />);
    expect(
      await screen.findByText("Start coherence breathing", {}, { timeout: 3000 })
    ).toBeInTheDocument();
  });

  it("shows the action description when no health data", async () => {
    renderWithProviders(<DailyBrainReport />);
    expect(
      await screen.findByText(/4-min session to centre your nervous system/, {}, { timeout: 3000 })
    ).toBeInTheDocument();
  });

  it("shows a Start button for the recommended action", async () => {
    renderWithProviders(<DailyBrainReport />);
    expect(
      await screen.findByRole("button", { name: /Start/ }, { timeout: 3000 })
    ).toBeInTheDocument();
  });

  it("does not show 'Last night' card when no sleep data", async () => {
    renderWithProviders(<DailyBrainReport />);
    // Wait for load to complete by finding a known post-load element
    await screen.findByText("Do this now", {}, { timeout: 3000 });
    expect(screen.queryByText("Last night")).not.toBeInTheDocument();
  });

  it("does not show 'Your pattern' card when no pattern data", async () => {
    renderWithProviders(<DailyBrainReport />);
    await screen.findByText("Do this now", {}, { timeout: 3000 });
    expect(screen.queryByText("Your pattern")).not.toBeInTheDocument();
  });

  it("does not show Yesterday's insight card when health data is empty", async () => {
    renderWithProviders(<DailyBrainReport />);
    await screen.findByText("Do this now", {}, { timeout: 3000 });
    expect(screen.queryByText("Yesterday's insight")).not.toBeInTheDocument();
  });

  it("does not show stress risk when no health data", async () => {
    renderWithProviders(<DailyBrainReport />);
    await screen.findByText("Do this now", {}, { timeout: 3000 });
    expect(screen.queryByText("Stress risk")).not.toBeInTheDocument();
  });

  it("does not show streak badge when session count is zero", async () => {
    renderWithProviders(<DailyBrainReport />);
    await screen.findByText("Do this now", {}, { timeout: 3000 });
    expect(screen.queryByText(/-day streak/)).not.toBeInTheDocument();
  });
});

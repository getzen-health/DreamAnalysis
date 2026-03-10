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

  it("shows the 'No data yet' card when no health data", async () => {
    renderWithProviders(<DailyBrainReport />);
    expect(
      await screen.findByText("No data yet", {}, { timeout: 3000 })
    ).toBeInTheDocument();
  });

  it("shows the 'Do this now' card section", async () => {
    renderWithProviders(<DailyBrainReport />);
    expect(
      await screen.findByText("Do this now", {}, { timeout: 3000 })
    ).toBeInTheDocument();
  });

  it("shows voice check-in CTA as default action when no health data", async () => {
    renderWithProviders(<DailyBrainReport />);
    expect(
      await screen.findByText("Run a voice check-in", {}, { timeout: 3000 })
    ).toBeInTheDocument();
  });

  it("shows voice check-in description when no health data", async () => {
    renderWithProviders(<DailyBrainReport />);
    // Note: the page uses a curly apostrophe (U+2019) in "today\u2019s"
    expect(
      await screen.findByText(
        /Capture a quick emotion snapshot to personalize today/i,
        {},
        { timeout: 3000 }
      )
    ).toBeInTheDocument();
  });

  it("shows a Start button for the recommended action", async () => {
    renderWithProviders(<DailyBrainReport />);
    // Wait for Do this now card first, then confirm button is present
    await screen.findByText("Do this now", {}, { timeout: 3000 });
    const buttons = screen.getAllByRole("button");
    expect(buttons.some((b) => /start/i.test(b.textContent ?? ""))).toBe(true);
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

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import DailyBrainReport from "@/pages/daily-brain-report";

vi.mock("wouter", () => ({
  useLocation: () => ["/brain-report", vi.fn()],
  Link: (props: any) => <a href={props.href}>{props.children}</a>,
}));

// Stub fetch: all three API calls return empty arrays (no data state)
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

  it("shows the 'Last night' section header", async () => {
    renderWithProviders(<DailyBrainReport />);
    expect(
      await screen.findByText("Last night", {}, { timeout: 3000 })
    ).toBeInTheDocument();
  });

  it("shows no overnight session message when no data", async () => {
    renderWithProviders(<DailyBrainReport />);
    expect(
      await screen.findByText(/No overnight session recorded yet/, {}, { timeout: 3000 })
    ).toBeInTheDocument();
  });

  it("shows 'Connect your Muse 2' link in empty sleep state", async () => {
    renderWithProviders(<DailyBrainReport />);
    expect(
      await screen.findByText("Connect your Muse 2", {}, { timeout: 3000 })
    ).toBeInTheDocument();
  });

  it("shows the Today's forecast section header", async () => {
    renderWithProviders(<DailyBrainReport />);
    expect(
      await screen.findByText("Today's forecast", {}, { timeout: 3000 })
    ).toBeInTheDocument();
  });

  it("shows Peak focus row", async () => {
    renderWithProviders(<DailyBrainReport />);
    expect(
      await screen.findByText("Peak focus", {}, { timeout: 3000 })
    ).toBeInTheDocument();
  });

  it("shows the peak focus time window", async () => {
    renderWithProviders(<DailyBrainReport />);
    expect(
      await screen.findByText("9:30 am – 12:00 pm", {}, { timeout: 3000 })
    ).toBeInTheDocument();
  });

  it("shows Likely slump row", async () => {
    renderWithProviders(<DailyBrainReport />);
    expect(
      await screen.findByText("Likely slump", {}, { timeout: 3000 })
    ).toBeInTheDocument();
  });

  it("shows the slump time window", async () => {
    renderWithProviders(<DailyBrainReport />);
    expect(
      await screen.findByText("2:30 pm – 3:30 pm", {}, { timeout: 3000 })
    ).toBeInTheDocument();
  });

  it("shows the Recommended now section header", async () => {
    renderWithProviders(<DailyBrainReport />);
    expect(
      await screen.findByText("Recommended now", {}, { timeout: 3000 })
    ).toBeInTheDocument();
  });

  it("shows default recommended action when no health data", async () => {
    renderWithProviders(<DailyBrainReport />);
    expect(
      await screen.findByText("Start coherence breathing", {}, { timeout: 3000 })
    ).toBeInTheDocument();
  });

  it("shows a Start button for the recommended action", async () => {
    renderWithProviders(<DailyBrainReport />);
    expect(
      await screen.findByRole("button", { name: /Start/ }, { timeout: 3000 })
    ).toBeInTheDocument();
  });

  it("shows the View full session history footer link", async () => {
    renderWithProviders(<DailyBrainReport />);
    expect(
      await screen.findByText("View full session history →", {}, { timeout: 3000 })
    ).toBeInTheDocument();
  });

  it("does not show Yesterday's insight card when health data is empty", async () => {
    renderWithProviders(<DailyBrainReport />);
    // Wait for load to complete by finding a known post-load element
    await screen.findByText("Recommended now", {}, { timeout: 3000 });
    expect(screen.queryByText("Yesterday's insight")).not.toBeInTheDocument();
  });

  it("does not show stress risk row when no health data", async () => {
    renderWithProviders(<DailyBrainReport />);
    await screen.findByText("Today's forecast", {}, { timeout: 3000 });
    expect(screen.queryByText("Stress risk")).not.toBeInTheDocument();
  });
});

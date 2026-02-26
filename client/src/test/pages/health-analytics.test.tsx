import { describe, it, expect, vi, beforeEach, beforeAll } from "vitest";
import { screen, waitFor, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import HealthAnalytics from "@/pages/health-analytics";

// ResizeObserver is used by Recharts
beforeAll(() => {
  global.ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
});

vi.mock("@/hooks/use-device", () => ({
  useDevice: () => ({
    latestFrame: null,
    state: "disconnected",
    deviceStatus: null,
  }),
}));

vi.mock("@/lib/ml-api", () => ({
  listSessions: vi.fn().mockResolvedValue([]),
}));

vi.mock("@/lib/participant", () => ({
  getParticipantId: () => "test-user-123",
}));

vi.mock("@/components/score-circle", () => ({
  ScoreCircle: ({ label, value }: { label: string; value: number }) => (
    <div data-testid={`score-circle-${label}`}>{value}</div>
  ),
}));

vi.mock("@/components/chart-tooltip", () => ({
  ChartTooltip: () => null,
}));

describe("HealthAnalytics page", () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ([]),
    }) as unknown as typeof fetch;
  });

  it("renders without crashing", async () => {
    renderWithProviders(<HealthAnalytics />);
    await waitFor(() => {
      expect(document.body).toBeTruthy();
    });
  });

  it("shows connection banner when device is not streaming", async () => {
    renderWithProviders(<HealthAnalytics />);
    await waitFor(() => {
      expect(
        screen.getByText(/Connect your Muse 2 from the sidebar to see live health analytics/)
      ).toBeInTheDocument();
    });
  });

  it("shows three score gauge cards", async () => {
    renderWithProviders(<HealthAnalytics />);
    await waitFor(() => {
      expect(screen.getByTestId("score-circle-Brain Health")).toBeInTheDocument();
      expect(screen.getByTestId("score-circle-Cognitive")).toBeInTheDocument();
      expect(screen.getByTestId("score-circle-Wellbeing")).toBeInTheDocument();
    });
  });

  it("shows score gauge labels", async () => {
    renderWithProviders(<HealthAnalytics />);
    await waitFor(() => {
      expect(screen.getByText("Focus + Relaxation + Low Stress")).toBeInTheDocument();
      expect(screen.getByText("Focus + Creativity + Memory")).toBeInTheDocument();
      expect(screen.getByText("Relaxation + Low Stress + Flow")).toBeInTheDocument();
    });
  });

  it("shows Brain Health Trends card", async () => {
    renderWithProviders(<HealthAnalytics />);
    await waitFor(() => {
      expect(screen.getByText("Brain Health Trends")).toBeInTheDocument();
    });
  });

  it("shows period selector tabs", async () => {
    renderWithProviders(<HealthAnalytics />);
    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Today" })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Week" })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Month" })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "3 Months" })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Year" })).toBeInTheDocument();
    });
  });

  it("shows empty state message when no data and not streaming", async () => {
    renderWithProviders(<HealthAnalytics />);
    await waitFor(() => {
      expect(screen.getByText("Connect device to see trends")).toBeInTheDocument();
    });
  });

  it("switching period tabs does not crash", async () => {
    renderWithProviders(<HealthAnalytics />);
    const weekTab = await screen.findByRole("button", { name: "Week" });
    fireEvent.click(weekTab);
    await waitFor(() => {
      expect(screen.getByText("Brain Health Trends")).toBeInTheDocument();
    });
  });

  it("switching to non-live period shows no-sessions empty state", async () => {
    renderWithProviders(<HealthAnalytics />);
    const weekTab = await screen.findByRole("button", { name: "Week" });
    fireEvent.click(weekTab);
    await waitFor(() => {
      expect(screen.getByText("No sessions in this period")).toBeInTheDocument();
    });
  });

  it("score gauge values are zero when not streaming", async () => {
    renderWithProviders(<HealthAnalytics />);
    await waitFor(() => {
      const brainHealth = screen.getByTestId("score-circle-Brain Health");
      expect(brainHealth.textContent).toBe("0");
    });
  });
});

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

vi.mock("@/hooks/use-voice-emotion", () => ({
  useVoiceEmotion: () => ({ lastResult: null }),
}));

vi.mock("@/hooks/use-auth", () => ({
  useAuth: () => ({ user: null }),
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
        screen.getByText(/Run a voice check-in or connect EEG/)
      ).toBeInTheDocument();
    });
  });

  it("shows placeholder dashes when no real data", async () => {
    renderWithProviders(<HealthAnalytics />);
    await waitFor(() => {
      // When disconnected with no voice data, shows "—" placeholders
      const dashes = screen.getAllByText("—");
      expect(dashes.length).toBe(3);
    });
  });

  it("shows score label names in placeholders", async () => {
    renderWithProviders(<HealthAnalytics />);
    await waitFor(() => {
      // When no real data, placeholder circles show label text
      expect(screen.getByText("Brain")).toBeInTheDocument();
      expect(screen.getByText("Cognitive")).toBeInTheDocument();
      expect(screen.getByText("Wellbeing")).toBeInTheDocument();
    });
  });

  it("shows Trends card", async () => {
    renderWithProviders(<HealthAnalytics />);
    await waitFor(() => {
      expect(screen.getByText("Trends")).toBeInTheDocument();
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
      expect(screen.getByText("Trends")).toBeInTheDocument();
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

  it("shows Today's Scores section when not streaming", async () => {
    renderWithProviders(<HealthAnalytics />);
    await waitFor(() => {
      expect(screen.getByText("Today's Scores")).toBeInTheDocument();
    });
  });
});

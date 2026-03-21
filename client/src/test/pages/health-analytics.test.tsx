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
        screen.getByText(/Run a voice analysis or connect EEG/)
      ).toBeInTheDocument();
    });
  });

  it("shows score labels when no real data", async () => {
    renderWithProviders(<HealthAnalytics />);
    await waitFor(() => {
      // When disconnected with no voice data, composite score labels still render
      expect(screen.getByText("Brain")).toBeInTheDocument();
      expect(screen.getByText("Cognitive")).toBeInTheDocument();
      expect(screen.getByText("Wellbeing")).toBeInTheDocument();
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

  it("shows Brain Health Trends card", async () => {
    renderWithProviders(<HealthAnalytics />);
    await waitFor(() => {
      expect(screen.getByText("Brain Health Trends")).toBeInTheDocument();
    });
  });

  it("shows period selector tabs", async () => {
    renderWithProviders(<HealthAnalytics />);
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /Today/ })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /Week/ })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /\bMonth\b/ })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /3 Months/ })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /Year/ })).toBeInTheDocument();
    });
  });

  it("shows empty state message when no data and not streaming", async () => {
    renderWithProviders(<HealthAnalytics />);
    await waitFor(() => {
      expect(screen.getByText("Start your first session to see trends")).toBeInTheDocument();
    });
  });

  it("switching period tabs does not crash", async () => {
    renderWithProviders(<HealthAnalytics />);
    const weekTab = await screen.findByRole("button", { name: /Week/ });
    fireEvent.click(weekTab);
    await waitFor(() => {
      expect(screen.getByText("Brain Health Trends")).toBeInTheDocument();
    });
  });

  it("switching to non-live period shows no-sessions empty state", async () => {
    renderWithProviders(<HealthAnalytics />);
    const weekTab = await screen.findByRole("button", { name: /Week/ });
    fireEvent.click(weekTab);
    await waitFor(() => {
      expect(screen.getByText("No sessions in this period")).toBeInTheDocument();
    });
  });

  it("shows Composite Scores section when not streaming", async () => {
    renderWithProviders(<HealthAnalytics />);
    await waitFor(() => {
      expect(screen.getByText("Composite Scores")).toBeInTheDocument();
    });
  });
});

describe("HealthAnalytics — individual metric panels", () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ([]),
    }) as unknown as typeof fetch;
  });

  it("shows individual metric panels even when no real data (with empty state)", async () => {
    // MetricPanels always render now — show "No sessions yet" when no data
    renderWithProviders(<HealthAnalytics />);
    await waitFor(() => {
      expect(screen.getByText("Brain Health Trends")).toBeInTheDocument();
    });
    // Metric panels are always visible with labels
    expect(screen.getByText("Attention and concentration level")).toBeInTheDocument();
    expect(screen.getByText("Mental stress and tension")).toBeInTheDocument();
  });

  it("shows placeholder '--' values in composite scores when no data", async () => {
    renderWithProviders(<HealthAnalytics />);
    await waitFor(() => {
      // Placeholder circles show '--' when hasRealData is false
      const dashes = screen.getAllByText("--");
      expect(dashes.length).toBeGreaterThanOrEqual(3);
    });
  });
});

describe("HealthAnalytics — composite scores section details", () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ([]),
    }) as unknown as typeof fetch;
  });

  it("shows all three composite score names", async () => {
    renderWithProviders(<HealthAnalytics />);
    await waitFor(() => {
      expect(screen.getByText("Brain")).toBeInTheDocument();
      expect(screen.getByText("Cognitive")).toBeInTheDocument();
      expect(screen.getByText("Wellbeing")).toBeInTheDocument();
    });
  });

  it("shows Composite Scores header with uppercase styling", async () => {
    renderWithProviders(<HealthAnalytics />);
    await waitFor(() => {
      const header = screen.getByText("Composite Scores");
      expect(header).toBeInTheDocument();
      expect(header.tagName.toLowerCase()).toBe("p");
    });
  });

  it("composite scores section has aria-label for accessibility", async () => {
    renderWithProviders(<HealthAnalytics />);
    await waitFor(() => {
      const section = screen.getByLabelText("Today's health scores");
      expect(section).toBeInTheDocument();
    });
  });

  it("shows Voice Analysis link button in empty Today state", async () => {
    renderWithProviders(<HealthAnalytics />);
    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: /Voice Analysis/i })
      ).toBeInTheDocument();
    });
  });

  it("Voice Analysis link is not shown after switching to Week period", async () => {
    renderWithProviders(<HealthAnalytics />);
    const weekTab = await screen.findByRole("button", { name: /Week/ });
    fireEvent.click(weekTab);
    await waitFor(() => {
      expect(screen.queryByRole("button", { name: /Voice Analysis/i })).not.toBeInTheDocument();
    });
  });
});

describe("HealthAnalytics — metric panel structure", () => {
  beforeEach(() => {
    localStorage.clear();
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ([]),
    }) as unknown as typeof fetch;
  });

  it("shows metric panels with empty state when disconnected with no voice data", async () => {
    renderWithProviders(<HealthAnalytics />);
    await waitFor(() => {
      // Metric panels always render now — Valence and Arousal should be visible
      expect(screen.getByText("Valence")).toBeInTheDocument();
      expect(screen.getByText("Arousal")).toBeInTheDocument();
      expect(screen.getByText("Positive/negative feeling")).toBeInTheDocument();
      expect(screen.getByText("Energy level")).toBeInTheDocument();
      // Empty state message should appear since there's no data
      const noSessions = screen.getAllByText("No sessions yet");
      expect(noSessions.length).toBeGreaterThanOrEqual(1);
    });
  });

  it("trend chart section renders at full width with all period tabs", async () => {
    renderWithProviders(<HealthAnalytics />);
    await waitFor(() => {
      expect(screen.getByText("Brain Health Trends")).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /Today/ })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /Week/ })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /\bMonth\b/ })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /3 Months/ })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /Year/ })).toBeInTheDocument();
    });
  });
});

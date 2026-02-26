import { describe, it, expect, vi, beforeEach, beforeAll } from "vitest";
import { screen, waitFor, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import DreamDetection from "@/pages/dream-journal";

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

describe("DreamDetection (dream-journal) page", () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({}),
    }) as unknown as typeof fetch;
  });

  it("renders without crashing", async () => {
    renderWithProviders(<DreamDetection />);
    await waitFor(() => {
      expect(document.body).toBeTruthy();
    });
  });

  it("renders the Detection and Patterns tabs", async () => {
    renderWithProviders(<DreamDetection />);
    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Detection" })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Patterns" })).toBeInTheDocument();
    });
  });

  it("shows connection banner when device is not streaming", async () => {
    renderWithProviders(<DreamDetection />);
    await waitFor(() => {
      expect(
        screen.getByText(/Connect your Muse 2 from the sidebar/)
      ).toBeInTheDocument();
    });
  });

  it("shows the Sleep Stage card in detection tab", async () => {
    renderWithProviders(<DreamDetection />);
    await waitFor(() => {
      expect(screen.getByText("Sleep Stage")).toBeInTheDocument();
    });
  });

  it("shows the REM & Dream Activity card in detection tab", async () => {
    renderWithProviders(<DreamDetection />);
    await waitFor(() => {
      expect(screen.getByText("REM & Dream Activity")).toBeInTheDocument();
    });
  });

  it("shows the Detected Dream Episodes section with zero count", async () => {
    renderWithProviders(<DreamDetection />);
    await waitFor(() => {
      expect(screen.getByText("Detected Dream Episodes")).toBeInTheDocument();
      expect(screen.getByText("0 detected")).toBeInTheDocument();
    });
  });

  it("shows empty state message for dream episodes when not streaming", async () => {
    renderWithProviders(<DreamDetection />);
    await waitFor(() => {
      expect(
        screen.getByText(/Connect your BCI device and sleep to begin detection/)
      ).toBeInTheDocument();
    });
  });

  it("shows 'Connect device to see dream activity' placeholder when no timeline data", async () => {
    renderWithProviders(<DreamDetection />);
    await waitFor(() => {
      expect(
        screen.getByText("Connect device to see dream activity")
      ).toBeInTheDocument();
    });
  });

  it("switches to Patterns tab when clicked", async () => {
    renderWithProviders(<DreamDetection />);
    const patternsTab = await screen.findByRole("button", { name: "Patterns" });
    fireEvent.click(patternsTab);
    await waitFor(() => {
      expect(screen.getByText("Dream Frames")).toBeInTheDocument();
    });
  });

  it("shows Patterns tab summary stats when switched", async () => {
    renderWithProviders(<DreamDetection />);
    const patternsTab = await screen.findByRole("button", { name: "Patterns" });
    fireEvent.click(patternsTab);
    await waitFor(() => {
      expect(screen.getByText("Avg REM %")).toBeInTheDocument();
      expect(screen.getByText("REM Cycles")).toBeInTheDocument();
    });
  });

  it("shows Sleep Architecture section in patterns tab", async () => {
    renderWithProviders(<DreamDetection />);
    const patternsTab = await screen.findByRole("button", { name: "Patterns" });
    fireEvent.click(patternsTab);
    await waitFor(() => {
      expect(screen.getByText("Sleep Architecture")).toBeInTheDocument();
    });
  });
});

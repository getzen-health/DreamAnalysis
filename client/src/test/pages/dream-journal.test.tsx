import { describe, it, expect, vi, beforeEach, beforeAll } from "vitest";
import { screen, waitFor } from "@testing-library/react";
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

  it("shows Tonight card header", () => {
    renderWithProviders(<DreamDetection />);
    expect(screen.getByText("Tonight")).toBeInTheDocument();
  });

  it("shows Connect your device heading when not streaming", () => {
    renderWithProviders(<DreamDetection />);
    expect(screen.getByText("Connect your device")).toBeInTheDocument();
  });

  it("shows Muse 2 sleep wear message when not streaming", () => {
    renderWithProviders(<DreamDetection />);
    expect(
      screen.getByText(/Wear your Muse 2 while you sleep/)
    ).toBeInTheDocument();
  });

  it("shows automatic dream detection message", () => {
    renderWithProviders(<DreamDetection />);
    expect(
      screen.getByText(/the app will detect your dreams automatically/)
    ).toBeInTheDocument();
  });

  it("shows Episodes tonight section", () => {
    renderWithProviders(<DreamDetection />);
    expect(screen.getByText("Episodes tonight")).toBeInTheDocument();
  });

  it("shows 0 detected episode count", () => {
    renderWithProviders(<DreamDetection />);
    expect(screen.getByText("0 detected")).toBeInTheDocument();
  });

  it("shows empty state message when not streaming", () => {
    renderWithProviders(<DreamDetection />);
    expect(
      screen.getByText("Start a sleep session to record dream episodes.")
    ).toBeInTheDocument();
  });

  it("shows morning dream record button", () => {
    renderWithProviders(<DreamDetection />);
    expect(screen.getByText("Record this morning's dream")).toBeInTheDocument();
  });

  it("shows dream journal entry hint", () => {
    renderWithProviders(<DreamDetection />);
    expect(
      screen.getByText("Write what you remember — even a word counts")
    ).toBeInTheDocument();
  });

  it("renders the full page without errors", () => {
    renderWithProviders(<DreamDetection />);
    expect(screen.getByText("Tonight")).toBeInTheDocument();
    expect(screen.getByText("Episodes tonight")).toBeInTheDocument();
  });
});

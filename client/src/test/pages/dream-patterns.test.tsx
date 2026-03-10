import { describe, it, expect, vi, beforeEach, beforeAll } from "vitest";
import { screen, waitFor, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import DreamPatterns from "@/pages/dream-patterns";

beforeAll(() => {
  global.ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
});

vi.mock("@/lib/participant", () => ({ getParticipantId: () => "test-user-123" }));

vi.mock("@/hooks/use-device", () => ({
  useDevice: () => ({
    state: "disconnected",
    latestFrame: null,
    connect: vi.fn(),
    disconnect: vi.fn(),
  }),
}));

vi.mock("@/lib/ml-api", () => ({
  listSessions: vi.fn().mockResolvedValue([]),
}));

vi.mock("@/components/chart-tooltip", () => ({
  ChartTooltip: () => null,
}));

describe("DreamPatterns page", () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({}),
    }) as unknown as typeof fetch;
  });

  it("renders without crashing", async () => {
    renderWithProviders(<DreamPatterns />);
    await waitFor(() => expect(document.body).toBeTruthy());
  });

  it("shows Dream Patterns heading", async () => {
    renderWithProviders(<DreamPatterns />);
    await waitFor(() => {
      expect(screen.getByText("Dream Patterns")).toBeInTheDocument();
    });
  });

  it("shows connection banner on Today view when not streaming", async () => {
    renderWithProviders(<DreamPatterns />);
    await waitFor(() => {
      expect(
        screen.getByText(/Connect Muse 2 for live sleep staging and automatic dream detection/)
      ).toBeInTheDocument();
    });
  });

  it("shows period selector tabs (Today, Week, Month)", async () => {
    renderWithProviders(<DreamPatterns />);
    await waitFor(() => {
      expect(screen.getByText("Today")).toBeInTheDocument();
      expect(screen.getByText("Week")).toBeInTheDocument();
      expect(screen.getByText("Month")).toBeInTheDocument();
    });
  });

  it("shows all period tabs including 3 Months and Year", async () => {
    renderWithProviders(<DreamPatterns />);
    await waitFor(() => {
      expect(screen.getByText("3 Months")).toBeInTheDocument();
      expect(screen.getByText("Year")).toBeInTheDocument();
    });
  });

  it("shows live Today summary cards (Dream Frames, Avg REM %, Avg Intensity, REM Cycles)", async () => {
    renderWithProviders(<DreamPatterns />);
    await waitFor(() => {
      expect(screen.getByText("Dream Frames")).toBeInTheDocument();
      expect(screen.getByText("Avg REM %")).toBeInTheDocument();
      expect(screen.getByText("Avg Intensity")).toBeInTheDocument();
      expect(screen.getByText("REM Cycles")).toBeInTheDocument();
    });
  });

  it("shows Sleep Architecture chart section (today view)", async () => {
    renderWithProviders(<DreamPatterns />);
    await waitFor(() => {
      expect(screen.getByText("Sleep Architecture")).toBeInTheDocument();
    });
  });

  it("shows Dream Activity chart section", async () => {
    renderWithProviders(<DreamPatterns />);
    await waitFor(() => {
      expect(screen.getByText("Dream Activity")).toBeInTheDocument();
    });
  });

  it("shows REM Likelihood chart section", async () => {
    renderWithProviders(<DreamPatterns />);
    await waitFor(() => {
      expect(screen.getByText("REM Likelihood")).toBeInTheDocument();
    });
  });

  it("switching to Week tab shows historical summary cards", async () => {
    renderWithProviders(<DreamPatterns />);
    await waitFor(() => expect(screen.getByText("Week")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Week"));
    await waitFor(() => {
      expect(screen.getByText("Sessions")).toBeInTheDocument();
      expect(screen.getByText("Total Hours")).toBeInTheDocument();
    });
  });

  it("switching to Week tab shows Sleep Session Wellness chart", async () => {
    renderWithProviders(<DreamPatterns />);
    await waitFor(() => expect(screen.getByText("Week")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Week"));
    await waitFor(() => {
      expect(screen.getByText("Sleep Session Wellness")).toBeInTheDocument();
    });
  });
});

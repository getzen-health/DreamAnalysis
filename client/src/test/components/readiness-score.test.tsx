import { describe, it, expect, vi, afterEach, beforeAll } from "vitest";
import { screen, waitFor, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import { ReadinessScore } from "@/components/readiness-score";

beforeAll(() => {
  global.ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
});

vi.mock("@/hooks/use-theme", () => ({
  useTheme: () => ({ theme: "dark", setTheme: vi.fn() }),
}));

// ── helpers ───────────────────────────────────────────────────────────────────

const MOCK_DATA = {
  user_id: "test-user",
  score: 78,
  color: "green" as const,
  label: "Ready",
  factors: {
    sleep_quality: 85,
    stress_avg: 30,
    hrv_trend: 72,
    voice_emotion: 65,
  },
  history: [
    { date: "2026-03-06", score: 60 },
    { date: "2026-03-07", score: 65 },
    { date: "2026-03-08", score: 55 },
    { date: "2026-03-09", score: 70 },
    { date: "2026-03-10", score: 72 },
    { date: "2026-03-11", score: 75 },
    { date: "2026-03-12", score: 78 },
  ],
};

function mockFetch(data: typeof MOCK_DATA) {
  global.fetch = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => data,
  } as Response);
}

afterEach(() => {
  vi.restoreAllMocks();
});

// ── tests ─────────────────────────────────────────────────────────────────────

describe("ReadinessScore", () => {
  it("shows a loading spinner while fetching", () => {
    global.fetch = vi.fn().mockReturnValue(new Promise(() => {}));
    renderWithProviders(<ReadinessScore userId="test-user" />);
    expect(document.querySelector(".animate-spin")).toBeInTheDocument();
  });

  it("renders score and label after successful fetch", async () => {
    mockFetch(MOCK_DATA);
    renderWithProviders(<ReadinessScore userId="test-user" />);
    await waitFor(() => {
      expect(screen.getByText("78")).toBeInTheDocument();
    });
    expect(screen.getByText("Ready")).toBeInTheDocument();
    expect(screen.getByText("Brain Readiness")).toBeInTheDocument();
  });

  it("does not render content on fetch error", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      json: async () => ({}),
    } as Response);
    renderWithProviders(<ReadinessScore userId="test-user" />);
    // Score text should never appear when fetch fails
    await waitFor(
      () => {
        expect(screen.queryByText("Brain Readiness")).not.toBeInTheDocument();
      },
      { timeout: 200 }
    );
  });

  it("shows factor breakdown when expand button is clicked", async () => {
    mockFetch(MOCK_DATA);
    renderWithProviders(<ReadinessScore userId="test-user" />);

    await waitFor(() => {
      expect(screen.getByText("Show breakdown")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText("Show breakdown"));

    expect(screen.getByText("Sleep quality")).toBeInTheDocument();
    expect(screen.getByText("Stress (inverted)")).toBeInTheDocument();
    expect(screen.getByText("HRV trend")).toBeInTheDocument();
    expect(screen.getByText("Voice emotion")).toBeInTheDocument();
  });

  it("hides factor breakdown after collapsing", async () => {
    mockFetch(MOCK_DATA);
    renderWithProviders(<ReadinessScore userId="test-user" />);

    await waitFor(() => {
      expect(screen.getByText("Show breakdown")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText("Show breakdown"));
    expect(screen.getByText("Sleep quality")).toBeInTheDocument();

    fireEvent.click(screen.getByText("Hide breakdown"));
    expect(screen.queryByText("Sleep quality")).not.toBeInTheDocument();
  });

  it("shows 'no data' for null factor values", async () => {
    const nullFactors = {
      ...MOCK_DATA,
      factors: { sleep_quality: null, stress_avg: null, hrv_trend: null, voice_emotion: null },
    };
    mockFetch(nullFactors as typeof MOCK_DATA);
    renderWithProviders(<ReadinessScore userId="test-user" />);

    await waitFor(() => {
      expect(screen.getByText("Show breakdown")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText("Show breakdown"));
    const noDataElements = screen.getAllByText("no data");
    expect(noDataElements.length).toBe(4);
  });

  it("renders score circle SVG with aria-label", async () => {
    mockFetch(MOCK_DATA);
    renderWithProviders(<ReadinessScore userId="test-user" />);
    await waitFor(() => {
      expect(screen.getByLabelText("Readiness score: 78")).toBeInTheDocument();
    });
  });
});

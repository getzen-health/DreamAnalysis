import { describe, it, expect, vi, afterEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import { EFSMiniCard } from "@/components/efs-mini-card";

vi.mock("wouter", () => ({
  useLocation: () => ["/", vi.fn()],
}));

// ── Helpers ───────────────────────────────────────────────────────────────────

const MOCK_EFS_DATA = {
  score: 82,
  color: "green" as const,
  label: "Strong",
  confidence: "full",
  trend: { direction: "up" as const, delta: 8, period: "30d" },
  vitals: {},
  dailyInsight: null,
  computedAt: "2026-03-19T08:00:00Z",
};

const MOCK_BUILDING_DATA = {
  score: null,
  color: null,
  label: null,
  confidence: "building",
  progress: { daysTracked: 1, daysRequired: 3, percentage: 33, message: "Keep tracking" },
  trend: null,
  vitals: {},
  dailyInsight: null,
  computedAt: null,
};

function mockFetch(data: any) {
  global.fetch = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => data,
  } as Response);
}

afterEach(() => {
  vi.restoreAllMocks();
});

// ── Tests ─────────────────────────────────────────────────────────────────────

describe("EFSMiniCard", () => {
  it("shows loading state while fetching", () => {
    global.fetch = vi.fn().mockReturnValue(new Promise(() => {}));
    renderWithProviders(<EFSMiniCard userId="test-user" />);
    expect(document.querySelector(".animate-spin")).toBeInTheDocument();
  });

  it("renders score when data is available", async () => {
    mockFetch(MOCK_EFS_DATA);
    renderWithProviders(<EFSMiniCard userId="test-user" />);
    await waitFor(() => {
      expect(screen.getByText("82")).toBeInTheDocument();
    });
    expect(screen.getByText("Emotional Fitness")).toBeInTheDocument();
  });

  it("shows 'Building...' text when score is null with progress", async () => {
    mockFetch(MOCK_BUILDING_DATA);
    renderWithProviders(<EFSMiniCard userId="test-user" />);
    await waitFor(() => {
      expect(screen.getByText("Building...")).toBeInTheDocument();
    });
  });

  it("shows progress ring percentage when building", async () => {
    mockFetch(MOCK_BUILDING_DATA);
    renderWithProviders(<EFSMiniCard userId="test-user" />);
    await waitFor(() => {
      expect(screen.getByText("33%")).toBeInTheDocument();
    });
  });

  it("shows 'No data yet' when score is null and no progress", async () => {
    const noProgress = { ...MOCK_BUILDING_DATA, progress: undefined };
    mockFetch(noProgress);
    renderWithProviders(<EFSMiniCard userId="test-user" />);
    await waitFor(() => {
      expect(screen.getByText("No data yet")).toBeInTheDocument();
    });
  });

  it("applies correct color class for score", async () => {
    mockFetch(MOCK_EFS_DATA);
    renderWithProviders(<EFSMiniCard userId="test-user" />);
    await waitFor(() => {
      const scoreEl = screen.getByText("82");
      expect(scoreEl.className).toContain("text-cyan-400");
    });
  });

  it("does not render content on fetch error", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      json: async () => ({}),
    } as Response);
    renderWithProviders(<EFSMiniCard userId="test-user" />);
    await waitFor(
      () => {
        expect(screen.queryByText("Emotional Fitness")).not.toBeInTheDocument();
      },
      { timeout: 200 }
    );
  });
});

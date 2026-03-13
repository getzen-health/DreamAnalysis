import { describe, it, expect, vi, afterEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import { StreakBadge } from "@/components/streak-badge";

// ── helpers ───────────────────────────────────────────────────────────────────

function mockFetch(data: object) {
  global.fetch = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => data,
  } as Response);
}

const BASE_STREAK = {
  user_id: "test-user",
  current_streak: 5,
  best_streak: 12,
  today_checked_in: false,
  milestones: [7, 30, 100],
  next_milestone: 7,
  total_checkins: 20,
};

afterEach(() => {
  vi.restoreAllMocks();
});

// ── tests ─────────────────────────────────────────────────────────────────────

describe("StreakBadge", () => {
  it("shows a loading spinner while fetching", () => {
    global.fetch = vi.fn().mockReturnValue(new Promise(() => {}));
    renderWithProviders(<StreakBadge userId="test-user" />);
    expect(document.querySelector(".animate-spin")).toBeInTheDocument();
  });

  it("renders current streak count", async () => {
    mockFetch(BASE_STREAK);
    renderWithProviders(<StreakBadge userId="test-user" />);
    await waitFor(() => {
      expect(screen.getByText("5 day streak")).toBeInTheDocument();
    });
  });

  it("renders best streak subtitle", async () => {
    mockFetch(BASE_STREAK);
    renderWithProviders(<StreakBadge userId="test-user" />);
    await waitFor(() => {
      expect(screen.getByText(/Best: 12 days/)).toBeInTheDocument();
    });
  });

  it("shows days-to-next-milestone hint", async () => {
    mockFetch(BASE_STREAK);
    renderWithProviders(<StreakBadge userId="test-user" />);
    await waitFor(() => {
      // 7 - 5 = 2 days to 7-day milestone
      expect(screen.getByText(/2 days to 7-day milestone/)).toBeInTheDocument();
    });
  });

  it("shows check-in nudge when not checked in today", async () => {
    mockFetch(BASE_STREAK);
    renderWithProviders(<StreakBadge userId="test-user" />);
    await waitFor(() => {
      expect(screen.getByText(/Do a voice check-in today/)).toBeInTheDocument();
    });
  });

  it("does not show nudge when already checked in", async () => {
    mockFetch({ ...BASE_STREAK, today_checked_in: true });
    renderWithProviders(<StreakBadge userId="test-user" />);
    await waitFor(() => {
      expect(screen.getByText("5 day streak")).toBeInTheDocument();
    });
    expect(screen.queryByText(/Do a voice check-in today/)).not.toBeInTheDocument();
  });

  it("renders 3 milestone dots", async () => {
    mockFetch(BASE_STREAK);
    renderWithProviders(<StreakBadge userId="test-user" />);
    await waitFor(() => {
      expect(screen.getByText("5 day streak")).toBeInTheDocument();
    });
    expect(screen.getByText("7")).toBeInTheDocument();
    expect(screen.getByText("30")).toBeInTheDocument();
    expect(screen.getByText("100")).toBeInTheDocument();
  });

  it("shows 'Start your streak' when streak is 0", async () => {
    mockFetch({ ...BASE_STREAK, current_streak: 0, best_streak: 0 });
    renderWithProviders(<StreakBadge userId="test-user" />);
    await waitFor(() => {
      expect(screen.getByText("Start your streak")).toBeInTheDocument();
    });
  });

  it("does not render streak count on fetch error", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      json: async () => ({}),
    } as Response);
    renderWithProviders(<StreakBadge userId="test-user" />);
    // Component stays in loading/error state — streak text should never appear
    await waitFor(
      () => {
        expect(screen.queryByText("5 day streak")).not.toBeInTheDocument();
      },
      { timeout: 200 }
    );
  });

  it("compact mode renders inline badge", async () => {
    mockFetch(BASE_STREAK);
    renderWithProviders(<StreakBadge userId="test-user" compact />);
    await waitFor(() => {
      expect(screen.getByText("5d")).toBeInTheDocument();
    });
  });

  it("compact mode shows best streak when lower than current is not shown", async () => {
    mockFetch({ ...BASE_STREAK, current_streak: 12, best_streak: 12 });
    renderWithProviders(<StreakBadge userId="test-user" compact />);
    await waitFor(() => {
      expect(screen.getByText("12d")).toBeInTheDocument();
    });
    // best is same as current — subtitle not shown
    expect(screen.queryByText(/· best/)).not.toBeInTheDocument();
  });
});

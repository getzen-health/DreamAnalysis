import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach, beforeAll } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import Discover from "@/pages/discover";

beforeAll(() => {
  global.ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
});

vi.mock("@/lib/participant", () => ({ getParticipantId: () => "test-user-123" }));

vi.mock("wouter", () => ({
  useLocation: () => ["/discover", vi.fn()],
}));

vi.mock("@/lib/animations", () => ({
  pageTransition: { initial: {}, animate: {}, transition: {} },
  cardVariants: { hidden: {}, visible: {} },
}));

vi.mock("@/hooks/use-health-sync", () => ({
  useHealthSync: () => ({ latestPayload: null }),
}));

vi.mock("@/lib/mood-patterns", () => ({
  detectMoodPatterns: () => [],
}));

vi.mock("@/lib/ml-api", () => ({
  listSessions: vi.fn().mockResolvedValue([]),
}));

vi.mock("@/lib/queryClient", () => ({
  resolveUrl: (url: string) => url,
}));

vi.mock("framer-motion", () => ({
  motion: {
    main: React.forwardRef(({ children, ...props }: React.PropsWithChildren<Record<string, unknown>>, ref: React.Ref<HTMLElement>) =>
      React.createElement("main", { ...props, ref }, children)
    ),
    button: React.forwardRef(({ children, ...props }: React.PropsWithChildren<Record<string, unknown>>, ref: React.Ref<HTMLButtonElement>) =>
      React.createElement("button", { ...props, ref }, children)
    ),
  },
}));

describe("Discover page", () => {
  beforeEach(() => {
    localStorage.clear();
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => [],
    }) as unknown as typeof fetch;
  });

  afterEach(() => {
    localStorage.clear();
  });

  it("renders without crashing", async () => {
    renderWithProviders(<Discover />);
    await waitFor(() => {
      expect(screen.getByText("Discover")).toBeInTheDocument();
    });
  });

  it("shows empty state when no emotion data", async () => {
    renderWithProviders(<Discover />);
    await waitFor(() => {
      expect(screen.getByText("Complete a voice analysis to see trends")).toBeInTheDocument();
    });
  });

  it("shows chart when localStorage emotion history has data", async () => {
    // Seed localStorage with multiple emotion history entries for TODAY
    // (the default time range is "today" so entries must be from today)
    const now = new Date();
    const history = [
      { stress: 0.3, happiness: 0.7, focus: 0.5, dominantEmotion: "happy", timestamp: new Date(now.getTime() - 3600000).toISOString() },
      { stress: 0.4, happiness: 0.6, focus: 0.6, dominantEmotion: "happy", timestamp: new Date(now.getTime() - 1800000).toISOString() },
      { stress: 0.2, happiness: 0.8, focus: 0.7, dominantEmotion: "neutral", timestamp: now.toISOString() },
    ];
    localStorage.setItem("ndw_emotion_history", JSON.stringify(history));

    renderWithProviders(<Discover />);
    await waitFor(() => {
      // Chart should be visible (the empty state message should NOT appear)
      expect(screen.queryByText("Complete a voice analysis to see trends")).not.toBeInTheDocument();
    });
  });

  it("accumulates emotion history from voice checkin events", async () => {
    // Simulate a voice checkin result in localStorage
    localStorage.setItem("ndw_last_emotion", JSON.stringify({
      result: {
        emotion: "happy",
        valence: 0.6,
        stress_index: 0.3,
        focus_index: 0.7,
        relaxation_index: 0.7,
        confidence: 0.85,
      },
      timestamp: Date.now(),
    }));

    renderWithProviders(<Discover />);

    await waitFor(() => {
      // Verify emotion history was created in localStorage
      const historyRaw = localStorage.getItem("ndw_emotion_history");
      expect(historyRaw).not.toBeNull();
      const history = JSON.parse(historyRaw!);
      expect(history.length).toBeGreaterThanOrEqual(1);
      expect(history[0].dominantEmotion).toBe("happy");
    });
  });

  it("shows Emotions heading", async () => {
    renderWithProviders(<Discover />);
    await waitFor(() => {
      expect(screen.getByText("Emotions")).toBeInTheDocument();
    });
  });

  it("shows navigation cards for all categories", async () => {
    renderWithProviders(<Discover />);
    await waitFor(() => {
      // "Sleep" appears in both EEG Music fallback and nav grid, so use getAllByText
      expect(screen.getAllByText("Sleep").length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText("Brain")).toBeInTheDocument();
      expect(screen.getByText("Health")).toBeInTheDocument();
      expect(screen.getByText("Inner Energy")).toBeInTheDocument();
      expect(screen.getByText("Wellness")).toBeInTheDocument();
    });
  });

  it("shows legend labels for Stress and Focus", async () => {
    renderWithProviders(<Discover />);
    await waitFor(() => {
      // Legend items — Mood was removed from chart, only Stress and Focus remain
      expect(screen.getAllByText("Stress").length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText("Focus").length).toBeGreaterThanOrEqual(1);
    });
  });

  it("shows continuous trend subtitle on emotions card", async () => {
    renderWithProviders(<Discover />);
    await waitFor(() => {
      expect(screen.getByText("Stress & Focus — continuous trend")).toBeInTheDocument();
    });
  });
});

describe("localStorage emotion history", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  afterEach(() => {
    localStorage.clear();
  });

  it("caps history at 200 entries", () => {
    // Pre-fill with 200 entries
    const entries = Array.from({ length: 200 }, (_, i) => ({
      stress: 0.3,
      happiness: 0.5,
      focus: 0.5,
      dominantEmotion: "neutral",
      timestamp: new Date(Date.now() - (200 - i) * 60000).toISOString(),
    }));
    localStorage.setItem("ndw_emotion_history", JSON.stringify(entries));

    // Trigger the import to access the functions
    // We test this indirectly through the Discover page rendering
    localStorage.setItem("ndw_last_emotion", JSON.stringify({
      result: {
        emotion: "happy",
        valence: 0.5,
        stress_index: 0.2,
        focus_index: 0.6,
      },
      timestamp: Date.now(),
    }));

    renderWithProviders(<Discover />);

    waitFor(() => {
      const raw = localStorage.getItem("ndw_emotion_history");
      const history = JSON.parse(raw!);
      expect(history.length).toBeLessThanOrEqual(200);
    });
  });

  it("prunes entries older than 7 days", () => {
    const oldEntry = {
      stress: 0.5,
      happiness: 0.5,
      focus: 0.5,
      dominantEmotion: "sad",
      timestamp: new Date(Date.now() - 8 * 86400000).toISOString(), // 8 days ago
    };
    const recentEntry = {
      stress: 0.3,
      happiness: 0.7,
      focus: 0.6,
      dominantEmotion: "happy",
      timestamp: new Date(Date.now() - 1 * 86400000).toISOString(), // 1 day ago
    };
    localStorage.setItem("ndw_emotion_history", JSON.stringify([oldEntry, recentEntry]));

    // Trigger via adding new emotion
    localStorage.setItem("ndw_last_emotion", JSON.stringify({
      result: {
        emotion: "neutral",
        valence: 0,
        stress_index: 0.4,
        focus_index: 0.5,
      },
      timestamp: Date.now(),
    }));

    renderWithProviders(<Discover />);

    waitFor(() => {
      const raw = localStorage.getItem("ndw_emotion_history");
      const history = JSON.parse(raw!);
      // Old entry (8 days) should be pruned
      const emotions = history.map((e: { dominantEmotion: string }) => e.dominantEmotion);
      expect(emotions).not.toContain("sad");
      expect(emotions).toContain("happy");
    });
  });
});

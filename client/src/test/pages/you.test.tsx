import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach, beforeAll } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import You from "@/pages/you";

beforeAll(() => {
  global.ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
});

vi.mock("framer-motion", () => ({
  motion: {
    main: React.forwardRef(
      (
        { children, ...props }: React.PropsWithChildren<Record<string, unknown>>,
        ref: React.Ref<HTMLElement>,
      ) => React.createElement("main", { ...props, ref }, children),
    ),
  },
}));

vi.mock("@/lib/animations", () => ({
  pageTransition: { initial: {}, animate: {}, transition: {} },
}));

vi.mock("@/hooks/use-auth", () => ({
  useAuth: () => ({
    user: { id: 1, username: "TestUser", createdAt: "2026-01-01" },
    logout: vi.fn(),
  }),
}));

vi.mock("@/hooks/use-theme", () => ({
  useTheme: () => ({
    theme: "dark",
    themeSetting: "dark",
    setTheme: vi.fn(),
  }),
}));

vi.mock("@/lib/participant", () => ({
  getParticipantId: () => "test-user-123",
}));

vi.mock("@/lib/ml-api", () => ({
  listSessions: vi.fn().mockResolvedValue([]),
}));

vi.mock("@/components/chronotype-quiz", () => ({
  ChronotypeQuiz: () => <div data-testid="chronotype-quiz" />,
}));

vi.mock("@/lib/chronotype", () => ({
  getStoredChronotype: () => null,
}));

vi.mock("@/components/notification-prefs-sheet", () => ({
  NotificationPrefsSheet: () => <div data-testid="notification-prefs-sheet" />,
}));

vi.mock("wouter", () => ({
  useLocation: () => ["/you", vi.fn()],
}));

describe("You page", () => {
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

  it("renders the user name", () => {
    renderWithProviders(<You />);
    expect(screen.getByText("TestUser")).toBeInTheDocument();
  });

  it("shows achievements link card instead of inline badges", () => {
    renderWithProviders(<You />);
    // Should have the link card text
    expect(screen.getByText("Achievements")).toBeInTheDocument();
    expect(
      screen.getByText("View your badges and milestones"),
    ).toBeInTheDocument();
  });

  it("does NOT render inline AchievementBadges component", () => {
    renderWithProviders(<You />);
    // The AchievementBadges component has a testid "overall-progress"
    // which should NOT be present since we replaced it with a link card
    expect(screen.queryByTestId("overall-progress")).not.toBeInTheDocument();
  });

  it("shows Connected Assets section with single consolidated item", () => {
    renderWithProviders(<You />);
    expect(screen.getByText("Connected Assets")).toBeInTheDocument();
    // Should show the single "Connected Devices" item
    expect(screen.getByText("Connected Devices")).toBeInTheDocument();
  });

  it("does NOT show separate Health Connect, BCI, or Wearables items", () => {
    renderWithProviders(<You />);
    // These 3 separate items should no longer exist
    expect(screen.queryByText("Health Connect")).not.toBeInTheDocument();
    expect(screen.queryByText("Google Health Connect")).not.toBeInTheDocument();
    expect(screen.queryByText("Apple HealthKit")).not.toBeInTheDocument();
    expect(screen.queryByText("BCI / EEG")).not.toBeInTheDocument();
    expect(screen.queryByText("Wearables")).not.toBeInTheDocument();
  });

  it("shows connected device count in the subtitle", () => {
    renderWithProviders(<You />);
    // Default state: 0 devices connected out of 7
    expect(screen.getByText("0 of 7 connected")).toBeInTheDocument();
  });

  it("shows correct device count when devices are connected", () => {
    localStorage.setItem("ndw_health_connect_granted", "true");
    localStorage.setItem("ndw_muse_connected", "true");
    renderWithProviders(<You />);
    expect(screen.getByText("2 of 7 connected")).toBeInTheDocument();
  });

  it("shows Settings section", () => {
    renderWithProviders(<You />);
    expect(screen.getByText("Settings")).toBeInTheDocument();
  });

  it("shows Sign Out button", () => {
    renderWithProviders(<You />);
    expect(screen.getByText("Sign Out")).toBeInTheDocument();
  });
});

import { describe, it, expect, vi, beforeAll } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import Dashboard from "@/pages/dashboard";

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

vi.mock("@/hooks/use-theme", () => ({
  useTheme: () => ({ theme: "dark", setTheme: vi.fn() }),
}));

vi.mock("wouter", () => ({
  useLocation: () => ["/", vi.fn()],
  Link: (props: any) => <a href={props.href}>{props.children}</a>,
}));

vi.mock("@/hooks/use-auth", () => ({
  useAuth: () => ({ user: null }),
}));

vi.mock("@/hooks/use-health-sync", () => ({
  useHealthSync: () => ({ latestPayload: null }),
}));

vi.mock("@/lib/ml-api", () => ({
  getHealthInsights: vi.fn().mockResolvedValue([]),
  getBaselineStatus: vi.fn().mockResolvedValue({ ready: true }),
  listSessions: vi.fn().mockResolvedValue([]),
}));

vi.mock("@/lib/participant", () => ({
  getParticipantId: () => "test-participant-uuid",
}));

describe("Dashboard page", () => {
  it("renders without crashing", () => {
    renderWithProviders(<Dashboard />);
  });

  it("shows all four quick action cards", () => {
    renderWithProviders(<Dashboard />);
    expect(screen.getByText("Daily Report")).toBeInTheDocument();
    expect(screen.getByText("Dream Journal")).toBeInTheDocument();
    expect(screen.getByText("AI Companion")).toBeInTheDocument();
    expect(screen.getByText("Breathe")).toBeInTheDocument();
  });

  it("shows connect device banner when not streaming", () => {
    renderWithProviders(<Dashboard />);
    expect(
      screen.getByText(/Start with a voice check-in or sync health data\. EEG is optional\./)
    ).toBeInTheDocument();
  });

  it("shows a personalized greeting header", () => {
    renderWithProviders(<Dashboard />);
    // Greeting depends on time of day — one of these must be present
    const greeting = screen.getByText(/Good (morning|afternoon|evening)/);
    expect(greeting).toBeInTheDocument();
  });

  // Active ML Models section is hidden (consumer mode, gated by `false &&`)
  it("hides ML model badges (consumer mode)", async () => {
    renderWithProviders(<Dashboard />);
    await waitFor(() => {
      expect(screen.queryByText("Active ML Models")).not.toBeInTheDocument();
    });
  });

  it("hides Brain State Now card when not streaming and no health data", () => {
    renderWithProviders(<Dashboard />);
    expect(screen.queryByText("Brain State Now")).not.toBeInTheDocument();
  });

  it("hides Brain-Health Insights for new users with no sessions", async () => {
    renderWithProviders(<Dashboard />);
    await waitFor(() => {
      expect(screen.queryByText("Brain-Health Insights")).not.toBeInTheDocument();
    });
  });

  it("hides Weekly Stress Landscape for new users with no sessions", async () => {
    renderWithProviders(<Dashboard />);
    await waitFor(() => {
      expect(screen.queryByText("Weekly Stress Landscape")).not.toBeInTheDocument();
    });
  });
});

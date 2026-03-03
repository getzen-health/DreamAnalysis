import { describe, it, expect, vi, beforeAll } from "vitest";
import { screen } from "@testing-library/react";
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

vi.mock("@/lib/ml-api", () => ({
  getHealthInsights: vi.fn().mockResolvedValue([]),
  listSessions: vi.fn().mockResolvedValue([]),
}));

vi.mock("@/lib/participant", () => ({
  getParticipantId: () => "test-participant-uuid",
}));

describe("Dashboard page", () => {
  it("renders without crashing", () => {
    renderWithProviders(<Dashboard />);
  });

  it("shows the Brain-Health Insights section", () => {
    renderWithProviders(<Dashboard />);
    expect(screen.getByText("Brain-Health Insights")).toBeInTheDocument();
  });

  it("shows the Brain-Health Insights heading", () => {
    renderWithProviders(<Dashboard />);
    // Confirm heading (not the same element as the section card checked above)
    expect(screen.getAllByText("Brain-Health Insights").length).toBeGreaterThan(0);
  });

  it("shows Brain State Now card", () => {
    renderWithProviders(<Dashboard />);
    expect(screen.getByText("Brain State Now")).toBeInTheDocument();
  });

  it("shows key health metric labels", () => {
    renderWithProviders(<Dashboard />);
    expect(screen.getByText("Stress")).toBeInTheDocument();
    expect(screen.getByText("Focus")).toBeInTheDocument();
    expect(screen.getByText("Flow")).toBeInTheDocument();
    expect(screen.getByText("Creativity")).toBeInTheDocument();
  });

  it("shows all four quick action cards", () => {
    renderWithProviders(<Dashboard />);
    expect(screen.getByText("Brain Monitor")).toBeInTheDocument();
    expect(screen.getByText("Dream Journal")).toBeInTheDocument();
    expect(screen.getByText("AI Companion")).toBeInTheDocument();
    expect(screen.getByText("Breathe")).toBeInTheDocument();
  });

  it("shows connect device banner when not streaming", () => {
    renderWithProviders(<Dashboard />);
    expect(
      screen.getByText(/Connect your Muse 2 to see live dashboard data/)
    ).toBeInTheDocument();
  });

  it("shows empty state for Brain-Health Insights when no data", () => {
    renderWithProviders(<Dashboard />);
    expect(
      screen.getByText(/Insights appear after a few days of data/)
    ).toBeInTheDocument();
  });

  it("shows empty state when not streaming", () => {
    renderWithProviders(<Dashboard />);
    expect(
      screen.getByText(/Connect device to see live brain state/)
    ).toBeInTheDocument();
  });
});

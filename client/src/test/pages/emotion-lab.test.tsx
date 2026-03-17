import React from "react";
import { describe, it, expect, vi, beforeAll } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import EmotionLab from "@/pages/emotion-lab";

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

vi.mock("@/hooks/use-auth", () => ({
  useAuth: () => ({ user: null, login: vi.fn(), logout: vi.fn() }),
}));

vi.mock("@/hooks/use-voice-emotion", () => ({
  useVoiceEmotion: () => ({
    startRecording: vi.fn(),
    stopRecording: vi.fn(),
    isRecording: false,
    isAnalyzing: false,
    lastResult: null,
    error: null,
  }),
}));

vi.mock("@/hooks/use-toast", () => ({
  useToast: () => ({ toast: vi.fn(), dismiss: vi.fn(), toasts: [] }),
}));

vi.mock("@/components/simulation-mode-banner", () => ({
  SimulationModeBanner: () => <div data-testid="simulation-mode-banner" />,
}));

vi.mock("wouter", () => ({
  useLocation: () => ["/emotions", vi.fn()],
  Link: (props: any) => <a href={props.href}>{props.children}</a>,
}));

vi.mock("@/lib/ml-api", () => ({
  listSessions: vi.fn().mockResolvedValue([]),
  getBaselineStatus: vi.fn().mockResolvedValue({ is_ready: false, frames_collected: 0 }),
  getMultimodalStatus: vi.fn().mockResolvedValue({}),
  getEmotionHistory: vi.fn().mockResolvedValue([]),
}));

vi.mock("@/components/emotion-wheel", () => ({
  EmotionWheel: () => <div data-testid="emotion-wheel" />,
}));

vi.mock("@/components/signal-quality-badge", () => ({
  SignalQualityBadge: () => <div data-testid="signal-quality-badge" />,
}));

vi.mock("@/components/chart-tooltip", () => ({
  ChartTooltip: () => null,
}));

vi.mock("@/hooks/use-ml-connection", () => ({
  useMLConnection: () => ({
    status: "ready",
    latencyMs: 42,
    warmupProgress: 100,
    retryCount: 0,
    reconnect: vi.fn(),
  }),
  MLConnectionProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

describe("EmotionLab page", () => {
  it("renders without crashing", () => {
    renderWithProviders(<EmotionLab />);
  });

  it("shows no-data message when no voice check-in exists", () => {
    renderWithProviders(<EmotionLab />);
    expect(
      screen.getByText(/No emotion data yet/)
    ).toBeInTheDocument();
  });

  it("shows Recent Emotions section header", () => {
    renderWithProviders(<EmotionLab />);
    expect(screen.getByText("Recent Emotions")).toBeInTheDocument();
  });

  it("shows Right now card section", () => {
    renderWithProviders(<EmotionLab />);
    expect(screen.getByText("Right now")).toBeInTheDocument();
  });

  it("shows dashboard redirect message when not streaming", () => {
    renderWithProviders(<EmotionLab />);
    expect(screen.getByText(/Do a voice check-in on the Dashboard/)).toBeInTheDocument();
  });

  it("shows empty session message when not streaming", () => {
    renderWithProviders(<EmotionLab />);
    expect(
      screen.getByText("Log a mood or start a session to see your emotion history.")
    ).toBeInTheDocument();
  });
});

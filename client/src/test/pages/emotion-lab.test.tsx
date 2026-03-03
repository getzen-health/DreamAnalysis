import React from "react";
import { describe, it, expect, vi, beforeAll } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
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

  it("shows connect-device banner when Muse 2 not streaming", () => {
    renderWithProviders(<EmotionLab />);
    expect(
      screen.getByText(/Connect your Muse 2 from the sidebar to see live emotion data/)
    ).toBeInTheDocument();
  });

  it("shows Emotion History section header", () => {
    renderWithProviders(<EmotionLab />);
    expect(screen.getByText("Emotion History")).toBeInTheDocument();
  });

  it("shows all period tabs", () => {
    renderWithProviders(<EmotionLab />);
    expect(screen.getByRole("button", { name: "Today" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Week" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Month" })).toBeInTheDocument();
  });

  it("shows the Brainwave Snapshot section", () => {
    renderWithProviders(<EmotionLab />);
    expect(screen.getByText("Brainwave Snapshot")).toBeInTheDocument();
  });

  it("period tab click does not throw", () => {
    renderWithProviders(<EmotionLab />);
    const weekTab = screen.getByRole("button", { name: "Week" });
    fireEvent.click(weekTab);
    expect(screen.getByRole("button", { name: "Week" })).toBeInTheDocument();
  });
});

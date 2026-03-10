import React from "react";
import { describe, it, expect, vi, beforeEach, beforeAll } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import BrainMonitor from "@/pages/brain-monitor";

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
    deviceStatus: null,
  }),
}));

vi.mock("@/hooks/use-inference", () => ({
  useInference: () => ({
    isLocal: false,
    latencyMs: 0,
    isReady: false,
  }),
}));

vi.mock("@/lib/ml-api", () => ({
  analyzeWavelet: vi.fn().mockResolvedValue(null),
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

vi.mock("@/components/charts/eeg-waveform-canvas", () => ({
  EEGWaveformCanvas: () => <canvas data-testid="eeg-waveform-canvas" />,
}));

vi.mock("@/components/charts/spectrogram-chart", () => ({
  SpectrogramChart: () => <div data-testid="spectrogram-chart" />,
}));

vi.mock("@/components/signal-quality-badge", () => ({
  SignalQualityBadge: () => <div data-testid="signal-quality-badge" />,
}));

vi.mock("@/components/alert-banner", () => ({
  AlertBanner: () => <div data-testid="alert-banner" />,
}));

vi.mock("@/components/session-controls", () => ({
  SessionControls: () => <div data-testid="session-controls" />,
}));

vi.mock("@/components/mood-music-player", () => ({
  MoodMusicPlayer: () => <div data-testid="mood-music-player" />,
}));

describe("BrainMonitor page", () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({}),
    }) as unknown as typeof fetch;
  });

  it("renders without crashing", async () => {
    renderWithProviders(<BrainMonitor />);
    await waitFor(() => expect(document.body).toBeTruthy());
  });

  it("shows connection banner when device is disconnected", async () => {
    renderWithProviders(<BrainMonitor />);
    await waitFor(() => {
      expect(
        screen.getByText(/EEG is offline/)
      ).toBeInTheDocument();
    });
  });

  it("shows EEG Brain Wave Activity heading", async () => {
    renderWithProviders(<BrainMonitor />);
    await waitFor(() => {
      expect(screen.getByText("EEG Brain Wave Activity")).toBeInTheDocument();
    });
  });

  it("shows Alpha Waves and Beta Waves display", async () => {
    renderWithProviders(<BrainMonitor />);
    await waitFor(() => {
      expect(screen.getByText("Alpha Waves")).toBeInTheDocument();
      expect(screen.getByText("Beta Waves")).toBeInTheDocument();
    });
  });

  it("shows dashes for alpha/beta hz when not streaming", async () => {
    renderWithProviders(<BrainMonitor />);
    await waitFor(() => {
      const dashes = screen.getAllByText("—");
      expect(dashes.length).toBeGreaterThanOrEqual(2);
    });
  });

  it("shows Brain State Now panel", async () => {
    renderWithProviders(<BrainMonitor />);
    await waitFor(() => {
      expect(screen.getByText("Brain State Now")).toBeInTheDocument();
    });
  });

  it("shows prompt to connect device inside Brain State Now panel", async () => {
    renderWithProviders(<BrainMonitor />);
    await waitFor(() => {
      const els = screen.getAllByText("No EEG signal — showing voice + health estimates");
      expect(els.length).toBeGreaterThan(0);
    });
  });

  it("shows Electrode Status Grid section", async () => {
    renderWithProviders(<BrainMonitor />);
    await waitFor(() => {
      expect(screen.getByText("Electrode Status Grid")).toBeInTheDocument();
    });
  });

  it("shows No Device label when not streaming", async () => {
    renderWithProviders(<BrainMonitor />);
    await waitFor(() => {
      expect(screen.getByText("No Device")).toBeInTheDocument();
    });
  });

  it("shows alert banner component", async () => {
    renderWithProviders(<BrainMonitor />);
    await waitFor(() => {
      expect(screen.getByTestId("alert-banner")).toBeInTheDocument();
    });
  });
});

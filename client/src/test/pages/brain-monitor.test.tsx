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

vi.mock("wouter", () => ({
  useLocation: () => ["/brain-monitor", vi.fn()],
  Link: (props: any) => <a href={props.href} className={props.className}>{props.children}</a>,
}));

// Default: disconnected. Individual tests override via vi.mocked.
const mockUseDevice = vi.fn().mockReturnValue({
  state: "disconnected",
  latestFrame: null,
  connect: vi.fn(),
  disconnect: vi.fn(),
  deviceStatus: null,
  selectedDevice: null,
  reconnectCount: 0,
  epochReady: false,
  bleReconnect: { attempt: 0, isReconnecting: false, lastError: null, gaveUp: false },
});

vi.mock("@/hooks/use-device", () => ({
  useDevice: () => mockUseDevice(),
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
    mockUseDevice.mockReturnValue({
      state: "disconnected",
      latestFrame: null,
      connect: vi.fn(),
      disconnect: vi.fn(),
      deviceStatus: null,
      selectedDevice: null,
      reconnectCount: 0,
      epochReady: false,
      bleReconnect: { attempt: 0, isReconnecting: false, lastError: null, gaveUp: false },
    });
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

  it("does not show duplicate band labels when not streaming (removed top grid)", async () => {
    renderWithProviders(<BrainMonitor />);
    await waitFor(() => {
      // Band Powers section only renders when streaming with data — so offline
      // should not show Delta/Theta/Alpha/Beta/Gamma labels at all in the EEG section
      expect(screen.queryByText("Band Powers")).not.toBeInTheDocument();
    });
  });

  it("shows Brain State panel", async () => {
    renderWithProviders(<BrainMonitor />);
    await waitFor(() => {
      expect(screen.getByText("Brain State")).toBeInTheDocument();
    });
  });

  it("shows offline message when not streaming", async () => {
    renderWithProviders(<BrainMonitor />);
    await waitFor(() => {
      const els = screen.getAllByText(/No EEG signal/);
      expect(els.length).toBeGreaterThan(0);
    });
  });

  it("shows alert banner component", async () => {
    renderWithProviders(<BrainMonitor />);
    await waitFor(() => {
      expect(screen.getByTestId("alert-banner")).toBeInTheDocument();
    });
  });

  describe("when streaming with band powers", () => {
    const streamingFrame = {
      signals: [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
      analysis: {
        band_powers: {
          delta: 0.25,
          theta: 0.15,
          alpha: 0.30,
          beta: 0.20,
          gamma: 0.10,
        },
        features: {},
        emotions: {
          emotion: "happy",
          confidence: 0.8,
          valence: 0.5,
          arousal: 0.4,
          stress_index: 0.3,
          focus_index: 0.6,
          relaxation_index: 0.7,
          ready: true,
          buffered_sec: 30,
          window_sec: 30,
        },
        epoch_ready: true,
        sleep_staging: { stage: "Wake", stage_index: 0, confidence: 0.9, probabilities: {} },
        stress: { level: "relaxed", stress_index: 0.3, confidence: 0.7 },
        flow_state: { in_flow: false, flow_score: 0.4, confidence: 0.6 },
        creativity: { creativity_score: 0.3, state: "normal", confidence: 0.5 },
        attention: { state: "focused", attention_score: 0.6, confidence: 0.6 },
        cognitive_load: { level: "medium", load_index: 0.5, confidence: 0.5 },
        drowsiness: { state: "alert", drowsiness_index: 0.1, confidence: 0.8 },
        meditation: { depth: "light", meditation_score: 0.3, confidence: 0.5 },
        memory_encoding: { encoding_active: false, encoding_score: 0.3, state: "normal", confidence: 0.4 },
        dream_detection: { is_dreaming: false, probability: 0.05, rem_likelihood: 0.02, dream_intensity: 0.1, lucidity_estimate: 0 },
        lucid_dream: { state: "awake", lucidity_score: 0, confidence: 0.9 },
      },
      quality: { sqi: 85, artifacts_detected: [], clean_ratio: 0.95, channel_quality: [90, 88, 87, 91] },
      timestamp: Date.now() / 1000,
      n_channels: 4,
      sample_rate: 256,
    };

    beforeEach(() => {
      mockUseDevice.mockReturnValue({
        state: "streaming",
        latestFrame: streamingFrame,
        connect: vi.fn(),
        disconnect: vi.fn(),
        deviceStatus: { connected: true, streaming: true, device_type: "synthetic", n_channels: 4, sample_rate: 256, brainflow_available: false },
        selectedDevice: "synthetic",
        reconnectCount: 0,
        epochReady: true,
        bleReconnect: { attempt: 0, isReconnecting: false, lastError: null, gaveUp: false },
      });
    });

    it("shows Band Powers section with all 5 bands when streaming", async () => {
      renderWithProviders(<BrainMonitor />);
      await waitFor(() => {
        expect(screen.getByText("Band Powers")).toBeInTheDocument();
        expect(screen.getByTestId("band-delta")).toBeInTheDocument();
        expect(screen.getByTestId("band-theta")).toBeInTheDocument();
        expect(screen.getByTestId("band-alpha")).toBeInTheDocument();
        expect(screen.getByTestId("band-beta")).toBeInTheDocument();
        expect(screen.getByTestId("band-gamma")).toBeInTheDocument();
      });
    });

    it("shows band power percentages when streaming", async () => {
      renderWithProviders(<BrainMonitor />);
      await waitFor(() => {
        // Band powers should show as percentage values (e.g. "25%", "15%", etc.)
        expect(screen.getByTestId("band-delta")).toHaveTextContent("25%");
        expect(screen.getByTestId("band-theta")).toHaveTextContent("15%");
        expect(screen.getByTestId("band-alpha")).toHaveTextContent("30%");
        expect(screen.getByTestId("band-beta")).toHaveTextContent("20%");
        expect(screen.getByTestId("band-gamma")).toHaveTextContent("10%");
      });
    });

    it("shows LIVE label when streaming", async () => {
      renderWithProviders(<BrainMonitor />);
      await waitFor(() => {
        expect(screen.getByText("LIVE")).toBeInTheDocument();
      });
    });

    it("shows Band Powers frequency ranges", async () => {
      renderWithProviders(<BrainMonitor />);
      await waitFor(() => {
        expect(screen.getByTestId("band-delta")).toHaveTextContent("0.5-4 Hz");
        expect(screen.getByTestId("band-theta")).toHaveTextContent("4-8 Hz");
        expect(screen.getByTestId("band-alpha")).toHaveTextContent("8-12 Hz");
        expect(screen.getByTestId("band-beta")).toHaveTextContent("12-30 Hz");
        expect(screen.getByTestId("band-gamma")).toHaveTextContent("30-100 Hz");
      });
    });

    it("renders EEG waveform canvas when streaming", async () => {
      renderWithProviders(<BrainMonitor />);
      await waitFor(() => {
        expect(screen.getByTestId("eeg-waveform-canvas")).toBeInTheDocument();
      });
    });

    it("shows compact signal quality status with colored dot when streaming", async () => {
      renderWithProviders(<BrainMonitor />);
      await waitFor(() => {
        const badge = screen.getByTestId("signal-quality-status");
        expect(badge).toBeInTheDocument();
        // Should show "Good Signal" (score is 100 by default)
        expect(badge).toHaveTextContent("Good Signal");
      });
    });

    it("does not render SignalQualityBadge component (removed for cleaner mobile)", async () => {
      renderWithProviders(<BrainMonitor />);
      await waitFor(() => {
        // SignalQualityBadge mock renders with data-testid="signal-quality-badge"
        expect(screen.queryByTestId("signal-quality-badge")).not.toBeInTheDocument();
      });
    });

    it("band labels appear only in Band Powers section (no duplicate top grid)", async () => {
      renderWithProviders(<BrainMonitor />);
      await waitFor(() => {
        // Delta should appear exactly once — inside the Band Powers bar chart
        const deltas = screen.getAllByText("Delta");
        expect(deltas).toHaveLength(1);
        // And it should be inside the band-delta test id element
        expect(screen.getByTestId("band-delta")).toHaveTextContent("Delta");
      });
    });
  });

  describe("Task #21: ML model cards link to history pages", () => {
    it("renders Emotion model card linking to brain-monitor", async () => {
      renderWithProviders(<BrainMonitor />);
      await waitFor(() => {
        const link = screen.getByTestId("model-link-Emotion").closest("a");
        expect(link).toHaveAttribute("href", "/brain-monitor");
      });
    });

    it("renders Stress model card as a link to /stress", async () => {
      renderWithProviders(<BrainMonitor />);
      await waitFor(() => {
        const link = screen.getByTestId("model-link-Stress").closest("a");
        expect(link).toHaveAttribute("href", "/stress");
      });
    });

    it("renders Focus model card as a link to /focus", async () => {
      renderWithProviders(<BrainMonitor />);
      await waitFor(() => {
        const link = screen.getByTestId("model-link-Focus").closest("a");
        expect(link).toHaveAttribute("href", "/focus");
      });
    });

    it("renders Sleep model card as a link to /sleep", async () => {
      renderWithProviders(<BrainMonitor />);
      await waitFor(() => {
        const link = screen.getByTestId("model-link-Sleep").closest("a");
        expect(link).toHaveAttribute("href", "/sleep");
      });
    });

    // Remaining models are collapsed by default — only visible after expanding
    it("shows expand button for remaining models", async () => {
      renderWithProviders(<BrainMonitor />);
      await waitFor(() => {
        expect(screen.getByText("Show all 16 models")).toBeInTheDocument();
      });
    });

    it("linked model cards have hover styling classes", async () => {
      renderWithProviders(<BrainMonitor />);
      await waitFor(() => {
        const card = screen.getByTestId("model-link-Emotion");
        expect(card.className).toContain("hover:border-primary/50");
        expect(card.className).toContain("cursor-pointer");
      });
    });
  });

  describe("Task #22: Try Synthetic Device button", () => {
    it("shows Try Synthetic Device button when not streaming", async () => {
      renderWithProviders(<BrainMonitor />);
      await waitFor(() => {
        expect(screen.getByText("Try Synthetic Device")).toBeInTheDocument();
      });
    });

    it("Try Synthetic Device button calls device.connect('synthetic')", async () => {
      const connectFn = vi.fn();
      mockUseDevice.mockReturnValue({
        state: "disconnected",
        latestFrame: null,
        connect: connectFn,
        disconnect: vi.fn(),
        deviceStatus: null,
        selectedDevice: null,
        reconnectCount: 0,
        epochReady: false,
        bleReconnect: { attempt: 0, isReconnecting: false, lastError: null, gaveUp: false },
      });
      renderWithProviders(<BrainMonitor />);
      const btn = await screen.findByText("Try Synthetic Device");
      btn.closest("button")?.click();
      expect(connectFn).toHaveBeenCalledWith("synthetic");
    });

    it("does not show Try Synthetic Device button when streaming", async () => {
      mockUseDevice.mockReturnValue({
        state: "streaming",
        latestFrame: {
          signals: [[1, 2], [3, 4], [5, 6], [7, 8]],
          analysis: { band_powers: {}, features: {} },
          quality: {},
          timestamp: Date.now() / 1000,
          n_channels: 4,
          sample_rate: 256,
        },
        connect: vi.fn(),
        disconnect: vi.fn(),
        deviceStatus: { connected: true, streaming: true, device_type: "synthetic", n_channels: 4, sample_rate: 256, brainflow_available: false },
        selectedDevice: "synthetic",
        reconnectCount: 0,
        epochReady: false,
        bleReconnect: { attempt: 0, isReconnecting: false, lastError: null, gaveUp: false },
      });
      renderWithProviders(<BrainMonitor />);
      await waitFor(() => {
        expect(screen.queryByText("Try Synthetic Device")).not.toBeInTheDocument();
      });
    });
  });

  describe("Task #39: Signal quality status labels", () => {
    function makeStreamingDevice(overrides: Record<string, unknown> = {}) {
      return {
        state: "streaming",
        latestFrame: {
          signals: [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
          analysis: { band_powers: { delta: 0.2, theta: 0.2, alpha: 0.2, beta: 0.2, gamma: 0.2 }, features: {} },
          quality: {},
          timestamp: Date.now() / 1000,
          n_channels: 4,
          sample_rate: 256,
          ...overrides,
        },
        connect: vi.fn(),
        disconnect: vi.fn(),
        deviceStatus: { connected: true, streaming: true, device_type: "muse2", n_channels: 4, sample_rate: 256, brainflow_available: true },
        selectedDevice: "muse2",
        reconnectCount: 0,
        epochReady: true,
        bleReconnect: { attempt: 0, isReconnecting: false, lastError: null, gaveUp: false },
      };
    }

    it("shows 'Good Signal' for high quality score (>70)", async () => {
      mockUseDevice.mockReturnValue(makeStreamingDevice({ signal_quality_score: 85 }));
      renderWithProviders(<BrainMonitor />);
      await waitFor(() => {
        const badge = screen.getByTestId("signal-quality-status");
        expect(badge).toHaveTextContent("Good Signal");
      });
    });

    it("shows 'Fair Signal' for medium quality score (40-70)", async () => {
      mockUseDevice.mockReturnValue(makeStreamingDevice({ signal_quality_score: 55 }));
      renderWithProviders(<BrainMonitor />);
      await waitFor(() => {
        const badge = screen.getByTestId("signal-quality-status");
        expect(badge).toHaveTextContent("Fair Signal");
      });
    });

    it("shows 'Poor Signal' for low quality score (<40)", async () => {
      mockUseDevice.mockReturnValue(makeStreamingDevice({ signal_quality_score: 20 }));
      renderWithProviders(<BrainMonitor />);
      await waitFor(() => {
        const badge = screen.getByTestId("signal-quality-status");
        expect(badge).toHaveTextContent("Poor Signal");
      });
    });

    it("signal status badge has a colored dot element", async () => {
      mockUseDevice.mockReturnValue(makeStreamingDevice({ signal_quality_score: 85 }));
      renderWithProviders(<BrainMonitor />);
      await waitFor(() => {
        const badge = screen.getByTestId("signal-quality-status");
        const dot = badge.querySelector("span.rounded-full");
        expect(dot).toBeTruthy();
      });
    });
  });
});

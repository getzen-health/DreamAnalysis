import { describe, it, expect, vi, beforeEach, beforeAll } from "vitest";
import { screen, waitFor, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import Neurofeedback from "@/pages/neurofeedback";

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

vi.mock("@/lib/ml-api", () => ({
  getNeurofeedbackProtocols: vi.fn().mockResolvedValue({
    alpha_up: { name: "Alpha Enhancement", description: "Boost alpha waves for calm focus." },
    smr_up: { name: "SMR Training", description: "Enhance sensorimotor rhythm." },
    theta_beta_ratio: { name: "Theta/Beta Ratio", description: "Balance theta and beta." },
    alpha_asymmetry: { name: "Alpha Asymmetry", description: "Balance left/right frontal alpha." },
  }),
  startNeurofeedback: vi.fn().mockResolvedValue({ status: "calibrating" }),
  evaluateNeurofeedback: vi.fn().mockResolvedValue({ status: "calibrating", progress: 0.5 }),
  stopNeurofeedback: vi.fn().mockResolvedValue({
    stats: {
      total_rewards: 5,
      reward_rate: 0.6,
      avg_score: 72,
      max_streak: 3,
      total_evaluations: 10,
    },
  }),
}));

vi.mock("@/lib/participant", () => ({
  getParticipantId: () => "test-user-123",
}));

describe("Neurofeedback page", () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({}),
    }) as unknown as typeof fetch;
  });

  it("renders without crashing", async () => {
    renderWithProviders(<Neurofeedback />);
    await waitFor(() => {
      expect(document.body).toBeTruthy();
    });
  });

  it("shows the page heading", async () => {
    renderWithProviders(<Neurofeedback />);
    await waitFor(() => {
      expect(screen.getByText("Neurofeedback Training")).toBeInTheDocument();
    });
  });

  it("shows connection banner when device is not streaming", async () => {
    renderWithProviders(<Neurofeedback />);
    await waitFor(() => {
      expect(
        screen.getByText(/Neurofeedback is an EEG-only mode/)
      ).toBeInTheDocument();
    });
  });

  it("shows Select Protocol card in idle phase", async () => {
    renderWithProviders(<Neurofeedback />);
    await waitFor(() => {
      expect(screen.getByText("Select Protocol")).toBeInTheDocument();
    });
  });

  it("shows How It Works card in idle phase", async () => {
    renderWithProviders(<Neurofeedback />);
    await waitFor(() => {
      expect(screen.getByText("How It Works")).toBeInTheDocument();
    });
  });

  it("shows Audio Feedback toggle in idle phase", async () => {
    renderWithProviders(<Neurofeedback />);
    await waitFor(() => {
      expect(screen.getByText("Audio Feedback")).toBeInTheDocument();
    });
  });

  it("shows Connect Device First button when device not connected", async () => {
    renderWithProviders(<Neurofeedback />);
    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: /EEG Required For Training/ })
      ).toBeInTheDocument();
    });
  });

  it("Start button is disabled when device not connected", async () => {
    renderWithProviders(<Neurofeedback />);
    await waitFor(() => {
      const btn = screen.getByRole("button", { name: /EEG Required For Training/ });
      expect(btn).toBeDisabled();
    });
  });

  it("shows step-by-step instructions in How It Works", async () => {
    renderWithProviders(<Neurofeedback />);
    await waitFor(() => {
      expect(screen.getByText(/Connect Muse 2/)).toBeInTheDocument();
      expect(screen.getByText(/Calibration/)).toBeInTheDocument();
    });
  });

  it("protocol dropdown combobox is present in idle phase", async () => {
    renderWithProviders(<Neurofeedback />);
    await waitFor(() => {
      expect(screen.getByRole("combobox")).toBeInTheDocument();
    });
  });
});

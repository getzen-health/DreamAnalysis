import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import SettingsPage from "@/pages/settings";

vi.mock("@/lib/participant", () => ({ getParticipantId: () => "test-user-123" }));

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
  useAuth: () => ({ user: { id: 1, username: "testuser", email: "test@test.com" }, logout: vi.fn() }),
}));

vi.mock("@/hooks/use-toast", () => ({
  useToast: () => ({ toast: vi.fn() }),
}));

vi.mock("@/lib/ml-api", () => ({
  getBaselineStatus: vi.fn().mockResolvedValue({ ready: false, n_frames: 0 }),
  addBaselineFrame: vi.fn().mockResolvedValue({ ok: true }),
  resetBaselineCalibration: vi.fn().mockResolvedValue({ ok: true }),
  getCalibrationStatus: vi.fn().mockResolvedValue({
    personal_model_active: false,
    personalization_progress_pct: 40,
    total_sessions: 2,
    total_labeled_epochs: 2,
    activation_threshold_sessions: 5,
    accuracy_improvement_pct: 0,
    personal_blend_weight_pct: 70,
    feature_priors: { alpha_mean: 0.1, beta_mean: 0.05, theta_mean: 0.08 },
    message: "2/5 corrected sessions collected. Keep correcting labels to activate personalization.",
  }),
}));

describe("Settings page", () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({}),
      blob: async () => new Blob(["data"], { type: "text/csv" }),
      text: async () => "",
    }) as unknown as typeof fetch;

    // Stub localStorage
    Object.defineProperty(window, "localStorage", {
      value: {
        getItem: vi.fn().mockReturnValue(null),
        setItem: vi.fn(),
        removeItem: vi.fn(),
      },
      writable: true,
    });
  });

  it("renders without crashing", async () => {
    renderWithProviders(<SettingsPage />);
    await waitFor(() => expect(document.body).toBeTruthy());
  });

  it("shows ML Backend section", async () => {
    renderWithProviders(<SettingsPage />);
    await waitFor(() => {
      expect(screen.getByText("ML Backend")).toBeInTheDocument();
    });
  });

  it("shows Model Personalization section", async () => {
    renderWithProviders(<SettingsPage />);
    await waitFor(() => {
      expect(screen.getByText("Model Personalization")).toBeInTheDocument();
      expect(screen.getByText("2 / 5 corrected sessions")).toBeInTheDocument();
    });
  });

  it("shows Connected Assets link", async () => {
    renderWithProviders(<SettingsPage />);
    await waitFor(() => {
      expect(screen.getByText("Connected Assets")).toBeInTheDocument();
      expect(screen.getByText("Manage health, BCI, and wearable connections")).toBeInTheDocument();
    });
  });

  it("shows Interface Settings section with theme buttons", async () => {
    renderWithProviders(<SettingsPage />);
    await waitFor(() => {
      expect(screen.getByText("Interface Settings")).toBeInTheDocument();
      expect(screen.getByTestId("button-theme-dark")).toBeInTheDocument();
      expect(screen.getByTestId("button-theme-light")).toBeInTheDocument();
    });
  });

  it("shows toggle switches for chart animations, neural effects, health alerts", async () => {
    renderWithProviders(<SettingsPage />);
    await waitFor(() => {
      expect(screen.getByTestId("switch-chart-animations")).toBeInTheDocument();
      expect(screen.getByTestId("switch-neural-effects")).toBeInTheDocument();
      expect(screen.getByTestId("switch-health-alerts")).toBeInTheDocument();
    });
  });

  it("shows Data Export section", async () => {
    renderWithProviders(<SettingsPage />);
    await waitFor(() => {
      expect(screen.getByText("Data Export")).toBeInTheDocument();
    });
  });

  it("shows Export Health Data and Export Dream Analysis buttons", async () => {
    renderWithProviders(<SettingsPage />);
    await waitFor(() => {
      expect(screen.getByTestId("button-export-health-data")).toBeInTheDocument();
      expect(screen.getByTestId("button-export-dream-analysis")).toBeInTheDocument();
    });
  });

  it("shows Privacy Policy link", async () => {
    renderWithProviders(<SettingsPage />);
    await waitFor(() => {
      expect(screen.getByText("Privacy Policy")).toBeInTheDocument();
      expect(screen.getByText("How we collect, store, and protect your data")).toBeInTheDocument();
    });
  });

  it("shows Privacy & Security section with toggle switches", async () => {
    renderWithProviders(<SettingsPage />);
    await waitFor(() => {
      expect(screen.getByText("Privacy & Security")).toBeInTheDocument();
      expect(screen.getByTestId("switch-local-processing")).toBeInTheDocument();
      expect(screen.getByTestId("switch-data-encryption")).toBeInTheDocument();
    });
  });

  it("shows Clear All Data button", async () => {
    renderWithProviders(<SettingsPage />);
    await waitFor(() => {
      expect(screen.getByTestId("button-clear-data")).toBeInTheDocument();
    });
  });

  it("shows Morning Reminders section", async () => {
    renderWithProviders(<SettingsPage />);
    await waitFor(() => {
      expect(screen.getByText("Morning Reminders")).toBeInTheDocument();
    });
  });

  it("shows Wellness Disclaimer card", async () => {
    renderWithProviders(<SettingsPage />);
    await waitFor(() => {
      expect(screen.getByTestId("wellness-disclaimer")).toBeInTheDocument();
      expect(screen.getByText("Wellness Disclaimer")).toBeInTheDocument();
    });
  });

  it("wellness disclaimer states app is not a medical device", async () => {
    renderWithProviders(<SettingsPage />);
    await waitFor(() => {
      expect(screen.getByText(/NOT a medical device/)).toBeInTheDocument();
    });
  });

  it("wellness disclaimer mentions not FDA cleared", async () => {
    renderWithProviders(<SettingsPage />);
    await waitFor(() => {
      expect(screen.getByText(/Not FDA cleared/)).toBeInTheDocument();
    });
  });
});

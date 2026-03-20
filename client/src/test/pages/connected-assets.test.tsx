import { describe, it, expect, vi, beforeEach, beforeAll } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import ConnectedAssets from "@/pages/connected-assets";

beforeAll(() => {
  global.ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
});

vi.mock("@/hooks/use-toast", () => ({
  useToast: () => ({ toast: vi.fn() }),
}));

vi.mock("@/hooks/use-device", () => ({
  useDevice: () => ({
    state: "disconnected",
    deviceStatus: null,
    latestFrame: null,
  }),
}));

vi.mock("@/lib/participant", () => ({
  getParticipantId: () => "test-user-123",
}));

vi.mock("@/lib/queryClient", () => ({
  resolveUrl: (path: string) => `http://localhost:5000${path}`,
}));

vi.mock("@/lib/ml-api", () => ({
  ingestHealthData: vi.fn().mockResolvedValue({ stored: 0, metrics: [] }),
}));

vi.mock("wouter", () => ({
  useLocation: () => ["/connected-assets", vi.fn()],
}));

describe("ConnectedAssets page", () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ devices: [] }),
    }) as unknown as typeof fetch;
    // Clear localStorage mocks
    localStorage.removeItem("ndw_health_connect_granted");
    localStorage.removeItem("ndw_apple_health_granted");
  });

  it("renders without crashing", async () => {
    renderWithProviders(<ConnectedAssets />);
    await waitFor(() => {
      expect(document.body).toBeTruthy();
    });
  });

  it("shows the page heading", async () => {
    renderWithProviders(<ConnectedAssets />);
    await waitFor(() => {
      expect(screen.getByText("Connected Assets")).toBeInTheDocument();
    });
  });

  it("shows subtitle text", async () => {
    renderWithProviders(<ConnectedAssets />);
    await waitFor(() => {
      expect(
        screen.getByText(
          "Manage your health, brain, and wearable connections"
        )
      ).toBeInTheDocument();
    });
  });

  it("shows Connect Health section", async () => {
    renderWithProviders(<ConnectedAssets />);
    await waitFor(() => {
      expect(screen.getByText("Connect Health")).toBeInTheDocument();
    });
  });

  it("shows BCI / EEG section", async () => {
    renderWithProviders(<ConnectedAssets />);
    await waitFor(() => {
      expect(screen.getByText("BCI / EEG")).toBeInTheDocument();
    });
  });

  it("shows Connect Wearable section", async () => {
    renderWithProviders(<ConnectedAssets />);
    await waitFor(() => {
      expect(screen.getByText("Connect Wearable")).toBeInTheDocument();
    });
  });

  it("shows health section subtitle", async () => {
    renderWithProviders(<ConnectedAssets />);
    await waitFor(() => {
      // On web platform, shows combined subtitle
      expect(
        screen.getByText(
          "Google Health Connect (Android) / Apple HealthKit (iOS)"
        )
      ).toBeInTheDocument();
    });
  });

  it("shows BCI subtitle", async () => {
    renderWithProviders(<ConnectedAssets />);
    await waitFor(() => {
      expect(
        screen.getByText("Brain-computer interface headbands")
      ).toBeInTheDocument();
    });
  });

  it("shows wearable subtitle", async () => {
    renderWithProviders(<ConnectedAssets />);
    await waitFor(() => {
      expect(screen.getByText("Oura, WHOOP, Garmin")).toBeInTheDocument();
    });
  });

  it("shows supported BCI devices", async () => {
    renderWithProviders(<ConnectedAssets />);
    await waitFor(() => {
      expect(screen.getByText("Muse 2")).toBeInTheDocument();
      expect(screen.getByText("Muse S")).toBeInTheDocument();
      expect(screen.getByText("Synthetic (Demo)")).toBeInTheDocument();
    });
  });

  it("shows all three wearable providers", async () => {
    renderWithProviders(<ConnectedAssets />);
    await waitFor(() => {
      expect(screen.getByText("Oura Ring")).toBeInTheDocument();
      expect(screen.getByText("WHOOP")).toBeInTheDocument();
      expect(screen.getByText("Garmin")).toBeInTheDocument();
    });
  });

  it("shows wearable descriptions", async () => {
    renderWithProviders(<ConnectedAssets />);
    await waitFor(() => {
      expect(
        screen.getByText("Readiness, sleep, activity, heart rate")
      ).toBeInTheDocument();
      expect(
        screen.getByText("Recovery, strain, sleep, HRV")
      ).toBeInTheDocument();
      expect(
        screen.getByText("Steps, stress, body battery, workouts")
      ).toBeInTheDocument();
    });
  });

  it("shows EEG device state as disconnected", async () => {
    renderWithProviders(<ConnectedAssets />);
    await waitFor(() => {
      expect(
        screen.getByText("No device connected")
      ).toBeInTheDocument();
    });
  });

  it("shows Setup button when device is disconnected", async () => {
    renderWithProviders(<ConnectedAssets />);
    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: /Setup/i })
      ).toBeInTheDocument();
    });
  });

  it("shows health data sync description", async () => {
    renderWithProviders(<ConnectedAssets />);
    await waitFor(() => {
      expect(
        screen.getByText(
          "Syncs heart rate, sleep stages, steps, workouts, and mindfulness data."
        )
      ).toBeInTheDocument();
    });
  });

  it("shows web platform hint for health connect", async () => {
    renderWithProviders(<ConnectedAssets />);
    await waitFor(() => {
      expect(
        screen.getByText("Use the mobile app to connect")
      ).toBeInTheDocument();
    });
  });

  it("shows Connect buttons for wearable providers", async () => {
    renderWithProviders(<ConnectedAssets />);
    await waitFor(() => {
      // Each wearable has a Connect button
      const connectButtons = screen.getAllByRole("button", {
        name: /Connect/i,
      });
      // Health Connect button + 3 wearable Connect buttons = 4 total
      expect(connectButtons.length).toBeGreaterThanOrEqual(3);
    });
  });
});

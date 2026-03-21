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

vi.mock("@/lib/health-connect", () => ({
  requestHealthWritePermissions: vi.fn().mockResolvedValue(undefined),
}));

describe("ConnectedAssets page — flat device list", () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ devices: [] }),
    }) as unknown as typeof fetch;
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
          "Manage your health, brain, and wearable connections",
        ),
      ).toBeInTheDocument();
    });
  });

  it("shows all devices in a flat list", async () => {
    renderWithProviders(<ConnectedAssets />);
    await waitFor(() => {
      // Health
      expect(screen.getByText("Health Connect")).toBeInTheDocument();
      // EEG devices
      expect(screen.getByText("Muse 2")).toBeInTheDocument();
      expect(screen.getByText("Muse S")).toBeInTheDocument();
      expect(screen.getByText("Synthetic Demo")).toBeInTheDocument();
      // Wearables
      expect(screen.getByText("Oura Ring")).toBeInTheDocument();
      expect(screen.getByText("WHOOP")).toBeInTheDocument();
      expect(screen.getByText("Garmin")).toBeInTheDocument();
    });
  });

  it("shows device descriptions", async () => {
    renderWithProviders(<ConnectedAssets />);
    await waitFor(() => {
      expect(
        screen.getByText("Heart rate, sleep, steps, workouts, mindfulness"),
      ).toBeInTheDocument();
      expect(
        screen.getByText("EEG brain wave monitoring at 256 Hz"),
      ).toBeInTheDocument();
      expect(
        screen.getByText("Readiness, sleep, activity, heart rate"),
      ).toBeInTheDocument();
      expect(
        screen.getByText("Recovery, strain, sleep, HRV"),
      ).toBeInTheDocument();
      expect(
        screen.getByText("Steps, stress, body battery, workouts"),
      ).toBeInTheDocument();
    });
  });

  it("shows device test ids for each row", async () => {
    renderWithProviders(<ConnectedAssets />);
    await waitFor(() => {
      expect(screen.getByTestId("device-health")).toBeInTheDocument();
      expect(screen.getByTestId("device-muse2")).toBeInTheDocument();
      expect(screen.getByTestId("device-muse-s")).toBeInTheDocument();
      expect(screen.getByTestId("device-synthetic")).toBeInTheDocument();
      expect(screen.getByTestId("device-oura")).toBeInTheDocument();
      expect(screen.getByTestId("device-whoop")).toBeInTheDocument();
      expect(screen.getByTestId("device-garmin")).toBeInTheDocument();
    });
  });

  it("shows web platform hint for health connect", async () => {
    renderWithProviders(<ConnectedAssets />);
    await waitFor(() => {
      expect(
        screen.getByText("Use the mobile app to connect"),
      ).toBeInTheDocument();
    });
  });

  it("does NOT have separate section headers", async () => {
    renderWithProviders(<ConnectedAssets />);
    await waitFor(() => {
      // Old page had "Connect Health", "BCI / EEG", "Connect Wearable" headers
      expect(screen.queryByText("Connect Health")).not.toBeInTheDocument();
      expect(screen.queryByText("BCI / EEG")).not.toBeInTheDocument();
      expect(screen.queryByText("Connect Wearable")).not.toBeInTheDocument();
    });
  });

  it("shows Connect buttons for wearable providers", async () => {
    renderWithProviders(<ConnectedAssets />);
    await waitFor(() => {
      const connectButtons = screen.getAllByRole("button", {
        name: /Connect/i,
      });
      // Health Connect (disabled on web) + 3 wearables = 4
      expect(connectButtons.length).toBeGreaterThanOrEqual(3);
    });
  });
});

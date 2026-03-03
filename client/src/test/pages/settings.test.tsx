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

vi.mock("@/hooks/use-toast", () => ({
  useToast: () => ({ toast: vi.fn() }),
}));

vi.mock("@/lib/ml-api", () => ({
  ingestHealthData: vi.fn().mockResolvedValue({ stored: 10, metrics: ["heart_rate"] }),
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

  it("shows Health Connections section", async () => {
    renderWithProviders(<SettingsPage />);
    await waitFor(() => {
      expect(screen.getByText("Health Connections")).toBeInTheDocument();
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

  it("shows Apple Health Upload Export button", async () => {
    renderWithProviders(<SettingsPage />);
    await waitFor(() => {
      const uploadButtons = screen.getAllByText("Upload Export");
      expect(uploadButtons.length).toBeGreaterThanOrEqual(1);
    });
  });

  it("shows Export to HealthKit button", async () => {
    renderWithProviders(<SettingsPage />);
    await waitFor(() => {
      expect(screen.getByText("Export to HealthKit")).toBeInTheDocument();
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
});

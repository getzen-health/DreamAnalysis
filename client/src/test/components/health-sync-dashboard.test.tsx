/**
 * Tests for HealthSyncDashboard component.
 *
 * Covers:
 *  - All 6 sources rendered
 *  - Connected / Disconnected badge display
 *  - Color-coded freshness indicators
 *  - Sync button presence and disabled state
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, cleanup, waitFor } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import HealthSyncDashboard from "@/components/health-sync-dashboard";

// ── Mocks ─────────────────────────────────────────────────────────────────────

vi.mock("@/lib/ml-api", () => ({
  getMLApiUrl: () => "http://localhost:8080",
}));

// Default: all sources disconnected (fetch fails or returns empty)
function mockFetchDisconnected() {
  global.fetch = vi.fn().mockResolvedValue({
    ok: false,
    status: 503,
    json: async () => ({}),
  } as Response);
}

function mockFetchConnected() {
  const now = new Date().toISOString();
  const thirtyMinAgo = new Date(Date.now() - 30 * 60 * 1000).toISOString();
  const twoHoursAgo = new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString();
  const twoDaysAgo = new Date(Date.now() - 48 * 60 * 60 * 1000).toISOString();

  global.fetch = vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: async () => ({
      sources: [
        {
          source: "apple_health",
          connected: true,
          last_sync: thirtyMinAgo,
          data_types: ["HR", "HRV", "sleep", "steps"],
          freshness: "fresh",
        },
        {
          source: "google_health",
          connected: false,
          last_sync: null,
          data_types: ["steps", "HR", "calories"],
          freshness: "disconnected",
        },
        {
          source: "oura",
          connected: true,
          last_sync: twoHoursAgo,
          data_types: ["sleep", "readiness", "activity"],
          freshness: "stale",
        },
        {
          source: "garmin",
          connected: true,
          last_sync: twoDaysAgo,
          data_types: ["Body Battery", "stress", "HRV"],
          freshness: "old",
        },
        {
          source: "whoop",
          connected: false,
          last_sync: null,
          data_types: ["recovery", "strain", "sleep"],
          freshness: "disconnected",
        },
        {
          source: "muse_eeg",
          connected: true,
          last_sync: now,
          data_types: ["connection", "battery", "signal"],
          freshness: "fresh",
        },
      ],
      fetched_at: now,
    }),
  } as Response);
}

beforeEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

// ── Tests ─────────────────────────────────────────────────────────────────────

describe("HealthSyncDashboard", () => {
  it("renders the card heading", async () => {
    mockFetchDisconnected();
    renderWithProviders(<HealthSyncDashboard />);
    expect(screen.getByText(/Health Sync Status/i)).toBeInTheDocument();
  });

  it("renders all 6 source rows in default (disconnected) state", async () => {
    mockFetchDisconnected();
    renderWithProviders(<HealthSyncDashboard />);

    // The 6 source rows are always rendered from the static SOURCE_ORDER list
    // even when the API call fails, so they appear immediately without waiting.
    expect(screen.getByTestId("health-source-apple_health")).toBeInTheDocument();
    expect(screen.getByTestId("health-source-google_health")).toBeInTheDocument();
    expect(screen.getByTestId("health-source-oura")).toBeInTheDocument();
    expect(screen.getByTestId("health-source-garmin")).toBeInTheDocument();
    expect(screen.getByTestId("health-source-whoop")).toBeInTheDocument();
    expect(screen.getByTestId("health-source-muse_eeg")).toBeInTheDocument();
  });

  it("shows Disconnected badges for all sources when API fails", async () => {
    mockFetchDisconnected();
    renderWithProviders(<HealthSyncDashboard />);

    const badges = await screen.findAllByText("Disconnected");
    // At minimum all 6 sources should show disconnected
    expect(badges.length).toBeGreaterThanOrEqual(6);
  });

  it("shows Connected badge for connected sources and Disconnected for others", async () => {
    mockFetchConnected();
    renderWithProviders(<HealthSyncDashboard />);

    // Wait for fetch + re-render
    await waitFor(() => {
      expect(screen.queryAllByText("Connected").length).toBeGreaterThan(0);
    });

    const connectedBadges = screen.getAllByText("Connected");
    const disconnectedBadges = screen.getAllByText("Disconnected");

    // 4 connected (apple, oura, garmin, muse) + 2 disconnected (google, whoop)
    expect(connectedBadges.length).toBe(4);
    expect(disconnectedBadges.length).toBe(2);
  });

  it("applies green freshness class for fresh sources", async () => {
    mockFetchConnected();
    renderWithProviders(<HealthSyncDashboard />);

    await waitFor(() => {
      expect(screen.queryAllByText("Connected").length).toBeGreaterThan(0);
    });

    const appleBadge = screen.getByTestId("badge-apple_health");
    expect(appleBadge.className).toContain("green");
  });

  it("applies yellow freshness class for stale sources", async () => {
    mockFetchConnected();
    renderWithProviders(<HealthSyncDashboard />);

    await waitFor(() => {
      expect(screen.queryAllByText("Connected").length).toBeGreaterThan(0);
    });

    const ouraBadge = screen.getByTestId("badge-oura");
    expect(ouraBadge.className).toContain("yellow");
  });

  it("applies red freshness class for old sources", async () => {
    mockFetchConnected();
    renderWithProviders(<HealthSyncDashboard />);

    await waitFor(() => {
      expect(screen.queryAllByText("Connected").length).toBeGreaterThan(0);
    });

    const garminBadge = screen.getByTestId("badge-garmin");
    expect(garminBadge.className).toContain("red");
  });

  it("applies red freshness class for disconnected sources", async () => {
    mockFetchConnected();
    renderWithProviders(<HealthSyncDashboard />);

    await waitFor(() => {
      expect(screen.queryAllByText("Connected").length).toBeGreaterThan(0);
    });

    const googleBadge = screen.getByTestId("badge-google_health");
    expect(googleBadge.className).toContain("red");
  });

  it("renders a Sync button for each source", async () => {
    mockFetchDisconnected();
    renderWithProviders(<HealthSyncDashboard />);

    for (const source of [
      "apple_health",
      "google_health",
      "oura",
      "garmin",
      "whoop",
      "muse_eeg",
    ]) {
      expect(
        screen.getByTestId(`btn-sync-${source}`)
      ).toBeInTheDocument();
    }
  });

  it("Sync button is disabled when source is disconnected", async () => {
    mockFetchDisconnected();
    renderWithProviders(<HealthSyncDashboard />);

    // All sources disconnected — all Sync buttons should be disabled
    const syncBtn = screen.getByTestId("btn-sync-apple_health");
    expect(syncBtn).toBeDisabled();
  });

  it("Sync button is enabled when source is connected", async () => {
    mockFetchConnected();
    renderWithProviders(<HealthSyncDashboard />);

    await waitFor(() => {
      expect(screen.queryAllByText("Connected").length).toBeGreaterThan(0);
    });

    const syncBtn = screen.getByTestId("btn-sync-apple_health");
    expect(syncBtn).not.toBeDisabled();
  });

  it("Disconnect button appears only for connected sources", async () => {
    mockFetchConnected();
    renderWithProviders(<HealthSyncDashboard />);

    await waitFor(() => {
      expect(screen.queryAllByText("Connected").length).toBeGreaterThan(0);
    });

    // apple_health is connected → disconnect button present
    expect(screen.getByTestId("btn-disconnect-apple_health")).toBeInTheDocument();

    // google_health is disconnected → no disconnect button
    expect(
      screen.queryByTestId("btn-disconnect-google_health")
    ).not.toBeInTheDocument();
  });

  it("shows connected count summary badge", async () => {
    mockFetchConnected();
    renderWithProviders(<HealthSyncDashboard />);

    await waitFor(() => {
      expect(screen.getByText(/4\/6 connected/i)).toBeInTheDocument();
    });
  });

  it("shows 0/6 connected when API fails", async () => {
    mockFetchDisconnected();
    renderWithProviders(<HealthSyncDashboard />);

    // Immediately visible from static SOURCE_ORDER defaults
    expect(screen.getByText(/0\/6 connected/i)).toBeInTheDocument();
  });

  it("renders human-readable source labels", async () => {
    mockFetchDisconnected();
    renderWithProviders(<HealthSyncDashboard />);

    expect(screen.getByText("Apple HealthKit")).toBeInTheDocument();
    expect(screen.getByText("Google Health Connect")).toBeInTheDocument();
    expect(screen.getByText("Oura Ring")).toBeInTheDocument();
    expect(screen.getByText("Garmin")).toBeInTheDocument();
    expect(screen.getByText("Whoop")).toBeInTheDocument();
    expect(screen.getByText("Muse 2 EEG")).toBeInTheDocument();
  });
});

import { describe, it, expect, vi, beforeEach, beforeAll } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import Insights from "@/pages/insights";

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
  }),
}));

describe("Insights page", () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({}),
    }) as unknown as typeof fetch;
  });

  it("renders without crashing", async () => {
    renderWithProviders(<Insights />);
    await waitFor(() => expect(document.body).toBeTruthy());
  });

  it("shows connection banner when device is disconnected", async () => {
    renderWithProviders(<Insights />);
    await waitFor(() => {
      expect(
        screen.getByText(/Showing voice-based insights\. Health and watch signals can refine this further, and EEG is optional later\./)
      ).toBeInTheDocument();
    });
  });

  it("shows empty state prompt when no voice data and not streaming", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      json: async () => null,
    }) as unknown as typeof fetch;
    renderWithProviders(<Insights />);
    await waitFor(() => {
      expect(
        screen.getByText("Complete a voice check-in to unlock your first insights")
      ).toBeInTheDocument();
    });
  });

  it("shows Brain State Narrative feature preview in empty state", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      json: async () => null,
    }) as unknown as typeof fetch;
    renderWithProviders(<Insights />);
    await waitFor(() => {
      expect(screen.getByText("Brain State Narrative")).toBeInTheDocument();
    });
  });

  it("shows AI-Generated Insights feature preview in empty state", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      json: async () => null,
    }) as unknown as typeof fetch;
    renderWithProviders(<Insights />);
    await waitFor(() => {
      expect(screen.getByText("AI-Generated Insights")).toBeInTheDocument();
    });
  });

  it("shows Recommended Actions feature preview in empty state", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      json: async () => null,
    }) as unknown as typeof fetch;
    renderWithProviders(<Insights />);
    await waitFor(() => {
      expect(screen.getByText("Recommended Actions")).toBeInTheDocument();
    });
  });

  it("shows Trends Over Time feature preview in empty state", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      json: async () => null,
    }) as unknown as typeof fetch;
    renderWithProviders(<Insights />);
    await waitFor(() => {
      expect(screen.getByText("Trends Over Time")).toBeInTheDocument();
    });
  });

  it("shows Brain Wave Trends chart card", async () => {
    renderWithProviders(<Insights />);
    await waitFor(() => {
      expect(screen.getByText("Brain Wave Trends")).toBeInTheDocument();
    });
  });

  it("shows Cognitive Profile chart card", async () => {
    renderWithProviders(<Insights />);
    await waitFor(() => {
      expect(screen.getByText("Cognitive Profile")).toBeInTheDocument();
    });
  });

  it("shows connect device message in Brain Wave Trends when not streaming", async () => {
    renderWithProviders(<Insights />);
    await waitFor(() => {
      expect(
        screen.getByText("Live trends appear when optional EEG is connected")
      ).toBeInTheDocument();
    });
  });

  it("shows connect device message in Cognitive Profile when not streaming", async () => {
    renderWithProviders(<Insights />);
    await waitFor(() => {
      expect(
        screen.getByText("Add EEG later to see the live cognitive profile")
      ).toBeInTheDocument();
    });
  });
});

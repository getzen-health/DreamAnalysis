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
        screen.getByText(/Connect your Muse 2 from the sidebar to unlock your live brain narrative/)
      ).toBeInTheDocument();
    });
  });

  it("shows What You'll See Here card when not streaming", async () => {
    renderWithProviders(<Insights />);
    await waitFor(() => {
      expect(screen.getByText("What You'll See Here")).toBeInTheDocument();
    });
  });

  it("shows Brain State Narrative feature preview", async () => {
    renderWithProviders(<Insights />);
    await waitFor(() => {
      expect(screen.getByText("Brain State Narrative")).toBeInTheDocument();
    });
  });

  it("shows AI-Generated Insights feature preview", async () => {
    renderWithProviders(<Insights />);
    await waitFor(() => {
      expect(screen.getByText("AI-Generated Insights")).toBeInTheDocument();
    });
  });

  it("shows Recommended Actions feature preview", async () => {
    renderWithProviders(<Insights />);
    await waitFor(() => {
      expect(screen.getByText("Recommended Actions")).toBeInTheDocument();
    });
  });

  it("shows Band Wave Trends feature preview", async () => {
    renderWithProviders(<Insights />);
    await waitFor(() => {
      expect(screen.getByText("Band Wave Trends")).toBeInTheDocument();
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
        screen.getByText("Connect Muse 2 to see live trends")
      ).toBeInTheDocument();
    });
  });

  it("shows connect device message in Cognitive Profile when not streaming", async () => {
    renderWithProviders(<Insights />);
    await waitFor(() => {
      expect(
        screen.getByText("Connect Muse 2 to see your cognitive profile")
      ).toBeInTheDocument();
    });
  });
});

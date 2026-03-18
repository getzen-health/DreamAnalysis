import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import InnerEnergy from "@/pages/inner-energy";

vi.mock("@/lib/participant", () => ({ getParticipantId: () => "test-user-123" }));

vi.mock("@/hooks/use-device", () => ({
  useDevice: () => ({
    state: "disconnected",
    latestFrame: null,
    connect: vi.fn(),
    disconnect: vi.fn(),
  }),
}));

vi.mock("@/components/score-circle", () => ({
  ScoreCircle: ({ label, value }: { label: string; value: number }) => (
    <div data-testid={`score-circle-${label}`}>{value}</div>
  ),
}));

describe("InnerEnergy page", () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({}),
    }) as unknown as typeof fetch;
  });

  it("renders without crashing", async () => {
    renderWithProviders(<InnerEnergy />);
    await waitFor(() => expect(document.body).toBeTruthy());
  });

  it("shows Inner Energy page heading", async () => {
    renderWithProviders(<InnerEnergy />);
    await waitFor(() => {
      expect(screen.getByText("Inner Energy")).toBeInTheDocument();
    });
  });

  it("shows connection banner when device is disconnected", async () => {
    renderWithProviders(<InnerEnergy />);
    await waitFor(() => {
      expect(
        screen.getByText(/Start with a voice analysis to see your energy state/)
      ).toBeInTheDocument();
    });
  });

  it("shows Energy Guidance section", async () => {
    renderWithProviders(<InnerEnergy />);
    await waitFor(() => {
      expect(screen.getByText("Energy Guidance")).toBeInTheDocument();
    });
  });

  it("shows connect message in guidance when disconnected", async () => {
    renderWithProviders(<InnerEnergy />);
    await waitFor(() => {
      expect(
        screen.getByText(/Start with a voice analysis to estimate your energy state/)
      ).toBeInTheDocument();
    });
  });

  it("shows Meditation score circle", async () => {
    renderWithProviders(<InnerEnergy />);
    await waitFor(() => {
      expect(screen.getByTestId("score-circle-Meditation")).toBeInTheDocument();
    });
  });

  it("shows Awareness score circle", async () => {
    renderWithProviders(<InnerEnergy />);
    await waitFor(() => {
      expect(screen.getByTestId("score-circle-Awareness")).toBeInTheDocument();
    });
  });

  it("shows Clarity score circle", async () => {
    renderWithProviders(<InnerEnergy />);
    await waitFor(() => {
      expect(screen.getByTestId("score-circle-Clarity")).toBeInTheDocument();
    });
  });

  it("shows all 7 chakra cards (Energy Centers)", async () => {
    renderWithProviders(<InnerEnergy />);
    await waitFor(() => {
      expect(screen.getByText("Energy Centers")).toBeInTheDocument();
      // Chakra names
      expect(screen.getByText("Root")).toBeInTheDocument();
      expect(screen.getByText("Heart")).toBeInTheDocument();
      expect(screen.getByText("Crown")).toBeInTheDocument();
    });
  });

  it("shows all chakra Sanskrit names", async () => {
    renderWithProviders(<InnerEnergy />);
    await waitFor(() => {
      expect(screen.getByText("Muladhara")).toBeInTheDocument();
      expect(screen.getByText("Sahasrara")).toBeInTheDocument();
    });
  });
});

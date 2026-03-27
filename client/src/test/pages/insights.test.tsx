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
    // Insights reads from localStorage, not fetch
    localStorage.clear();
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({}),
    }) as unknown as typeof fetch;
  });

  it("renders without crashing", async () => {
    renderWithProviders(<Insights />);
    await waitFor(() => expect(document.body).toBeTruthy());
  });

  it("shows Insights heading", () => {
    renderWithProviders(<Insights />);
    expect(screen.getByText("Insights")).toBeInTheDocument();
  });

  it("shows empty state when no emotion history in localStorage", async () => {
    renderWithProviders(<Insights />);
    await waitFor(() => {
      expect(screen.getByText("Your insight engine is waiting")).toBeInTheDocument();
    });
  });

  it("shows CTA button to start check-in in empty state", async () => {
    renderWithProviders(<Insights />);
    await waitFor(() => {
      expect(screen.getByText("Go to Today's check-in")).toBeInTheDocument();
    });
  });

  it("shows start a check-in subtitle in empty state", async () => {
    renderWithProviders(<Insights />);
    await waitFor(() => {
      expect(
        screen.getByText(/Start a voice check-in to build your insight engine/)
      ).toBeInTheDocument();
    });
  });

  it("shows emotion history data when localStorage has readings", async () => {
    const fakeHistory = [
      {
        stress: 0.4,
        happiness: 0.7,
        focus: 0.6,
        dominantEmotion: "happy",
        timestamp: new Date().toISOString(),
        valence: 0.5,
        arousal: 0.6,
      },
      {
        stress: 0.3,
        happiness: 0.8,
        focus: 0.7,
        dominantEmotion: "happy",
        timestamp: new Date(Date.now() - 86400000).toISOString(),
        valence: 0.6,
        arousal: 0.5,
      },
    ];
    localStorage.setItem("ndw_emotion_history", JSON.stringify(fakeHistory));
    renderWithProviders(<Insights />);
    await waitFor(() => {
      expect(screen.getByText(/readings · personal baseline active/)).toBeInTheDocument();
    });
  });

  it("shows Explore Patterns section when data is present", async () => {
    const fakeHistory = [
      {
        stress: 0.4,
        happiness: 0.7,
        focus: 0.6,
        dominantEmotion: "happy",
        timestamp: new Date().toISOString(),
      },
    ];
    localStorage.setItem("ndw_emotion_history", JSON.stringify(fakeHistory));
    renderWithProviders(<Insights />);
    await waitFor(() => {
      expect(screen.getByText("Explore Patterns")).toBeInTheDocument();
    });
  });

  it("shows Emotion Trends link when data is present", async () => {
    const fakeHistory = [
      {
        stress: 0.4,
        happiness: 0.7,
        focus: 0.6,
        dominantEmotion: "neutral",
        timestamp: new Date().toISOString(),
      },
    ];
    localStorage.setItem("ndw_emotion_history", JSON.stringify(fakeHistory));
    renderWithProviders(<Insights />);
    await waitFor(() => {
      expect(screen.getByText("Emotion Trends")).toBeInTheDocument();
    });
  });

  it("shows Brain States link when data is present", async () => {
    const fakeHistory = [
      {
        stress: 0.5,
        happiness: 0.5,
        focus: 0.5,
        dominantEmotion: "neutral",
        timestamp: new Date().toISOString(),
      },
    ];
    localStorage.setItem("ndw_emotion_history", JSON.stringify(fakeHistory));
    renderWithProviders(<Insights />);
    await waitFor(() => {
      expect(screen.getByText("Brain States")).toBeInTheDocument();
    });
  });

  it("does not show empty state when emotion history exists", async () => {
    const fakeHistory = [
      {
        stress: 0.3,
        happiness: 0.8,
        focus: 0.7,
        dominantEmotion: "happy",
        timestamp: new Date().toISOString(),
      },
    ];
    localStorage.setItem("ndw_emotion_history", JSON.stringify(fakeHistory));
    renderWithProviders(<Insights />);
    await waitFor(() => {
      expect(screen.queryByText("Your insight engine is waiting")).not.toBeInTheDocument();
    });
  });
});

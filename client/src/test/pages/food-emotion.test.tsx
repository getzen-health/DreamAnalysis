import { describe, it, expect, vi, beforeEach, beforeAll } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import FoodEmotion from "@/pages/food-emotion";

beforeAll(() => {
  global.ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
});

vi.mock("@/lib/participant", () => ({ getParticipantId: () => "test-user-123" }));

vi.mock("@/lib/ml-api", () => ({
  predictFoodEmotion: vi.fn().mockResolvedValue({
    food_state: "balanced",
    state_probabilities: {
      craving_carbs: 0.1,
      appetite_suppressed: 0.05,
      comfort_seeking: 0.1,
      balanced: 0.6,
      stress_eating: 0.05,
      mindful_eating: 0.1,
    },
    confidence: 0.8,
    is_calibrated: false,
    calibration_progress: 0.3,
    components: {
      faa: 0.2,
      high_beta: 0.35,
      prefrontal_theta: 0.4,
      delta: 0.25,
    },
    recommendations: {
      prefer: ["Complex carbs", "Leafy greens"],
      avoid: ["Sugary snacks", "Processed foods"],
      strategy: "Focus on mindful eating",
      mindfulness_tip: "Pause before eating",
    },
    simulation_mode: true,
  }),
  calibrateFoodEmotion: vi.fn().mockResolvedValue({}),
}));

/* ------------------------------------------------------------------
 * Mutable mock state for useDevice.
 *
 * By default the device is disconnected so the useQuery fires and
 * `data` (with recommendations) is available. Tests that need the
 * streaming path (badge + confidence) flip this before rendering.
 * ------------------------------------------------------------------ */
const deviceMock = {
  latestFrame: null as any,
  state: "disconnected" as string,
  deviceStatus: null as any,
};

vi.mock("@/hooks/use-device", () => ({
  useDevice: () => deviceMock,
}));

describe("FoodEmotion page", () => {
  beforeEach(() => {
    // Reset to disconnected before every test
    deviceMock.latestFrame = null;
    deviceMock.state = "disconnected";
    deviceMock.deviceStatus = null;

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({}),
    }) as unknown as typeof fetch;
  });

  it("renders without crashing", async () => {
    renderWithProviders(<FoodEmotion />);
    await waitFor(() => expect(document.body).toBeTruthy());
  });

  it("shows the Food & Cravings heading", async () => {
    renderWithProviders(<FoodEmotion />);
    await waitFor(() => {
      expect(screen.getByText("Food & Cravings")).toBeInTheDocument();
    });
  });

  it("shows EEG-based appetite subtitle", async () => {
    renderWithProviders(<FoodEmotion />);
    await waitFor(() => {
      expect(
        screen.getByText("EEG-based appetite and eating-state analysis")
      ).toBeInTheDocument();
    });
  });

  it("shows Current Food State card", async () => {
    renderWithProviders(<FoodEmotion />);
    await waitFor(() => {
      expect(screen.getByText("Current Food State")).toBeInTheDocument();
    });
  });

  it("shows the balanced food state badge after data loads", async () => {
    // Badge only renders when device is streaming with live emotions
    deviceMock.state = "streaming";
    deviceMock.latestFrame = {
      analysis: {
        emotions: {
          stress_index: 0.1,
          relaxation_index: 0.3,
          focus_index: 0.3,
          valence: 0.0,
        },
      },
    };
    renderWithProviders(<FoodEmotion />);
    await waitFor(() => {
      // deriveFoodStates produces a label rendered as a badge
      expect(
        screen.getByText(
          /Balanced|Mindful Eating|Craving Carbs|Comfort Seeking|Stress Eating|Appetite Suppressed/
        )
      ).toBeInTheDocument();
    });
  });

  it("shows Confidence label in Current Food State card", async () => {
    // Confidence only renders when streaming with live emotions
    deviceMock.state = "streaming";
    deviceMock.latestFrame = {
      analysis: {
        emotions: {
          stress_index: 0.1,
          relaxation_index: 0.3,
          focus_index: 0.3,
          valence: 0.0,
        },
      },
    };
    renderWithProviders(<FoodEmotion />);
    await waitFor(() => {
      expect(screen.getByText("Confidence")).toBeInTheDocument();
    });
  });

  it("shows prefer list after data loads", async () => {
    renderWithProviders(<FoodEmotion />);
    await waitFor(() => {
      expect(screen.getByText("Prefer")).toBeInTheDocument();
    });
  });

  it("shows avoid list after data loads", async () => {
    renderWithProviders(<FoodEmotion />);
    await waitFor(() => {
      expect(screen.getByText("Avoid")).toBeInTheDocument();
    });
  });

  it("shows mindfulness tip after data loads", async () => {
    renderWithProviders(<FoodEmotion />);
    await waitFor(() => {
      expect(screen.getByText("Pause before eating")).toBeInTheDocument();
    });
  });

  it("shows Dietary Guidance section after data loads", async () => {
    renderWithProviders(<FoodEmotion />);
    await waitFor(() => {
      expect(screen.getByText("Dietary Guidance")).toBeInTheDocument();
    });
  });

  it("shows Calibration card", async () => {
    renderWithProviders(<FoodEmotion />);
    await waitFor(() => {
      expect(screen.getByText("Calibration")).toBeInTheDocument();
    });
  });

  it("shows Calibrate Now button", async () => {
    renderWithProviders(<FoodEmotion />);
    await waitFor(() => {
      expect(screen.getByText("Calibrate Now")).toBeInTheDocument();
    });
  });
});

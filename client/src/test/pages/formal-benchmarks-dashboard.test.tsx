import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import FormalBenchmarksDashboard from "@/pages/formal-benchmarks-dashboard";

vi.mock("@/lib/participant", () => ({ getParticipantId: () => "test-user-123" }));

describe("FormalBenchmarksDashboard page", () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({}),
    }) as unknown as typeof fetch;
  });

  it("renders without crashing", async () => {
    renderWithProviders(<FormalBenchmarksDashboard />);
    await waitFor(() => expect(document.body).toBeTruthy());
  });

  it("shows the Full System Overview heading", async () => {
    renderWithProviders(<FormalBenchmarksDashboard />);
    await waitFor(() => {
      expect(screen.getByText("Full System Overview")).toBeInTheDocument();
    });
  });

  it("shows the NeuralDreamWorkshop Research Dashboard label", async () => {
    renderWithProviders(<FormalBenchmarksDashboard />);
    await waitFor(() => {
      expect(
        screen.getByText(/NeuralDreamWorkshop · Research Dashboard/)
      ).toBeInTheDocument();
    });
  });

  it("shows hero stat cards (ML Models, API Endpoints, etc.)", async () => {
    renderWithProviders(<FormalBenchmarksDashboard />);
    await waitFor(() => {
      // Multiple "ML Models" elements may exist — just check at least one
      const mlModelEls = screen.getAllByText("ML Models");
      expect(mlModelEls.length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText("API Endpoints")).toBeInTheDocument();
      expect(screen.getByText("EEG Datasets")).toBeInTheDocument();
    });
  });

  it("shows 18 as ML Models count", async () => {
    renderWithProviders(<FormalBenchmarksDashboard />);
    await waitFor(() => {
      expect(screen.getByText("18")).toBeInTheDocument();
    });
  });

  it("shows Section 1 Models header", async () => {
    renderWithProviders(<FormalBenchmarksDashboard />);
    await waitFor(() => {
      expect(screen.getByText("All 21 ML Models")).toBeInTheDocument();
    });
  });

  it("shows model accuracy numbers for Emotion Classifier", async () => {
    renderWithProviders(<FormalBenchmarksDashboard />);
    await waitFor(() => {
      expect(screen.getByText("71.52% CV")).toBeInTheDocument();
    });
  });

  it("shows model cards for key models (Sleep Staging, Dream Detector)", async () => {
    renderWithProviders(<FormalBenchmarksDashboard />);
    await waitFor(() => {
      // Model names appear in model cards — may be split across nodes, use getAllByText
      expect(screen.getAllByText(/Sleep Staging/).length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText(/Dream Detector/).length).toBeGreaterThanOrEqual(1);
    });
  });

  it("shows Section 2 Datasets header", async () => {
    renderWithProviders(<FormalBenchmarksDashboard />);
    await waitFor(() => {
      expect(
        screen.getByText("8 EEG Datasets — Honest Accuracy")
      ).toBeInTheDocument();
    });
  });

  it("shows Section 3 Food-Emotion header", async () => {
    renderWithProviders(<FormalBenchmarksDashboard />);
    await waitFor(() => {
      expect(screen.getByText("Food-Emotion Biomarker System")).toBeInTheDocument();
    });
  });

  it("shows category badges (Emotion, Sleep, Cognition)", async () => {
    renderWithProviders(<FormalBenchmarksDashboard />);
    await waitFor(() => {
      // Category legend badges exist
      const emotionBadges = screen.getAllByText("Emotion");
      expect(emotionBadges.length).toBeGreaterThanOrEqual(1);
      const sleepBadges = screen.getAllByText("Sleep");
      expect(sleepBadges.length).toBeGreaterThanOrEqual(1);
    });
  });

  it("shows Live Accuracy label in model cards", async () => {
    renderWithProviders(<FormalBenchmarksDashboard />);
    await waitFor(() => {
      const liveAccuracyLabels = screen.getAllByText("Live Accuracy");
      expect(liveAccuracyLabels.length).toBeGreaterThan(0);
    });
  });

  it("shows research roadmap section (Publishing Plan)", async () => {
    renderWithProviders(<FormalBenchmarksDashboard />);
    await waitFor(() => {
      // IRB Ethics Approval is a publishing plan step title
      const irbEls = screen.getAllByText(/IRB Ethics Approval/);
      expect(irbEls.length).toBeGreaterThanOrEqual(1);
    });
  });
});

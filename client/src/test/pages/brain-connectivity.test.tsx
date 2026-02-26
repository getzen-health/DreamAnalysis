import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor, fireEvent } from "@testing-library/react";
import { renderWithProviders } from "../test-utils";
import BrainConnectivity from "@/pages/brain-connectivity";

vi.mock("@/lib/participant", () => ({ getParticipantId: () => "test-user-123" }));

vi.mock("@/lib/ml-api", () => ({
  analyzeConnectivity: vi.fn().mockResolvedValue({
    connectivity_matrix: [[1, 0.5], [0.5, 1]],
    graph_metrics: {
      clustering_coefficient: 0.72,
      avg_path_length: 1.5,
      small_world_index: 2.1,
      hub_nodes: [0],
      modularity: 0.38,
    },
    directed_flow: {
      granger: { matrix: [[0, 0.3], [0.3, 0]], significant_pairs: [] },
      dtf_matrix: [[0, 0.2], [0.2, 0]],
      dominant_direction: "frontal_to_temporal",
    },
  }),
  simulateEEG: vi.fn().mockResolvedValue({
    signals: [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
  }),
}));

vi.mock("@/components/charts/connectivity-chart", () => ({
  ConnectivityChart: () => <div data-testid="connectivity-chart" />,
}));

describe("BrainConnectivity page", () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({}),
    }) as unknown as typeof fetch;
  });

  it("renders without crashing", async () => {
    renderWithProviders(<BrainConnectivity />);
    await waitFor(() => expect(document.body).toBeTruthy());
  });

  it("shows Brain Network Connectivity heading", async () => {
    renderWithProviders(<BrainConnectivity />);
    await waitFor(() => {
      expect(screen.getByText("Brain Network Connectivity")).toBeInTheDocument();
    });
  });

  it("shows controls card with Channels and Method selectors", async () => {
    renderWithProviders(<BrainConnectivity />);
    await waitFor(() => {
      expect(screen.getByText("Channels:")).toBeInTheDocument();
      expect(screen.getByText("Method:")).toBeInTheDocument();
    });
  });

  it("shows the Analyze button", async () => {
    renderWithProviders(<BrainConnectivity />);
    await waitFor(() => {
      expect(screen.getByText("Analyze")).toBeInTheDocument();
    });
  });

  it("does not show results before clicking Analyze", async () => {
    renderWithProviders(<BrainConnectivity />);
    await waitFor(() => {
      expect(screen.queryByText("Graph Metrics")).not.toBeInTheDocument();
    });
  });

  it("shows Graph Metrics after clicking Analyze", async () => {
    renderWithProviders(<BrainConnectivity />);
    const analyzeBtn = await screen.findByText("Analyze");
    fireEvent.click(analyzeBtn);
    await waitFor(() => {
      expect(screen.getByText("Graph Metrics")).toBeInTheDocument();
    });
  });

  it("shows Connectivity Graph heading after analyze", async () => {
    renderWithProviders(<BrainConnectivity />);
    const analyzeBtn = await screen.findByText("Analyze");
    fireEvent.click(analyzeBtn);
    await waitFor(() => {
      expect(screen.getByText("Connectivity Graph")).toBeInTheDocument();
    });
  });

  it("shows Clustering Coefficient metric after analyze", async () => {
    renderWithProviders(<BrainConnectivity />);
    const analyzeBtn = await screen.findByText("Analyze");
    fireEvent.click(analyzeBtn);
    await waitFor(() => {
      expect(screen.getByText("Clustering Coefficient")).toBeInTheDocument();
    });
  });

  it("shows Connectivity Matrix heading after analyze", async () => {
    renderWithProviders(<BrainConnectivity />);
    const analyzeBtn = await screen.findByText("Analyze");
    fireEvent.click(analyzeBtn);
    await waitFor(() => {
      expect(screen.getByText("Connectivity Matrix")).toBeInTheDocument();
    });
  });

  it("shows Hub Nodes after analyze", async () => {
    renderWithProviders(<BrainConnectivity />);
    const analyzeBtn = await screen.findByText("Analyze");
    fireEvent.click(analyzeBtn);
    await waitFor(() => {
      expect(screen.getByText("Hub Nodes")).toBeInTheDocument();
    });
  });
});

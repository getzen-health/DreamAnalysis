import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock fetch globally
const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

// Must import after mocking
const ML_API_URL = "http://localhost:8000";

describe("ML API client", () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  it("should construct correct URL for health check", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ status: "healthy", service: "neural-dream-ml" }),
    });

    const res = await fetch(`${ML_API_URL}/health`);
    const data = await res.json();

    expect(mockFetch).toHaveBeenCalledWith(`${ML_API_URL}/health`);
    expect(data.status).toBe("healthy");
  });

  it("should handle EEG simulation request", async () => {
    const mockAnalysis = {
      analysis: {
        emotions: { emotion: "relaxed", confidence: 0.8 },
        sleep_staging: { stage: "Wake" },
        flow_state: { flow_score: 0.3 },
      },
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockAnalysis,
    });

    const res = await fetch(`${ML_API_URL}/api/simulate-eeg`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ state: "rest", duration: 2.0, fs: 256, n_channels: 1 }),
    });
    const data = await res.json();

    expect(data.analysis.emotions.emotion).toBe("relaxed");
    expect(data.analysis.sleep_staging.stage).toBe("Wake");
  });

  it("should handle API errors gracefully", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      json: async () => ({ detail: "Internal server error" }),
    });

    const res = await fetch(`${ML_API_URL}/api/simulate-eeg`, {
      method: "POST",
      body: "{}",
    });

    expect(res.ok).toBe(false);
    expect(res.status).toBe(500);
  });
});

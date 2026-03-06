import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// Mock fetch globally before importing the module under test
const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

// Import an exported function that goes through mlFetch.
// simulateEEG calls mlFetch internally.
import { simulateEEG } from "@/lib/ml-api";

// Helper: build a minimal successful Response-like object
function okResponse(body: unknown = {}) {
  return {
    ok: true,
    status: 200,
    clone: () => okResponse(body),
    json: async () => body,
    text: async () => JSON.stringify(body),
  };
}

// Helper: build a failing Response-like object.
// When `detail` is provided the response body includes a `detail` field;
// mlFetch treats 5xx responses with a detail string as non-retryable
// (descriptive server error like "Bluetooth not available").
// Omit `detail` (the default for 5xx) to exercise the retry path.
function failResponse(status: number, detail?: string) {
  const body = detail !== undefined ? { detail } : {};
  return {
    ok: false,
    status,
    statusText: String(status),
    clone: function () { return failResponse(status, detail); },
    json: async () => body,
    text: async () => JSON.stringify(body),
  };
}

describe("mlFetch retry logic", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    mockFetch.mockReset();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("retries 3 times on 503 then throws (total 4 fetch calls)", async () => {
    mockFetch.mockResolvedValue(failResponse(503));

    // Attach a no-op rejection handler immediately to prevent unhandledRejection
    // before we await it below.
    const promise = simulateEEG("rest", 2, 256, 1);
    // Prevent unhandled rejection warning while timers run
    promise.catch(() => undefined);

    // Advance through all three retry delays: 1s + 3s + 9s = 13s total
    await vi.advanceTimersByTimeAsync(1_000); // retry 1 fires
    await vi.advanceTimersByTimeAsync(3_000); // retry 2 fires
    await vi.advanceTimersByTimeAsync(9_000); // retry 3 fires

    await expect(promise).rejects.toThrow();

    // 1 initial + 3 retries = 4 total calls
    expect(mockFetch).toHaveBeenCalledTimes(4);
  });

  it("does NOT retry on 422 (total 1 fetch call)", async () => {
    mockFetch.mockResolvedValueOnce(failResponse(422, "Unprocessable Entity"));

    const promise = simulateEEG("rest", 2, 256, 1);

    // No timer advancement needed — 4xx should fail immediately without retrying
    await expect(promise).rejects.toThrow();

    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  it("succeeds on 2nd attempt after 1 failure (total 2 fetch calls)", async () => {
    const successBody = {
      simulation: { signals: [[0, 1, 2]], fs: 256 },
      analysis: {
        emotions: { emotion: "relaxed", confidence: 0.8 },
        sleep_staging: { stage: "Wake" },
        flow_state: { flow_score: 0.3 },
      },
    };

    mockFetch
      .mockResolvedValueOnce(failResponse(503))
      .mockResolvedValueOnce(okResponse(successBody));

    const promise = simulateEEG("rest", 2, 256, 1);

    // Advance 1s to trigger the first retry
    await vi.advanceTimersByTimeAsync(1_000);

    const result = await promise;

    // Should have resolved successfully
    expect(result).toBeDefined();

    // 1 failure + 1 success = 2 total calls
    expect(mockFetch).toHaveBeenCalledTimes(2);
  });
});

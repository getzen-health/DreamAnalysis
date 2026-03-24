import { describe, it, expect } from "vitest";
import {
  InferenceLatencyTracker,
  type InferenceStats,
  type ModelLatencyStats,
} from "@/lib/inference-latency";

// ── Constructor / initial state ─────────────────────────────────────────────

describe("InferenceLatencyTracker — initial state", () => {
  it("returns empty stats when no samples recorded", () => {
    const tracker = new InferenceLatencyTracker();
    const stats = tracker.getStats();
    expect(stats).toEqual({});
  });

  it("getModelStats returns null for unknown model", () => {
    const tracker = new InferenceLatencyTracker();
    expect(tracker.getModelStats("eegnet")).toBeNull();
  });
});

// ── Single model recording ──────────────────────────────────────────────────

describe("InferenceLatencyTracker — single model", () => {
  it("records a single sample and returns correct stats", () => {
    const tracker = new InferenceLatencyTracker();
    tracker.record("eegnet", 10);

    const stats = tracker.getModelStats("eegnet");
    expect(stats).not.toBeNull();
    expect(stats!.avg).toBe(10);
    expect(stats!.p95).toBe(10);
    expect(stats!.min).toBe(10);
    expect(stats!.max).toBe(10);
    expect(stats!.count).toBe(1);
  });

  it("computes correct average over multiple samples", () => {
    const tracker = new InferenceLatencyTracker();
    tracker.record("eegnet", 10);
    tracker.record("eegnet", 20);
    tracker.record("eegnet", 30);

    const stats = tracker.getModelStats("eegnet")!;
    expect(stats.avg).toBe(20);
    expect(stats.min).toBe(10);
    expect(stats.max).toBe(30);
    expect(stats.count).toBe(3);
  });

  it("computes correct p95 for 20 samples", () => {
    const tracker = new InferenceLatencyTracker();
    // 19 samples at 10ms, 1 sample at 100ms
    for (let i = 0; i < 19; i++) {
      tracker.record("eegnet", 10);
    }
    tracker.record("eegnet", 100);

    const stats = tracker.getModelStats("eegnet")!;
    // p95 index = ceil(20 * 0.95) - 1 = ceil(19) - 1 = 18 (0-indexed)
    // sorted: [10 x19, 100] -> sorted[18] = 10
    // The 100ms outlier is above the 95th percentile
    expect(stats.p95).toBe(10);
    expect(stats.count).toBe(20);
  });

  it("computes correct p95 for 100 samples", () => {
    const tracker = new InferenceLatencyTracker();
    // 95 samples at 5ms, 5 samples at 50ms
    for (let i = 0; i < 95; i++) {
      tracker.record("eegnet", 5);
    }
    // But buffer is 50 — only last 50 samples are kept
    // So after 95 records of 5ms, buffer has 50 samples of 5ms
    for (let i = 0; i < 5; i++) {
      tracker.record("eegnet", 50);
    }

    const stats = tracker.getModelStats("eegnet")!;
    // Buffer has last 50 samples: 45 x 5ms + 5 x 50ms
    expect(stats.count).toBe(50);
    expect(stats.p95).toBe(50);
  });
});

// ── Rolling buffer eviction ─────────────────────────────────────────────────

describe("InferenceLatencyTracker — rolling buffer", () => {
  it("caps at bufferSize samples (default 50)", () => {
    const tracker = new InferenceLatencyTracker();
    for (let i = 0; i < 100; i++) {
      tracker.record("eegnet", i);
    }
    const stats = tracker.getModelStats("eegnet")!;
    expect(stats.count).toBe(50);
  });

  it("evicts oldest samples when buffer is full", () => {
    const tracker = new InferenceLatencyTracker();
    // Record 50 samples of 1000ms (these will all be evicted)
    for (let i = 0; i < 50; i++) {
      tracker.record("eegnet", 1000);
    }
    // Record 50 samples of 5ms (these fill the buffer, evicting all 1000ms)
    for (let i = 0; i < 50; i++) {
      tracker.record("eegnet", 5);
    }

    const stats = tracker.getModelStats("eegnet")!;
    expect(stats.avg).toBe(5);
    expect(stats.min).toBe(5);
    expect(stats.max).toBe(5);
  });

  it("respects custom buffer size", () => {
    const tracker = new InferenceLatencyTracker(10);
    for (let i = 0; i < 20; i++) {
      tracker.record("eegnet", i);
    }
    const stats = tracker.getModelStats("eegnet")!;
    expect(stats.count).toBe(10);
    // Last 10 samples are 10..19
    expect(stats.min).toBe(10);
    expect(stats.max).toBe(19);
    expect(stats.avg).toBe(14.5);
  });
});

// ── Multiple models ─────────────────────────────────────────────────────────

describe("InferenceLatencyTracker — multiple models", () => {
  it("tracks different models independently", () => {
    const tracker = new InferenceLatencyTracker();
    tracker.record("eegnet", 5);
    tracker.record("voice", 15);
    tracker.record("crossAttention", 25);

    const all = tracker.getStats();
    expect(Object.keys(all)).toHaveLength(3);
    expect(all.eegnet!.avg).toBe(5);
    expect(all.voice!.avg).toBe(15);
    expect(all.crossAttention!.avg).toBe(25);
  });

  it("getStats returns all models in one call", () => {
    const tracker = new InferenceLatencyTracker();
    tracker.record("eegnet", 10);
    tracker.record("eegnet", 20);
    tracker.record("emotion", 30);

    const stats = tracker.getStats();
    expect("eegnet" in stats).toBe(true);
    expect("emotion" in stats).toBe(true);
    expect(stats.eegnet!.avg).toBe(15);
    expect(stats.emotion!.avg).toBe(30);
  });
});

// ── Edge cases ──────────────────────────────────────────────────────────────

describe("InferenceLatencyTracker — edge cases", () => {
  it("handles zero-ms latency", () => {
    const tracker = new InferenceLatencyTracker();
    tracker.record("eegnet", 0);
    const stats = tracker.getModelStats("eegnet")!;
    expect(stats.avg).toBe(0);
    expect(stats.p95).toBe(0);
  });

  it("handles fractional ms values", () => {
    const tracker = new InferenceLatencyTracker();
    tracker.record("eegnet", 1.5);
    tracker.record("eegnet", 2.7);
    const stats = tracker.getModelStats("eegnet")!;
    expect(stats.avg).toBeCloseTo(2.1, 5);
  });

  it("reset clears all data", () => {
    const tracker = new InferenceLatencyTracker();
    tracker.record("eegnet", 10);
    tracker.record("voice", 20);
    tracker.reset();
    expect(tracker.getStats()).toEqual({});
    expect(tracker.getModelStats("eegnet")).toBeNull();
  });

  it("reset for single model clears only that model", () => {
    const tracker = new InferenceLatencyTracker();
    tracker.record("eegnet", 10);
    tracker.record("voice", 20);
    tracker.resetModel("eegnet");
    expect(tracker.getModelStats("eegnet")).toBeNull();
    expect(tracker.getModelStats("voice")).not.toBeNull();
  });
});

// ── p95 correctness ─────────────────────────────────────────────────────────

describe("InferenceLatencyTracker — p95 precision", () => {
  it("p95 of uniform distribution [1..50] is 48", () => {
    const tracker = new InferenceLatencyTracker(50);
    for (let i = 1; i <= 50; i++) {
      tracker.record("eegnet", i);
    }
    const stats = tracker.getModelStats("eegnet")!;
    // p95 index = ceil(50 * 0.95) - 1 = ceil(47.5) - 1 = 48 - 1 = 47 (0-indexed)
    // sorted[47] = 48
    expect(stats.p95).toBe(48);
  });

  it("p95 of all-same values equals that value", () => {
    const tracker = new InferenceLatencyTracker();
    for (let i = 0; i < 50; i++) {
      tracker.record("eegnet", 42);
    }
    expect(tracker.getModelStats("eegnet")!.p95).toBe(42);
  });
});

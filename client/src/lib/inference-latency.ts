/**
 * Inference Latency Instrumentation — tracks per-model ONNX inference timing.
 *
 * Maintains a rolling buffer of latency samples per model name.
 * Exposes avg, p95, min, max stats for diagnosing performance bottlenecks
 * before users notice lag.
 *
 * Usage:
 *   const tracker = new InferenceLatencyTracker();
 *   const t0 = performance.now();
 *   await session.run(input);
 *   tracker.record("eegnet", performance.now() - t0);
 *   console.log(tracker.getStats());
 */

// ── Types ────────────────────────────────────────────────────────────────────

export interface ModelLatencyStats {
  /** Average latency in ms across the buffer. */
  avg: number;
  /** 95th percentile latency in ms. */
  p95: number;
  /** Minimum observed latency in ms. */
  min: number;
  /** Maximum observed latency in ms. */
  max: number;
  /** Number of samples currently in the buffer. */
  count: number;
}

export type InferenceStats = Record<string, ModelLatencyStats>;

// ── Implementation ───────────────────────────────────────────────────────────

export class InferenceLatencyTracker {
  private _buffers: Map<string, number[]> = new Map();
  private _bufferSize: number;

  /**
   * @param bufferSize Maximum number of latency samples to retain per model.
   *                   Oldest samples are evicted when the buffer is full.
   *                   Default: 50.
   */
  constructor(bufferSize: number = 50) {
    this._bufferSize = bufferSize;
  }

  /**
   * Record a latency sample for a given model.
   *
   * @param model  Model name (e.g. "eegnet", "voice", "crossAttention")
   * @param ms     Inference duration in milliseconds
   */
  record(model: string, ms: number): void {
    let buf = this._buffers.get(model);
    if (!buf) {
      buf = [];
      this._buffers.set(model, buf);
    }
    buf.push(ms);
    // Evict oldest if over capacity
    if (buf.length > this._bufferSize) {
      buf.splice(0, buf.length - this._bufferSize);
    }
  }

  /**
   * Get latency stats for a single model, or null if no samples exist.
   */
  getModelStats(model: string): ModelLatencyStats | null {
    const buf = this._buffers.get(model);
    if (!buf || buf.length === 0) return null;
    return computeStats(buf);
  }

  /**
   * Get latency stats for all tracked models.
   */
  getStats(): InferenceStats {
    const result: InferenceStats = {};
    this._buffers.forEach((buf, model) => {
      if (buf.length > 0) {
        result[model] = computeStats(buf);
      }
    });
    return result;
  }

  /** Clear all latency data for all models. */
  reset(): void {
    this._buffers.clear();
  }

  /** Clear latency data for a single model. */
  resetModel(model: string): void {
    this._buffers.delete(model);
  }
}

// ── Internal helpers ─────────────────────────────────────────────────────────

function computeStats(samples: number[]): ModelLatencyStats {
  const n = samples.length;
  let sum = 0;
  let min = Infinity;
  let max = -Infinity;

  for (let i = 0; i < n; i++) {
    const v = samples[i];
    sum += v;
    if (v < min) min = v;
    if (v > max) max = v;
  }

  // Sort a copy for p95
  const sorted = samples.slice().sort((a, b) => a - b);
  // p95: the value at the 95th percentile index
  const p95Idx = Math.ceil(n * 0.95) - 1;
  const p95 = sorted[Math.min(p95Idx, n - 1)];

  return {
    avg: sum / n,
    p95,
    min,
    max,
    count: n,
  };
}

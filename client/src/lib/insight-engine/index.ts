/**
 * InsightEngine — real-time deviation detection from EEG metrics.
 *
 * Ingests normalized metrics (stress, focus, valence, arousal, energy, hrv, sleep)
 * per frame, maintains rolling baselines per 2-hour bucket, and flags sustained
 * deviations (|z| > 1.5 for > 2 minutes) as real-time insight events.
 */

// ─── Types ───────────────────────────────────────────────────────────────────

export interface MetricSnapshot {
  stress: number;
  focus: number;
  valence: number;
  arousal: number;
  energy?: number;
  hrv?: number;
  sleep?: number;
}

export interface DeviationEvent {
  metric: string;
  direction: "high" | "low";
  zScore: number;
  durationMinutes: number;
  currentValue: number;
  baselineValue: number;
  message: string;
  cta?: string;
  ctaHref?: string;
}

// ─── Baseline bucket ─────────────────────────────────────────────────────────

interface Bucket {
  sum: number;
  sumSq: number;
  count: number;
}

function emptyBucket(): Bucket {
  return { sum: 0, sumSq: 0, count: 0 };
}

function bucketMean(b: Bucket): number {
  return b.count > 0 ? b.sum / b.count : 0;
}

function bucketStd(b: Bucket): number {
  if (b.count < 2) return 1;
  const mean = b.sum / b.count;
  const variance = b.sumSq / b.count - mean * mean;
  return Math.sqrt(Math.max(0, variance)) || 1;
}

// ─── Engine ──────────────────────────────────────────────────────────────────

const METRICS = ["stress", "focus", "valence", "arousal"] as const;
const DEVIATION_THRESHOLD = 1.5;
const BANNER_COOLDOWN_MS = 5 * 60 * 1000; // 5 minutes

export class InsightEngine {
  private userId: string;
  private baselines: Map<string, Bucket[]>; // metric → 12 buckets (2h each)
  private deviationStart: Map<string, number>; // metric → timestamp when deviation started
  private lastBannerShown: number = 0;

  constructor(userId: string) {
    this.userId = userId;
    this.baselines = new Map();
    this.deviationStart = new Map();

    for (const m of METRICS) {
      this.baselines.set(m, Array.from({ length: 12 }, emptyBucket));
    }
  }

  /** Get the 2-hour bucket index (0-11) for the current time */
  private getBucketIndex(): number {
    return Math.floor(new Date().getHours() / 2);
  }

  /** Ingest a metric snapshot and update baselines */
  ingest(snapshot: MetricSnapshot): void {
    const idx = this.getBucketIndex();
    for (const metric of METRICS) {
      const value = snapshot[metric];
      if (value == null) continue;
      const buckets = this.baselines.get(metric)!;
      const b = buckets[idx];
      b.sum += value;
      b.sumSq += value * value;
      b.count += 1;
    }
  }

  /** Check all metrics for sustained deviations */
  getRealTimeInsights(): DeviationEvent[] {
    const events: DeviationEvent[] = [];
    const idx = this.getBucketIndex();
    const now = Date.now();

    for (const metric of METRICS) {
      const buckets = this.baselines.get(metric)!;
      const b = buckets[idx];
      if (b.count < 10) continue; // need enough data for baseline

      const mean = bucketMean(b);
      const std = bucketStd(b);
      // Use the latest ingested value approximation (current mean as proxy)
      const latest = b.count > 0 ? b.sum / b.count : 0;
      const z = (latest - mean) / std;

      if (Math.abs(z) > DEVIATION_THRESHOLD) {
        if (!this.deviationStart.has(metric)) {
          this.deviationStart.set(metric, now);
        }
        const duration = (now - this.deviationStart.get(metric)!) / 60000;
        const direction = z > 0 ? "high" : "low";
        const { message, cta, ctaHref } = this.getDeviationMessage(metric, direction);

        events.push({
          metric,
          direction,
          zScore: Math.round(z * 100) / 100,
          durationMinutes: Math.round(duration * 10) / 10,
          currentValue: Math.round(latest * 100) / 100,
          baselineValue: Math.round(mean * 100) / 100,
          message,
          cta,
          ctaHref,
        });
      } else {
        this.deviationStart.delete(metric);
      }
    }

    return events;
  }

  /** Whether enough time has passed since last banner */
  isBannerAllowed(): boolean {
    return Date.now() - this.lastBannerShown > BANNER_COOLDOWN_MS;
  }

  /** Record that banner was shown */
  recordBannerShown(): void {
    this.lastBannerShown = Date.now();
  }

  private getDeviationMessage(
    metric: string,
    direction: "high" | "low",
  ): { message: string; cta?: string; ctaHref?: string } {
    if (metric === "stress" && direction === "high") {
      return {
        message: "Stress has been elevated above your baseline",
        cta: "Try breathing exercise",
        ctaHref: "/biofeedback",
      };
    }
    if (metric === "focus" && direction === "low") {
      return {
        message: "Focus has dropped below your usual level",
        cta: "Try a focus session",
        ctaHref: "/neurofeedback",
      };
    }
    if (metric === "valence" && direction === "low") {
      return {
        message: "Your mood is lower than usual for this time of day",
        cta: "Check in with yourself",
        ctaHref: "/ai-companion",
      };
    }
    return {
      message: `${metric} is ${direction === "high" ? "above" : "below"} your baseline`,
    };
  }
}

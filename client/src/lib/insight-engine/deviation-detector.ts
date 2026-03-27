import { BaselineStore, normalize, type DeviationMetric, type NormalizedReading } from "./baseline-store";

export interface DeviationEvent {
  metric: DeviationMetric;
  currentValue: number;
  baselineMean: number;
  zScore: number;
  direction: "high" | "low";
  durationMinutes: number;
  baselineQuality: number;
  relatedPattern?: {
    passType: string;
    correlationStrength: number;
    summary: string;
  };
}

interface TimerEntry {
  startedAt: string; // ISO
  zScore: number;
}

const TIMERS_KEY = "ndw_deviation_timers";
const Z_THRESHOLD = 1.5;
const RECOVERY_THRESHOLD = 1.0;

function loadTimers(): Record<string, TimerEntry> {
  try {
    return JSON.parse(localStorage.getItem(TIMERS_KEY) || "{}");
  } catch {
    return {};
  }
}

function saveTimers(timers: Record<string, TimerEntry>): void {
  try {
    localStorage.setItem(TIMERS_KEY, JSON.stringify(timers));
  } catch {}
}

export class DeviationDetector {
  constructor(private baseline: BaselineStore) {}

  detect(reading: NormalizedReading, timestamp?: string): DeviationEvent[] {
    const ts = timestamp || new Date().toISOString();
    const bucket = new Date(ts).getUTCHours();
    const timers = loadTimers();
    const events: DeviationEvent[] = [];

    const metrics: Array<[DeviationMetric, number | undefined]> = [
      ["stress",  reading.stress],
      ["focus",   reading.focus],
      ["valence", reading.valence],
      ["arousal", reading.arousal],
      ["energy",  reading.energy],
      ["hrv",     reading.hrv],
      ["sleep",   reading.sleep],
      ["steps",   reading.steps],
    ];

    for (const [metric, rawValue] of metrics) {
      if (rawValue === undefined || rawValue === null) continue;

      // Normalize for hrv/sleep/steps
      const normalized = normalize(metric, rawValue);

      const z = this.baseline.getZScore(metric, normalized, bucket);
      const absZ = Math.abs(z);

      if (absZ <= RECOVERY_THRESHOLD) {
        delete timers[metric]; // clear timer on recovery
        continue;
      }

      // Hysteresis band: 1.0 < |z| <= 1.5 — timer preserved but no event fired.
      // Duration accumulates across this band; this is intentional to avoid
      // false recoveries when signal briefly dips below the fire threshold.
      if (absZ > Z_THRESHOLD) {
        if (!timers[metric]) {
          timers[metric] = { startedAt: ts, zScore: z };
        }
        const started = new Date(timers[metric].startedAt).getTime();
        const now = new Date(ts).getTime();
        if (isNaN(started)) {
          // Corrupted timer entry — reset and treat as fresh start
          timers[metric] = { startedAt: ts, zScore: z };
        }
        const durationMinutes = Math.max(0, (new Date(ts).getTime() - new Date(timers[metric].startedAt).getTime()) / 60000);

        const cell = this.baseline.getCell(metric, bucket);
        events.push({
          metric,
          currentValue: normalized,
          baselineMean: cell?.mean ?? 0,
          zScore: z,
          direction: z > 0 ? "high" : "low",
          durationMinutes,
          baselineQuality: this.baseline.getBaselineQuality(metric, bucket),
        });
      }
    }

    saveTimers(timers);
    return events;
  }
}

export type DeviationMetric = "stress" | "focus" | "valence" | "arousal" | "hrv" | "sleep" | "steps" | "energy";

export interface BaselineCell {
  mean: number;
  std: number;
  sampleCount: number;
  lastUpdated: string;
  rawSamples: Array<{ value: number; timestamp: string }>; // kept for rolling recalc
}

type BaselineMap = Record<string, BaselineCell>;

const STORAGE_KEY = "ndw_baseline_map";
const WINDOW_DAYS = 7;
const MIN_SAMPLES = 7;

// Population defaults (all on 0-1 scale)
const POPULATION_DEFAULTS: Record<DeviationMetric, { mean: number; std: number }> = {
  stress:  { mean: 0.40, std: 0.15 },
  focus:   { mean: 0.55, std: 0.18 },
  valence: { mean: 0.55, std: 0.20 },
  arousal: { mean: 0.50, std: 0.18 },
  hrv:     { mean: 0.42, std: 0.15 },
  sleep:   { mean: 0.65, std: 0.15 },
  steps:   { mean: 0.35, std: 0.20 },
  energy:  { mean: 0.50, std: 0.18 },
};

export interface NormalizedReading {
  stress: number;
  focus: number;
  valence: number;   // already 0-1 (NormalizedReading contract)
  arousal: number;
  energy?: number;
  hrv?: number;      // raw ms
  sleep?: number;    // raw score 0-100
  steps?: number;    // raw step count
  source?: "eeg" | "health" | "voice";
  timestamp?: string;
}

function normalize(metric: DeviationMetric, rawValue: number): number {
  switch (metric) {
    case "hrv":   return Math.min(rawValue / 120, 1);
    case "sleep": return Math.min(rawValue / 100, 1);
    case "steps": return Math.min(rawValue / 15000, 1);
    default:      return rawValue; // stress, focus, valence, arousal, energy already 0-1
  }
}

function hourBucket(isoTimestamp: string): number {
  return new Date(isoTimestamp).getUTCHours();
}

function cellKey(metric: DeviationMetric, bucket: number): string {
  return `${metric}_${bucket}`;
}

function computeStats(samples: number[]): { mean: number; std: number } {
  const n = samples.length;
  if (n === 0) return { mean: 0, std: 0 };
  const mean = samples.reduce((a, b) => a + b, 0) / n;
  const variance = samples.reduce((acc, v) => acc + (v - mean) ** 2, 0) / n;
  return { mean, std: Math.sqrt(variance) };
}

function cutoff(): string {
  const d = new Date();
  d.setUTCDate(d.getUTCDate() - WINDOW_DAYS);
  return d.toISOString();
}

export class BaselineStore {
  private map: BaselineMap;

  constructor() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      this.map = raw ? JSON.parse(raw) : {};
    } catch {
      this.map = {};
    }
  }

  update(reading: NormalizedReading, timestamp: string): void {
    const ts = timestamp || new Date().toISOString();
    const bucket = hourBucket(ts);
    const cutoffTs = cutoff();

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

    for (const [metric, raw] of metrics) {
      if (raw === undefined || raw === null) continue;
      const value = normalize(metric, raw);
      const key = cellKey(metric, bucket);
      const cell: BaselineCell = this.map[key] ?? {
        mean: 0, std: 0, sampleCount: 0,
        lastUpdated: ts, rawSamples: [],
      };

      // Add sample and prune old entries
      cell.rawSamples.push({ value, timestamp: ts });
      cell.rawSamples = cell.rawSamples.filter(s => s.timestamp >= cutoffTs);

      const stats = computeStats(cell.rawSamples.map(s => s.value));
      this.map[key] = {
        ...stats,
        sampleCount: cell.rawSamples.length,
        lastUpdated: ts,
        rawSamples: cell.rawSamples,
      };
    }

    this.persist();
  }

  getCell(metric: DeviationMetric, bucket: number): BaselineCell | null {
    return this.map[cellKey(metric, bucket)] ?? null;
  }

  /**
   * @param normalizedValue - must be pre-normalized to 0-1 range. Callers
   *   are responsible for converting raw hrv/sleep/steps before calling.
   */
  getZScore(metric: DeviationMetric, normalizedValue: number, bucket: number): number {
    const cell = this.getCell(metric, bucket);
    const usable = cell && cell.sampleCount >= MIN_SAMPLES;
    const mean = usable ? cell.mean : POPULATION_DEFAULTS[metric].mean;
    const std  = usable ? cell.std  : POPULATION_DEFAULTS[metric].std;
    return (normalizedValue - mean) / Math.max(std, 0.01);
  }

  getBaselineQuality(metric: DeviationMetric, bucket: number): number {
    const cell = this.getCell(metric, bucket);
    if (!cell) return 0;
    return Math.min(cell.sampleCount / 30, 1); // 30 samples = full quality
  }

  private persist(): void {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(this.map));
    } catch { /* storage full — ignore */ }
  }
}

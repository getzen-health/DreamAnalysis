/**
 * dream-quality-score.ts
 *
 * Pure functions for computing and displaying a 0-100 Dream Quality Score.
 *
 * Formula components (all optional / nullable):
 *   Base                      = 50
 *   Sleep quality (10-90)     ± 20 pts  (centered at 50)
 *   Lucidity score (0-100)    0 to +15 pts
 *   Nightmare penalty (0-1)   0 to −25 pts
 *   Dream recall (bool)       +10 pts
 *
 * Score is clamped to [0, 100].
 */

export interface DreamQualityInput {
  /** Stored as slider(1-9) × 10, so range 10–90. Null = not recorded. */
  sleepQuality: number | null | undefined;
  /** 0-100. Null = not recorded. */
  lucidityScore: number | null | undefined;
  /** 0-1 nightmare severity from multi-pass analysis. Null = not a nightmare. */
  threatSimulationIndex: number | null | undefined;
  /** Whether the user recorded dream text (recall indicator). */
  hasDreamText: boolean;
}

export interface DailyQualityPoint {
  date: string;   // YYYY-MM-DD
  score: number;  // 0-100 daily average
  count: number;  // number of dreams that day
}

export interface DreamQualityTrend {
  current: number | null;
  avgScore: number | null;
  trend: DailyQualityPoint[];
  totalDreams: number;
}

// ── Score computation ────────────────────────────────────────────────────────

/**
 * Compute a single dream quality score in [0, 100].
 * Entries with no data at all return null (nothing to score).
 */
export function computeDreamQualityScore(entry: DreamQualityInput): number | null {
  const hasAnyData =
    entry.sleepQuality != null ||
    entry.lucidityScore != null ||
    entry.threatSimulationIndex != null ||
    entry.hasDreamText;

  if (!hasAnyData) return null;

  let score = 50;

  // Sleep quality: ±20 points centred at stored value 50 (slider mid-point 5/9)
  if (entry.sleepQuality != null && entry.sleepQuality > 0) {
    score += ((entry.sleepQuality - 50) / 40) * 20;
  }

  // Lucidity bonus: 0 – +15
  if (entry.lucidityScore != null && entry.lucidityScore >= 0) {
    score += (entry.lucidityScore / 100) * 15;
  }

  // Nightmare penalty: 0 – −25
  score -= (entry.threatSimulationIndex ?? 0) * 25;

  // Dream recall bonus: +10
  if (entry.hasDreamText) score += 10;

  return Math.round(Math.max(0, Math.min(100, score)));
}

// ── Aggregation ──────────────────────────────────────────────────────────────

/**
 * Group scored entries by calendar date and average them per day.
 * Entries whose score is null are excluded from averages.
 * Result is sorted ascending by date.
 */
export function aggregateDailyScores(
  entries: Array<{ timestampIso: string; score: number | null }>,
): DailyQualityPoint[] {
  const map: Record<string, number[]> = {};

  for (const e of entries) {
    if (e.score == null) continue;
    const date = e.timestampIso.slice(0, 10); // YYYY-MM-DD
    if (!map[date]) map[date] = [];
    map[date].push(e.score);
  }

  return Object.entries(map)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([date, scores]) => ({
      date,
      score: Math.round(scores.reduce((s, v) => s + v, 0) / scores.length),
      count: scores.length,
    }));
}

// ── Display helpers ───────────────────────────────────────────────────────────

export type QualityBand = "poor" | "fair" | "good" | "great";

export function qualityBand(score: number): QualityBand {
  if (score < 35) return "poor";
  if (score < 55) return "fair";
  if (score < 75) return "good";
  return "great";
}

export function qualityLabel(score: number): string {
  switch (qualityBand(score)) {
    case "poor":  return "Poor";
    case "fair":  return "Fair";
    case "good":  return "Good";
    case "great": return "Great";
  }
}

/** Tailwind text colour class for a quality band. */
export function qualityColorClass(score: number): string {
  switch (qualityBand(score)) {
    case "poor":  return "text-destructive";
    case "fair":  return "text-amber-400";
    case "good":  return "text-cyan-400";
    case "great": return "text-emerald-400";
  }
}

/** Tailwind bg colour class (subtle tint) for a quality band. */
export function qualityBgClass(score: number): string {
  switch (qualityBand(score)) {
    case "poor":  return "bg-destructive/10";
    case "fair":  return "bg-amber-500/10";
    case "good":  return "bg-cyan-500/10";
    case "great": return "bg-emerald-500/10";
  }
}

/**
 * Compute average score from a list of nullable scores.
 * Returns null if no valid scores exist.
 */
export function averageScore(scores: Array<number | null>): number | null {
  const valid = scores.filter((s): s is number => s != null);
  if (valid.length === 0) return null;
  return Math.round(valid.reduce((s, v) => s + v, 0) / valid.length);
}

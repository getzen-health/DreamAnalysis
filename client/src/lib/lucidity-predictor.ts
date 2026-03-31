/**
 * lucidity-predictor.ts
 * Pure helpers for estimating tonight's lucid dream likelihood.
 *
 * Inputs: historical dream rows (lucidityScore, timestamp) + presleep
 * intention history.  No network calls.
 *
 * Scoring model (evidence-based weightings):
 *   40 % — recent avg lucidity   (direct past-performance signal)
 *   25 % — 7-day dream recall rate (awareness / recall correlates with lucidity)
 *   20 % — recall streak momentum  (consistent recording habit → more lucidity)
 *   15 % — presleep intention set  (MILD-style intention = +5-10% lucid probability)
 *
 * References:
 *   - Stumbrys et al. 2012 (MILD effectiveness: 21.9% lucid nights when practised)
 *   - Schredl & Erlacher 2011 (dream recall frequency → lucid frequency, r = 0.37)
 *   - LaBerge 1990 (reality testing + intention setting)
 */

// ── types ─────────────────────────────────────────────────────────────────────

export interface DreamForLucidity {
  lucidityScore: number | null;  // 0-100
  timestamp: string;             // ISO
}

export interface IntentionForLucidity {
  date: string;  // "YYYY-MM-DD"
  text: string;
}

export type LucidityBand = "low" | "moderate" | "high" | "optimal";

export interface LucidityFactor {
  label: string;
  value: number;   // 0-100 (already scaled for display)
  weight: number;  // fractional weight used in score (0-1, informational)
  tip: string;     // how to improve this factor
}

export interface LucidityPrediction {
  /** Overall score 0-100 */
  likelihood: number;
  band: LucidityBand;
  factors: LucidityFactor[];
  /** Primary recommendation for tonight */
  recommendation: string;
  /** One-line human-readable summary */
  summary: string;
}

// ── internal helpers ──────────────────────────────────────────────────────────

/** ISO timestamp → "YYYY-MM-DD" (UTC-safe) */
function toDateStr(ts: string): string {
  return ts.slice(0, 10);
}

/** UTC today as "YYYY-MM-DD" */
function utcTodayStr(): string {
  return new Date().toISOString().slice(0, 10);
}

/** 7-day recall rate: fraction of days in last 7 with ≥ 1 dream. */
function recentRecallRate(dreams: DreamForLucidity[]): number {
  const today = utcTodayStr();
  const todayMs = new Date(today + "T00:00:00Z").getTime();
  const sevenAgo = todayMs - 6 * 24 * 60 * 60 * 1000;

  const daySet = new Set<string>();
  for (const d of dreams) {
    const ms = new Date(toDateStr(d.timestamp) + "T00:00:00Z").getTime();
    if (ms >= sevenAgo && ms <= todayMs) daySet.add(toDateStr(d.timestamp));
  }
  return daySet.size / 7;
}

/** Consecutive-day recall streak ending today or yesterday. */
function recallStreak(dreams: DreamForLucidity[]): number {
  const days = new Set(dreams.map((d) => toDateStr(d.timestamp)));
  const todayMs = new Date(utcTodayStr() + "T00:00:00Z").getTime();

  let streak = 0;
  // Start from today; allow "yesterday" start too
  let cursor = days.has(utcTodayStr()) ? todayMs : todayMs - 24 * 60 * 60 * 1000;
  const cursorDate = new Date(cursor).toISOString().slice(0, 10);
  if (!days.has(cursorDate)) return 0;

  let current = cursor;
  while (true) {
    const dateStr = new Date(current).toISOString().slice(0, 10);
    if (!days.has(dateStr)) break;
    streak++;
    current -= 24 * 60 * 60 * 1000;
  }
  return streak;
}

/** Average lucidityScore over the last N days (null entries excluded). */
function avgLucidity(dreams: DreamForLucidity[], days: number): number {
  const todayMs = new Date(utcTodayStr() + "T00:00:00Z").getTime();
  const cutoff  = todayMs - (days - 1) * 24 * 60 * 60 * 1000;

  const scores = dreams
    .filter((d) => {
      const ms = new Date(toDateStr(d.timestamp) + "T00:00:00Z").getTime();
      return ms >= cutoff && ms <= todayMs && d.lucidityScore != null;
    })
    .map((d) => d.lucidityScore as number);

  return scores.length > 0 ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;
}

// ── band classification ───────────────────────────────────────────────────────

export function lucidityBand(score: number): LucidityBand {
  if (score >= 75) return "optimal";
  if (score >= 50) return "high";
  if (score >= 25) return "moderate";
  return "low";
}

export const BAND_LABEL: Record<LucidityBand, string> = {
  optimal:  "Optimal",
  high:     "High",
  moderate: "Moderate",
  low:      "Low",
};

export const BAND_COLOR: Record<LucidityBand, string> = {
  optimal:  "text-emerald-400",
  high:     "text-sky-400",
  moderate: "text-amber-400",
  low:      "text-muted-foreground",
};

export const BAND_BG: Record<LucidityBand, string> = {
  optimal:  "bg-emerald-500/10 border-emerald-500/20",
  high:     "bg-sky-500/10 border-sky-500/20",
  moderate: "bg-amber-500/10 border-amber-500/20",
  low:      "bg-white/5 border-white/10",
};

// ── recommendations ───────────────────────────────────────────────────────────

const TECHNIQUE_BY_BAND: Record<LucidityBand, string> = {
  optimal:  "WILD (Wake-Initiated): lie still after waking at 6 am and re-enter sleep consciously.",
  high:     "MILD: before sleep, visualise becoming lucid in a recent dream and affirm 'I will know I'm dreaming.'",
  moderate: "Reality testing: check your hands 10× today; before sleep recall your last dream clearly.",
  low:      "Build recall first: keep a journal bedside and write ≥ 3 dream fragments each morning.",
};

const SUMMARY_BY_BAND: Record<LucidityBand, string> = {
  optimal:  "Conditions are excellent — your recall and awareness are primed for lucidity tonight.",
  high:     "Good momentum: a clear intention tonight could tip you into lucidity.",
  moderate: "Potential is building — focus on recall consistency and reality checks.",
  low:      "Start with dream recall; lucidity follows naturally once recall is strong.",
};

// ── main export ───────────────────────────────────────────────────────────────

/**
 * Compute a 0-100 lucidity likelihood for tonight based on dream history
 * and presleep intention history.
 */
export function computeLucidityPrediction(
  dreams: DreamForLucidity[],
  intentions: IntentionForLucidity[],
): LucidityPrediction {
  // --- individual factor values (0-100) ---
  const lucidAvg    = avgLucidity(dreams, 7);          // 0-100
  const recallRate  = recentRecallRate(dreams) * 100;  // 0-100
  const streakScore = Math.min(recallStreak(dreams), 14) / 14 * 100; // cap at 14 days
  const todayDate   = utcTodayStr();
  const intentionSet = intentions.some((i) => i.date === todayDate) ? 100 : 0;

  // --- weighted composite (weights sum to 1.0) ---
  const W_LUCID   = 0.40;
  const W_RECALL  = 0.25;
  const W_STREAK  = 0.20;
  const W_INTENT  = 0.15;

  const likelihood = Math.round(
    W_LUCID  * lucidAvg   +
    W_RECALL * recallRate +
    W_STREAK * streakScore +
    W_INTENT * intentionSet,
  );

  const band = lucidityBand(likelihood);

  const factors: LucidityFactor[] = [
    {
      label:  "Recent lucidity (7d avg)",
      value:  Math.round(lucidAvg),
      weight: W_LUCID,
      tip:    "Increases with MILD practice and longer REM windows.",
    },
    {
      label:  "Dream recall rate (7d)",
      value:  Math.round(recallRate),
      weight: W_RECALL,
      tip:    "Record at least 3 dream fragments each morning to strengthen recall.",
    },
    {
      label:  "Recall streak",
      value:  Math.round(streakScore),
      weight: W_STREAK,
      tip:    "Record dreams every day to build streak momentum.",
    },
    {
      label:  "Presleep intention set",
      value:  intentionSet,
      weight: W_INTENT,
      tip:    "Write tonight's lucid-dream intention in the Presleep Intention card.",
    },
  ];

  return {
    likelihood,
    band,
    factors,
    recommendation: TECHNIQUE_BY_BAND[band],
    summary:        SUMMARY_BY_BAND[band],
  };
}

/**
 * 7-day trend of lucidity scores — oldest first, one point per day that
 * has at least one dream.  Used for the sparkline.
 */
export interface LucidityTrendPoint {
  date: string;   // "YYYY-MM-DD"
  avgScore: number; // 0-100
}

export function lucidityTrend(
  dreams: DreamForLucidity[],
  days = 14,
): LucidityTrendPoint[] {
  const todayMs = new Date(utcTodayStr() + "T00:00:00Z").getTime();
  const cutoff  = todayMs - (days - 1) * 24 * 60 * 60 * 1000;

  const byDay = new Map<string, number[]>();
  for (const d of dreams) {
    if (d.lucidityScore == null) continue;
    const ms = new Date(toDateStr(d.timestamp) + "T00:00:00Z").getTime();
    if (ms < cutoff || ms > todayMs) continue;
    const key = toDateStr(d.timestamp);
    if (!byDay.has(key)) byDay.set(key, []);
    byDay.get(key)!.push(d.lucidityScore);
  }

  return Array.from(byDay.entries())
    .map(([date, scores]) => ({
      date,
      avgScore: Math.round(scores.reduce((a, b) => a + b, 0) / scores.length),
    }))
    .sort((a, b) => a.date.localeCompare(b.date));
}

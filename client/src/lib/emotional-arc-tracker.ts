/**
 * emotional-arc-tracker.ts
 * Pure helpers for scoring and aggregating the "emotionalArc" narrative field
 * stored per dream by the multi-pass LLM (Pass-2).  No network calls.
 *
 * Arc valence: -1 (highly distressing) → 0 (neutral/mixed) → +1 (uplifting)
 */

// ── keyword vocabulary ────────────────────────────────────────────────────────

// Stemmed/partial terms so "resolving", "resolved", "resolution" all match "resolv"
const POSITIVE_STEMS = [
  "resolv", "peaceful", "tranquil", "wonder", "joy", "joyful", "curious",
  "hopeful", "hope", "calm", "content", "inspir", "triumph", "uplift",
  "clarif", "free", "liberat", "heal", "reconnect", "discover", "accept",
  "awaken", "transform", "grow", "expand", "relief", "satisfy", "delight",
];

const NEGATIVE_STEMS = [
  "fear", "terror", "terrif", "anxious", "anxiety", "escalat", "trap",
  "chas", "helpless", "dark", "panic", "desperat", "dread", "threat",
  "loom", "running", "stuck", "overwhelm", "collaps", "spiral", "bleak",
  "confus", "doom", "distress", "anguish", "sorrow", "grief", "rage",
  "frustrat", "disappoint", "isolat", "alon",
];

function countMatches(lower: string, stems: string[]): number {
  return stems.reduce((n, s) => n + (lower.includes(s) ? 1 : 0), 0);
}

// ── core scoring ──────────────────────────────────────────────────────────────

/**
 * Score a single `emotionalArc` string.
 * Returns a number in [-1, 1]:
 *   positive → uplifting arc (resolving, peaceful…)
 *   negative → distressing arc (escalating fear, trapped…)
 *   near 0   → neutral or mixed
 * Returns null when arcText is absent or blank.
 */
export function scoreArcValence(arcText: string | null | undefined): number | null {
  if (!arcText || arcText.trim().length === 0) return null;
  const lower = arcText.toLowerCase();
  const pos = countMatches(lower, POSITIVE_STEMS);
  const neg = countMatches(lower, NEGATIVE_STEMS);
  if (pos === 0 && neg === 0) return 0; // no keywords — neutral
  return (pos - neg) / (pos + neg);    // already in [-1, 1]
}

// ── label helpers ─────────────────────────────────────────────────────────────

export type ArcLabel = "distressing" | "tense" | "neutral" | "mixed" | "uplifting";

/**
 * Map valence number to a human-readable label.
 */
export function arcLabel(valence: number): ArcLabel {
  if (valence <= -0.4) return "distressing";
  if (valence <= -0.1) return "tense";
  if (valence <= 0.1)  return "mixed";
  if (valence <= 0.4)  return "neutral";
  return "uplifting";
}

export const ARC_LABEL_COLOR: Record<ArcLabel, string> = {
  distressing: "text-red-400",
  tense:       "text-orange-400",
  mixed:       "text-amber-400",
  neutral:     "text-muted-foreground",
  uplifting:   "text-emerald-400",
};

export const ARC_LABEL_BG: Record<ArcLabel, string> = {
  distressing: "bg-red-500/15",
  tense:       "bg-orange-500/15",
  mixed:       "bg-amber-500/15",
  neutral:     "bg-white/10",
  uplifting:   "bg-emerald-500/15",
};

// ── trend aggregation ─────────────────────────────────────────────────────────

export interface DreamForArc {
  emotionalArc: string | null;
  timestamp: string; // ISO date string
}

export interface ArcTrendPoint {
  date: string;        // "2026-03-28"
  avgValence: number;  // mean valence for that day, null days omitted
  dreamCount: number;  // dreams with an arc that day
}

/**
 * Group dreams by calendar date (UTC) and compute the mean arc valence.
 * Dreams without an emotionalArc are excluded from the average but do not
 * produce a data point at all (keeps chart clean).
 * Returns points sorted oldest-first.
 */
export function aggregateArcTrend(dreams: DreamForArc[]): ArcTrendPoint[] {
  const byDate = new Map<string, number[]>();

  for (const d of dreams) {
    const v = scoreArcValence(d.emotionalArc);
    if (v === null) continue;
    const day = d.timestamp.slice(0, 10); // "YYYY-MM-DD"
    const arr = byDate.get(day) ?? [];
    arr.push(v);
    byDate.set(day, arr);
  }

  return Array.from(byDate.entries())
    .map(([date, vals]) => ({
      date,
      avgValence: vals.reduce((s, v) => s + v, 0) / vals.length,
      dreamCount: vals.length,
    }))
    .sort((a, b) => a.date.localeCompare(b.date));
}

// ── pattern extraction ────────────────────────────────────────────────────────

/**
 * Extract a short, normalised summary phrase from an arc description.
 * Strips filler words, trims to ≤ 5 words, lowercases.
 *
 * Examples:
 *   "starts anxious then resolving"  → "anxious then resolving"
 *   "escalating fear throughout"     → "escalating fear"
 *   "wonder and peaceful"            → "wonder and peaceful"
 */
export function extractArcPattern(arcText: string): string {
  const FILLER = /^(starts?\s+|begins?\s+|feels?\s+|sense\s+of\s+)/i;
  const trimmed = arcText.replace(FILLER, "").trim().toLowerCase();
  const words = trimmed.split(/\s+/).slice(0, 5);
  return words.join(" ");
}

/**
 * Return the top-N most-frequent arc patterns across a list of dreams.
 * Patterns are normalised via `extractArcPattern`.
 */
export function topArcPatterns(
  dreams: DreamForArc[],
  topN = 5,
): { pattern: string; count: number }[] {
  const freq = new Map<string, number>();

  for (const d of dreams) {
    if (!d.emotionalArc || d.emotionalArc.trim().length === 0) continue;
    const p = extractArcPattern(d.emotionalArc);
    if (p.length === 0) continue;
    freq.set(p, (freq.get(p) ?? 0) + 1);
  }

  return Array.from(freq.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, topN)
    .map(([pattern, count]) => ({ pattern, count }));
}

// ── overall summary ───────────────────────────────────────────────────────────

export interface ArcSummary {
  /** Mean valence across all dreams with an arc, or null if none */
  meanValence: number | null;
  /** Fraction of dreams that are net-positive arc (valence > 0.1) */
  positiveRate: number;
  /** Fraction of dreams that are net-negative arc (valence < -0.1) */
  negativeRate: number;
  /** Total dreams that had an emotionalArc field */
  arcCount: number;
}

export function computeArcSummary(dreams: DreamForArc[]): ArcSummary {
  const scored = dreams
    .map((d) => scoreArcValence(d.emotionalArc))
    .filter((v): v is number => v !== null);

  if (scored.length === 0) {
    return { meanValence: null, positiveRate: 0, negativeRate: 0, arcCount: 0 };
  }

  const mean = scored.reduce((s, v) => s + v, 0) / scored.length;
  const pos  = scored.filter((v) => v > 0.1).length / scored.length;
  const neg  = scored.filter((v) => v < -0.1).length / scored.length;

  return {
    meanValence:  mean,
    positiveRate: pos,
    negativeRate: neg,
    arcCount:     scored.length,
  };
}

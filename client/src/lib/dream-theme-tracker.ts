/**
 * Dream Theme Tracker — longitudinal analysis of dream journal entries.
 *
 * Pure functions: takes an array of dream entries + a period in days,
 * returns recurring themes, emotion distribution, and trend analysis.
 * No side effects, no API calls.
 *
 * Issue #549: 7/30/90 day pattern tracking.
 */

// ── Interfaces ───────────────────────────────────────────────────────────────

export interface DreamTheme {
  theme: string;
  count: number;
  firstSeen: string;
  lastSeen: string;
  trend: "increasing" | "stable" | "decreasing";
  associatedEmotions: string[];
}

export interface DreamPatternSummary {
  topThemes: DreamTheme[];
  emotionDistribution: Record<string, number>;
  lucidDreamCount: number;
  totalDreams: number;
  periodDays: number;
}

export interface DreamEntry {
  dreamText: string;
  emotions: string[];
  symbols: string[];
  lucidityScore?: number;
  timestamp: string;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Filter entries to those within the given period (days from now). */
function filterByPeriod(dreams: DreamEntry[], periodDays: number): DreamEntry[] {
  const cutoff = Date.now() - periodDays * 86_400_000;
  return dreams.filter((d) => new Date(d.timestamp).getTime() >= cutoff);
}

/** Determine trend by comparing first-half frequency vs second-half frequency. */
function computeTrend(
  timestamps: string[],
  periodDays: number,
): "increasing" | "stable" | "decreasing" {
  if (timestamps.length < 2) return "stable";

  const now = Date.now();
  const midpoint = now - (periodDays * 86_400_000) / 2;

  let firstHalf = 0;
  let secondHalf = 0;
  for (const ts of timestamps) {
    if (new Date(ts).getTime() < midpoint) {
      firstHalf++;
    } else {
      secondHalf++;
    }
  }

  if (secondHalf > firstHalf) return "increasing";
  if (secondHalf < firstHalf) return "decreasing";
  return "stable";
}

/** Find the top N emotions co-occurring with a given theme across entries. */
function findAssociatedEmotions(
  theme: string,
  dreams: DreamEntry[],
  maxEmotions: number = 3,
): string[] {
  const emotionCounts: Record<string, number> = {};

  for (const dream of dreams) {
    const symbols = dream.symbols.map((s) => s.toLowerCase());
    if (!symbols.includes(theme.toLowerCase())) continue;

    for (const emotion of dream.emotions) {
      const normalized = emotion.toLowerCase();
      emotionCounts[normalized] = (emotionCounts[normalized] ?? 0) + 1;
    }
  }

  return Object.entries(emotionCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, maxEmotions)
    .map(([emotion]) => emotion);
}

// ── Main Analysis ────────────────────────────────────────────────────────────

export function analyzeDreamPatterns(
  dreams: DreamEntry[],
  periodDays: number,
): DreamPatternSummary {
  const filtered = filterByPeriod(dreams, periodDays);

  if (filtered.length === 0) {
    return {
      topThemes: [],
      emotionDistribution: {},
      lucidDreamCount: 0,
      totalDreams: 0,
      periodDays,
    };
  }

  // Count symbol occurrences and collect timestamps per symbol
  const symbolCounts: Record<string, number> = {};
  const symbolTimestamps: Record<string, string[]> = {};
  const symbolFirstSeen: Record<string, string> = {};
  const symbolLastSeen: Record<string, string> = {};

  for (const dream of filtered) {
    for (const rawSymbol of dream.symbols) {
      const symbol = rawSymbol.toLowerCase();
      symbolCounts[symbol] = (symbolCounts[symbol] ?? 0) + 1;

      if (!symbolTimestamps[symbol]) symbolTimestamps[symbol] = [];
      symbolTimestamps[symbol].push(dream.timestamp);

      const ts = dream.timestamp;
      if (!symbolFirstSeen[symbol] || ts < symbolFirstSeen[symbol]) {
        symbolFirstSeen[symbol] = ts;
      }
      if (!symbolLastSeen[symbol] || ts > symbolLastSeen[symbol]) {
        symbolLastSeen[symbol] = ts;
      }
    }
  }

  // Build top 5 themes sorted by count descending
  const topThemes: DreamTheme[] = Object.entries(symbolCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([theme, count]) => ({
      theme,
      count,
      firstSeen: symbolFirstSeen[theme],
      lastSeen: symbolLastSeen[theme],
      trend: computeTrend(symbolTimestamps[theme], periodDays),
      associatedEmotions: findAssociatedEmotions(theme, filtered),
    }));

  // Emotion distribution across all dreams in the period
  const emotionDistribution: Record<string, number> = {};
  for (const dream of filtered) {
    for (const emotion of dream.emotions) {
      const normalized = emotion.toLowerCase();
      emotionDistribution[normalized] =
        (emotionDistribution[normalized] ?? 0) + 1;
    }
  }

  // Count lucid dreams (lucidityScore > 0.5 threshold)
  const lucidDreamCount = filtered.filter(
    (d) => d.lucidityScore != null && d.lucidityScore > 0.5,
  ).length;

  return {
    topThemes,
    emotionDistribution,
    lucidDreamCount,
    totalDreams: filtered.length,
    periodDays,
  };
}

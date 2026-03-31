/**
 * dream-symbol-context.ts
 * Pure helpers for analysing what emotional/thematic context surrounds each
 * recurring symbol across a user's dream history.
 *
 * No network calls. Operates on the DreamEntry shape already fetched by
 * other components (shared TanStack Query cache key "dream-analysis").
 */

export interface DreamEntryForSymbol {
  id: string;
  symbols: string[] | null;
  themes: string[] | null;
  emotionalArc: string | null;
  threatSimulationIndex: number | null;  // 0-1
  lucidityScore: number | null;          // 0-100
  timestamp: string;
}

export interface SymbolContext {
  symbol: string;
  /** Number of dreams containing this symbol */
  count: number;
  /** Mean threat simulation index across appearances (0-1) */
  avgTsi: number;
  /** Mean lucidity score across appearances (0-100) */
  avgLucidity: number;
  /** Top 3 co-occurring themes, most frequent first */
  topThemes: string[];
  /** Most common arc phrases when this symbol appears (≤ 3) */
  topArcs: string[];
  /** ISO timestamp of most recent appearance */
  lastSeen: string;
  /** Dream IDs that contain this symbol (for drill-down) */
  dreamIds: string[];
}

export type SymbolMood = "uplifting" | "dark" | "neutral" | "mixed";

// ── helpers ───────────────────────────────────────────────────────────────────

function normSymbol(s: string): string {
  return s.toLowerCase().trim();
}

/** Trim an arc description to a short phrase (≤ 5 words). */
function arcPhrase(arc: string): string {
  return arc.toLowerCase().trim().split(/\s+/).slice(0, 5).join(" ");
}

function topN<T>(freqMap: Map<T, number>, n: number): T[] {
  return Array.from(freqMap.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, n)
    .map(([k]) => k);
}

// ── core builder ──────────────────────────────────────────────────────────────

/**
 * Build a context map for every symbol that appears in at least one dream.
 * Symbols are normalised to lowercase.
 */
export function buildSymbolContextMap(
  dreams: DreamEntryForSymbol[],
): Map<string, SymbolContext> {
  const map = new Map<string, {
    count: number;
    tsiSum: number;
    tsiCount: number;
    lucidSum: number;
    lucidCount: number;
    themeFreq: Map<string, number>;
    arcFreq: Map<string, number>;
    lastSeen: string;
    dreamIds: string[];
  }>();

  for (const dream of dreams) {
    const syms = (dream.symbols ?? []).map(normSymbol).filter(Boolean);
    if (syms.length === 0) continue;

    for (const sym of syms) {
      if (!map.has(sym)) {
        map.set(sym, {
          count: 0,
          tsiSum: 0, tsiCount: 0,
          lucidSum: 0, lucidCount: 0,
          themeFreq: new Map(),
          arcFreq: new Map(),
          lastSeen: dream.timestamp,
          dreamIds: [],
        });
      }

      const ctx = map.get(sym)!;
      ctx.count++;
      ctx.dreamIds.push(dream.id);

      if (dream.timestamp > ctx.lastSeen) ctx.lastSeen = dream.timestamp;

      if (dream.threatSimulationIndex != null) {
        ctx.tsiSum += dream.threatSimulationIndex;
        ctx.tsiCount++;
      }
      if (dream.lucidityScore != null) {
        ctx.lucidSum += dream.lucidityScore;
        ctx.lucidCount++;
      }
      for (const t of dream.themes ?? []) {
        ctx.themeFreq.set(t, (ctx.themeFreq.get(t) ?? 0) + 1);
      }
      if (dream.emotionalArc?.trim()) {
        const phrase = arcPhrase(dream.emotionalArc);
        ctx.arcFreq.set(phrase, (ctx.arcFreq.get(phrase) ?? 0) + 1);
      }
    }
  }

  // Convert to SymbolContext
  const result = new Map<string, SymbolContext>();
  for (const [sym, raw] of map.entries()) {
    result.set(sym, {
      symbol: sym,
      count: raw.count,
      avgTsi: raw.tsiCount > 0 ? raw.tsiSum / raw.tsiCount : 0,
      avgLucidity: raw.lucidCount > 0 ? raw.lucidSum / raw.lucidCount : 0,
      topThemes: topN(raw.themeFreq, 3),
      topArcs: topN(raw.arcFreq, 3),
      lastSeen: raw.lastSeen,
      dreamIds: raw.dreamIds,
    });
  }

  return result;
}

// ── mood classification ───────────────────────────────────────────────────────

/**
 * Classify the overall "mood" of a symbol based on its average TSI and lucidity.
 *
 * - avgTsi > 0.5  → tends dark (nightmare-associated)
 * - avgLucidity > 60 → tends uplifting (associated with lucid/aware dreams)
 * - Both elevated → mixed
 * - Neither → neutral
 */
export function symbolMood(ctx: SymbolContext): SymbolMood {
  const dark    = ctx.avgTsi > 0.5;
  const uplift  = ctx.avgLucidity > 60;

  if (dark && uplift) return "mixed";
  if (dark)           return "dark";
  if (uplift)         return "uplifting";
  return "neutral";
}

export const MOOD_COLOR: Record<SymbolMood, string> = {
  uplifting: "text-emerald-400",
  dark:      "text-red-400",
  neutral:   "text-muted-foreground",
  mixed:     "text-amber-400",
};

export const MOOD_BG: Record<SymbolMood, string> = {
  uplifting: "bg-emerald-500/10 border-emerald-500/20",
  dark:      "bg-red-500/10 border-red-500/20",
  neutral:   "bg-white/5 border-white/10",
  mixed:     "bg-amber-500/10 border-amber-500/20",
};

export const MOOD_LABEL: Record<SymbolMood, string> = {
  uplifting: "uplifting",
  dark:      "dark",
  neutral:   "neutral",
  mixed:     "mixed",
};

// ── sorting & filtering ───────────────────────────────────────────────────────

export type SymbolSortKey = "frequency" | "recent" | "darkest" | "brightest";

/**
 * Sort symbol contexts and return as an array (most notable first).
 */
export function sortedSymbols(
  map: Map<string, SymbolContext>,
  sortBy: SymbolSortKey = "frequency",
  limit = 8,
): SymbolContext[] {
  const arr = Array.from(map.values());

  switch (sortBy) {
    case "frequency":
      arr.sort((a, b) => b.count - a.count);
      break;
    case "recent":
      arr.sort((a, b) => b.lastSeen.localeCompare(a.lastSeen));
      break;
    case "darkest":
      arr.sort((a, b) => b.avgTsi - a.avgTsi);
      break;
    case "brightest":
      arr.sort((a, b) => b.avgLucidity - a.avgLucidity);
      break;
  }

  return arr.slice(0, limit);
}

// ── summary insight ───────────────────────────────────────────────────────────

export interface SymbolInsight {
  /** e.g. "water" */
  symbol: string;
  /** Human-readable summary, e.g. "Appears in 5 dreams, usually in dark contexts." */
  summary: string;
}

/**
 * Generate a one-line human-readable summary for a symbol context.
 */
export function symbolSummary(ctx: SymbolContext): string {
  const moodWord = MOOD_LABEL[symbolMood(ctx)];
  const themeStr = ctx.topThemes.slice(0, 2).join(", ");
  const freq = ctx.count === 1 ? "1 dream" : `${ctx.count} dreams`;
  const themeClause = themeStr ? `, often with ${themeStr}` : "";
  return `Appears in ${freq}, usually ${moodWord}${themeClause}.`;
}

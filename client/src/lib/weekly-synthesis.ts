/**
 * weekly-synthesis.ts
 *
 * Pure functions for building the weekly dream synthesis prompt context.
 * No network calls — all functions operate on already-fetched dream data.
 *
 * Used by the WeeklySynthesisCard component and testable in isolation.
 */

export interface SynthesisDream {
  /** ISO timestamp string */
  date: string;
  /** Hall/Van de Castle themes e.g. ["threat-simulation","self-exploration"] */
  themes?: string[] | null;
  /** Free-text emotional arc from pass-2 analysis */
  emotionalArc?: string | null;
  /** Specific insight from pass-3 */
  keyInsight?: string | null;
  /** 0-1 nightmare severity */
  threatSimulationIndex?: number | null;
  /** Symbol names */
  symbols?: string[] | null;
}

export interface WeeklySynthesisResponse {
  synthesis: string;
  topThemes: string[];
  dreamCount: number;
  nightmareCount: number;
  generatedAt: string;
}

// ── Formatting helpers ────────────────────────────────────────────────────────

/** Format a single dream as a compact context line for the LLM prompt. */
export function formatDreamRow(dream: SynthesisDream, index: number): string {
  const parts: string[] = [`Dream ${index + 1} (${dream.date.slice(0, 10)})`];

  const themes = dream.themes?.filter(Boolean) ?? [];
  if (themes.length > 0) parts.push(`themes: ${themes.join(", ")}`);

  if (dream.emotionalArc?.trim()) parts.push(`arc: ${dream.emotionalArc.trim()}`);

  if (dream.keyInsight?.trim()) parts.push(`insight: "${dream.keyInsight.trim()}"`);

  const symbols = dream.symbols?.filter(Boolean) ?? [];
  if (symbols.length > 0) parts.push(`symbols: ${symbols.slice(0, 3).join(", ")}`);

  const isNightmare = (dream.threatSimulationIndex ?? 0) > 0.5;
  if (isNightmare) parts.push("(nightmare)");

  return parts.join(" | ");
}

/**
 * Build the full structured context string to inject into the synthesis prompt.
 * Uses only metadata — not raw dream text — to preserve privacy and save tokens.
 */
export function buildSynthesisContext(dreams: SynthesisDream[]): string {
  if (dreams.length === 0) return "No dream data available for this week.";

  const lines = dreams.map((d, i) => formatDreamRow(d, i));
  return lines.join("\n");
}

// ── Aggregation helpers ───────────────────────────────────────────────────────

/**
 * Aggregate and deduplicate themes across all dreams.
 * Returns the top N most-frequent themes in frequency order.
 */
export function extractWeekTopThemes(
  dreams: SynthesisDream[],
  topN = 5,
): string[] {
  const freq: Record<string, number> = {};
  for (const d of dreams) {
    for (const t of d.themes ?? []) {
      if (!t) continue;
      freq[t] = (freq[t] ?? 0) + 1;
    }
  }
  return Object.entries(freq)
    .sort(([, a], [, b]) => b - a)
    .slice(0, topN)
    .map(([theme]) => theme);
}

/** Count dreams classified as nightmares (threatSimulationIndex > 0.5). */
export function countWeekNightmares(dreams: SynthesisDream[]): number {
  return dreams.filter((d) => (d.threatSimulationIndex ?? 0) > 0.5).length;
}

/** Collect all unique symbols seen across the week, deduped. */
export function extractWeekSymbols(dreams: SynthesisDream[]): string[] {
  const seen = new Set<string>();
  for (const d of dreams) {
    for (const s of d.symbols ?? []) {
      if (s) seen.add(s.toLowerCase());
    }
  }
  return Array.from(seen);
}

/**
 * Build the full system + user prompt pair for the weekly synthesis LLM call.
 * Exported so the server can consume it without duplicating logic.
 */
export function buildWeeklySynthesisPrompt(
  dreams: SynthesisDream[],
): { system: string; user: string } {
  const nightmares = countWeekNightmares(dreams);
  const themes = extractWeekTopThemes(dreams);
  const context = buildSynthesisContext(dreams);

  const system =
    "You are a compassionate dream therapist and researcher. Synthesize a week of dream metadata into a warm, insightful 2-3 paragraph report. Focus on psychological patterns, emotional growth, and practical insights. Do not invent content not present in the data.";

  const user =
    `Here is this week's dream data (${dreams.length} dream${dreams.length !== 1 ? "s" : ""}, ${nightmares} nightmare${nightmares !== 1 ? "s" : ""}):\n\n` +
    context +
    (themes.length > 0
      ? `\n\nTop recurring themes: ${themes.join(", ")}.`
      : "") +
    '\n\nWrite a concise synthesis (2-3 paragraphs) that: (1) identifies the dominant emotional or psychological thread across these dreams, (2) highlights any patterns worth the dreamer\'s attention, and (3) offers one actionable suggestion for the week ahead. Return ONLY valid JSON: {"synthesis":"<paragraphs>","topThemes":["<theme1>","<theme2>","<theme3>"]}';

  return { system, user };
}

/**
 * dream-history.ts — pure helpers for browsing enriched dream journal entries.
 * No network calls; all functions operate on plain data from the API.
 */

export interface DreamEntry {
  id: string;
  userId: string | null;
  dreamText: string;
  symbols: string[] | null;
  aiAnalysis: string | null;
  lucidityScore: number | null;   // 0-100 integer
  sleepQuality: number | null;    // 0-100 integer (stored ×10 from 0-10 input)
  tags: string[] | null;
  sleepDuration: number | null;   // hours
  themes: string[] | null;
  emotionalArc: string | null;
  keyInsight: string | null;
  threatSimulationIndex: number | null; // 0-1
  timestamp: string;              // ISO date string
}

export type DreamFilter = "all" | "nightmares" | "lucid" | "with-insight";

// ── nightmare / lucid helpers ────────────────────────────────────────────────

export function isNightmare(entry: DreamEntry): boolean {
  return (entry.threatSimulationIndex ?? 0) > 0.5;
}

export function isLucid(entry: DreamEntry): boolean {
  return (entry.lucidityScore ?? 0) >= 70;
}

export function hasInsight(entry: DreamEntry): boolean {
  return typeof entry.keyInsight === "string" && entry.keyInsight.trim().length > 0;
}

// ── quality band (mirrors dream-quality-score.ts thresholds) ─────────────────

export type QualityBand = "poor" | "fair" | "good" | "great";

export function entryQualityBand(entry: DreamEntry): QualityBand | null {
  const sq = entry.sleepQuality ?? null;
  const tsi = entry.threatSimulationIndex ?? 0;
  const lucid = entry.lucidityScore ?? 0;
  const hasText = entry.dreamText.trim().length > 0;

  if (sq === null && tsi === 0 && lucid === 0) return null;

  // base 50 + sleepQuality±20 (sq is 0-100, centered at 50) + lucidity*0.15 − tsi*25 + recall+10
  let score = 50;
  if (sq !== null) score += ((sq - 50) / 50) * 20;
  score += lucid * 0.15;
  score -= tsi * 25;
  if (hasText) score += 10;
  score = Math.max(0, Math.min(100, score));

  if (score >= 75) return "great";
  if (score >= 55) return "good";
  if (score >= 35) return "fair";
  return "poor";
}

const BAND_COLOR: Record<QualityBand, string> = {
  poor:  "text-red-400",
  fair:  "text-amber-400",
  good:  "text-emerald-400",
  great: "text-cyan-400",
};

export function bandColorClass(band: QualityBand): string {
  return BAND_COLOR[band];
}

// ── date formatting ──────────────────────────────────────────────────────────

/** "Mon, Mar 28" */
export function formatEntryDate(isoTimestamp: string): string {
  const d = new Date(isoTimestamp);
  if (isNaN(d.getTime())) return isoTimestamp.slice(0, 10);
  return d.toLocaleDateString("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
  });
}

/** "Mar 28 at 7:32 AM" */
export function formatEntryDatetime(isoTimestamp: string): string {
  const d = new Date(isoTimestamp);
  if (isNaN(d.getTime())) return isoTimestamp;
  return d.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

// ── text helpers ─────────────────────────────────────────────────────────────

/** Trim dream text to an excerpt, preserving whole words. */
export function truncateDreamText(text: string, maxLen = 120): string {
  if (text.length <= maxLen) return text;
  const cut = text.slice(0, maxLen);
  const lastSpace = cut.lastIndexOf(" ");
  return (lastSpace > 0 ? cut.slice(0, lastSpace) : cut) + "…";
}

/** Return up to `n` themes (default 3) */
export function topThemes(entry: DreamEntry, n = 3): string[] {
  return (entry.themes ?? []).slice(0, n);
}

// ── filtering & sorting ──────────────────────────────────────────────────────

export function applyFilter(entries: DreamEntry[], filter: DreamFilter): DreamEntry[] {
  switch (filter) {
    case "nightmares": return entries.filter(isNightmare);
    case "lucid":      return entries.filter(isLucid);
    case "with-insight": return entries.filter(hasInsight);
    default:           return entries;
  }
}

/** Case-insensitive search across dreamText, themes, keyInsight */
export function searchDreams(entries: DreamEntry[], query: string): DreamEntry[] {
  const q = query.trim().toLowerCase();
  if (!q) return entries;
  return entries.filter((e) => {
    if (e.dreamText.toLowerCase().includes(q)) return true;
    if ((e.keyInsight ?? "").toLowerCase().includes(q)) return true;
    if ((e.themes ?? []).some((t) => t.toLowerCase().includes(q))) return true;
    if ((e.symbols ?? []).some((s) => s.toLowerCase().includes(q))) return true;
    return false;
  });
}

/** Sort newest-first (stable sort). */
export function sortNewest(entries: DreamEntry[]): DreamEntry[] {
  return [...entries].sort(
    (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
  );
}

// ── stats ────────────────────────────────────────────────────────────────────

export interface DreamHistoryStats {
  total: number;
  nightmares: number;
  lucid: number;
  withInsight: number;
}

export function computeStats(entries: DreamEntry[]): DreamHistoryStats {
  return {
    total:       entries.length,
    nightmares:  entries.filter(isNightmare).length,
    lucid:       entries.filter(isLucid).length,
    withInsight: entries.filter(hasInsight).length,
  };
}

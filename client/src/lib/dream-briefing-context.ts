/**
 * dream-briefing-context.ts
 *
 * Fetches the most recent dream analysis for a user and extracts the fields
 * needed by the morning-briefing endpoint's dreamContext payload.
 *
 * Pure helper functions are exported separately so they can be unit-tested
 * without network calls.
 */

import { apiRequest } from "@/lib/queryClient";

// ── Types ─────────────────────────────────────────────────────────────────────

export interface DreamBriefingContext {
  keyInsight: string | null;
  themes: string[];
  emotionalArc: string | null;
  /** true when threatSimulationIndex > 0.5 */
  isSleepDistress: boolean;
}

interface RawDreamAnalysis {
  keyInsight?: string | null;
  themes?: string[] | null;
  emotionalArc?: string | null;
  threatSimulationIndex?: number | null;
  timestamp?: string;
}

// ── Pure helpers (exported for tests) ─────────────────────────────────────────

/**
 * Extract briefing context from a raw dream analysis row.
 * Returns null when the row has no useful content.
 */
export function extractDreamBriefingContext(
  row: RawDreamAnalysis | null | undefined,
): DreamBriefingContext | null {
  if (!row) return null;

  const themes = Array.isArray(row.themes) ? row.themes.filter(Boolean) : [];
  const keyInsight = row.keyInsight?.trim() || null;
  const emotionalArc = row.emotionalArc?.trim() || null;
  const isSleepDistress = (row.threatSimulationIndex ?? 0) > 0.5;

  // Only return context when there's at least one meaningful field
  if (!keyInsight && themes.length === 0 && !emotionalArc) return null;

  return { keyInsight, themes, emotionalArc, isSleepDistress };
}

/**
 * Return true when the dream timestamp is from the past 24 hours.
 * Keeps the briefing relevant — don't surface a week-old dream.
 */
export function isDreamRecent(timestampIso: string | undefined, nowMs = Date.now()): boolean {
  if (!timestampIso) return false;
  const dreamMs = new Date(timestampIso).getTime();
  if (Number.isNaN(dreamMs)) return false;
  return nowMs - dreamMs < 24 * 60 * 60 * 1000;
}

// ── Network fetch ──────────────────────────────────────────────────────────────

/**
 * Fetch the most recent dream analysis for a user and extract briefing context.
 * Silently returns null on any network or parse error — briefing still works
 * without dream data.
 */
export async function fetchLatestDreamContext(
  userId: string,
): Promise<DreamBriefingContext | null> {
  if (!userId || userId === "anonymous") return null;
  try {
    const res = await apiRequest("GET", `/api/dream-analysis/${userId}?limit=1`);
    const rows: RawDreamAnalysis[] = await res.json();
    const row = rows?.[0];
    if (!isDreamRecent(row?.timestamp)) return null;
    return extractDreamBriefingContext(row);
  } catch {
    return null;
  }
}

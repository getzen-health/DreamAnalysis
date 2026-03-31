/**
 * sleep-intention.ts
 * Presleep intention tracking — store, retrieve, and score nightly intentions
 * against the following morning's dream entry.
 *
 * Storage: localStorage only (no backend required).
 * Key format: `ndw_intention_YYYY-MM-DD` — one intention per night.
 */

export interface IntentionEntry {
  date: string;               // "YYYY-MM-DD" of the night
  text: string;               // freeform intention text
  savedAt: string;            // ISO timestamp
  alignmentScore: number | null;   // 0-1 after morning scoring, null if not yet scored
  alignedDreamId: string | null;   // id of the dream that was scored
}

const KEY_PREFIX = "ndw_intention_";
const MAX_HISTORY_DAYS = 30;

// ── storage helpers ───────────────────────────────────────────────────────────

function todayKey(): string {
  return KEY_PREFIX + new Date().toISOString().slice(0, 10);
}

function dateKey(date: string): string {
  return KEY_PREFIX + date;
}

/** Save tonight's intention (overwrites any existing one for today). */
export function saveIntention(text: string): IntentionEntry {
  const trimmed = text.trim().slice(0, 500); // cap at 500 chars
  const today = new Date().toISOString().slice(0, 10);
  const entry: IntentionEntry = {
    date: today,
    text: trimmed,
    savedAt: new Date().toISOString(),
    alignmentScore: null,
    alignedDreamId: null,
  };
  try {
    localStorage.setItem(todayKey(), JSON.stringify(entry));
  } catch {
    // quota exceeded or SSR — silently ignore
  }
  return entry;
}

/** Retrieve today's intention if it exists. */
export function getTodayIntention(): IntentionEntry | null {
  try {
    const raw = localStorage.getItem(todayKey());
    return raw ? (JSON.parse(raw) as IntentionEntry) : null;
  } catch {
    return null;
  }
}

/** Retrieve the intention for a specific date. */
export function getIntentionForDate(date: string): IntentionEntry | null {
  try {
    const raw = localStorage.getItem(dateKey(date));
    return raw ? (JSON.parse(raw) as IntentionEntry) : null;
  } catch {
    return null;
  }
}

/** Return up to MAX_HISTORY_DAYS of intentions, newest-first. */
export function getIntentionHistory(): IntentionEntry[] {
  const results: IntentionEntry[] = [];
  try {
    for (let i = 0; i < MAX_HISTORY_DAYS; i++) {
      const d = new Date();
      d.setDate(d.getDate() - i);
      const date = d.toISOString().slice(0, 10);
      const raw = localStorage.getItem(dateKey(date));
      if (raw) {
        try { results.push(JSON.parse(raw) as IntentionEntry); } catch { /* skip */ }
      }
    }
  } catch {
    // localStorage unavailable
  }
  return results;
}

/** Persist an alignment score back onto an existing intention entry. */
export function persistAlignmentScore(
  date: string,
  score: number,
  dreamId: string,
): IntentionEntry | null {
  try {
    const key = dateKey(date);
    const raw = localStorage.getItem(key);
    if (!raw) return null;
    const entry: IntentionEntry = JSON.parse(raw);
    entry.alignmentScore = Math.max(0, Math.min(1, score));
    entry.alignedDreamId = dreamId;
    localStorage.setItem(key, JSON.stringify(entry));
    return entry;
  } catch {
    return null;
  }
}

/** Remove intentions older than MAX_HISTORY_DAYS. */
export function clearOldIntentions(): void {
  try {
    const cutoff = new Date();
    cutoff.setDate(cutoff.getDate() - MAX_HISTORY_DAYS);
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (!key?.startsWith(KEY_PREFIX)) continue;
      const dateStr = key.slice(KEY_PREFIX.length); // "YYYY-MM-DD"
      if (new Date(dateStr) < cutoff) {
        localStorage.removeItem(key);
        i--; // index shifts after removal
      }
    }
  } catch {
    // ignore
  }
}

// ── alignment scoring ─────────────────────────────────────────────────────────

/**
 * Score how well a dream (text + themes) aligns with a presleep intention.
 *
 * Method: normalise both sides to lowercase token sets, compute
 * Jaccard-like overlap weighted by intention token length.
 *
 * Returns a value in [0, 1]:
 *   0   = zero overlap
 *   1   = perfect overlap (every intention word found in dream)
 */
export function scoreAlignment(
  intentionText: string,
  dreamText: string,
  themes: string[] | null = null,
): number {
  if (!intentionText.trim() || !dreamText.trim()) return 0;

  // Tokenise: lower, letters+digits only, 3+ chars, deduplicated
  const tokenise = (s: string): Set<string> => {
    const words = s.toLowerCase().match(/[a-z]{3,}/g) ?? [];
    return new Set(words);
  };

  const STOP = new Set([
    "the", "and", "but", "for", "with", "about", "that", "this",
    "have", "will", "want", "dream", "night", "sleep", "today",
    "just", "feel", "more", "some", "from", "been", "are", "was",
  ]);

  const filterStop = (set: Set<string>): Set<string> => {
    const out = new Set<string>();
    for (const w of set) { if (!STOP.has(w)) out.add(w); }
    return out;
  };

  const intentTokens = filterStop(tokenise(intentionText));
  if (intentTokens.size === 0) return 0;

  // Build dream corpus from text + themes
  const corpusParts = [dreamText, ...(themes ?? [])].join(" ");
  const dreamTokens = filterStop(tokenise(corpusParts));

  // Stemmed partial match: intention token is "present" if dream contains a word
  // that starts with the same 4-char prefix (handles "resolv"→"resolving" etc.)
  let matches = 0;
  for (const tok of intentTokens) {
    const prefix = tok.slice(0, Math.min(tok.length, 5));
    let found = dreamTokens.has(tok);
    if (!found) {
      for (const dt of dreamTokens) {
        if (dt.startsWith(prefix)) { found = true; break; }
      }
    }
    if (found) matches++;
  }

  return matches / intentTokens.size;
}

// ── label helpers ─────────────────────────────────────────────────────────────

export type AlignmentLabel = "strong" | "partial" | "weak" | "none";

export function alignmentLabel(score: number): AlignmentLabel {
  if (score >= 0.5) return "strong";
  if (score >= 0.25) return "partial";
  if (score > 0)    return "weak";
  return "none";
}

export const ALIGNMENT_COLOR: Record<AlignmentLabel, string> = {
  strong:  "text-emerald-400",
  partial: "text-amber-400",
  weak:    "text-orange-400",
  none:    "text-muted-foreground/50",
};

export const ALIGNMENT_BG: Record<AlignmentLabel, string> = {
  strong:  "bg-emerald-500/15",
  partial: "bg-amber-500/15",
  weak:    "bg-orange-500/15",
  none:    "bg-white/5",
};

/** One-line description of alignment quality. */
export function alignmentDescription(label: AlignmentLabel): string {
  switch (label) {
    case "strong":  return "Dream closely matched your intention";
    case "partial": return "Dream partially reflected your intention";
    case "weak":    return "Faint trace of your intention in the dream";
    case "none":    return "Dream did not reflect tonight's intention";
  }
}

// ── stats ────────────────────────────────────────────────────────────────────

export interface IntentionStats {
  totalSet: number;        // intentions set in history
  totalScored: number;     // scored (morning dream recorded)
  avgAlignment: number | null; // mean score among scored entries
  strongRate: number;      // fraction of scored entries with score ≥ 0.5
}

export function computeIntentionStats(history: IntentionEntry[]): IntentionStats {
  const scored = history.filter((e) => e.alignmentScore !== null);
  const scores = scored.map((e) => e.alignmentScore as number);
  return {
    totalSet:     history.length,
    totalScored:  scored.length,
    avgAlignment: scores.length > 0
      ? scores.reduce((s, v) => s + v, 0) / scores.length
      : null,
    strongRate: scores.length > 0
      ? scores.filter((v) => v >= 0.5).length / scores.length
      : 0,
  };
}

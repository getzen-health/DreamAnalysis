/**
 * feedback-sync — Persists emotion corrections to Supabase + ML backend.
 *
 * Flow:
 *   1. Try Supabase insert (persistent cloud storage)
 *   2. On failure, queue in localStorage for later sync
 *   3. Fire-and-forget to ML backend /api/feedback (online learning)
 *
 * Offline queue is flushed on app startup via flushPendingCorrections().
 */

import { getSupabase } from "./supabase-browser";
import { getMLApiUrl } from "./ml-api";

export interface CorrectionRecord {
  userId: string;
  predictedEmotion: string;
  correctedEmotion: string;
  source: "manual" | "voice" | "eeg";
  confidence?: number;
  features?: Record<string, number>;
  sessionId?: string;
}

const QUEUE_KEY = "ndw_pending_corrections";

/**
 * Write correction to Supabase + ML backend + local queue fallback.
 */
export async function recordCorrection(record: CorrectionRecord): Promise<void> {
  // 1. Try Supabase (persistent)
  try {
    const supabase = await getSupabase();
    if (supabase) {
      const { error } = await supabase.from("user_feedback").insert({
        user_id: record.userId,
        predicted_emotion: record.predictedEmotion,
        corrected_emotion: record.correctedEmotion,
        source: record.source,
        confidence: record.confidence ?? null,
        features: record.features ?? null,
        session_id: record.sessionId ?? null,
      });
      if (error) {
        queueCorrection(record);
      }
    } else {
      // No Supabase configured — queue for later
      queueCorrection(record);
    }
  } catch {
    // Network error or other failure — queue for later sync
    queueCorrection(record);
  }

  // 2. Also fire to ML backend (immediate online learning)
  try {
    const mlUrl = getMLApiUrl();
    await fetch(`${mlUrl}/api/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: record.userId,
        predicted_label: record.predictedEmotion,
        correct_label: record.correctedEmotion,
      }),
    });
  } catch {
    // fire and forget — ML backend may be offline
  }

  // 3. Trigger retrain check (fire-and-forget, non-blocking)
  triggerRetrainCheck(record.userId).catch(() => {});
}

/**
 * Queue a correction in localStorage for later sync.
 */
function queueCorrection(record: CorrectionRecord): void {
  try {
    const pending = JSON.parse(localStorage.getItem(QUEUE_KEY) || "[]");
    pending.push({ ...record, queuedAt: new Date().toISOString() });
    localStorage.setItem(QUEUE_KEY, JSON.stringify(pending));
  } catch {
    // localStorage unavailable (private browsing, quota exceeded)
  }
}

/**
 * Flush queued corrections to Supabase. Call on app startup after auth.
 * Returns the number of successfully flushed items.
 */
export async function flushPendingCorrections(): Promise<number> {
  let pending: Array<CorrectionRecord & { queuedAt?: string }>;
  try {
    pending = JSON.parse(localStorage.getItem(QUEUE_KEY) || "[]");
  } catch {
    return 0;
  }
  if (pending.length === 0) return 0;

  const supabase = await getSupabase();
  if (!supabase) return 0;

  let flushed = 0;
  const remaining: typeof pending = [];

  for (const record of pending) {
    try {
      const { error } = await supabase.from("user_feedback").insert({
        user_id: record.userId,
        predicted_emotion: record.predictedEmotion,
        corrected_emotion: record.correctedEmotion,
        source: record.source,
        confidence: record.confidence ?? null,
        features: record.features ?? null,
        session_id: record.sessionId ?? null,
      });
      if (error) {
        remaining.push(record);
      } else {
        flushed++;
      }
    } catch {
      remaining.push(record);
    }
  }

  try {
    localStorage.setItem(QUEUE_KEY, JSON.stringify(remaining));
  } catch {
    // ignore
  }
  return flushed;
}

/**
 * Get total correction count for a user from Supabase.
 */
export async function getCorrectionCount(userId: string): Promise<number> {
  try {
    const supabase = await getSupabase();
    if (!supabase) return 0;
    const { count } = await supabase
      .from("user_feedback")
      .select("*", { count: "exact", head: true })
      .eq("user_id", userId);
    return count ?? 0;
  } catch {
    return 0;
  }
}

// ─── Auto-retrain triggers ──────────────────────────────────────────────────

const CORRECTION_COUNT_KEY = "ndw_correction_count";
const LAST_SYNC_KEY = "ndw_last_training_sync";
const LAST_RETRAINED_KEY = "ndw_last_retrained";

/**
 * Check if the ML backend should be pinged for a retrain.
 * Only fires every 10th correction to avoid spamming.
 * Fire-and-forget — errors are swallowed silently.
 */
export async function triggerRetrainCheck(userId: string): Promise<void> {
  try {
    const count = parseInt(localStorage.getItem(CORRECTION_COUNT_KEY) || "0", 10) + 1;
    localStorage.setItem(CORRECTION_COUNT_KEY, String(count));

    if (count % 10 !== 0) return;

    const mlUrl = getMLApiUrl();
    const resp = await fetch(`${mlUrl}/training/sync/${userId}`, { method: "POST" });
    if (resp.ok) {
      try {
        const data = await resp.json();
        if (data.retrain_triggered) {
          localStorage.setItem(LAST_RETRAINED_KEY, new Date().toISOString());
        }
      } catch {
        // ignore parse errors
      }
    }
  } catch {
    // fire and forget — ML backend may be offline
  }
}

/**
 * Sync corrections and trigger retrain on app startup (once per day).
 * Should be called after auth is confirmed and user ID is available.
 * Fire-and-forget — errors are swallowed silently so the app still loads.
 */
export async function syncOnStartup(userId: string): Promise<void> {
  try {
    const lastSync = localStorage.getItem(LAST_SYNC_KEY);
    const today = new Date().toISOString().split("T")[0];
    if (lastSync === today) return;

    const mlUrl = getMLApiUrl();
    // Only set the lastSync date AFTER a successful fetch, so failures retry next time.
    const resp = await fetch(`${mlUrl}/training/sync/${userId}`, { method: "POST" });
    if (resp.ok) {
      localStorage.setItem(LAST_SYNC_KEY, today);
      try {
        const data = await resp.json();
        if (data.retrain_triggered) {
          localStorage.setItem(LAST_RETRAINED_KEY, new Date().toISOString());
        }
      } catch {
        // ignore parse errors
      }
    }
  } catch {
    // fire and forget — ML backend may be offline
  }
}

/**
 * Get local retraining status for display in the UI.
 */
export function getRetrainingStatus(): {
  lastRetrained: string | null;
  correctionsCount: number;
  nextRetrainAt: number;
} {
  const count = parseInt(localStorage.getItem(CORRECTION_COUNT_KEY) || "0", 10);
  const lastRetrained = localStorage.getItem(LAST_RETRAINED_KEY);
  // Next retrain fires at the next multiple of 10 above the current count
  const nextRetrainAt = (Math.floor(count / 10) + 1) * 10;
  return { lastRetrained, correctionsCount: count, nextRetrainAt };
}

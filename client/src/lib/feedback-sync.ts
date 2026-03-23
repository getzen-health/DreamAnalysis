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

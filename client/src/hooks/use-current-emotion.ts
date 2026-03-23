/**
 * useCurrentEmotion -- single source of truth for the current emotional state.
 *
 * Reads from localStorage (ndw_last_emotion) and listens for:
 *   - "ndw-emotion-update" custom event (fired by voice analysis or EEG updates)
 *   - "ndw-voice-updated" legacy event (fired by bottom tab mic)
 *   - "storage" event (cross-tab sync)
 *
 * All pages that need emotion data should use this hook.
 */

import { useState, useEffect, useCallback } from "react";
import { sbGetSetting } from "../lib/supabase-store";

export interface CurrentEmotion {
  /** Primary emotion label */
  emotion: string;
  /** Valence: -1 (negative) to +1 (positive) */
  valence: number;
  /** Arousal: 0 (calm) to 1 (energetic) */
  arousal: number;
  /** Stress: 0 (relaxed) to 1 (stressed) */
  stress: number;
  /** Focus: 0 (unfocused) to 1 (focused) */
  focus: number;
  /** Model confidence: 0 to 1 */
  confidence: number;
  /** Where this reading came from */
  source: "voice" | "eeg" | "manual";
  /** ISO timestamp of when the reading was taken */
  timestamp: string;
  /** True if the reading is older than 30 minutes */
  isStale: boolean;
}

const STALE_THRESHOLD_MS = 30 * 60 * 1000; // 30 minutes

function readFromStorage(): CurrentEmotion | null {
  try {
    const raw = sbGetSetting("ndw_last_emotion");
    if (!raw) return null;

    const parsed = JSON.parse(raw);
    // Handle both {result: {...}, timestamp} and flat format
    const data = parsed?.result ?? parsed;
    if (!data?.emotion) return null;

    const ts = parsed?.timestamp ?? data?.timestamp ?? Date.now();
    // Normalize: timestamp can be epoch ms or epoch seconds
    const epochMs = ts > 1e12 ? ts : ts * 1000;
    const isStale = Date.now() - epochMs > STALE_THRESHOLD_MS;

    return {
      emotion: data.emotion,
      valence: data.valence ?? 0,
      arousal: data.arousal ?? 0.5,
      stress: data.stress_index ?? data.stress ?? 0.5,
      focus: data.focus_index ?? data.focus ?? 0.5,
      confidence: data.confidence ?? 0.5,
      source: data.model_type === "eeg" ? "eeg" : "voice",
      timestamp: new Date(epochMs).toISOString(),
      isStale,
    };
  } catch {
    return null;
  }
}

export function useCurrentEmotion(): {
  emotion: CurrentEmotion | null;
  refresh: () => void;
} {
  const [emotion, setEmotion] = useState<CurrentEmotion | null>(readFromStorage);

  const refresh = useCallback(() => {
    setEmotion(readFromStorage());
  }, []);

  useEffect(() => {
    // Listen for all emotion update events
    window.addEventListener("ndw-emotion-update", refresh);
    window.addEventListener("ndw-voice-updated", refresh);
    window.addEventListener("storage", refresh);

    return () => {
      window.removeEventListener("ndw-emotion-update", refresh);
      window.removeEventListener("ndw-voice-updated", refresh);
      window.removeEventListener("storage", refresh);
    };
  }, [refresh]);

  return { emotion, refresh };
}

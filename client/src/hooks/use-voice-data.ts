/**
 * useVoiceData — shared hook for reading voice check-in data across all pages.
 *
 * Reads from localStorage (ndw_last_emotion) and listens for the
 * "ndw-voice-updated" event dispatched by the bottom tab mic button.
 * All pages that display voice emotion data should use this hook
 * instead of reading localStorage directly.
 */

import { useState, useEffect } from "react";

export interface VoiceCheckinData {
  emotion?: string;
  valence?: number;
  arousal?: number;
  stress_index?: number;
  focus_index?: number;
  relaxation_index?: number;
  confidence?: number;
  model_type?: string;
  probabilities?: Record<string, number>;
  biomarkers?: Record<string, unknown>;
  timestamp?: number;
}

function readCheckin(): VoiceCheckinData | null {
  try {
    const raw = localStorage.getItem("ndw_last_emotion");
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    // Handle both {result: {...}, timestamp} and direct format
    const data = parsed?.result ?? parsed;
    if (!data?.emotion) return null;
    return {
      ...data,
      timestamp: parsed?.timestamp ?? Date.now(),
    };
  } catch {
    return null;
  }
}

/**
 * Returns the latest voice check-in data, auto-updating when a new
 * check-in happens via the bottom tab mic button.
 */
export function useVoiceData(): VoiceCheckinData | null {
  const [data, setData] = useState<VoiceCheckinData | null>(readCheckin);

  useEffect(() => {
    // Re-read on voice update event (from bottom tab mic)
    function handler() {
      setData(readCheckin());
    }
    window.addEventListener("ndw-voice-updated", handler);
    // Also re-read on storage change (cross-tab)
    window.addEventListener("storage", handler);
    return () => {
      window.removeEventListener("ndw-voice-updated", handler);
      window.removeEventListener("storage", handler);
    };
  }, []);

  return data;
}

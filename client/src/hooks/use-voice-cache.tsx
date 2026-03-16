/**
 * VoiceCacheContext — shared voice emotion result across all pages.
 *
 * On mount, fetches /voice-watch/latest/{userId} to hydrate any result
 * cached within the last 5 minutes (server TTL). Polls every 30 seconds
 * so new check-ins on other pages propagate automatically.
 *
 * When use-voice-emotion completes a recording, it calls
 * setVoiceCacheResult() to update the context immediately (no wait for poll).
 */
import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  type ReactNode,
} from "react";
import { resolveUrl } from "@/lib/queryClient";
import { getParticipantId } from "@/lib/participant";
import type { VoiceEmotionResult } from "./use-voice-emotion";

interface VoiceCacheState {
  /** Most recent voice emotion result (from server cache or local recording). */
  cachedEmotion: VoiceEmotionResult | null;
  /** Unix timestamp (ms) when the result was captured. null if no result yet. */
  lastCheckInTime: number | null;
  /** True if a result exists and is < 5 minutes old. */
  hasRecentCheckIn: boolean;
  /** Update the cache immediately after a local recording completes. */
  setVoiceCacheResult: (result: VoiceEmotionResult) => void;
}

const FIVE_MINUTES_MS = 5 * 60 * 1000;
const POLL_INTERVAL_MS = 30_000;

export const VoiceCacheContext = createContext<VoiceCacheState>({
  cachedEmotion: null,
  lastCheckInTime: null,
  hasRecentCheckIn: false,
  setVoiceCacheResult: () => {},
});

export function VoiceCacheProvider({ children }: { children: ReactNode }) {
  const userId = getParticipantId();
  const [cachedEmotion, setCachedEmotion] = useState<VoiceEmotionResult | null>(null);
  const [lastCheckInTime, setLastCheckInTime] = useState<number | null>(null);

  const fetchLatest = useCallback(async () => {
    try {
      const res = await fetch(resolveUrl(`/api/ml/voice-watch/latest/${userId}`));
      if (!res.ok) return;
      const data = await res.json();
      if (
        data &&
        typeof data === "object" &&
        !Array.isArray(data) &&
        typeof data.emotion === "string"
      ) {
        setCachedEmotion(data as VoiceEmotionResult);
        // Use current time as approximation — server TTL check already ensures
        // the result is < 5 min old, so treating fetch time as check-in time is safe.
        setLastCheckInTime((prev) => prev ?? Date.now());
      }
    } catch {
      // Network error — leave existing cache in place
    }
  }, [userId]);

  // Fetch on mount, then poll every 30 s
  useEffect(() => {
    fetchLatest();
    const interval = setInterval(fetchLatest, POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [fetchLatest]);

  const setVoiceCacheResult = useCallback((result: VoiceEmotionResult) => {
    setCachedEmotion(result);
    setLastCheckInTime(Date.now());
  }, []);

  const hasRecentCheckIn =
    cachedEmotion !== null &&
    lastCheckInTime !== null &&
    Date.now() - lastCheckInTime < FIVE_MINUTES_MS;

  return (
    <VoiceCacheContext.Provider
      value={{ cachedEmotion, lastCheckInTime, hasRecentCheckIn, setVoiceCacheResult }}
    >
      {children}
    </VoiceCacheContext.Provider>
  );
}

/** Read the shared voice cache state from any page. */
export function useVoiceCache(): VoiceCacheState {
  return useContext(VoiceCacheContext);
}

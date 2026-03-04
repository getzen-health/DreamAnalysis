import {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
  useCallback,
  type ReactNode,
} from "react";
import { pingBackend } from "@/lib/ml-api";

// ── Types ─────────────────────────────────────────────────────────────────────

export type MLStatus = "idle" | "connecting" | "warming" | "ready" | "error";

export interface MLConnectionState {
  status: MLStatus;
  latencyMs: number | null;
  warmupProgress: number; // 0-100
  retryCount: number;
  reconnect: () => void;
}

// ── Context ───────────────────────────────────────────────────────────────────

const MLConnectionContext = createContext<MLConnectionState | null>(null);

// ── Constants ─────────────────────────────────────────────────────────────────

const FAST_PING_INTERVAL_MS = 5_000;
const SLOW_PING_INTERVAL_MS = 30_000;
/** Render free-tier cold start can take ~50 s — keep trying for 8 × 5s = 40 s */
const MAX_FAILURES = 8;

/** Total warmup duration over which progress climbs 0→95 */
const WARMUP_DURATION_MS = 35_000;
/** How often the progress bar advances (ms) */
const PROGRESS_TICK_MS = 300;
/** Progress increment per tick so we reach ~95 in WARMUP_DURATION_MS */
const PROGRESS_STEP = (95 / WARMUP_DURATION_MS) * PROGRESS_TICK_MS;

// ── Provider ──────────────────────────────────────────────────────────────────

export function MLConnectionProvider({
  children,
}: {
  children: ReactNode;
}): JSX.Element {
  const [status, setStatus] = useState<MLStatus>("connecting");
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [warmupProgress, setWarmupProgress] = useState(0);
  const [retryCount, setRetryCount] = useState(0);

  // Refs let interval callbacks read current state without stale closures.
  const failuresRef = useRef(0);
  const readyRef = useRef(false);
  const pingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const progressIntervalRef = useRef<ReturnType<typeof setInterval> | null>(
    null
  );
  // Guard: ignore ping results that arrive after a reconnect() was issued.
  const reconnectGenerationRef = useRef(0);
  // Stable ref to reconnect() so the slow-poll closure never captures a stale copy.
  const reconnectRef = useRef<() => void>(() => {});

  const clearPingInterval = () => {
    if (pingIntervalRef.current !== null) {
      clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = null;
    }
  };

  const clearProgressInterval = () => {
    if (progressIntervalRef.current !== null) {
      clearInterval(progressIntervalRef.current);
      progressIntervalRef.current = null;
    }
  };

  const startProgressBar = useCallback(() => {
    clearProgressInterval();
    progressIntervalRef.current = setInterval(() => {
      setWarmupProgress((prev) => {
        const next = prev + PROGRESS_STEP;
        return next >= 95 ? 95 : next;
      });
    }, PROGRESS_TICK_MS);
  }, []);

  const stopProgressBar = useCallback((finalValue: number) => {
    clearProgressInterval();
    setWarmupProgress(finalValue);
  }, []);

  const startPinging = useCallback((generation: number) => {
    clearPingInterval();

    const doPing = async () => {
      // If a new reconnect happened while this ping was in-flight, discard.
      if (generation !== reconnectGenerationRef.current) return;

      const t0 = Date.now();
      const ok = await pingBackend();

      if (generation !== reconnectGenerationRef.current) return;

      if (ok) {
        const elapsed = Date.now() - t0;
        readyRef.current = true;
        failuresRef.current = 0;

        setStatus("ready");
        setLatencyMs(elapsed);
        stopProgressBar(100);

        // Switch to slow keep-alive polling
        clearPingInterval();
        pingIntervalRef.current = setInterval(async () => {
          if (generation !== reconnectGenerationRef.current) return;
          const alive = await pingBackend();
          if (!alive && generation === reconnectGenerationRef.current) {
            reconnectRef.current();
          }
        }, SLOW_PING_INTERVAL_MS);
      } else {
        failuresRef.current += 1;
        setRetryCount(failuresRef.current);

        if (failuresRef.current >= MAX_FAILURES) {
          setStatus("error");
          clearProgressInterval();
          // Keep slow-polling so the app auto-recovers when Render finishes starting
          clearPingInterval();
          pingIntervalRef.current = setInterval(async () => {
            if (generation !== reconnectGenerationRef.current) return;
            const alive = await pingBackend();
            if (alive && generation === reconnectGenerationRef.current) {
              reconnectRef.current();
            }
          }, SLOW_PING_INTERVAL_MS);
        } else {
          // Show warm-up in progress on first failure
          setStatus("warming");
        }
      }
    };

    // Assign the interval BEFORE calling doPing() so that if doPing resolves
    // synchronously (e.g. in tests), clearPingInterval() can find and clear it.
    pingIntervalRef.current = setInterval(doPing, FAST_PING_INTERVAL_MS);
    doPing();
  }, [stopProgressBar]);

  const reconnect = useCallback(() => {
    // Bump generation so in-flight pings from previous sessions are discarded.
    reconnectGenerationRef.current += 1;
    const gen = reconnectGenerationRef.current;

    clearPingInterval();
    clearProgressInterval();

    failuresRef.current = 0;
    readyRef.current = false;

    setStatus("connecting");
    setLatencyMs(null);
    setWarmupProgress(0);
    setRetryCount(0);

    startProgressBar();
    startPinging(gen);
  }, [startPinging, startProgressBar]);

  // Keep the ref in sync so slow-poll closures always call the current reconnect.
  reconnectRef.current = reconnect;

  // Mount: kick everything off
  useEffect(() => {
    const gen = reconnectGenerationRef.current;
    startProgressBar();
    startPinging(gen);

    return () => {
      // Invalidate all in-flight pings on unmount
      reconnectGenerationRef.current += 1;
      clearPingInterval();
      clearProgressInterval();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <MLConnectionContext.Provider
      value={{ status, latencyMs, warmupProgress, retryCount, reconnect }}
    >
      {children}
    </MLConnectionContext.Provider>
  );
}

// ── Consumer hook ─────────────────────────────────────────────────────────────

export function useMLConnection(): MLConnectionState {
  const ctx = useContext(MLConnectionContext);
  if (!ctx) {
    throw new Error("useMLConnection must be used within an MLConnectionProvider");
  }
  return ctx;
}

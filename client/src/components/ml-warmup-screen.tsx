import { useEffect, useRef, useState } from "react";
import { Brain } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useMLConnection } from "@/hooks/use-ml-connection";

// ── Types ─────────────────────────────────────────────────────────────────────

interface MLWarmupScreenProps {
  onSimulationMode?: () => void;
}

// ── Constants ─────────────────────────────────────────────────────────────────

const ROTATING_MESSAGES = [
  "Initializing neural engines...",
  "Loading EEG models...",
  "Calibrating signal pipeline...",
  "Almost ready...",
] as const;

const MESSAGE_INTERVAL_MS = 8_000;
const ELAPSED_TICK_MS = 1_000;
/** Show "Browse app while loading" after this many seconds */
const SIMULATION_MODE_THRESHOLD_S = 3;

// ── Component ─────────────────────────────────────────────────────────────────

export function MLWarmupScreen({ onSimulationMode }: MLWarmupScreenProps): JSX.Element | null {
  const { status, warmupProgress } = useMLConnection();

  const [messageIndex, setMessageIndex] = useState(0);
  const [elapsed, setElapsed] = useState(0);

  const startTimeRef = useRef<number>(Date.now());
  const elapsedIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const messageIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Only render during active warm-up phases.
  const isVisible = status === "connecting" || status === "warming";

  useEffect(() => {
    if (!isVisible) return;

    // Reset counters whenever the overlay becomes visible.
    startTimeRef.current = Date.now();
    setElapsed(0);
    setMessageIndex(0);

    // Elapsed counter — ticks every second.
    elapsedIntervalRef.current = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startTimeRef.current) / ELAPSED_TICK_MS));
    }, ELAPSED_TICK_MS);

    // Rotating message — cycles every 8 seconds.
    messageIntervalRef.current = setInterval(() => {
      setMessageIndex((prev) => (prev + 1) % ROTATING_MESSAGES.length);
    }, MESSAGE_INTERVAL_MS);

    return () => {
      if (elapsedIntervalRef.current !== null) {
        clearInterval(elapsedIntervalRef.current);
        elapsedIntervalRef.current = null;
      }
      if (messageIntervalRef.current !== null) {
        clearInterval(messageIntervalRef.current);
        messageIntervalRef.current = null;
      }
    };
  }, [isVisible]);

  if (!isVisible) {
    return null;
  }

  const showSimulationButton =
    onSimulationMode !== undefined && elapsed >= SIMULATION_MODE_THRESHOLD_S;

  return (
    <div className="fixed inset-0 z-50 bg-background flex flex-col items-center justify-center gap-6 p-8 animate-in fade-in duration-500">
      {/* Brain icon — small and static */}
      <Brain className="h-8 w-8 text-primary/70" aria-hidden="true" />

      {/* Title */}
      <h1 className="text-2xl font-semibold text-foreground tracking-tight">
        AntarAI
      </h1>

      {/* Progress bar — prominent, wider */}
      <div className="w-full max-w-md rounded-full bg-muted h-1.5 overflow-hidden">
        <div
          className="h-full bg-primary transition-all duration-500 ease-out rounded-full"
          style={{ width: `${warmupProgress}%` }}
          role="progressbar"
          aria-valuenow={warmupProgress}
          aria-valuemin={0}
          aria-valuemax={100}
        />
      </div>

      {/* Rotating status message */}
      <p className="text-sm text-muted-foreground/80 text-center">
        {ROTATING_MESSAGES[messageIndex]}
      </p>

      {/* Elapsed time */}
      <p className="text-xs text-muted-foreground/40">
        {elapsed}s
      </p>

      {/* Simulation mode button — appears after 40 s */}
      {showSimulationButton && (
        <Button
          variant="outline"
          onClick={onSimulationMode}
          className="mt-4"
        >
          Browse app while loading
        </Button>
      )}
    </div>
  );
}

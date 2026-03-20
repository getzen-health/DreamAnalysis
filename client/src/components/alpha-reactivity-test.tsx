import { useState, useEffect, useRef, useCallback } from "react";
import { runAlphaReactivityTest } from "@/lib/signal-quality";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { CheckCircle, AlertTriangle, Eye, EyeOff, Loader2 } from "lucide-react";

type TestStep = "idle" | "eyes-open" | "eyes-closed" | "processing" | "result";

interface AlphaReactivityTestProps {
  /** Provide current EEG data as Float32Array[] (one per channel, 256 Hz). */
  getCurrentData: () => Float32Array[] | null;
  /** Sampling rate (default 256 Hz). */
  fs?: number;
  /** Duration of each phase in seconds (default 5). */
  phaseDurationSec?: number;
  /** Called when the test completes. */
  onComplete?: (signalVerified: boolean) => void;
}

const COLLECTION_INTERVAL_MS = 250; // collect data every 250ms

/**
 * Guided alpha reactivity test flow:
 *   Step 1: "Keep your eyes open for 5 seconds" (with countdown)
 *   Step 2: "Now close your eyes for 5 seconds" (with countdown)
 *   Step 3: Show result — "Signal verified!" or "Signal may be noisy"
 *
 * Science: When eyes close, posterior alpha rhythm (8-12 Hz) increases.
 * If alpha power increases by > 20%, the signal is real neural EEG.
 * This is the gold standard for consumer EEG signal verification.
 */
export function AlphaReactivityTest({
  getCurrentData,
  fs = 256,
  phaseDurationSec = 5,
  onComplete,
}: AlphaReactivityTestProps) {
  const [step, setStep] = useState<TestStep>("idle");
  const [countdown, setCountdown] = useState(0);
  const [result, setResult] = useState<boolean | null>(null);

  const eyesOpenRef = useRef<Float32Array[][]>([]);
  const eyesClosedRef = useRef<Float32Array[][]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const collectionRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const cleanup = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (collectionRef.current) {
      clearInterval(collectionRef.current);
      collectionRef.current = null;
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => cleanup, [cleanup]);

  const collectData = useCallback(
    (target: Float32Array[][]) => {
      const data = getCurrentData();
      if (data && data.length > 0) {
        target.push(data);
      }
    },
    [getCurrentData],
  );

  const mergeCollected = useCallback(
    (collected: Float32Array[][]): Float32Array[] => {
      if (collected.length === 0) return [];
      const nChannels = collected[0].length;
      const merged: Float32Array[] = [];

      for (let ch = 0; ch < nChannels; ch++) {
        // Concatenate all collected chunks for this channel
        const chunks = collected.map((frame) => frame[ch]).filter(Boolean);
        const totalLen = chunks.reduce((sum, c) => sum + c.length, 0);
        const out = new Float32Array(totalLen);
        let offset = 0;
        for (const chunk of chunks) {
          out.set(chunk, offset);
          offset += chunk.length;
        }
        merged.push(out);
      }

      return merged;
    },
    [],
  );

  const startTest = useCallback(() => {
    cleanup();
    eyesOpenRef.current = [];
    eyesClosedRef.current = [];
    setResult(null);
    setStep("eyes-open");
    setCountdown(phaseDurationSec);

    // Start collecting eyes-open data
    collectionRef.current = setInterval(
      () => collectData(eyesOpenRef.current),
      COLLECTION_INTERVAL_MS,
    );

    // Countdown timer
    let remaining = phaseDurationSec;
    timerRef.current = setInterval(() => {
      remaining--;
      setCountdown(remaining);

      if (remaining <= 0) {
        // Switch to eyes-closed phase
        cleanup();
        setStep("eyes-closed");
        setCountdown(phaseDurationSec);

        collectionRef.current = setInterval(
          () => collectData(eyesClosedRef.current),
          COLLECTION_INTERVAL_MS,
        );

        let remaining2 = phaseDurationSec;
        timerRef.current = setInterval(() => {
          remaining2--;
          setCountdown(remaining2);

          if (remaining2 <= 0) {
            cleanup();
            setStep("processing");

            // Process collected data
            const openData = mergeCollected(eyesOpenRef.current);
            const closedData = mergeCollected(eyesClosedRef.current);

            const verified =
              openData.length > 0 && closedData.length > 0
                ? runAlphaReactivityTest(openData, closedData, fs)
                : false;

            setResult(verified);
            setStep("result");
            onComplete?.(verified);
          }
        }, 1000);
      }
    }, 1000);
  }, [cleanup, collectData, mergeCollected, fs, phaseDurationSec, onComplete]);

  return (
    <Card className="w-full max-w-md">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Alpha Reactivity Test</CardTitle>
      </CardHeader>
      <CardContent>
        {step === "idle" && (
          <div className="space-y-3">
            <p className="text-sm text-muted-foreground">
              This test verifies your EEG signal is real by checking if alpha waves
              increase when you close your eyes (a known neurological response).
            </p>
            <Button onClick={startTest} className="w-full">
              Start Test
            </Button>
          </div>
        )}

        {step === "eyes-open" && (
          <div className="space-y-3 text-center">
            <Eye className="h-10 w-10 mx-auto text-blue-500" />
            <p className="text-lg font-medium">Keep your eyes open</p>
            <p className="text-4xl font-mono font-bold tabular-nums">{countdown}</p>
            <p className="text-sm text-muted-foreground">
              Look at the screen naturally. Try not to blink.
            </p>
          </div>
        )}

        {step === "eyes-closed" && (
          <div className="space-y-3 text-center">
            <EyeOff className="h-10 w-10 mx-auto text-purple-500" />
            <p className="text-lg font-medium">Now close your eyes</p>
            <p className="text-4xl font-mono font-bold tabular-nums">{countdown}</p>
            <p className="text-sm text-muted-foreground">
              Relax and keep your eyes gently closed.
            </p>
          </div>
        )}

        {step === "processing" && (
          <div className="space-y-3 text-center">
            <Loader2 className="h-10 w-10 mx-auto text-muted-foreground animate-spin" />
            <p className="text-sm text-muted-foreground">Analyzing alpha reactivity...</p>
          </div>
        )}

        {step === "result" && result !== null && (
          <div className="space-y-3 text-center">
            {result ? (
              <>
                <CheckCircle className="h-10 w-10 mx-auto text-green-500" />
                <p className="text-lg font-medium text-green-600 dark:text-green-400">
                  Signal verified!
                </p>
                <p className="text-sm text-muted-foreground">
                  Alpha reactivity detected. Your headband is reading real brain signals.
                </p>
              </>
            ) : (
              <>
                <AlertTriangle className="h-10 w-10 mx-auto text-amber-500" />
                <p className="text-lg font-medium text-amber-600 dark:text-amber-400">
                  Signal may be noisy
                </p>
                <p className="text-sm text-muted-foreground">
                  No alpha reactivity detected. Try adjusting your headband position
                  and ensuring the electrode pads are making good contact.
                </p>
              </>
            )}
            <Button variant="outline" onClick={startTest} className="w-full mt-2">
              Run Again
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

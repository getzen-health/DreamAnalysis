/**
 * Hybrid inference hook — tries local ONNX first, falls back to server API.
 *
 * ONNX Runtime (403KB) is lazy-loaded on first analyze() call, not at mount.
 * This keeps the initial bundle lean — inference pages load without pulling in
 * the ONNX chunk until the user actually triggers an EEG analysis.
 *
 * Returns `analyze()` function, `isLocal` flag, `latencyMs`, and `isReady` status.
 */

import { useState, useCallback, useRef } from "react";
import { analyzeEEG, type EEGAnalysisResult } from "@/lib/ml-api";

interface InferenceResult {
  analyze: (signals: number[][], fs?: number) => Promise<EEGAnalysisResult | null>;
  isLocal: boolean;
  latencyMs: number;
  isReady: boolean;
}

export function useInference(): InferenceResult {
  const [isLocal, setIsLocal] = useState(false);
  const [latencyMs, setLatencyMs] = useState(0);
  // isReady starts true — server API is always available as fallback.
  // Will be updated to reflect local ONNX readiness once first analyze() fires.
  const [isReady, setIsReady] = useState(true);
  // Track whether local ML has been initialized yet (lazy — deferred to first call)
  const localInitDone = useRef(false);

  const analyze = useCallback(
    async (signals: number[][], fs: number = 256): Promise<EEGAnalysisResult | null> => {
      const start = performance.now();

      // Lazy-initialize local ML on first analyze() call.
      // This defers the onnxruntime-web dynamic import until inference is actually needed,
      // keeping the initial page load free of the 403KB ONNX bundle.
      if (!localInitDone.current) {
        localInitDone.current = true;
        // Dynamic import of ml-local triggers the onnxruntime-web chunk load.
        // We do not await the full initialization here to avoid blocking the
        // first inference call — the localML singleton handles concurrent init.
        const { localML: ml } = await import("@/lib/ml-local");
        ml.initialize().then(() => {
          setIsReady(true);
        });
      }

      // Dynamically import ml-local only when needed (cached after first import)
      const { localML, extractFeatures } = await import("@/lib/ml-local");

      // Try local inference if ONNX is ready
      if (localML.isReady()) {
        try {
          const signal = signals[0] || [];
          localML.setLastSignal(signal, fs);
          const features = extractFeatures(signal, fs);

          const [sleep, emotion, dream] = await Promise.all([
            localML.analyzeSleep(features),
            localML.analyzeEmotion(features),
            localML.detectDream(features),
          ]);

          if (sleep && emotion && dream) {
            const elapsed = performance.now() - start;
            setLatencyMs(elapsed);
            setIsLocal(true);

            return {
              sleep_stage: sleep,
              emotions: {
                emotion: emotion.emotion,
                confidence: emotion.confidence,
                probabilities: emotion.probabilities,
                valence: 0,
                arousal: 0,
                stress_index: 0,
                focus_index: 0,
                relaxation_index: 0,
                band_powers: {},
              },
              dream_detection: {
                is_dreaming: dream.is_dreaming,
                probability: dream.probability,
                rem_likelihood: dream.probability * 0.8,
                dream_intensity: dream.probability * 0.7,
                lucidity_estimate: 0.1,
              },
              features: {},
              band_powers: {},
            };
          }
        } catch {
          // Fall through to server
        }
      }

      // Fallback to server API
      try {
        const result = await analyzeEEG(signals, fs);
        const elapsed = performance.now() - start;
        setLatencyMs(elapsed);
        setIsLocal(false);
        return result;
      } catch {
        return null;
      }
    },
    []
  );

  return { analyze, isLocal, latencyMs, isReady };
}

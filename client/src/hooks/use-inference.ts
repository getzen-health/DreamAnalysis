/**
 * Hybrid inference hook — tries local ONNX first, falls back to server API.
 *
 * Returns `analyze()` function, `isLocal` flag, `latencyMs`, and `isReady` status.
 */

import { useState, useEffect, useCallback, useRef } from "react";
import { localML, extractFeatures } from "@/lib/ml-local";
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
  const [isReady, setIsReady] = useState(false);
  const initRef = useRef(false);

  useEffect(() => {
    if (initRef.current) return;
    initRef.current = true;

    localML.initialize().then(() => {
      const ready = localML.isReady();
      setIsLocal(ready);
      setIsReady(true);
    });
  }, []);

  const analyze = useCallback(
    async (signals: number[][], fs: number = 256): Promise<EEGAnalysisResult | null> => {
      const start = performance.now();

      // Try local inference first
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

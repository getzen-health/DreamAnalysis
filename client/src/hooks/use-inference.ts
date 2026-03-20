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
import { computeAndCacheBrainAge } from "@/lib/brain-age";

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

          // Try EEGNet first (4-channel, raw EEG input, 4KB model)
          // then fall back to generic emotion model (17-feature, 2.2MB)
          let emotion: Awaited<ReturnType<typeof localML.analyzeEmotion>> = null;
          let eegnetValence = 0;
          let eegnetArousal = 0;

          if (localML.isEEGNetReady() && signals.length >= 4) {
            // Convert number[][] to Float32Array[] for EEGNet
            const rawChannels = signals.slice(0, 4).map((ch) => new Float32Array(ch));
            const eegnetResult = await localML.analyzeEmotionEEGNet(rawChannels, fs);
            if (eegnetResult) {
              emotion = eegnetResult;
              // Import utils to get valence/arousal from probabilities
              const { eegnetEmotionFromProbabilities } = await import("@/lib/eegnet-utils");
              const probs = Object.values(eegnetResult.probabilities);
              // Probabilities are already ordered by EEGNET_EMOTIONS in the dict,
              // but dict ordering may not match — extract in order
              const { EEGNET_EMOTIONS } = await import("@/lib/eegnet-utils");
              const orderedProbs = EEGNET_EMOTIONS.map(
                (e: string) => eegnetResult.probabilities[e] ?? 0
              );
              const full = eegnetEmotionFromProbabilities(orderedProbs);
              eegnetValence = full.valence;
              eegnetArousal = full.arousal;
            }
          }

          // Fall back to generic emotion model if EEGNet didn't produce a result
          if (!emotion) {
            emotion = await localML.analyzeEmotion(features);
          }

          const [sleep, dream] = await Promise.all([
            localML.analyzeSleep(features),
            localML.detectDream(features),
          ]);

          if (sleep && emotion && dream) {
            const elapsed = performance.now() - start;
            setLatencyMs(elapsed);
            setIsLocal(true);

            // Compute and cache brain age from extracted band powers
            const { extractBandPowers } = await import("@/lib/eeg-features");
            const bandPowers = extractBandPowers(signal, fs);
            computeAndCacheBrainAge(bandPowers);

            return {
              sleep_stage: sleep,
              emotions: {
                emotion: emotion.emotion,
                confidence: emotion.confidence,
                probabilities: emotion.probabilities,
                valence: eegnetValence,
                arousal: eegnetArousal,
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

        // Compute and cache brain age from server-returned band powers
        if (result?.band_powers) {
          const se = result.features?.spectral_entropy ?? 0.65;
          computeAndCacheBrainAge(result.band_powers, se);
        }

        return result;
      } catch {
        return null;
      }
    },
    []
  );

  return { analyze, isLocal, latencyMs, isReady };
}

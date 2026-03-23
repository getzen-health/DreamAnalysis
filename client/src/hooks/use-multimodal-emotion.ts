/**
 * useMultimodalEmotion — React hook that fuses EEG, voice, and health signals
 * into a single adaptive emotion estimate.
 *
 * Reads from:
 *   - localStorage "ndw_last_eeg_emotion" (EEG emotion from use-inference)
 *   - localStorage "ndw_last_emotion" (voice emotion from voice analysis)
 *   - useHealthSync → analyzeHealthState() (health-derived emotion)
 *
 * Calls fuseModalities() whenever any input updates.
 * Returns the fused result and a readiness flag.
 */

import { useState, useEffect, useCallback, useMemo } from "react";
import { useHealthSync } from "./use-health-sync";
import { analyzeHealthState, type HealthSnapshot } from "@/lib/health-inference";
import {
  fuseModalities,
  recordFusionFeedback,
  type ModalityInput,
  type FusedResult,
} from "@/lib/multimodal-fusion";
import {
  loadPersonalAdapter,
  savePersonalAdapter,
  updateFromCorrection,
} from "@/lib/personal-adapter";
import { EEGNET_EMOTIONS } from "@/lib/eegnet-utils";
import { sbGetSetting } from "../lib/supabase-store";
import { recordCorrection } from "@/lib/feedback-sync";
import { getParticipantId } from "@/lib/participant";

export interface UseMultimodalEmotionReturn {
  emotion: FusedResult | null;
  isReady: boolean;
  /** Record user correction to adapt per-modality weights. */
  correctEmotion: (userCorrectedEmotion: string) => void;
}

// ── Helpers to read cached modality data ───────────────────────────────────

interface CachedEmotion {
  emotion: string;
  valence: number;
  arousal: number;
  confidence: number;
  stress_index?: number;
  stress?: number;
  focus_index?: number;
}

function readEEGEmotion(): ModalityInput | null {
  try {
    const raw = sbGetSetting("ndw_last_eeg_emotion");
    if (!raw) return null;
    const data: CachedEmotion = JSON.parse(raw);
    if (!data.emotion) return null;
    return {
      valence: data.valence ?? 0,
      arousal: data.arousal ?? 0.5,
      stress: data.stress_index ?? data.stress ?? 0.3,
      confidence: data.confidence ?? 0.5,
      emotion: data.emotion,
      source: "eeg",
    };
  } catch {
    return null;
  }
}

function readVoiceEmotion(): ModalityInput | null {
  try {
    const raw = sbGetSetting("ndw_last_emotion");
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    const data: CachedEmotion = parsed?.result ?? parsed;
    if (!data.emotion) return null;
    return {
      valence: data.valence ?? 0,
      arousal: data.arousal ?? 0.5,
      stress: data.stress_index ?? data.stress ?? 0.3,
      confidence: data.confidence ?? 0.5,
      emotion: data.emotion,
      source: "voice",
    };
  } catch {
    return null;
  }
}

function buildHealthInput(payload: Record<string, unknown> | null): ModalityInput | null {
  if (!payload) return null;

  const snapshot: HealthSnapshot = {
    hrv: payload.hrv_rmssd as number | undefined,
    restingHR: (payload.resting_heart_rate ?? payload.current_heart_rate) as number | undefined,
    sleepQuality: undefined,
    sleepDuration: payload.sleep_total_hours as number | undefined,
    steps: payload.steps_today as number | undefined,
    activeEnergyBurned: payload.active_energy_kcal as number | undefined,
  };

  // Only proceed if we have at least one metric
  const hasData =
    snapshot.hrv !== undefined ||
    snapshot.restingHR !== undefined ||
    snapshot.sleepDuration !== undefined ||
    snapshot.steps !== undefined ||
    snapshot.activeEnergyBurned !== undefined;

  if (!hasData) return null;

  const healthState = analyzeHealthState(snapshot);

  // Map health-derived emotion from valence/arousal
  let emotion = "neutral";
  if (healthState.valence > 0.2 && healthState.arousal < 0.4) emotion = "calm";
  else if (healthState.valence > 0.2 && healthState.arousal >= 0.4) emotion = "happy";
  else if (healthState.valence < -0.1 && healthState.arousal > 0.5) emotion = "anxious";
  else if (healthState.valence < -0.1) emotion = "sad";

  return {
    valence: healthState.valence,
    arousal: healthState.arousal,
    stress: healthState.stressIndex,
    confidence: healthState.confidence,
    emotion,
    source: "health",
  };
}

// ── Hook ──────────────────────────────────────────────────────────────────

export function useMultimodalEmotion(): UseMultimodalEmotionReturn {
  const { latestPayload } = useHealthSync();
  const [fusedEmotion, setFusedEmotion] = useState<FusedResult | null>(null);
  const [lastInputs, setLastInputs] = useState<ModalityInput[]>([]);

  // Recompute fusion whenever any source updates
  const recompute = useCallback(() => {
    const inputs: ModalityInput[] = [];

    const eeg = readEEGEmotion();
    if (eeg) inputs.push(eeg);

    const voice = readVoiceEmotion();
    if (voice) inputs.push(voice);

    const health = buildHealthInput(latestPayload as Record<string, unknown> | null);
    if (health) inputs.push(health);

    setLastInputs(inputs);
    setFusedEmotion(fuseModalities(inputs));
  }, [latestPayload]);

  // Listen for EEG and voice updates via custom events + localStorage
  useEffect(() => {
    recompute();

    const handler = () => recompute();
    window.addEventListener("ndw-voice-updated", handler);
    window.addEventListener("ndw-emotion-update", handler);
    window.addEventListener("ndw-eeg-updated", handler);

    return () => {
      window.removeEventListener("ndw-voice-updated", handler);
      window.removeEventListener("ndw-emotion-update", handler);
      window.removeEventListener("ndw-eeg-updated", handler);
    };
  }, [recompute]);

  // Recompute when health data changes
  useEffect(() => {
    recompute();
  }, [latestPayload, recompute]);

  const isReady = fusedEmotion !== null;

  const correctEmotion = useCallback(
    (userCorrectedEmotion: string) => {
      if (fusedEmotion && lastInputs.length > 0) {
        recordFusionFeedback(
          fusedEmotion.emotion,
          userCorrectedEmotion,
          lastInputs,
        );

        // Update personal EEG adapter with the correction
        const predictedIdx = (EEGNET_EMOTIONS as readonly string[]).indexOf(fusedEmotion.emotion);
        const correctIdx = (EEGNET_EMOTIONS as readonly string[]).indexOf(userCorrectedEmotion);
        if (predictedIdx >= 0 && correctIdx >= 0) {
          const adapter = loadPersonalAdapter();
          const updated = updateFromCorrection(adapter, predictedIdx, correctIdx);
          savePersonalAdapter(updated);
        }

        // Persist correction to Supabase + ML backend
        recordCorrection({
          userId: getParticipantId(),
          predictedEmotion: fusedEmotion.emotion,
          correctedEmotion: userCorrectedEmotion,
          source: "manual",
        }).catch(() => {});

        // Recompute immediately with updated multipliers
        recompute();
      }
    },
    [fusedEmotion, lastInputs, recompute],
  );

  return { emotion: fusedEmotion, isReady, correctEmotion };
}

/**
 * On-device health inference — derives emotional state from Apple Health data.
 * All computation is local math, no ML model needed.
 *
 * Evidence-based derivations:
 *   - HRV (RMSSD) is the gold-standard stress/recovery biomarker
 *   - Resting HR correlates inversely with parasympathetic tone
 *   - Sleep quality/duration predicts next-day mood and focus
 *   - Physical activity correlates with positive affect
 */

import type { VoiceEmotionResult } from "./voice-inference";

export interface HealthSnapshot {
  hrv?: number;                 // RMSSD ms — higher = better recovery
  restingHR?: number;           // bpm — lower at rest = better
  sleepQuality?: number;        // 0-10 scale
  sleepDuration?: number;       // hours
  steps?: number;               // today's step count
  activeEnergyBurned?: number;  // kcal
  respiratoryRate?: number;     // breaths/min
}

export interface HealthEmotionResult {
  stressIndex: number;
  recoveryScore: number;
  valence: number;
  arousal: number;
  focusScore: number;
  source: "health";
  confidence: number;
  metrics_used: string[];   // list of which metrics were available
}

export interface FusedEmotionResult {
  valence: number;
  arousal: number;
  confidence: number;
  emotion?: string;
  stress_index?: number;
  focus_index?: number;
  source: "voice_only" | "health_only" | "voice_and_health";
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/** Clamp a value to [min, max]. */
function clip(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

// ─── Health state analysis ────────────────────────────────────────────────────

/**
 * Derive emotional/cognitive state from Apple Health metrics.
 * Returns confidence-weighted estimates with evidence-based formulas.
 *
 * References:
 *   - Shaffer & Ginsberg (2017): HRV reference values and stress correlation
 *   - Walker (2017): Sleep and emotional reactivity
 *   - Warburton et al. (2006): Physical activity and mental health
 */
export function analyzeHealthState(health: HealthSnapshot): HealthEmotionResult {
  const metricsUsed: string[] = [];

  // ── Stress index ──────────────────────────────────────────────────────────
  let stressIndex: number;
  if (health.hrv !== undefined) {
    metricsUsed.push("hrv");
    // 70 ms RMSSD is considered excellent; lower = more stressed
    stressIndex = clip(1 - health.hrv / 70, 0, 1);
  } else if (health.restingHR !== undefined) {
    metricsUsed.push("restingHR");
    // 55 bpm = healthy resting HR; >95 bpm considered high stress
    stressIndex = clip((health.restingHR - 55) / 40, 0, 1);
  } else {
    // No cardiac data — return a neutral pessimistic default
    stressIndex = 0.4;
  }

  // ── Component scores ──────────────────────────────────────────────────────

  // HRV score (40% of recovery)
  let hrvScore = 0.5; // default when unavailable
  if (health.hrv !== undefined) {
    hrvScore = clip(health.hrv / 70, 0, 1);
  }

  // Sleep score (40% of recovery)
  let sleepScore = 0.5; // default when unavailable
  if (health.sleepQuality !== undefined) {
    metricsUsed.push("sleepQuality");
    sleepScore = clip(health.sleepQuality / 10, 0, 1);
  } else if (health.sleepDuration !== undefined) {
    metricsUsed.push("sleepDuration");
    sleepScore = clip(health.sleepDuration / 8, 0, 1);
  }

  // Activity score (20% of recovery)
  let activityScore = 0.5; // default when unavailable
  if (health.steps !== undefined) {
    metricsUsed.push("steps");
    // 8,000 steps is the evidence-based daily target
    activityScore = clip(health.steps / 8000, 0, 1);
  } else if (health.activeEnergyBurned !== undefined) {
    metricsUsed.push("activeEnergyBurned");
    // 400 kcal active burn ≈ moderately active day
    activityScore = clip(health.activeEnergyBurned / 400, 0, 1);
  }

  // ── Recovery score: weighted average ─────────────────────────────────────
  const recoveryScore = clip(
    hrvScore * 0.4 + sleepScore * 0.4 + activityScore * 0.2,
    0,
    1
  );

  // ── Valence: recovery + inverse stress, with slight pessimistic offset ───
  // Offset of -0.3 means you need genuinely good recovery to reach neutral valence.
  const valence = clip(recoveryScore * 0.6 + (1 - stressIndex) * 0.4 - 0.3, -1, 1);

  // ── Arousal: blend of stress and physical activity ────────────────────────
  const arousal = clip(stressIndex * 0.5 + activityScore * 0.5, 0, 1);

  // ── Focus score ───────────────────────────────────────────────────────────
  const focusScore = clip(recoveryScore * 0.7 + (1 - stressIndex) * 0.3, 0, 1);

  // ── Confidence scales with number of available metrics ───────────────────
  // Min 0.50 (1 metric), up to 0.75 (5+ metrics)
  let confidence: number;
  const n = metricsUsed.length;
  if (n >= 5) {
    confidence = 0.75;
  } else if (n >= 3) {
    confidence = 0.65;
  } else {
    confidence = 0.50;
  }

  return {
    stressIndex,
    recoveryScore,
    valence,
    arousal,
    focusScore,
    source: "health",
    confidence,
    metrics_used: metricsUsed,
  };
}

// ─── Fusion ───────────────────────────────────────────────────────────────────

/**
 * Fuse voice emotion and health-derived emotion into a single estimate.
 *
 * Weights reflect relative reliability:
 *   - Voice is more immediate but noisier
 *   - Health data is more stable but lags current state
 *
 * When both sources are available:
 *   - valence = 0.65 * voice + 0.35 * health  (voice leads on momentary affect)
 *   - arousal = 0.55 * voice + 0.45 * health  (more balanced — HRV is strong arousal signal)
 */
export function fuseVoiceAndHealth(
  voice: VoiceEmotionResult | null,
  health: HealthEmotionResult | null
): FusedEmotionResult {
  if (voice !== null && health === null) {
    return {
      valence: voice.valence,
      arousal: voice.arousal,
      confidence: voice.confidence,
      emotion: voice.emotion,
      stress_index: voice.stress_index,
      focus_index: voice.focus_index,
      source: "voice_only",
    };
  }

  if (voice === null && health !== null) {
    return {
      valence: health.valence,
      arousal: health.arousal,
      confidence: health.confidence,
      stress_index: health.stressIndex,
      focus_index: health.focusScore,
      source: "health_only",
    };
  }

  if (voice !== null && health !== null) {
    const fusedValence = 0.65 * voice.valence + 0.35 * health.valence;
    const fusedArousal = 0.55 * voice.arousal + 0.45 * health.arousal;
    // Boosted confidence for multi-modal fusion, capped at 0.85
    const fusedConfidence = clip(
      Math.max(voice.confidence, health.confidence) * 1.1,
      0,
      0.85
    );

    return {
      valence: clip(fusedValence, -1, 1),
      arousal: clip(fusedArousal, 0, 1),
      confidence: fusedConfidence,
      emotion: voice.emotion,
      stress_index: voice.stress_index,
      focus_index: voice.focus_index,
      source: "voice_and_health",
    };
  }

  // Both null — return a zero-confidence neutral result
  return {
    valence: 0,
    arousal: 0,
    confidence: 0,
    source: "voice_only",
  };
}

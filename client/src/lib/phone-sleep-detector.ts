/**
 * phone-sleep-detector.ts — Estimate sleep stage and dream likelihood from
 * phone sensors (accelerometer, microphone, time-of-night, optional HR)
 * when no EEG headband is connected.
 *
 * This is a heuristic fallback — accuracy is significantly lower than EEG-based
 * staging. The UI should always indicate "Phone-based estimate" when this is used.
 *
 * Heuristic basis:
 *   - Actigraphy research (Sadeh 2011): wrist/phone movement strongly separates
 *     Wake from sleep, but cannot distinguish N1/N2/N3/REM directly.
 *   - Time-of-night cycling: adults cycle through ~90-min NREM-REM stages.
 *     Deep sleep (N3) concentrates in the first 1-3 hours; REM grows longer
 *     toward morning (4-7h+).
 *   - Heart rate: drops in deep sleep (~10-20% below resting), elevates and
 *     becomes variable in REM.
 *   - Microphone: ambient noise / snoring patterns correlate loosely with
 *     sleep depth (heavy snoring more common in deep NREM).
 */

export interface PhoneSleepInput {
  /** Movement intensity from accelerometer variance. 0 = perfectly still, 1 = active. */
  accelerometerVariance: number;
  /** Ambient noise / snoring level from microphone. 0 = silent, 1 = loud. */
  microphoneLevel: number;
  /** Hours elapsed since the user pressed "Start Sleep Session". */
  timeOfNight: number;
  /** Optional heart rate from Health Connect / HealthKit, in BPM. */
  heartRate?: number;
}

export interface PhoneSleepEstimate {
  /** Estimated sleep stage using simplified labels. */
  likelyStage: "Wake" | "Light" | "Deep" | "REM";
  /** Confidence in the estimate (0-1). Always lower than EEG-based. */
  confidence: number;
  /** Estimated probability of active dreaming (0-1). */
  dreamLikelihood: number;
  /** Human-readable explanation of why this stage was chosen. */
  reasoning: string;
}

/**
 * Estimate sleep stage and dream likelihood from phone sensor data.
 *
 * All inputs are clamped to [0, 1] (or [0, +inf] for timeOfNight/heartRate)
 * internally — callers do not need to pre-clamp.
 */
export function estimateSleepFromPhone(input: PhoneSleepInput): PhoneSleepEstimate {
  const accel = clamp(input.accelerometerVariance, 0, 1);
  const mic = clamp(input.microphoneLevel, 0, 1);
  const time = Math.max(0, input.timeOfNight);
  const hr = input.heartRate != null ? Math.max(0, input.heartRate) : undefined;

  // ── Rule 1: High movement → Wake ─────────────────────────────────────────
  if (accel > 0.5) {
    return {
      likelyStage: "Wake",
      confidence: 0.7 + 0.2 * accel, // more movement = higher confidence
      dreamLikelihood: 0.05,
      reasoning: "High movement detected — likely awake or restless.",
    };
  }

  // ── Rule 2: Moderate movement + moderate noise → Light sleep ─────────────
  if (accel > 0.2 || mic > 0.5) {
    return {
      likelyStage: "Light",
      confidence: 0.45,
      dreamLikelihood: 0.15,
      reasoning:
        "Some movement or elevated noise — likely light sleep (brief dreams possible).",
    };
  }

  // ── From here: accel <= 0.2 and mic <= 0.5 (still and quiet) ────────────

  // If heart rate is available, use it as a strong discriminator
  if (hr !== undefined) {
    return estimateWithHeartRate(accel, mic, time, hr);
  }

  // ── No HR: rely on time-of-night cycling ────────────────────────────────
  return estimateFromTimeOfNight(accel, mic, time);
}

// ── Internal helpers ──────────────────────────────────────────────────────────

function estimateWithHeartRate(
  accel: number,
  mic: number,
  time: number,
  hr: number,
): PhoneSleepEstimate {
  // HR thresholds: resting adult ~60-70 BPM.
  // Deep sleep: HR drops to ~50-60 BPM (low + still)
  // REM: HR elevates to ~65-80 BPM and becomes irregular (elevated + still)
  // These thresholds are approximate population averages.

  const isLowHR = hr < 60;
  const isElevatedHR = hr >= 65;

  if (isLowHR && accel <= 0.1) {
    // Low HR + very still → Deep sleep
    return {
      likelyStage: "Deep",
      confidence: 0.55,
      dreamLikelihood: 0.1,
      reasoning:
        "Very still with low heart rate — likely deep sleep (dreams rare in this stage).",
    };
  }

  if (isElevatedHR && accel <= 0.15) {
    // Elevated HR + still → REM
    const dreamScore = dreamLikelihoodFromTime(time, 0.65);
    return {
      likelyStage: "REM",
      confidence: 0.5,
      dreamLikelihood: dreamScore,
      reasoning:
        "Still body with elevated heart rate — likely REM sleep (dreaming probable).",
    };
  }

  // HR in the middle range — fall through to time-based estimate
  return estimateFromTimeOfNight(accel, mic, time);
}

function estimateFromTimeOfNight(
  accel: number,
  mic: number,
  time: number,
): PhoneSleepEstimate {
  // First 0.5h: falling asleep / light sleep
  if (time < 0.5) {
    return {
      likelyStage: "Light",
      confidence: 0.4,
      dreamLikelihood: 0.1,
      reasoning: "Early in sleep session — likely still falling asleep (light sleep).",
    };
  }

  // 0.5 - 3h: deep sleep dominates the first few cycles
  if (time >= 0.5 && time < 3) {
    return {
      likelyStage: "Deep",
      confidence: 0.4,
      dreamLikelihood: 0.1,
      reasoning:
        "1-3 hours into sleep — deep sleep is most common in early cycles (dreams unlikely).",
    };
  }

  // 3 - 4h: transition zone — could be deep or entering longer REM
  if (time >= 3 && time < 4) {
    return {
      likelyStage: "Light",
      confidence: 0.35,
      dreamLikelihood: 0.3,
      reasoning:
        "Mid-sleep transition — cycling between stages, some dreams possible.",
    };
  }

  // 4 - 7h: REM periods lengthen significantly in later cycles
  if (time >= 4 && time < 7) {
    return {
      likelyStage: "REM",
      confidence: 0.4,
      dreamLikelihood: 0.6,
      reasoning:
        "4-7 hours into sleep — REM periods are longest here, dreaming is likely.",
    };
  }

  // 7h+: late sleep, mostly light sleep and REM, frequent brief awakenings
  return {
    likelyStage: "Light",
    confidence: 0.35,
    dreamLikelihood: 0.4,
    reasoning:
      "Late sleep (7h+) — mostly light sleep and REM fragments, dreams common.",
  };
}

/**
 * Adjust dream likelihood based on time-of-night weighting.
 * REM dream probability increases as the night progresses.
 */
function dreamLikelihoodFromTime(time: number, base: number): number {
  // REM dream probability scales up with time — peaks around 5-6h
  if (time < 1) return clamp(base * 0.3, 0, 1);
  if (time < 3) return clamp(base * 0.5, 0, 1);
  if (time < 5) return clamp(base * 0.8, 0, 1);
  return clamp(base, 0, 1);
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

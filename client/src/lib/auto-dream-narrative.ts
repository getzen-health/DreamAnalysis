/**
 * Auto-generate a dream narrative from EEG session data — Issue #545.
 *
 * Produces a human-readable summary of a sleep session's dream activity
 * based purely on EEG-derived metrics (REM percentage, emotional valence,
 * arousal, dream episode count, etc.). No LLM needed — template-based.
 */

// ── Public types ──────────────────────────────────────────────────────────────

export interface EEGDreamData {
  /** Percentage of total sleep spent in REM (0-100) */
  remPercentage: number;
  /** Average emotional valence across session (-1 to 1) */
  avgValence: number;
  /** Average arousal level across session (0 to 1) */
  avgArousal: number;
  /** Count of detected dream episodes */
  dreamEpisodes: number;
  /** Dominant emotion detected (e.g. "happy", "neutral", "anxious") */
  dominantEmotion: string;
  /** Total sleep duration in hours */
  sleepDuration: number;
  /** Percentage of total sleep spent in deep sleep / N3 (0-100) */
  deepSleepPct: number;
}

// ── Thresholds ────────────────────────────────────────────────────────────────

const HIGH_REM_THRESHOLD = 20; // percent — normal adult REM is ~20-25%
const HIGH_AROUSAL_THRESHOLD = 0.6;
const POSITIVE_VALENCE_THRESHOLD = 0.15;
const NEGATIVE_VALENCE_THRESHOLD = -0.15;
const HIGH_DEEP_SLEEP_THRESHOLD = 20; // percent

// ── Narrative generator ───────────────────────────────────────────────────────

/**
 * Generate a plain-text dream narrative from EEG sleep session data.
 *
 * Uses template-based logic keyed on REM%, valence, arousal, and dream
 * episode count. Returns a short paragraph suitable for display in the
 * sleep session summary card.
 */
export function generateAutoDreamNarrative(data: EEGDreamData): string {
  const {
    remPercentage,
    avgValence,
    avgArousal,
    dreamEpisodes,
    dominantEmotion,
    sleepDuration,
    deepSleepPct,
  } = data;

  const durationStr = formatDuration(sleepDuration);
  const episodeWord = dreamEpisodes === 1 ? "episode" : "episodes";

  // ── No dream episodes detected ────────────────────────────────────────
  if (dreamEpisodes === 0) {
    if (deepSleepPct >= HIGH_DEEP_SLEEP_THRESHOLD) {
      return (
        `No dream episodes detected during your ${durationStr} session. ` +
        `Your sleep was primarily deep and restorative, with ${Math.round(deepSleepPct)}% spent in slow-wave sleep.`
      );
    }
    return (
      `No dream episodes detected during your ${durationStr} session. ` +
      `Your brain focused on recovery rather than dreaming tonight.`
    );
  }

  // ── Low REM — quiet night for dreams ──────────────────────────────────
  if (remPercentage < HIGH_REM_THRESHOLD) {
    return (
      `A quiet night for dreams — deep sleep dominated your ${durationStr} session. ` +
      `${dreamEpisodes} brief dream ${episodeWord} detected, but your brain focused on physical recovery ` +
      `with ${Math.round(deepSleepPct)}% deep sleep.`
    );
  }

  // ── High REM + positive valence ───────────────────────────────────────
  if (avgValence > POSITIVE_VALENCE_THRESHOLD) {
    const arousalNote =
      avgArousal > HIGH_AROUSAL_THRESHOLD
        ? "vivid and energetic"
        : "calm and pleasant";
    return (
      `Your brain was active with ${arousalNote} dreams across ${durationStr}. ` +
      `${dreamEpisodes} dream ${episodeWord} detected with a positive emotional tone ` +
      `(dominant emotion: ${dominantEmotion}). ` +
      `REM sleep made up ${Math.round(remPercentage)}% of your night.`
    );
  }

  // ── High REM + negative valence ───────────────────────────────────────
  if (avgValence < NEGATIVE_VALENCE_THRESHOLD) {
    const intensityNote =
      avgArousal > HIGH_AROUSAL_THRESHOLD
        ? "with heightened arousal — your brain was processing something significant"
        : "though arousal stayed moderate — more melancholic than distressing";
    return (
      `An emotionally intense night — your brain processed ${dreamEpisodes} dream ${episodeWord} ` +
      `across ${durationStr} ${intensityNote}. ` +
      `Dominant emotion: ${dominantEmotion}. ` +
      `REM sleep accounted for ${Math.round(remPercentage)}% of the session.`
    );
  }

  // ── High REM + neutral valence ────────────────────────────────────────
  const neutralNote =
    avgArousal > HIGH_AROUSAL_THRESHOLD
      ? "Your dreams were emotionally neutral but mentally active"
      : "Your dreams were gentle and emotionally balanced";
  return (
    `${neutralNote} across ${durationStr}. ` +
    `${dreamEpisodes} dream ${episodeWord} detected during ${Math.round(remPercentage)}% REM sleep. ` +
    `Dominant emotion: ${dominantEmotion}.`
  );
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function formatDuration(hours: number): string {
  if (hours <= 0) return "0m";
  const h = Math.floor(hours);
  const m = Math.round((hours - h) * 60);
  if (h === 0) return `${m}m`;
  if (m === 0) return `${h}h`;
  return `${h}h ${m}m`;
}

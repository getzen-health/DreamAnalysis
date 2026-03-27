/**
 * Dream + Biometric Fusion — connects dream content to overnight physiology.
 *
 * Pure function: takes a dream entry + overnight health data → fusion insight.
 * No side effects, no API calls.
 */

// ── Interfaces ───────────────────────────────────────────────────────────────

export interface DreamEntry {
  dreamText: string;
  emotions: string[];         // ["anxiety", "wonder"]
  lucidityScore?: number;     // 0-1
  sleepQuality?: number;      // 0-100
  timestamp: string;
}

export interface OvernightBiometrics {
  avgHeartRate?: number;
  minHeartRate?: number;
  hrvSdnn?: number;
  sleepDuration?: number;     // hours
  sleepEfficiency?: number;   // 0-100
  deepSleepPct?: number;      // 0-100
  remSleepPct?: number;       // 0-100
}

export interface BiometricHighlight {
  label: string;
  value: string;
  status: "normal" | "elevated" | "low";
}

export interface DreamFusionInsight {
  headline: string;
  body: string;
  dreamEmotions: string[];
  biometricHighlights: BiometricHighlight[];
  sleepContext: string;
}

// ── Constants ────────────────────────────────────────────────────────────────

const NEGATIVE_EMOTIONS = new Set([
  "anxiety", "fear", "sadness", "anger", "frustration",
  "panic", "dread", "grief", "terror", "despair",
  "worry", "stress", "nightmare", "scared", "nervous",
]);

const POSITIVE_EMOTIONS = new Set([
  "joy", "wonder", "happiness", "calm", "peace",
  "love", "excitement", "gratitude", "serenity", "bliss",
  "awe", "hope", "contentment", "delight", "euphoria",
]);

const DEFAULT_BASELINE_HR = 65;

// ── Helpers ──────────────────────────────────────────────────────────────────

function hasNegativeEmotions(emotions: string[]): boolean {
  return emotions.some((e) => NEGATIVE_EMOTIONS.has(e.toLowerCase()));
}

function hasPositiveEmotions(emotions: string[]): boolean {
  return emotions.some((e) => POSITIVE_EMOTIONS.has(e.toLowerCase()));
}

function hrStatus(hr: number, baseline: number): "normal" | "elevated" | "low" {
  if (hr > baseline + 5) return "elevated";
  if (hr < baseline - 8) return "low";
  return "normal";
}

function hrvStatus(hrv: number): "normal" | "elevated" | "low" {
  if (hrv >= 50) return "normal";
  if (hrv >= 30) return "elevated"; // reduced recovery — amber
  return "low";                      // poor recovery — red
}

function deepSleepStatus(pct: number): "normal" | "elevated" | "low" {
  if (pct >= 20) return "normal";
  if (pct >= 13) return "elevated"; // below ideal — amber
  return "low";                      // poor deep sleep — red
}

function remSleepStatus(pct: number): "normal" | "elevated" | "low" {
  if (pct >= 25) return "elevated";
  if (pct >= 18) return "normal";
  return "low";
}

function round1(n: number): string {
  return Number(n.toFixed(1)).toString();
}

// ── Build body narrative ─────────────────────────────────────────────────────

function buildBody(bio: OvernightBiometrics, baselineHr: number): string {
  const parts: string[] = [];

  if (bio.avgHeartRate != null) {
    const diff = bio.avgHeartRate - baselineHr;
    const comparison = diff > 2
      ? `${Math.round(Math.abs(diff))} bpm above your usual`
      : diff < -2
        ? `${Math.round(Math.abs(diff))} bpm below your usual`
        : "right around your baseline";
    parts.push(
      `Your heart rate averaged ${Math.round(bio.avgHeartRate)} bpm \u2014 ${comparison}.`,
    );
  }

  if (bio.hrvSdnn != null) {
    const suggestion = bio.hrvSdnn >= 50
      ? "suggesting good recovery"
      : bio.hrvSdnn >= 30
        ? "suggesting moderate nervous system activity"
        : "suggesting your nervous system was active during this dream";
    parts.push(`HRV was ${Math.round(bio.hrvSdnn)}ms, ${suggestion}.`);
  }

  if (bio.deepSleepPct != null && bio.remSleepPct != null) {
    parts.push(
      `You spent ${Math.round(bio.deepSleepPct)}% in deep sleep and ${Math.round(bio.remSleepPct)}% in REM.`,
    );
  } else if (bio.deepSleepPct != null) {
    parts.push(`You spent ${Math.round(bio.deepSleepPct)}% in deep sleep.`);
  } else if (bio.remSleepPct != null) {
    parts.push(`You spent ${Math.round(bio.remSleepPct)}% in REM.`);
  }

  return parts.join(" ") || "No biometric data available for this night.";
}

// ── Build sleep context ──────────────────────────────────────────────────────

function buildSleepContext(bio: OvernightBiometrics): string {
  const parts: string[] = [];
  if (bio.sleepDuration != null) parts.push(`${round1(bio.sleepDuration)}h sleep`);
  if (bio.deepSleepPct != null) parts.push(`${Math.round(bio.deepSleepPct)}% deep`);
  if (bio.remSleepPct != null) parts.push(`${Math.round(bio.remSleepPct)}% REM`);
  if (bio.sleepEfficiency != null) parts.push(`${Math.round(bio.sleepEfficiency)}% efficiency`);
  return parts.join(", ") || "No sleep data";
}

// ── Build biometric highlights ───────────────────────────────────────────────

function buildHighlights(
  bio: OvernightBiometrics,
  baselineHr: number,
): BiometricHighlight[] {
  const highlights: BiometricHighlight[] = [];

  if (bio.avgHeartRate != null) {
    highlights.push({
      label: "HR",
      value: `${Math.round(bio.avgHeartRate)} bpm`,
      status: hrStatus(bio.avgHeartRate, baselineHr),
    });
  }

  if (bio.hrvSdnn != null) {
    highlights.push({
      label: "HRV",
      value: `${Math.round(bio.hrvSdnn)}ms`,
      status: hrvStatus(bio.hrvSdnn),
    });
  }

  if (bio.deepSleepPct != null) {
    highlights.push({
      label: "Deep",
      value: `${Math.round(bio.deepSleepPct)}%`,
      status: deepSleepStatus(bio.deepSleepPct),
    });
  }

  if (bio.remSleepPct != null) {
    highlights.push({
      label: "REM",
      value: `${Math.round(bio.remSleepPct)}%`,
      status: remSleepStatus(bio.remSleepPct),
    });
  }

  return highlights;
}

// ── Determine headline ───────────────────────────────────────────────────────

function determineHeadline(
  dream: DreamEntry,
  bio: OvernightBiometrics,
  baselineHr: number,
): string {
  const negEmo = hasNegativeEmotions(dream.emotions);
  const posEmo = hasPositiveEmotions(dream.emotions);
  const hrElevated = bio.avgHeartRate != null && bio.avgHeartRate > baselineHr + 5;
  const goodHrv = bio.hrvSdnn != null && bio.hrvSdnn >= 50;
  const highLucidity = (dream.lucidityScore ?? 0) >= 0.6;
  const highRem = (bio.remSleepPct ?? 0) >= 25;
  const poorSleep = (dream.sleepQuality ?? 100) < 40;

  // Lucid dream + extended REM
  if (highLucidity && highRem) {
    return "Lucid dream detected \u2014 extended REM phase";
  }

  // Anxious dream + restless body
  if (negEmo && hrElevated) {
    return "Anxious dream, restless body";
  }

  // Peaceful dream + recovered body
  if (posEmo && goodHrv) {
    return "Peaceful dream, recovered body";
  }

  // Disrupted night
  if (poorSleep && negEmo) {
    return "Disrupted night \u2014 both sleep and dreams suffered";
  }

  // Negative emotions only
  if (negEmo) {
    return "Uneasy dream, body coping";
  }

  // Positive emotions only
  if (posEmo) {
    return "Pleasant dream, restful night";
  }

  return "Last night\u2019s dream";
}

// ── Main fusion function ─────────────────────────────────────────────────────

export function fuseDreamBiometrics(
  dream: DreamEntry,
  bio: OvernightBiometrics,
  baselineHr?: number,
): DreamFusionInsight | null {
  if (!dream.dreamText || dream.dreamText.trim().length === 0) {
    return null;
  }

  const baseline = baselineHr ?? DEFAULT_BASELINE_HR;

  return {
    headline: determineHeadline(dream, bio, baseline),
    body: buildBody(bio, baseline),
    dreamEmotions: dream.emotions,
    biometricHighlights: buildHighlights(bio, baseline),
    sleepContext: buildSleepContext(bio),
  };
}

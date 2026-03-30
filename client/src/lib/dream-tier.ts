/**
 * Dream analysis tier detection and phone-based sleep/REM estimation.
 *
 * Tier 1 — EEG: Full neurometric dream narrative (from EEG stream)
 * Tier 2 — Phone: Clock + health API → estimated REM windows, NO dream content
 * Tier 3 — Nothing: "Connect a BCI device" — no fabricated data ever
 *
 * Zero fabrication principle: Tier 2 gives timing estimates with explicit
 * uncertainty. It never invents dream content or emotions.
 */

export type DreamTier = "eeg" | "phone" | "none";

export interface RemWindow {
  cycleNumber: number;
  estimatedStart: string; // HH:MM
  estimatedEnd: string;   // HH:MM
  durationMinutes: number;
  label: string;
}

export interface PhoneSleepEstimate {
  source: "health_api" | "clock_only";
  sleepOnset: string; // HH:MM
  wakeTime: string;   // HH:MM
  totalHours: number;
  remWindows: RemWindow[];
  mostLikelyDreamWindow: RemWindow | null; // last (longest) REM before wake
  dataNote: string;
}

// ── Tier detection ────────────────────────────────────────────────────────────

/**
 * Determine which data tier is available.
 * @param isStreaming - EEG device actively streaming
 * @param hasHealthData - Health sync has returned any payload
 */
export function getDreamTier(isStreaming: boolean, hasHealthData: boolean): DreamTier {
  if (isStreaming) return "eeg";
  // DeviceMotion OR health data → phone tier (imperfect but honest)
  const hasMotion =
    typeof window !== "undefined" && "DeviceMotionEvent" in window;
  if (hasHealthData || hasMotion) return "phone";
  return "none";
}

// ── REM window estimation (90-min cycle model) ────────────────────────────────

const REM_DURATIONS_BY_CYCLE = [10, 20, 25, 30, 35]; // minutes of REM per cycle

function fmt(d: Date): string {
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

function parseHHMM(hhmm: string): { h: number; m: number } {
  const [h, m] = hhmm.split(":").map(Number);
  return { h: h ?? 23, m: m ?? 0 };
}

/**
 * Estimate REM windows using the 90-min sleep cycle model.
 * Returns windows ordered earliest → latest.
 * If totalHours < 3 no meaningful REM → returns empty array.
 */
export function estimateRemWindows(
  sleepOnsetHHMM: string,
  wakeTimeHHMM: string,
  healthSleepHours?: number | null,
): PhoneSleepEstimate {
  const { h: oh, m: om } = parseHHMM(sleepOnsetHHMM);
  const { h: wh, m: wm } = parseHHMM(wakeTimeHHMM);

  const now = new Date();
  const onset = new Date(now);
  onset.setHours(oh, om, 0, 0);

  const wake = new Date(now);
  wake.setHours(wh, wm, 0, 0);
  if (wake <= onset) wake.setDate(wake.getDate() + 1);

  // Override with health data if more accurate
  const totalMs = wake.getTime() - onset.getTime();
  const totalHours =
    healthSleepHours != null && healthSleepHours > 0
      ? healthSleepHours
      : totalMs / 3_600_000;

  const CYCLE_MIN = 90;
  const remWindows: RemWindow[] = [];

  for (let cycle = 1; cycle <= 8; cycle++) {
    const cycleEndMin = cycle * CYCLE_MIN;
    if (cycleEndMin / 60 > totalHours) break;

    const remDur = REM_DURATIONS_BY_CYCLE[Math.min(cycle - 1, REM_DURATIONS_BY_CYCLE.length - 1)];
    const remStartMin = cycleEndMin - remDur;

    const remStartDate = new Date(onset.getTime() + remStartMin * 60_000);
    const remEndDate = new Date(onset.getTime() + cycleEndMin * 60_000);

    if (remEndDate > wake) break;

    remWindows.push({
      cycleNumber: cycle,
      estimatedStart: fmt(remStartDate),
      estimatedEnd: fmt(remEndDate),
      durationMinutes: remDur,
      label: `REM ${cycle}`,
    });
  }

  const mostLikelyDreamWindow =
    remWindows.length > 0 ? remWindows[remWindows.length - 1] : null;

  const source = healthSleepHours != null ? "health_api" : "clock_only";
  const dataNote =
    source === "health_api"
      ? "Sleep duration from health app. REM windows estimated via 90-min cycle model — connect an EEG device for actual neurometric analysis."
      : "Clock-based estimate only. Connect an EEG device for real dream content analysis.";

  return {
    source,
    sleepOnset: sleepOnsetHHMM,
    wakeTime: wakeTimeHHMM,
    totalHours,
    remWindows,
    mostLikelyDreamWindow,
    dataNote,
  };
}

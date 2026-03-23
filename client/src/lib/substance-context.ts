/**
 * Pre-session substance/medication context for EEG baseline adjustment.
 *
 * Caffeine increases beta/decreases alpha. SSRIs decrease alpha/theta, increase beta.
 * Benzodiazepines increase beta. Without substance context, emotion readings are confounded.
 *
 * Storage: localStorage (ndw_substance_log) — never sent to any server.
 */

// ── Types ────────────────────────────────────────────────────────────────

export interface SubstanceLog {
  timestamp: string;
  caffeine: "none" | "1-2hrs_ago" | "3-6hrs_ago" | "6+hrs_ago";
  alcohol: "none" | "last_night" | "today";
  medications: string[]; // free text array, stored locally
  cannabis: "none" | "today" | "yesterday";
}

export interface BaselineAdjustment {
  alphaOffset: number;  // caffeine reduces alpha
  betaOffset: number;   // caffeine/SSRIs increase beta
  thetaOffset: number;  // SSRIs reduce theta
  note: string;         // human-readable explanation
}

// ── Constants ────────────────────────────────────────────────────────────

const STORAGE_KEY = "ndw_substance_log";

/**
 * SSRI medication keywords — matched case-insensitively against the
 * medications free-text array.
 */
const SSRI_KEYWORDS = [
  "ssri",
  "sertraline",
  "fluoxetine",
  "paroxetine",
  "citalopram",
  "escitalopram",
  "fluvoxamine",
  "venlafaxine",
  "duloxetine",
  "zoloft",
  "prozac",
  "paxil",
  "celexa",
  "lexapro",
  "luvox",
  "effexor",
  "cymbalta",
];

// ── Persistence (localStorage) ───────────────────────────────────────────

export function saveSubstanceLog(log: SubstanceLog): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(log));
  } catch {
    // localStorage full or unavailable
  }
}

export function getLatestSubstanceLog(): SubstanceLog | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object" || !parsed.timestamp) {
      return null;
    }
    return parsed as SubstanceLog;
  } catch {
    return null;
  }
}

// ── Once-per-day check ──────────────────────────────────────────────────

/**
 * Returns true if a substance log was already saved today (same calendar day).
 */
export function hasAnsweredToday(): boolean {
  const log = getLatestSubstanceLog();
  if (!log) return false;

  const logDate = new Date(log.timestamp);
  const now = new Date();

  return (
    logDate.getUTCFullYear() === now.getUTCFullYear() &&
    logDate.getUTCMonth() === now.getUTCMonth() &&
    logDate.getUTCDate() === now.getUTCDate()
  );
}

// ── Baseline adjustment logic ───────────────────────────────────────────

/**
 * Compute EEG baseline adjustments based on reported substances.
 *
 * Returns null if log is null (user skipped the questionnaire — no
 * adjustments should be applied).
 *
 * Offsets are expressed as proportional changes:
 *   -0.15 means "reduce by 15%"
 *   +0.20 means "increase by 20%"
 */
export function getBaselineAdjustment(
  log: SubstanceLog | null,
): BaselineAdjustment | null {
  if (log === null) return null;

  let alphaOffset = 0;
  let betaOffset = 0;
  let thetaOffset = 0;
  const notes: string[] = [];

  // ── Caffeine ────────────────────────────────────────────────────────
  if (log.caffeine === "1-2hrs_ago") {
    alphaOffset += -0.15;
    betaOffset += 0.20;
    notes.push("Caffeine may affect readings for 4-6 hours");
  } else if (log.caffeine === "3-6hrs_ago") {
    alphaOffset += -0.08;
    betaOffset += 0.10;
    notes.push("Residual caffeine may slightly affect readings");
  }

  // ── Alcohol ─────────────────────────────────────────────────────────
  if (log.alcohol === "last_night") {
    alphaOffset += -0.10;
    thetaOffset += 0.10;
    notes.push("Alcohol from last night may affect alpha/theta balance");
  } else if (log.alcohol === "today") {
    alphaOffset += -0.15;
    thetaOffset += 0.15;
    betaOffset += -0.10;
    notes.push("Alcohol significantly affects EEG readings");
  }

  // ── Medications (SSRI detection) ────────────────────────────────────
  const hasSSRI = log.medications.some((med) => {
    const lower = med.toLowerCase();
    return SSRI_KEYWORDS.some((kw) => lower.includes(kw));
  });

  if (hasSSRI) {
    alphaOffset += -0.10;
    betaOffset += 0.15;
    thetaOffset += -0.10;
    notes.push("SSRI medications affect alpha, beta, and theta bands");
  }

  // ── Cannabis ────────────────────────────────────────────────────────
  if (log.cannabis === "today") {
    alphaOffset += -0.10;
    thetaOffset += 0.15;
    notes.push("Cannabis may increase theta and reduce alpha");
  } else if (log.cannabis === "yesterday") {
    alphaOffset += -0.05;
    thetaOffset += 0.05;
    notes.push("Residual cannabis effects may slightly alter readings");
  }

  // Round to avoid floating-point drift
  alphaOffset = Math.round(alphaOffset * 100) / 100;
  betaOffset = Math.round(betaOffset * 100) / 100;
  thetaOffset = Math.round(thetaOffset * 100) / 100;

  return {
    alphaOffset,
    betaOffset,
    thetaOffset,
    note: notes.join(". "),
  };
}

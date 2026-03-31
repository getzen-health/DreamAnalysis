/**
 * nightmare-recurrence.ts
 *
 * Pure functions for recurring nightmare detection and IRT effectiveness display.
 * No network calls — all logic runs on data already fetched from
 * GET /api/nightmare-recurrence/:userId.
 */

export type NightmareTrend = "improving" | "stable" | "worsening" | "unknown";

export interface NightmareRecurrenceData {
  /** Nightmares in the most recent 7-day window */
  recentNightmares: number;
  /** Nightmares in the prior 7-day window (days 8-14) */
  olderNightmares: number;
  trend: NightmareTrend;
  /** ISO timestamp of the most recent nightmare */
  lastNightmareDate: string | null;
  /** Total IRT sessions saved by this user */
  irtSessionCount: number;
  /** ISO timestamp of the most recent IRT session */
  lastIrtDate: string | null;
  /** Nightmares that occurred AFTER the last IRT session */
  postIrtNightmares: number;
}

/** Determine trend direction from two 7-day window counts. */
export function computeNightmareTrend(
  recentCount: number,
  olderCount: number,
): NightmareTrend {
  if (recentCount + olderCount < 2) return "unknown";
  if (recentCount < olderCount) return "improving";
  if (recentCount > olderCount) return "worsening";
  return "stable";
}

/** Human-readable one-line description of the trend. */
export function trendLabel(trend: NightmareTrend): string {
  switch (trend) {
    case "improving":
      return "Nightmare frequency is decreasing";
    case "worsening":
      return "Nightmare frequency is increasing";
    case "stable":
      return "Nightmare frequency is unchanged";
    case "unknown":
      return "Not enough data to show a trend yet";
  }
}

/** Summarise IRT effectiveness as a short phrase. */
export function irtEffectivenessLabel(
  postIrtNightmares: number,
  irtSessionCount: number,
): string {
  if (irtSessionCount === 0) return "";
  if (postIrtNightmares === 0) return "No nightmares since your last IRT session";
  if (postIrtNightmares === 1)
    return "1 nightmare since your last IRT session";
  return `${postIrtNightmares} nightmares since your last IRT session`;
}

/**
 * Returns true when the nightmare situation warrants showing the IRT CTA:
 *  - trend is worsening or stable (not improving) AND recent nightmares > 0
 *  - OR: postIrt nightmares > 0 (nightmare recurred after IRT — needs repeat session)
 */
export function shouldSuggestIrt(data: NightmareRecurrenceData): boolean {
  if (data.recentNightmares === 0) return false;
  if (data.postIrtNightmares > 0) return true;
  return data.trend === "worsening" || data.trend === "stable" || data.trend === "unknown";
}

/** Format an ISO date to a short human label, e.g. "Mar 28". */
export function formatShortDate(isoDate: string | null): string {
  if (!isoDate) return "";
  const d = new Date(isoDate);
  if (isNaN(d.getTime())) return "";
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

/**
 * dream-recall.ts
 * Pure helpers for tracking dream recall consistency.
 * No network calls; operates on DreamEntry shapes (only needs `timestamp`).
 */

export interface RecallDay {
  date: string;       // "YYYY-MM-DD"
  count: number;      // number of dreams recorded that day (0 = no recall)
  isToday: boolean;
  isFuture: boolean;
}

export interface WeeklyRecallPoint {
  weekStart: string;  // "YYYY-MM-DD" (Monday)
  rate: number;       // 0-1 fraction of days with ≥1 dream
  dreamDays: number;
  totalDays: number;
}

// ── helpers ───────────────────────────────────────────────────────────────────

function toDateStr(d: Date): string {
  // Always produce a UTC-based YYYY-MM-DD so it matches timestamp.slice(0,10)
  return d.toISOString().slice(0, 10);
}

/** UTC-safe today: midnight UTC. */
function utcToday(): Date {
  const now = new Date();
  return new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()));
}

function addDays(d: Date, n: number): Date {
  // Use UTC arithmetic to avoid DST shifts
  return new Date(Date.UTC(
    d.getUTCFullYear(),
    d.getUTCMonth(),
    d.getUTCDate() + n,
  ));
}

/** Monday of the week containing `d` (UTC). */
function weekStart(d: Date): Date {
  const day = d.getUTCDay(); // 0=Sun … 6=Sat
  const diff = (day === 0 ? -6 : 1 - day); // shift to Monday
  return addDays(d, diff);
}

// ── calendar grid ─────────────────────────────────────────────────────────────

/**
 * Build a flat list of `RecallDay` entries covering the last `days` days
 * (oldest first, ending today).
 * Each entry carries the count of dreams recorded on that day.
 */
export function buildRecallCalendar(
  dreams: { timestamp: string }[],
  days = 28,
): RecallDay[] {
  const today = utcToday();

  // Build a lookup: "YYYY-MM-DD" → count
  const counts = new Map<string, number>();
  for (const d of dreams) {
    const key = d.timestamp.slice(0, 10);
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }

  const todayStr = toDateStr(today);
  const result: RecallDay[] = [];

  for (let i = days - 1; i >= 0; i--) {
    const day = addDays(today, -i);
    const date = toDateStr(day);
    result.push({
      date,
      count: counts.get(date) ?? 0,
      isToday: date === todayStr,
      isFuture: false,
    });
  }

  return result;
}

// ── streak ────────────────────────────────────────────────────────────────────

/**
 * Count consecutive days ending today (or yesterday if today has no entry)
 * where at least one dream was recorded.
 */
export function computeRecallStreak(dreams: { timestamp: string }[]): number {
  const seen = new Set(dreams.map((d) => d.timestamp.slice(0, 10)));
  const today = utcToday();

  let streak = 0;
  let current = new Date(today);

  // If today has no entry, allow starting from yesterday
  if (!seen.has(toDateStr(current))) {
    current = addDays(current, -1);
  }

  for (let i = 0; i < 365; i++) {
    if (seen.has(toDateStr(current))) {
      streak++;
      current = addDays(current, -1);
    } else {
      break;
    }
  }

  return streak;
}

// ── recall rate ───────────────────────────────────────────────────────────────

/**
 * Fraction of days in the last `days` days where at least one dream was recorded.
 * Days in the future are excluded.
 */
export function computeRecallRate(
  dreams: { timestamp: string }[],
  days = 7,
): number {
  const calendar = buildRecallCalendar(dreams, days);
  const active = calendar.filter((d) => !d.isFuture);
  if (active.length === 0) return 0;
  const withDream = active.filter((d) => d.count > 0).length;
  return withDream / active.length;
}

// ── weekly trend ──────────────────────────────────────────────────────────────

/**
 * Return per-week recall rates for the last `weeks` full weeks.
 * Weeks run Monday–Sunday. Oldest week first.
 */
export function recallWeeklyTrend(
  dreams: { timestamp: string }[],
  weeks = 4,
): WeeklyRecallPoint[] {
  const seen = new Set(dreams.map((d) => d.timestamp.slice(0, 10)));
  const today = utcToday();
  const todayStr = toDateStr(today);

  const result: WeeklyRecallPoint[] = [];

  for (let w = weeks - 1; w >= 0; w--) {
    const weekAnchor = addDays(today, -(w * 7));
    const monday = weekStart(weekAnchor);
    let dreamDays = 0;
    let totalDays = 0;

    for (let d = 0; d < 7; d++) {
      const day = addDays(monday, d);
      const dayStr = toDateStr(day);
      if (dayStr > todayStr) break; // don't count future days
      totalDays++;
      if (seen.has(dayStr)) dreamDays++;
    }

    if (totalDays === 0) continue;
    result.push({
      weekStart: toDateStr(monday),
      rate: dreamDays / totalDays,
      dreamDays,
      totalDays,
    });
  }

  return result;
}

// ── trend direction ───────────────────────────────────────────────────────────

export type RecallTrend = "improving" | "stable" | "declining" | "insufficient";

/**
 * Compare last 2 weeks of recall rate to determine trend.
 * Requires at least 2 weeks of data.
 */
export function recallTrendDirection(
  dreams: { timestamp: string }[],
): RecallTrend {
  const weeks = recallWeeklyTrend(dreams, 3);
  if (weeks.length < 2) return "insufficient";

  const prev = weeks[weeks.length - 2];
  const curr = weeks[weeks.length - 1];

  // No meaningful data if both comparison weeks have zero dream days
  if (prev.dreamDays === 0 && curr.dreamDays === 0) return "insufficient";

  const delta = curr.rate - prev.rate;
  if (delta > 0.15) return "improving";
  if (delta < -0.15) return "declining";
  return "stable";
}

// ── display helpers ───────────────────────────────────────────────────────────

/** Short "Mar 28" label for a YYYY-MM-DD string. */
export function shortDate(iso: string): string {
  const [, m, d] = iso.split("-");
  const months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  return `${months[parseInt(m, 10) - 1]} ${parseInt(d, 10)}`;
}

/** CSS class for a recall day cell based on dream count. */
export function recallCellClass(count: number, isToday: boolean): string {
  if (isToday && count === 0) return "bg-secondary/20 ring-1 ring-secondary/50";
  if (count === 0) return "bg-white/5";
  if (count === 1) return "bg-secondary/40";
  if (count === 2) return "bg-secondary/65";
  return "bg-secondary/90"; // 3+ dreams
}

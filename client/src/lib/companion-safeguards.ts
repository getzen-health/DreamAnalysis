import { sbGetSetting, sbSaveGeneric } from "./supabase-store";
/**
 * Companion usage safeguards — gentle nudges toward human connection
 * when AI chat usage becomes heavy.
 *
 * Based on Harvard/MIT 2025 research: higher AI chatbot usage correlates
 * with higher loneliness. Heavy emotional disclosure to AI = lower well-being.
 *
 * Design principles:
 * - NEVER block entirely — always allow "Continue anyway"
 * - NEVER guilt-trip — frame as caring suggestion
 * - Always offer a concrete alternative (call a friend, meditate, breathe)
 */

const STORAGE_KEY = "ndw_companion_usage";
const HEAVY_USE_THRESHOLD_MINUTES = 15;
const DAILY_LIMIT_MINUTES = 30;
const CONSECUTIVE_DAYS_THRESHOLD = 3;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface CompanionUsageStats {
  todayMinutes: number;
  todaySessions: number;
  consecutiveDaysHeavyUse: number;
  lastSessionEnd: string | null;
}

/** Internal storage format */
interface StoredUsage {
  date: string; // YYYY-MM-DD
  todayMinutes: number;
  todaySessions: number;
  consecutiveDaysHeavyUse: number;
  heavyUseDates: string[]; // YYYY-MM-DD entries
  lastSessionEnd: string | null;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function todayStr(): string {
  return new Date().toISOString().slice(0, 10);
}

function yesterdayStr(): string {
  const d = new Date();
  d.setDate(d.getDate() - 1);
  return d.toISOString().slice(0, 10);
}

function readStorage(): StoredUsage | null {
  try {
    const raw = sbGetSetting(STORAGE_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as StoredUsage;
  } catch {
    return null;
  }
}

function writeStorage(data: StoredUsage): void {
  sbSaveGeneric(STORAGE_KEY, data);
}

function emptyUsage(): StoredUsage {
  return {
    date: todayStr(),
    todayMinutes: 0,
    todaySessions: 0,
    consecutiveDaysHeavyUse: 0,
    heavyUseDates: [],
    lastSessionEnd: null,
  };
}

/**
 * If stored data is from a previous day, roll over:
 * - Reset todayMinutes and todaySessions to 0
 * - Update consecutiveDaysHeavyUse based on whether yesterday was heavy use
 */
function rolloverIfNeeded(stored: StoredUsage): StoredUsage {
  const today = todayStr();
  if (stored.date === today) return stored;

  const yesterday = yesterdayStr();
  const wasYesterdayHeavy = stored.date === yesterday && stored.todayMinutes > HEAVY_USE_THRESHOLD_MINUTES;

  const newConsecutive = wasYesterdayHeavy
    ? stored.consecutiveDaysHeavyUse + 1
    : 0;

  const rolled: StoredUsage = {
    date: today,
    todayMinutes: 0,
    todaySessions: 0,
    consecutiveDaysHeavyUse: newConsecutive,
    heavyUseDates: stored.heavyUseDates,
    lastSessionEnd: stored.lastSessionEnd,
  };

  writeStorage(rolled);
  return rolled;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Read current companion usage stats for today.
 * Automatically rolls over if the stored data is from a previous day.
 */
export function getCompanionUsage(): CompanionUsageStats {
  const stored = readStorage();
  if (!stored) {
    return {
      todayMinutes: 0,
      todaySessions: 0,
      consecutiveDaysHeavyUse: 0,
      lastSessionEnd: null,
    };
  }

  const current = rolloverIfNeeded(stored);
  return {
    todayMinutes: current.todayMinutes,
    todaySessions: current.todaySessions,
    consecutiveDaysHeavyUse: current.consecutiveDaysHeavyUse,
    lastSessionEnd: current.lastSessionEnd,
  };
}

/**
 * Record a companion chat session of the given duration.
 */
export function recordCompanionSession(durationMinutes: number): void {
  let stored = readStorage();
  if (!stored) {
    stored = emptyUsage();
  } else {
    stored = rolloverIfNeeded(stored);
  }

  stored.todayMinutes += durationMinutes;
  stored.todaySessions += 1;
  stored.lastSessionEnd = new Date().toISOString();

  writeStorage(stored);
}

/**
 * Returns a gentle nudge message if usage thresholds are exceeded,
 * or null if no nudge is needed.
 *
 * Priority order (highest severity first):
 * 1. Daily limit (30+ min total)
 * 2. Consecutive heavy-use days (3+)
 * 3. Single session length (15+ min)
 */
export function getUsageNudge(stats: CompanionUsageStats): string | null {
  // Daily limit — most restrictive
  if (stats.todayMinutes >= DAILY_LIMIT_MINUTES) {
    return "Daily limit reached. The AI companion will be available again tomorrow. In the meantime, try a meditation or reach out to a friend.";
  }

  // Consecutive heavy-use days
  if (stats.consecutiveDaysHeavyUse >= CONSECUTIVE_DAYS_THRESHOLD) {
    return "You've been using AI chat a lot lately. Real human connection can be even more helpful. Consider calling someone you care about today.";
  }

  // Single session length
  if (stats.todayMinutes >= HEAVY_USE_THRESHOLD_MINUTES) {
    return "You've been chatting for a while. Consider taking a break or reaching out to a friend.";
  }

  return null;
}

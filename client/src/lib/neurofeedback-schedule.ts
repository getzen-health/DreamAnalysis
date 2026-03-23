import { sbGetSetting, sbSaveGeneric } from "./supabase-store";
/**
 * Evidence-based neurofeedback session scheduling.
 *
 * Research basis:
 * - 2025 RCT: spaced practice (2-3 day intervals) outperforms daily massed training
 * - Clinical standard: 20-50 min sessions, 8-20 sessions for neuroplastic effects
 * - 20 sessions is the evidence-based minimum for measurable change
 *
 * Phase progression:
 *   1-8  sessions = beginner (learning the skill)
 *   9-20 sessions = building (consolidation)
 *   21+  sessions = maintaining
 */

const STORAGE_KEY = "ndw_nf_sessions";

// ── Types ────────────────────────────────────────────────────────────────────

export interface SessionSchedule {
  nextSessionDate: Date | null;
  daysSinceLastSession: number;
  isOptimalWindow: boolean;   // 2-3 days since last
  isTooSoon: boolean;         // < 2 days since last
  isTooLate: boolean;         // > 5 days since last (skill decay risk)
  totalSessions: number;
  currentPhase: "beginner" | "building" | "maintaining";
  message: string;
  progressPercent: number;    // 0-100 toward 20-session milestone
}

// ── Schedule Logic ───────────────────────────────────────────────────────────

export function getSessionSchedule(sessionHistory: Date[]): SessionSchedule {
  const totalSessions = sessionHistory.length;
  const progressPercent = Math.min(100, Math.round((totalSessions / 20) * 100));

  let currentPhase: SessionSchedule["currentPhase"];
  if (totalSessions >= 21) {
    currentPhase = "maintaining";
  } else if (totalSessions >= 9) {
    currentPhase = "building";
  } else {
    currentPhase = "beginner";
  }

  // No sessions yet
  if (totalSessions === 0) {
    return {
      nextSessionDate: null,
      daysSinceLastSession: 0,
      isOptimalWindow: false,
      isTooSoon: false,
      isTooLate: false,
      totalSessions: 0,
      currentPhase: "beginner",
      message: "Ready for your first session. Start whenever you are comfortable.",
      progressPercent: 0,
    };
  }

  // Sort descending to get most recent first
  const sorted = [...sessionHistory].sort(
    (a, b) => b.getTime() - a.getTime()
  );
  const lastSession = sorted[0];
  const now = new Date();
  const msSinceLast = now.getTime() - lastSession.getTime();
  const daysSinceLastSession = msSinceLast / (1000 * 60 * 60 * 24);

  // Recommended next session: 2-3 days after last session
  const nextSessionDate = new Date(
    lastSession.getTime() + 2.5 * 24 * 60 * 60 * 1000
  );

  // Use calendar day difference (avoids noon-to-noon fractional issues)
  const nowMidnight = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const lastMidnight = new Date(lastSession.getFullYear(), lastSession.getMonth(), lastSession.getDate());
  const wholeDays = Math.round((nowMidnight.getTime() - lastMidnight.getTime()) / (1000 * 60 * 60 * 24));
  const isTooSoon = wholeDays < 2;
  const isOptimalWindow = wholeDays >= 2 && wholeDays <= 3;
  const isGettingLate = wholeDays >= 4 && wholeDays <= 5;
  const isTooLate = wholeDays > 5;

  let message: string;
  if (isTooSoon) {
    message =
      "Your brain is still consolidating. Rest today for better results next session.";
  } else if (isOptimalWindow) {
    message =
      "Great timing! Your brain is ready for the next session.";
  } else if (isGettingLate) {
    message =
      "Time for your next session -- consistency matters for neuroplastic changes.";
  } else {
    // isTooLate
    message =
      "It's been a while. Pick up where you left off -- your progress is still there.";
  }

  return {
    nextSessionDate,
    daysSinceLastSession,
    isOptimalWindow,
    isTooSoon,
    isTooLate,
    totalSessions,
    currentPhase,
    message,
    progressPercent,
  };
}

// ── Persistence ──────────────────────────────────────────────────────────────

export function recordNeurofeedbackSession(): void {
  const sessions = JSON.parse(
    sbGetSetting(STORAGE_KEY) || "[]"
  ) as string[];
  sessions.push(new Date().toISOString());
  sbSaveGeneric(STORAGE_KEY, sessions);
}

export function getSessionHistory(): Date[] {
  const raw = JSON.parse(
    sbGetSetting(STORAGE_KEY) || "[]"
  ) as string[];
  return raw.map((s) => new Date(s));
}

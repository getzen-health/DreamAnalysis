import { sbGetSetting, sbSaveSetting } from "./supabase-store";
/**
 * meditation-narrative.ts — Story-based meditation progression system (#530)
 *
 * Maps cumulative session count to narrative chapters. Each chapter has a theme,
 * description, and unlock criteria (minimum sessions). Replaces raw streak counts
 * with a meaningful journey metaphor that deepens engagement.
 *
 * 10 chapters, loosely inspired by contemplative tradition stage models
 * (Theravada jhana factors, Zen ox-herding pictures) but kept secular.
 */

// ── Types ──────────────────────────────────────────────────────────────────

export interface NarrativeChapter {
  /** Chapter number (1-10) */
  chapter: number;
  /** Short chapter title */
  title: string;
  /** Evocative theme keyword */
  theme: string;
  /** 1-2 sentence description shown to the user */
  description: string;
  /** Minimum total sessions to unlock this chapter */
  unlockSessionCount: number;
}

export interface NarrativeProgress {
  /** The chapter the user is currently on */
  currentChapter: NarrativeChapter;
  /** How many sessions completed total */
  totalSessions: number;
  /** Sessions remaining to unlock next chapter (0 if at final chapter) */
  sessionsToNextChapter: number;
  /** Progress within current chapter as 0-1 fraction */
  chapterProgress: number;
  /** Whether the user has completed all chapters */
  journeyComplete: boolean;
}

// ── Chapter definitions ────────────────────────────────────────────────────

export const NARRATIVE_CHAPTERS: readonly NarrativeChapter[] = [
  {
    chapter: 1,
    title: "The Quiet Garden",
    theme: "stillness",
    description:
      "You step into a walled garden. The noise of the world fades. Here, the only task is to sit and notice your breath.",
    unlockSessionCount: 0,
  },
  {
    chapter: 2,
    title: "The Mountain Path",
    theme: "patience",
    description:
      "A narrow trail leads upward through mist. Each session is a step. The summit is not the goal — the walking is.",
    unlockSessionCount: 5,
  },
  {
    chapter: 3,
    title: "The Still Lake",
    theme: "clarity",
    description:
      "You reach a lake so still it mirrors the sky perfectly. When the mind quiets, reality becomes clearer.",
    unlockSessionCount: 12,
  },
  {
    chapter: 4,
    title: "The Night Sky",
    theme: "vastness",
    description:
      "Lying beneath infinite stars, the boundaries of self soften. Thoughts arise and dissolve like meteors.",
    unlockSessionCount: 20,
  },
  {
    chapter: 5,
    title: "The Deep Forest",
    theme: "courage",
    description:
      "The forest is dark and unfamiliar. Difficult emotions surface here. Sitting with discomfort is the practice.",
    unlockSessionCount: 30,
  },
  {
    chapter: 6,
    title: "The River Crossing",
    theme: "surrender",
    description:
      "A wide river blocks the path. You cannot force your way across. Let the current carry you.",
    unlockSessionCount: 45,
  },
  {
    chapter: 7,
    title: "The Inner Cave",
    theme: "insight",
    description:
      "Deep inside the mountain, silence is absolute. In this stillness, understanding arrives unbidden.",
    unlockSessionCount: 60,
  },
  {
    chapter: 8,
    title: "The Open Field",
    theme: "freedom",
    description:
      "No walls, no path, no destination. Just open space in every direction. The practice has become effortless.",
    unlockSessionCount: 80,
  },
  {
    chapter: 9,
    title: "The Return",
    theme: "integration",
    description:
      "You walk back through the garden gate. The world outside hasn't changed, but you see it differently now.",
    unlockSessionCount: 100,
  },
  {
    chapter: 10,
    title: "The Everyday",
    theme: "presence",
    description:
      "There is no special place to go. Every moment is the garden. Every breath is the practice.",
    unlockSessionCount: 130,
  },
] as const;

// ── Core logic ─────────────────────────────────────────────────────────────

/**
 * Given a total session count, return the current narrative chapter.
 * Returns the highest chapter whose unlockSessionCount <= totalSessions.
 */
export function getCurrentChapter(totalSessions: number): NarrativeChapter {
  const count = Math.max(0, Math.floor(totalSessions));
  let current = NARRATIVE_CHAPTERS[0];
  for (const chapter of NARRATIVE_CHAPTERS) {
    if (count >= chapter.unlockSessionCount) {
      current = chapter;
    } else {
      break;
    }
  }
  return current;
}

/**
 * Full narrative progress for a given session count.
 */
export function getNarrativeProgress(totalSessions: number): NarrativeProgress {
  const count = Math.max(0, Math.floor(totalSessions));
  const currentChapter = getCurrentChapter(count);
  const chapterIndex = NARRATIVE_CHAPTERS.indexOf(currentChapter);
  const isLastChapter = chapterIndex === NARRATIVE_CHAPTERS.length - 1;

  let sessionsToNextChapter = 0;
  let chapterProgress = 1;

  if (!isLastChapter) {
    const nextChapter = NARRATIVE_CHAPTERS[chapterIndex + 1];
    const rangeStart = currentChapter.unlockSessionCount;
    const rangeEnd = nextChapter.unlockSessionCount;
    const rangeSize = rangeEnd - rangeStart;
    const progressInRange = count - rangeStart;
    chapterProgress = rangeSize > 0 ? Math.min(progressInRange / rangeSize, 1) : 1;
    sessionsToNextChapter = Math.max(0, rangeEnd - count);
  }

  return {
    currentChapter,
    totalSessions: count,
    sessionsToNextChapter,
    chapterProgress,
    journeyComplete: isLastChapter && count >= currentChapter.unlockSessionCount,
  };
}

// ── Persistence helpers ────────────────────────────────────────────────────

const STORAGE_KEY = "ndw_meditation_session_count";

/** Read the stored session count from localStorage. */
export function getStoredSessionCount(): number {
  try {
    const raw = sbGetSetting(STORAGE_KEY);
    if (raw === null) return 0;
    const n = parseInt(raw, 10);
    return Number.isFinite(n) && n >= 0 ? n : 0;
  } catch {
    return 0;
  }
}

/** Increment the stored session count by 1 and return the new count. */
export function incrementSessionCount(): number {
  const current = getStoredSessionCount();
  const next = current + 1;
  try {
    sbSaveSetting(STORAGE_KEY, String(next));
  } catch {
    // localStorage full or unavailable — no-op
  }
  return next;
}

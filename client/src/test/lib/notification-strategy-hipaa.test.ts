/**
 * HIPAA compliance tests for notification content generators.
 *
 * Verifies that all notification generators produce text that is safe
 * for push notifications (no PHI visible on lock screen).
 */
import { describe, it, expect, beforeEach, vi } from "vitest";
import {
  generateMorningBrief,
  generateSessionReminder,
  generateWeeklyInsight,
  generateStreakEncouragement,
} from "@/lib/notification-strategy";
import { sanitizeNotificationText } from "@/lib/notification-sanitizer";

// ── PHI detection helper ─────────────────────────────────────────────────

/**
 * Checks if text contains specific health data that should NOT appear
 * in push notifications visible on lock screens.
 */
const PHI_CONTENT_PATTERNS = [
  /\d+\s*%\s*(?:stress|focus|relaxation|anxiety)/i,
  /(?:stress|focus|anxiety|relaxation)\s*(?:level|index|score)?[\s:]+\d/i,
  /\b(?:heart rate|HRV|bpm)\b.*\d/i,
  /\bEEG\s+(?:shows?|detected|recorded)/i,
  /\bmood\s+(?:was\s+)?detected\s+as/i,
  /\bemotion\s+detected/i,
  /\b(?:alpha|theta|beta|gamma|delta)\s+(?:waves?|power)\b/i,
];

function containsPHI(text: string): boolean {
  return PHI_CONTENT_PATTERNS.some((p) => p.test(text));
}

// ── Tests ────────────────────────────────────────────────────────────────

describe("Morning brief — HIPAA safety", () => {
  const inputs = [
    {},
    { lastEmotion: "happy" },
    { lastEmotion: "sad", streakDays: 10 },
    { streakDays: 0 },
    { lastEmotion: "angry", streakDays: 30, chronotype: "evening" },
  ];

  for (const input of inputs) {
    it(`input ${JSON.stringify(input)} — title + body survive sanitization unchanged or safely`, () => {
      const content = generateMorningBrief(input);
      const sanitizedTitle = sanitizeNotificationText(content.title);
      const sanitizedBody = sanitizeNotificationText(content.body);
      // The generated text should already be safe (sanitization is a no-op)
      expect(containsPHI(sanitizedTitle)).toBe(false);
      expect(containsPHI(sanitizedBody)).toBe(false);
    });
  }
});

describe("Session reminder — HIPAA safety", () => {
  it("generated text contains no PHI", () => {
    const content = generateSessionReminder();
    expect(containsPHI(content.title)).toBe(false);
    expect(containsPHI(content.body)).toBe(false);
  });
});

describe("Weekly insight — HIPAA safety", () => {
  const inputs = [
    { voiceCheckins: 0, neurofeedbackSessions: 0, stressTrend: "stable" as const },
    { voiceCheckins: 7, neurofeedbackSessions: 5, stressTrend: "improving" as const },
    { voiceCheckins: 1, neurofeedbackSessions: 0, stressTrend: "worsening" as const },
  ];

  for (const input of inputs) {
    it(`input trend=${input.stressTrend} — sanitized text contains no PHI`, () => {
      const content = generateWeeklyInsight(input);
      const sanitizedTitle = sanitizeNotificationText(content.title);
      const sanitizedBody = sanitizeNotificationText(content.body);
      expect(containsPHI(sanitizedTitle)).toBe(false);
      expect(containsPHI(sanitizedBody)).toBe(false);
    });
  }
});

describe("Streak encouragement — HIPAA safety", () => {
  for (const days of [0, 1, 3, 7, 14, 30, 100]) {
    it(`${days}-day streak — text contains no PHI`, () => {
      const content = generateStreakEncouragement(days);
      expect(containsPHI(content.title)).toBe(false);
      expect(containsPHI(content.body)).toBe(false);
    });
  }
});

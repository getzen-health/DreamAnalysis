/**
 * Tests that notification generators produce HIPAA-safe content.
 *
 * Push notifications are visible on lock screens. They must NEVER contain:
 * - Specific emotion labels (happy, sad, angry, etc.)
 * - Stress/health trend descriptions (stress improving, worsening)
 * - Specific health metric values (heart rate, HRV, etc.)
 *
 * Generators should use generic language that prompts users to open the app.
 */
import { describe, it, expect, beforeEach, vi } from "vitest";
import {
  generateMorningBrief,
  generateWeeklyInsight,
} from "@/lib/notification-strategy";

const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: vi.fn((key: string) => store[key] ?? null),
    setItem: vi.fn((key: string, value: string) => { store[key] = value; }),
    removeItem: vi.fn((key: string) => { delete store[key]; }),
    clear: vi.fn(() => { store = {}; }),
  };
})();
Object.defineProperty(globalThis, "localStorage", { value: localStorageMock });

beforeEach(() => {
  localStorageMock.clear();
  vi.clearAllMocks();
});

// Emotion words that should NEVER appear in push notification text
const EMOTION_WORDS = [
  "happy", "sad", "angry", "fearful", "fear", "anxious",
  "depressed", "excited", "surprised", "neutral", "stressed",
  "disgusted", "contempt",
];

function containsEmotionLabel(text: string): boolean {
  const lower = text.toLowerCase();
  return EMOTION_WORDS.some((e) => {
    // Match whole words only, not substrings like "neutral" in "neutralize"
    const regex = new RegExp(`\\b${e}\\b`, "i");
    return regex.test(lower);
  });
}

describe("generateMorningBrief — no emotion labels in output", () => {
  it("does not include lastEmotion value in title or body", () => {
    for (const emotion of ["happy", "sad", "angry", "fearful", "neutral"]) {
      const content = generateMorningBrief({ lastEmotion: emotion });
      expect(containsEmotionLabel(content.title)).toBe(false);
      expect(containsEmotionLabel(content.body)).toBe(false);
    }
  });

  it("does not include emotion when combined with streak", () => {
    const content = generateMorningBrief({ lastEmotion: "sad", streakDays: 10 });
    expect(containsEmotionLabel(content.body)).toBe(false);
  });
});

describe("generateWeeklyInsight — no stress trend in output", () => {
  it("does not include 'stress trend' wording in body for worsening", () => {
    const content = generateWeeklyInsight({
      voiceCheckins: 3,
      neurofeedbackSessions: 1,
      stressTrend: "worsening",
    });
    expect(content.body).not.toMatch(/stress trend/i);
  });

  it("does not include 'stress trend' wording in body for improving", () => {
    const content = generateWeeklyInsight({
      voiceCheckins: 5,
      neurofeedbackSessions: 3,
      stressTrend: "improving",
    });
    expect(content.body).not.toMatch(/stress trend/i);
  });
});

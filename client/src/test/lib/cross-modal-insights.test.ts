import { describe, it, expect } from "vitest";
import { generateInsights, type CorrelationInput, type PersonalInsight } from "@/lib/cross-modal-insights";

// ── Helpers ────────────────────────────────────────────────────────────────

function emptyInput(): CorrelationInput {
  return {
    emotionHistory: [],
    foodLogs: [],
    sleepData: [],
    steps: [],
  };
}

/**
 * All helpers use UTC dates to avoid timezone issues in tests.
 * The library uses toISOString().slice(0,10) for date keys, which is UTC-based.
 */

/** Parse "YYYY-MM-DD" into [year, month(0-indexed), day] */
function parseDate(s: string): [number, number, number] {
  const [y, m, d] = s.split("-").map(Number);
  return [y, m - 1, d];
}

/** Create N emotion readings at a specific UTC hour on consecutive days. */
function makeEmotionHistory(
  n: number,
  overrides: Partial<{ stress: number; focus: number; valence: number; energy: number; hour: number; startDate: string }> = {},
): CorrelationInput["emotionHistory"] {
  const { stress = 0.5, focus = 0.5, valence = 0, energy = 0.5, hour = 10, startDate = "2026-01-01" } = overrides;
  const [y, m, d] = parseDate(startDate);
  return Array.from({ length: n }, (_, i) => {
    const ts = new Date(Date.UTC(y, m, d + i, hour, 0, 0));
    return { stress, focus, valence, energy, timestamp: ts.toISOString() };
  });
}

function makeSleepData(
  n: number,
  overrides: Partial<{ quality: number; duration: number; startDate: string }> = {},
): CorrelationInput["sleepData"] {
  const { quality = 70, duration = 7, startDate = "2026-01-01" } = overrides;
  const [y, m, d] = parseDate(startDate);
  return Array.from({ length: n }, (_, i) => {
    const ts = new Date(Date.UTC(y, m, d + i, 12, 0, 0));
    return { quality, duration, date: ts.toISOString().slice(0, 10) };
  });
}

function makeSteps(
  n: number,
  overrides: Partial<{ count: number; startDate: string }> = {},
): CorrelationInput["steps"] {
  const { count = 5000, startDate = "2026-01-01" } = overrides;
  const [y, m, d] = parseDate(startDate);
  return Array.from({ length: n }, (_, i) => {
    const ts = new Date(Date.UTC(y, m, d + i, 12, 0, 0));
    return { count, date: ts.toISOString().slice(0, 10) };
  });
}

function makeFoodLogs(
  n: number,
  overrides: Partial<{ mealType: string; totalCalories: number; hour: number; startDate: string }> = {},
): CorrelationInput["foodLogs"] {
  const { mealType = "dinner", totalCalories = 600, hour = 19, startDate = "2026-01-01" } = overrides;
  const [y, m, d] = parseDate(startDate);
  return Array.from({ length: n }, (_, i) => {
    const ts = new Date(Date.UTC(y, m, d + i, hour, 0, 0));
    return { mealType, totalCalories, loggedAt: ts.toISOString() };
  });
}

// ── Tests ──────────────────────────────────────────────────────────────────

describe("generateInsights", () => {
  describe("empty input", () => {
    it("returns no insights for empty input", () => {
      const result = generateInsights(emptyInput());
      expect(result).toEqual([]);
    });

    it("returns no insights when all arrays are empty", () => {
      const input: CorrelationInput = {
        emotionHistory: [],
        foodLogs: [],
        sleepData: [],
        steps: [],
      };
      expect(generateInsights(input)).toHaveLength(0);
    });
  });

  // ── 1. Late Eating -> Sleep Quality ──────────────────────────────────────

  describe("late eating -> sleep quality", () => {
    it("detects better sleep without late eating", () => {
      // 6 nights with late eating + poor sleep
      const lateFoodDates = makeFoodLogs(6, { hour: 22, startDate: "2026-01-01" });
      const lateSleep = makeSleepData(6, { quality: 55, startDate: "2026-01-01" });

      // 6 nights with no late food + good sleep
      const noLateSleep = makeSleepData(6, { quality: 80, startDate: "2026-01-07" });

      const input: CorrelationInput = {
        emotionHistory: [],
        foodLogs: lateFoodDates,
        sleepData: [...lateSleep, ...noLateSleep],
        steps: [],
      };

      const insights = generateInsights(input);
      const sleepInsight = insights.find((i) => i.category === "food_sleep");
      expect(sleepInsight).toBeDefined();
      expect(sleepInsight!.text).toContain("sleep");
      expect(sleepInsight!.text).toContain("9pm");
      expect(sleepInsight!.dataPoints).toBeGreaterThanOrEqual(10);
    });

    it("requires minimum 5 samples per group", () => {
      // Only 4 in the late group
      const lateFoodDates = makeFoodLogs(4, { hour: 22, startDate: "2026-01-01" });
      const lateSleep = makeSleepData(4, { quality: 50, startDate: "2026-01-01" });
      const noLateSleep = makeSleepData(6, { quality: 80, startDate: "2026-01-05" });

      const input: CorrelationInput = {
        emotionHistory: [],
        foodLogs: lateFoodDates,
        sleepData: [...lateSleep, ...noLateSleep],
        steps: [],
      };

      const insights = generateInsights(input);
      const sleepInsight = insights.find((i) => i.category === "food_sleep");
      expect(sleepInsight).toBeUndefined();
    });

    it("produces an insight with exactly 5 samples per group", () => {
      const lateFoodDates = makeFoodLogs(5, { hour: 22, startDate: "2026-01-01" });
      const lateSleep = makeSleepData(5, { quality: 50, startDate: "2026-01-01" });
      const noLateSleep = makeSleepData(5, { quality: 80, startDate: "2026-01-06" });

      const input: CorrelationInput = {
        emotionHistory: [],
        foodLogs: lateFoodDates,
        sleepData: [...lateSleep, ...noLateSleep],
        steps: [],
      };

      const insights = generateInsights(input);
      const sleepInsight = insights.find((i) => i.category === "food_sleep");
      expect(sleepInsight).toBeDefined();
      expect(sleepInsight!.confidence).toBe("weak");
    });
  });

  // ── 2. Exercise -> Next-Day Mood ─────────────────────────────────────────

  describe("exercise -> next-day mood", () => {
    it("detects mood boost from active days", () => {
      // 6 active days (>7000 steps)
      const activeSteps = makeSteps(6, { count: 10000, startDate: "2026-01-01" });
      // 6 inactive days (<3000 steps)
      const inactiveSteps = makeSteps(6, { count: 1500, startDate: "2026-01-07" });

      // Next-day mood: high after active days, low after inactive
      const activeMood = makeEmotionHistory(6, { valence: 0.7, startDate: "2026-01-02" });
      const inactiveMood = makeEmotionHistory(6, { valence: -0.2, startDate: "2026-01-08" });

      const input: CorrelationInput = {
        emotionHistory: [...activeMood, ...inactiveMood],
        foodLogs: [],
        sleepData: [],
        steps: [...activeSteps, ...inactiveSteps],
      };

      const insights = generateInsights(input);
      const exerciseInsight = insights.find((i) => i.category === "exercise_mood");
      expect(exerciseInsight).toBeDefined();
      expect(exerciseInsight!.text).toContain("mood");
      expect(exerciseInsight!.dataPoints).toBeGreaterThanOrEqual(10);
    });

    it("requires minimum samples before producing insight", () => {
      const activeSteps = makeSteps(3, { count: 10000, startDate: "2026-01-01" });
      const inactiveSteps = makeSteps(3, { count: 1500, startDate: "2026-01-04" });
      const activeMood = makeEmotionHistory(3, { valence: 0.7, startDate: "2026-01-02" });
      const inactiveMood = makeEmotionHistory(3, { valence: -0.2, startDate: "2026-01-05" });

      const input: CorrelationInput = {
        emotionHistory: [...activeMood, ...inactiveMood],
        foodLogs: [],
        sleepData: [],
        steps: [...activeSteps, ...inactiveSteps],
      };

      const insights = generateInsights(input);
      expect(insights.find((i) => i.category === "exercise_mood")).toBeUndefined();
    });
  });

  // ── 3. Sleep Duration -> Focus ───────────────────────────────────────────

  describe("sleep duration -> focus", () => {
    it("detects better focus after good sleep", () => {
      // 6 nights of 8h sleep
      const goodSleep = makeSleepData(6, { duration: 8, startDate: "2026-01-01" });
      // 6 nights of 5h sleep
      const poorSleep = makeSleepData(6, { duration: 5, startDate: "2026-01-07" });

      // Focus next day: high after good sleep, low after poor
      const highFocus = makeEmotionHistory(6, { focus: 0.8, startDate: "2026-01-02" });
      const lowFocus = makeEmotionHistory(6, { focus: 0.3, startDate: "2026-01-08" });

      const input: CorrelationInput = {
        emotionHistory: [...highFocus, ...lowFocus],
        foodLogs: [],
        sleepData: [...goodSleep, ...poorSleep],
        steps: [],
      };

      const insights = generateInsights(input);
      const sleepFocus = insights.find((i) => i.category === "sleep_focus");
      expect(sleepFocus).toBeDefined();
      expect(sleepFocus!.text).toContain("focused");
      expect(sleepFocus!.text).toContain("7+ hours");
    });

    it("skips durations in the 6-7h range (neither group)", () => {
      // All sleep durations between 6-7h
      const midSleep = makeSleepData(10, { duration: 6.5, startDate: "2026-01-01" });
      const focus = makeEmotionHistory(10, { focus: 0.5, startDate: "2026-01-02" });

      const input: CorrelationInput = {
        emotionHistory: focus,
        foodLogs: [],
        sleepData: midSleep,
        steps: [],
      };

      const insights = generateInsights(input);
      expect(insights.find((i) => i.category === "sleep_focus")).toBeUndefined();
    });
  });

  // ── 4. Morning vs Evening Mood ───────────────────────────────────────────

  describe("morning vs evening mood", () => {
    it("detects higher morning mood", () => {
      // 8 morning readings with high valence
      const morningMood = makeEmotionHistory(8, { valence: 0.6, hour: 9 });
      // 8 evening readings with lower valence
      const eveningMood = makeEmotionHistory(8, { valence: -0.1, hour: 20 });

      const input: CorrelationInput = {
        emotionHistory: [...morningMood, ...eveningMood],
        foodLogs: [],
        sleepData: [],
        steps: [],
      };

      const insights = generateInsights(input);
      const timeInsight = insights.find((i) => i.category === "time_pattern");
      expect(timeInsight).toBeDefined();
      expect(timeInsight!.text).toContain("mornings");
    });

    it("detects higher evening mood", () => {
      const morningMood = makeEmotionHistory(8, { valence: -0.2, hour: 9 });
      const eveningMood = makeEmotionHistory(8, { valence: 0.7, hour: 20 });

      const input: CorrelationInput = {
        emotionHistory: [...morningMood, ...eveningMood],
        foodLogs: [],
        sleepData: [],
        steps: [],
      };

      const insights = generateInsights(input);
      const timeInsight = insights.find((i) => i.category === "time_pattern");
      expect(timeInsight).toBeDefined();
      expect(timeInsight!.text).toContain("evenings");
    });

    it("ignores midday readings (between noon and 6pm)", () => {
      // All readings at 2pm — neither morning nor evening
      const midday = makeEmotionHistory(20, { valence: 0.5, hour: 14 });

      const input: CorrelationInput = {
        emotionHistory: midday,
        foodLogs: [],
        sleepData: [],
        steps: [],
      };

      const insights = generateInsights(input);
      expect(insights.find((i) => i.category === "time_pattern")).toBeUndefined();
    });
  });

  // ── 5. Stress After Meals ────────────────────────────────────────────────

  describe("stress after meals", () => {
    it("detects stress rising after eating", () => {
      // 6 meals at noon UTC
      const meals = makeFoodLogs(6, { hour: 12, startDate: "2026-01-01" });

      // Post-meal readings (1h after meal at 13:00 UTC) with high stress
      const postMeal = Array.from({ length: 6 }, (_, i) => {
        const ts = new Date(Date.UTC(2026, 0, 1 + i, 13, 0, 0));
        return { stress: 0.8, focus: 0.5, valence: 0, energy: 0.5, timestamp: ts.toISOString() };
      });

      // Baseline readings (morning 8:00 UTC, not near meals) with low stress
      const baseline = Array.from({ length: 6 }, (_, i) => {
        const ts = new Date(Date.UTC(2026, 0, 1 + i, 8, 0, 0));
        return { stress: 0.3, focus: 0.5, valence: 0, energy: 0.5, timestamp: ts.toISOString() };
      });

      const input: CorrelationInput = {
        emotionHistory: [...postMeal, ...baseline],
        foodLogs: meals,
        sleepData: [],
        steps: [],
      };

      const insights = generateInsights(input);
      const stressInsight = insights.find((i) => i.category === "food_stress");
      expect(stressInsight).toBeDefined();
      expect(stressInsight!.text).toContain("stress");
      expect(stressInsight!.text).toContain("rises");
    });

    it("detects stress dropping after eating", () => {
      const meals = makeFoodLogs(6, { hour: 12, startDate: "2026-01-01" });

      // Post-meal: low stress (1h after meal at 13:00 UTC)
      const postMeal = Array.from({ length: 6 }, (_, i) => {
        const ts = new Date(Date.UTC(2026, 0, 1 + i, 13, 0, 0));
        return { stress: 0.2, focus: 0.5, valence: 0, energy: 0.5, timestamp: ts.toISOString() };
      });

      // Baseline: high stress (morning 8:00 UTC)
      const baseline = Array.from({ length: 6 }, (_, i) => {
        const ts = new Date(Date.UTC(2026, 0, 1 + i, 8, 0, 0));
        return { stress: 0.7, focus: 0.5, valence: 0, energy: 0.5, timestamp: ts.toISOString() };
      });

      const input: CorrelationInput = {
        emotionHistory: [...postMeal, ...baseline],
        foodLogs: meals,
        sleepData: [],
        steps: [],
      };

      const insights = generateInsights(input);
      const stressInsight = insights.find((i) => i.category === "food_stress");
      expect(stressInsight).toBeDefined();
      expect(stressInsight!.text).toContain("drops");
    });
  });

  // ── 6. Weekend vs Weekday Focus ──────────────────────────────────────────

  describe("weekend vs weekday focus", () => {
    it("detects higher weekday focus", () => {
      // Create 5 weekday readings (Mon-Fri) with high focus
      // 2026-01-05 is a Monday
      const weekdayFocus = makeEmotionHistory(5, {
        focus: 0.8,
        startDate: "2026-01-05",
        hour: 10,
      });

      // Create 5 weekend readings with low focus
      // 2026-01-03 is a Saturday (UTC), 2026-01-04 is Sunday
      const weekendEntries: CorrelationInput["emotionHistory"] = [];
      for (let i = 0; i < 5; i++) {
        // Alternate between Saturdays and Sundays across weeks
        const ts = new Date(Date.UTC(2026, 0, 3 + i * 7 + (i % 2), 10, 0, 0));
        weekendEntries.push({
          stress: 0.5,
          focus: 0.3,
          valence: 0,
          energy: 0.5,
          timestamp: ts.toISOString(),
        });
      }

      const input: CorrelationInput = {
        emotionHistory: [...weekdayFocus, ...weekendEntries],
        foodLogs: [],
        sleepData: [],
        steps: [],
      };

      const insights = generateInsights(input);
      const focusInsight = insights.find((i) => i.id === "weekday-focus");
      expect(focusInsight).toBeDefined();
      expect(focusInsight!.text).toContain("weekdays");
    });
  });

  // ── Confidence Levels ────────────────────────────────────────────────────

  describe("confidence levels (based on min group size)", () => {
    it("returns weak confidence when min group has 5-9 samples", () => {
      // 5 morning + 5 evening -> min group = 5 -> weak
      const morningMood = makeEmotionHistory(5, { valence: 0.8, hour: 9 });
      const eveningMood = makeEmotionHistory(5, { valence: -0.3, hour: 20 });

      const input: CorrelationInput = {
        emotionHistory: [...morningMood, ...eveningMood],
        foodLogs: [],
        sleepData: [],
        steps: [],
      };

      const insights = generateInsights(input);
      const timeInsight = insights.find((i) => i.category === "time_pattern");
      expect(timeInsight).toBeDefined();
      expect(timeInsight!.confidence).toBe("weak");
    });

    it("returns moderate confidence when min group has 10-14 samples", () => {
      // 10 morning + 10 evening -> min group = 10 -> moderate
      const morningMood = makeEmotionHistory(10, { valence: 0.8, hour: 9 });
      const eveningMood = makeEmotionHistory(10, { valence: -0.3, hour: 20 });

      const input: CorrelationInput = {
        emotionHistory: [...morningMood, ...eveningMood],
        foodLogs: [],
        sleepData: [],
        steps: [],
      };

      const insights = generateInsights(input);
      const timeInsight = insights.find((i) => i.category === "time_pattern");
      expect(timeInsight).toBeDefined();
      expect(timeInsight!.confidence).toBe("moderate");
    });

    it("returns strong confidence when min group has 15+ samples", () => {
      // 15 morning + 15 evening -> min group = 15 -> strong
      const morningMood = makeEmotionHistory(15, { valence: 0.8, hour: 9 });
      const eveningMood = makeEmotionHistory(15, { valence: -0.3, hour: 20 });

      const input: CorrelationInput = {
        emotionHistory: [...morningMood, ...eveningMood],
        foodLogs: [],
        sleepData: [],
        steps: [],
      };

      const insights = generateInsights(input);
      const timeInsight = insights.find((i) => i.category === "time_pattern");
      expect(timeInsight).toBeDefined();
      expect(timeInsight!.confidence).toBe("strong");
    });
  });

  // ── Percentage Accuracy ──────────────────────────────────────────────────

  describe("percentage calculation", () => {
    it("calculates correct percentage for sleep quality difference", () => {
      // Late eating group: quality = 50, no-late group: quality = 75
      // pctDiff(75, 50) = round(((75-50)/|50|)*100) = 50%
      const lateFoodDates = makeFoodLogs(5, { hour: 22, startDate: "2026-01-01" });
      const lateSleep = makeSleepData(5, { quality: 50, startDate: "2026-01-01" });
      const noLateSleep = makeSleepData(5, { quality: 75, startDate: "2026-01-06" });

      const input: CorrelationInput = {
        emotionHistory: [],
        foodLogs: lateFoodDates,
        sleepData: [...lateSleep, ...noLateSleep],
        steps: [],
      };

      const insights = generateInsights(input);
      const sleepInsight = insights.find((i) => i.category === "food_sleep");
      expect(sleepInsight).toBeDefined();
      expect(sleepInsight!.text).toContain("50%");
    });
  });

  // ── Noise Filtering (< 10% diff ignored) ────────────────────────────────

  describe("noise filtering", () => {
    it("filters out insights with less than 10% difference", () => {
      // Morning valence: 0.05 -> norm = 52.5
      // Evening valence: 0.0 -> norm = 50
      // pctDiff(52.5, 50) = 5% -> below threshold
      const morningMood = makeEmotionHistory(10, { valence: 0.05, hour: 9 });
      const eveningMood = makeEmotionHistory(10, { valence: 0.0, hour: 20 });

      const input: CorrelationInput = {
        emotionHistory: [...morningMood, ...eveningMood],
        foodLogs: [],
        sleepData: [],
        steps: [],
      };

      const insights = generateInsights(input);
      expect(insights.find((i) => i.category === "time_pattern")).toBeUndefined();
    });
  });

  // ── Sorting ──────────────────────────────────────────────────────────────

  describe("sorting", () => {
    it("sorts strong confidence before weak confidence", () => {
      // Two correlations: one with many samples (strong), one with few (weak)
      const morningMoodStrong = makeEmotionHistory(10, { valence: 0.8, hour: 9 });
      const eveningMoodStrong = makeEmotionHistory(10, { valence: -0.3, hour: 20 });

      // Weekday/weekend with fewer samples
      const weekdayFocus: CorrelationInput["emotionHistory"] = [];
      // 2026-01-05 is Monday (UTC)
      for (let i = 0; i < 5; i++) {
        const ts = new Date(Date.UTC(2026, 0, 5 + i, 10, 0, 0));
        weekdayFocus.push({ stress: 0.5, focus: 0.9, valence: 0, energy: 0.5, timestamp: ts.toISOString() });
      }
      const weekendFocus: CorrelationInput["emotionHistory"] = [];
      // 2026-01-03 is Saturday (UTC)
      for (let i = 0; i < 5; i++) {
        const ts = new Date(Date.UTC(2026, 0, 3 + i * 7, 10, 0, 0));
        weekendFocus.push({ stress: 0.5, focus: 0.3, valence: 0, energy: 0.5, timestamp: ts.toISOString() });
      }

      const input: CorrelationInput = {
        emotionHistory: [...morningMoodStrong, ...eveningMoodStrong, ...weekdayFocus, ...weekendFocus],
        foodLogs: [],
        sleepData: [],
        steps: [],
      };

      const insights = generateInsights(input);
      if (insights.length >= 2) {
        const confOrder = { strong: 0, moderate: 1, weak: 2 };
        expect(confOrder[insights[0].confidence]).toBeLessThanOrEqual(confOrder[insights[1].confidence]);
      }
    });
  });
});

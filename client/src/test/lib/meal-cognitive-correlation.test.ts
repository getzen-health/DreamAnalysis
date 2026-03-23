import { describe, it, expect, beforeEach, vi } from "vitest";
import {
  computeMealCognitiveCorrelation,
  generateInsight,
  type TimestampedReading,
  type MealTimestamp,
  type MealCognitiveInsight,
} from "@/lib/meal-cognitive-correlation";

// ── Helpers ──────────────────────────────────────────────────────────────────

function hoursAgo(h: number): string {
  return new Date(Date.now() - h * 60 * 60 * 1000).toISOString();
}

function makeReadings(entries: Array<{ hoursAgo: number; focus: number; stress: number }>): TimestampedReading[] {
  return entries.map((e) => ({
    timestamp: hoursAgo(e.hoursAgo),
    focus: e.focus,
    stress: e.stress,
  }));
}

function makeMeals(entries: Array<{ hoursAgo: number; label?: string }>): MealTimestamp[] {
  return entries.map((e) => ({
    timestamp: hoursAgo(e.hoursAgo),
    label: e.label ?? "lunch",
  }));
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe("computeMealCognitiveCorrelation", () => {
  it("returns null when no meals provided", () => {
    const readings = makeReadings([
      { hoursAgo: 0, focus: 0.7, stress: 0.3 },
      { hoursAgo: 1, focus: 0.6, stress: 0.4 },
    ]);
    const result = computeMealCognitiveCorrelation([], readings);
    expect(result).toBeNull();
  });

  it("returns null when no readings provided", () => {
    const meals = makeMeals([{ hoursAgo: 2 }]);
    const result = computeMealCognitiveCorrelation(meals, []);
    expect(result).toBeNull();
  });

  it("returns null when too few readings after meals", () => {
    // Only baseline readings, no post-meal readings
    const meals = makeMeals([{ hoursAgo: 0 }]); // meal right now
    const readings = makeReadings([
      { hoursAgo: 5, focus: 0.7, stress: 0.3 },
    ]);
    const result = computeMealCognitiveCorrelation(meals, readings);
    expect(result).toBeNull();
  });

  it("detects focus dip after meals", () => {
    // Baseline: high focus
    // Post-meal: lower focus
    const meals = makeMeals([{ hoursAgo: 3, label: "lunch" }]);
    const readings = makeReadings([
      // Baseline readings (before meal)
      { hoursAgo: 5, focus: 0.8, stress: 0.2 },
      { hoursAgo: 4, focus: 0.8, stress: 0.2 },
      // Post-meal readings (1-2 hours after meal)
      { hoursAgo: 2, focus: 0.4, stress: 0.5 },
      { hoursAgo: 1.5, focus: 0.35, stress: 0.55 },
      { hoursAgo: 1, focus: 0.5, stress: 0.4 },
    ]);

    const result = computeMealCognitiveCorrelation(meals, readings);
    expect(result).not.toBeNull();
    expect(result!.focusChange).toBeLessThan(0); // Focus dropped
    expect(result!.type).toBe("focus_dip");
  });

  it("detects focus boost after meals", () => {
    // Baseline: low focus
    // Post-meal: higher focus (e.g., breakfast)
    const meals = makeMeals([{ hoursAgo: 3, label: "breakfast" }]);
    const readings = makeReadings([
      // Baseline (before meal)
      { hoursAgo: 5, focus: 0.3, stress: 0.5 },
      { hoursAgo: 4, focus: 0.35, stress: 0.45 },
      // Post-meal (1-2 hours after)
      { hoursAgo: 2, focus: 0.7, stress: 0.2 },
      { hoursAgo: 1.5, focus: 0.75, stress: 0.15 },
      { hoursAgo: 1, focus: 0.7, stress: 0.2 },
    ]);

    const result = computeMealCognitiveCorrelation(meals, readings);
    expect(result).not.toBeNull();
    expect(result!.focusChange).toBeGreaterThan(0); // Focus improved
    expect(result!.type).toBe("focus_boost");
  });

  it("returns no_change when focus is stable", () => {
    const meals = makeMeals([{ hoursAgo: 3, label: "lunch" }]);
    const readings = makeReadings([
      { hoursAgo: 5, focus: 0.6, stress: 0.3 },
      { hoursAgo: 4, focus: 0.6, stress: 0.3 },
      { hoursAgo: 2, focus: 0.58, stress: 0.32 },
      { hoursAgo: 1.5, focus: 0.62, stress: 0.28 },
      { hoursAgo: 1, focus: 0.6, stress: 0.3 },
    ]);

    const result = computeMealCognitiveCorrelation(meals, readings);
    expect(result).not.toBeNull();
    expect(result!.type).toBe("no_change");
  });
});

// ── Insight generation ─────────────────────────────────────────────────────

describe("generateInsight", () => {
  it("generates dip insight with meal label", () => {
    const insight: MealCognitiveInsight = {
      type: "focus_dip",
      focusChange: -0.3,
      stressChange: 0.2,
      peakEffectHours: 1.5,
      mealLabel: "lunch",
      sampleCount: 5,
    };
    const text = generateInsight(insight);
    expect(text).toContain("focus");
    expect(text).toContain("lunch");
    expect(text.length).toBeGreaterThan(10);
  });

  it("generates boost insight", () => {
    const insight: MealCognitiveInsight = {
      type: "focus_boost",
      focusChange: 0.25,
      stressChange: -0.15,
      peakEffectHours: 1.0,
      mealLabel: "breakfast",
      sampleCount: 4,
    };
    const text = generateInsight(insight);
    expect(text).toContain("boost");
    expect(text).toContain("breakfast");
  });

  it("generates neutral insight for no_change", () => {
    const insight: MealCognitiveInsight = {
      type: "no_change",
      focusChange: 0.02,
      stressChange: 0.01,
      peakEffectHours: 1.5,
      mealLabel: "lunch",
      sampleCount: 3,
    };
    const text = generateInsight(insight);
    expect(text).toContain("stable");
  });
});

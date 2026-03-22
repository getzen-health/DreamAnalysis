import { describe, it, expect } from "vitest";
import {
  suggestInterventions,
  type Intervention,
  type InterventionTier,
} from "@/lib/intervention-engine";

describe("suggestInterventions", () => {
  // 1. High stress returns breathing as first suggestion
  it("returns breathing as first suggestion when stress is high", () => {
    const result = suggestInterventions("neutral", 0.7, true);
    expect(result.length).toBeGreaterThanOrEqual(2);
    expect(result[0].tier).toBe("breathing");
  });

  // 2. Sad emotion returns gratitude/calm suggestions
  it("returns gratitude or calm suggestions for sad emotion", () => {
    const result = suggestInterventions("sad", 0.3, true);
    expect(result.length).toBeGreaterThanOrEqual(2);
    const titles = result.map((r) => r.title.toLowerCase());
    const hasGratitudeOrCalm = titles.some(
      (t) => t.includes("gratitude") || t.includes("calm") || t.includes("music")
    );
    expect(hasGratitudeOrCalm).toBe(true);
  });

  // 3. Angry returns box breathing + reappraisal
  it("returns box breathing and reappraisal for angry emotion", () => {
    const result = suggestInterventions("angry", 0.4, true);
    expect(result.length).toBeGreaterThanOrEqual(2);
    const titles = result.map((r) => r.title.toLowerCase());
    const hasBoxBreathing = titles.some((t) => t.includes("box breathing"));
    const hasReappraisal = titles.some(
      (t) => t.includes("reappraisal") || t.includes("perspective") || t.includes("reframe")
    );
    expect(hasBoxBreathing).toBe(true);
    expect(hasReappraisal).toBe(true);
  });

  // 4. Always returns at least 1 intervention
  it("always returns at least 1 intervention", () => {
    const emotions = ["happy", "sad", "angry", "fear", "surprise", "neutral", "anxious", ""];
    for (const emo of emotions) {
      const result = suggestInterventions(emo, 0.3, false);
      expect(result.length).toBeGreaterThanOrEqual(1);
    }
  });

  // 5. When hasHeadband=false, no intervention requires headband
  it("returns no headband-required interventions when hasHeadband is false", () => {
    const emotions = ["happy", "sad", "angry", "fear", "surprise", "neutral", "anxious"];
    const stressLevels = [0.1, 0.5, 0.8];
    for (const emo of emotions) {
      for (const stress of stressLevels) {
        const result = suggestInterventions(emo, stress, false);
        const headbandRequired = result.filter((r) => r.requiresHeadband);
        expect(headbandRequired).toHaveLength(0);
      }
    }
  });

  // 6. All interventions have non-empty title, description, route
  it("returns interventions with non-empty title, description, and route", () => {
    const emotions = ["happy", "sad", "angry", "fear", "neutral", "anxious"];
    for (const emo of emotions) {
      const result = suggestInterventions(emo, 0.5, true);
      for (const intervention of result) {
        expect(intervention.title.length).toBeGreaterThan(0);
        expect(intervention.description.length).toBeGreaterThan(0);
        expect(intervention.route.length).toBeGreaterThan(0);
      }
    }
  });

  // 7. Duration strings are present
  it("returns interventions with non-empty duration strings", () => {
    const result = suggestInterventions("neutral", 0.5, true);
    for (const intervention of result) {
      expect(intervention.duration).toBeTruthy();
      expect(intervention.duration.length).toBeGreaterThan(0);
    }
  });

  // 8. Intervention titles never contain guilt language
  it("never uses guilt language in titles", () => {
    const guiltWords = ["should", "must", "need to", "have to", "ought to", "failure", "wrong"];
    const emotions = ["happy", "sad", "angry", "fear", "surprise", "neutral", "anxious"];
    const stressLevels = [0.1, 0.3, 0.5, 0.7, 0.9];
    for (const emo of emotions) {
      for (const stress of stressLevels) {
        const result = suggestInterventions(emo, stress, true);
        for (const intervention of result) {
          const lower = intervention.title.toLowerCase();
          for (const word of guiltWords) {
            expect(lower).not.toContain(word);
          }
        }
      }
    }
  });

  // Additional: anxious returns grounding exercise
  it("returns grounding exercise for anxious emotion", () => {
    const result = suggestInterventions("anxious", 0.5, false);
    const titles = result.map((r) => r.title.toLowerCase());
    const hasGrounding = titles.some(
      (t) => t.includes("grounding") || t.includes("5-4-3-2-1") || t.includes("senses")
    );
    expect(hasGrounding).toBe(true);
  });

  // Additional: neutral/calm returns maintenance + meditation suggestion
  it("returns maintenance and meditation suggestions for calm/neutral state", () => {
    const result = suggestInterventions("neutral", 0.15, true);
    expect(result.length).toBeGreaterThanOrEqual(2);
    const titles = result.map((r) => r.title.toLowerCase());
    const hasMaintainOrMeditation = titles.some(
      (t) => t.includes("maintain") || t.includes("meditation") || t.includes("deepen")
    );
    expect(hasMaintainOrMeditation).toBe(true);
  });

  // Additional: hasHeadband=true can include neurofeedback tier
  it("can include neurofeedback tier when headband is available", () => {
    // Neutral/calm with headband should offer neurofeedback as an option
    const result = suggestInterventions("neutral", 0.15, true);
    const hasNeurofeedback = result.some((r) => r.tier === "neurofeedback");
    expect(hasNeurofeedback).toBe(true);
  });

  // Additional: high stress returns reappraisal as second tier
  it("returns reappraisal tier alongside breathing for high stress", () => {
    const result = suggestInterventions("neutral", 0.8, false);
    const tiers = result.map((r) => r.tier);
    expect(tiers).toContain("breathing");
    expect(tiers).toContain("reappraisal");
  });

  // Additional: intervention icons are non-empty
  it("returns interventions with non-empty icon names", () => {
    const result = suggestInterventions("sad", 0.4, true);
    for (const intervention of result) {
      expect(intervention.icon.length).toBeGreaterThan(0);
    }
  });
});

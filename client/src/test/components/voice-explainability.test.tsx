import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "../test-utils";
import {
  EMOTION_EXPLANATIONS,
  computeFeatureBars,
} from "@/components/voice-checkin-card";

// ── unit tests: EMOTION_EXPLANATIONS map ─────────────────────────────────────

describe("EMOTION_EXPLANATIONS", () => {
  const KNOWN_EMOTIONS = [
    "happy",
    "sad",
    "angry",
    "neutral",
    "calm",
    "peaceful",
    "fear",
    "fearful",
    "surprise",
    "anxious",
    "nervous",
    "grateful",
    "proud",
    "lonely",
    "hopeful",
    "overwhelmed",
    "excited",
    "frustrated",
  ];

  it.each(KNOWN_EMOTIONS)(
    "returns a non-empty string for '%s'",
    (emotion) => {
      const explanation = EMOTION_EXPLANATIONS[emotion];
      expect(typeof explanation).toBe("string");
      expect(explanation.length).toBeGreaterThan(0);
    }
  );

  it("uses hedging language (not definitive claims)", () => {
    const hedgingPatterns = /suggest(?:s|ing)|typically|often|can |may |which /i;
    for (const emotion of KNOWN_EMOTIONS) {
      expect(EMOTION_EXPLANATIONS[emotion]).toMatch(hedgingPatterns);
    }
  });
});

// ── unit tests: computeFeatureBars ───────────────────────────────────────────

describe("computeFeatureBars", () => {
  it("returns 4 bars with correct labels", () => {
    const bars = computeFeatureBars({
      arousal: 0.7,
      stress_index: 0.3,
      valence: 0.5,
      confidence: 0.8,
    });
    expect(bars).toHaveLength(4);
    expect(bars.map((b) => b.label)).toEqual([
      "Energy level",
      "Voice pace",
      "Positivity",
      "Confidence",
    ]);
  });

  it("computes energy from arousal", () => {
    const bars = computeFeatureBars({
      arousal: 0.65,
      stress_index: 0,
      valence: 0,
      confidence: 0,
    });
    expect(bars[0].value).toBe(65);
  });

  it("computes voice pace as inverse of stress_index", () => {
    const bars = computeFeatureBars({
      arousal: 0,
      stress_index: 0.3,
      valence: 0,
      confidence: 0,
    });
    expect(bars[1].value).toBe(70);
  });

  it("maps valence from [-1,1] to [0,100] for positivity", () => {
    // valence = -1 -> 0%, valence = 0 -> 50%, valence = 1 -> 100%
    const barsNeg = computeFeatureBars({
      arousal: 0,
      stress_index: 0,
      valence: -1,
      confidence: 0,
    });
    expect(barsNeg[2].value).toBe(0);

    const barsZero = computeFeatureBars({
      arousal: 0,
      stress_index: 0,
      valence: 0,
      confidence: 0,
    });
    expect(barsZero[2].value).toBe(50);

    const barsPos = computeFeatureBars({
      arousal: 0,
      stress_index: 0,
      valence: 1,
      confidence: 0,
    });
    expect(barsPos[2].value).toBe(100);
  });

  it("uses violet color for positivity >= 50 and rose for < 50", () => {
    const barsPositive = computeFeatureBars({
      arousal: 0,
      stress_index: 0,
      valence: 0.2,
      confidence: 0,
    });
    expect(barsPositive[2].color).toBe("bg-violet-500");

    const barsNegative = computeFeatureBars({
      arousal: 0,
      stress_index: 0,
      valence: -0.5,
      confidence: 0,
    });
    expect(barsNegative[2].color).toBe("bg-rose-500");
  });

  it("computes confidence percentage correctly", () => {
    const bars = computeFeatureBars({
      arousal: 0,
      stress_index: 0,
      valence: 0,
      confidence: 0.85,
    });
    expect(bars[3].value).toBe(85);
  });

  it("clamps values to 0-100 range for out-of-bounds inputs", () => {
    const bars = computeFeatureBars({
      arousal: 1.5,
      stress_index: -0.3,
      valence: 2,
      confidence: 1.2,
    });
    expect(bars[0].value).toBe(100);
    expect(bars[1].value).toBeLessThanOrEqual(100);
    expect(bars[2].value).toBe(100);
    expect(bars[3].value).toBe(100);
  });
});

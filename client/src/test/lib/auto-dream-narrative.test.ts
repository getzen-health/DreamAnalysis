import { describe, it, expect } from "vitest";
import {
  generateAutoDreamNarrative,
  type EEGDreamData,
} from "@/lib/auto-dream-narrative";

// ── Helpers ──────────────────────────────────────────────────────────────────

function makeData(overrides: Partial<EEGDreamData> = {}): EEGDreamData {
  return {
    remPercentage: 25,
    avgValence: 0,
    avgArousal: 0.5,
    dreamEpisodes: 3,
    dominantEmotion: "neutral",
    sleepDuration: 7.5,
    deepSleepPct: 18,
    ...overrides,
  };
}

// ── No dream episodes ────────────────────────────────────────────────────────

describe("no dream episodes", () => {
  it("returns restorative message when deep sleep is high", () => {
    const result = generateAutoDreamNarrative(
      makeData({ dreamEpisodes: 0, deepSleepPct: 25 }),
    );
    expect(result).toContain("No dream episodes detected");
    expect(result).toContain("deep and restorative");
    expect(result).toContain("25%");
  });

  it("returns recovery message when deep sleep is low", () => {
    const result = generateAutoDreamNarrative(
      makeData({ dreamEpisodes: 0, deepSleepPct: 10 }),
    );
    expect(result).toContain("No dream episodes detected");
    expect(result).toContain("recovery");
    expect(result).not.toContain("deep and restorative");
  });
});

// ── Low REM ──────────────────────────────────────────────────────────────────

describe("low REM", () => {
  it("mentions quiet night and deep sleep dominance", () => {
    const result = generateAutoDreamNarrative(
      makeData({ remPercentage: 12, dreamEpisodes: 1, deepSleepPct: 30 }),
    );
    expect(result).toContain("quiet night for dreams");
    expect(result).toContain("physical recovery");
    expect(result).toContain("30% deep sleep");
    expect(result).toContain("1 brief dream episode");
  });

  it("uses plural for multiple episodes", () => {
    const result = generateAutoDreamNarrative(
      makeData({ remPercentage: 15, dreamEpisodes: 2, deepSleepPct: 22 }),
    );
    expect(result).toContain("2 brief dream episodes");
  });
});

// ── High REM + positive valence ──────────────────────────────────────────────

describe("high REM + positive valence", () => {
  it("describes vivid and energetic dreams with high arousal", () => {
    const result = generateAutoDreamNarrative(
      makeData({
        remPercentage: 28,
        avgValence: 0.5,
        avgArousal: 0.8,
        dreamEpisodes: 4,
        dominantEmotion: "happy",
      }),
    );
    expect(result).toContain("vivid and energetic");
    expect(result).toContain("4 dream episodes");
    expect(result).toContain("positive emotional tone");
    expect(result).toContain("happy");
    expect(result).toContain("28%");
  });

  it("describes calm and pleasant dreams with low arousal", () => {
    const result = generateAutoDreamNarrative(
      makeData({
        remPercentage: 22,
        avgValence: 0.3,
        avgArousal: 0.3,
        dreamEpisodes: 2,
        dominantEmotion: "neutral",
      }),
    );
    expect(result).toContain("calm and pleasant");
    expect(result).toContain("positive emotional tone");
  });
});

// ── High REM + negative valence ──────────────────────────────────────────────

describe("high REM + negative valence", () => {
  it("describes emotionally intense night with high arousal", () => {
    const result = generateAutoDreamNarrative(
      makeData({
        remPercentage: 30,
        avgValence: -0.4,
        avgArousal: 0.75,
        dreamEpisodes: 5,
        dominantEmotion: "fear",
      }),
    );
    expect(result).toContain("emotionally intense night");
    expect(result).toContain("heightened arousal");
    expect(result).toContain("processing something significant");
    expect(result).toContain("fear");
    expect(result).toContain("5 dream episodes");
  });

  it("describes melancholic night with low arousal", () => {
    const result = generateAutoDreamNarrative(
      makeData({
        remPercentage: 24,
        avgValence: -0.3,
        avgArousal: 0.4,
        dreamEpisodes: 2,
        dominantEmotion: "sad",
      }),
    );
    expect(result).toContain("emotionally intense night");
    expect(result).toContain("melancholic");
    expect(result).toContain("sad");
  });
});

// ── High REM + neutral valence ───────────────────────────────────────────────

describe("high REM + neutral valence", () => {
  it("describes mentally active dreams with high arousal", () => {
    const result = generateAutoDreamNarrative(
      makeData({
        remPercentage: 22,
        avgValence: 0.05,
        avgArousal: 0.7,
        dreamEpisodes: 3,
        dominantEmotion: "neutral",
      }),
    );
    expect(result).toContain("neutral but mentally active");
    expect(result).toContain("3 dream episodes");
    expect(result).toContain("22% REM");
  });

  it("describes gentle dreams with low arousal", () => {
    const result = generateAutoDreamNarrative(
      makeData({
        remPercentage: 20,
        avgValence: 0.0,
        avgArousal: 0.3,
        dreamEpisodes: 1,
        dominantEmotion: "neutral",
      }),
    );
    expect(result).toContain("gentle and emotionally balanced");
    expect(result).toContain("1 dream episode");
  });
});

// ── Duration formatting ──────────────────────────────────────────────────────

describe("duration formatting", () => {
  it("formats whole hours", () => {
    const result = generateAutoDreamNarrative(
      makeData({ sleepDuration: 8, dreamEpisodes: 0 }),
    );
    expect(result).toContain("8h session");
  });

  it("formats hours and minutes", () => {
    const result = generateAutoDreamNarrative(
      makeData({ sleepDuration: 6.5, dreamEpisodes: 0 }),
    );
    expect(result).toContain("6h 30m session");
  });

  it("formats sub-hour durations", () => {
    const result = generateAutoDreamNarrative(
      makeData({ sleepDuration: 0.5, dreamEpisodes: 0 }),
    );
    expect(result).toContain("30m session");
  });
});

// ── Edge cases ───────────────────────────────────────────────────────────────

describe("edge cases", () => {
  it("handles zero duration", () => {
    const result = generateAutoDreamNarrative(
      makeData({ sleepDuration: 0, dreamEpisodes: 0 }),
    );
    expect(result).toContain("0m session");
  });

  it("handles boundary REM percentage exactly at threshold", () => {
    // At exactly 20% — should NOT take the low-REM path
    const result = generateAutoDreamNarrative(
      makeData({ remPercentage: 20, avgValence: 0, dreamEpisodes: 2 }),
    );
    expect(result).not.toContain("quiet night for dreams");
  });

  it("handles boundary valence at positive threshold", () => {
    // At exactly 0.15 — not above threshold, should be neutral path
    const result = generateAutoDreamNarrative(
      makeData({ remPercentage: 25, avgValence: 0.15, dreamEpisodes: 2 }),
    );
    expect(result).not.toContain("positive emotional tone");
  });

  it("handles single dream episode with singular word", () => {
    const result = generateAutoDreamNarrative(
      makeData({ remPercentage: 25, avgValence: 0.5, dreamEpisodes: 1 }),
    );
    expect(result).toContain("1 dream episode");
    expect(result).not.toContain("1 dream episodes");
  });
});

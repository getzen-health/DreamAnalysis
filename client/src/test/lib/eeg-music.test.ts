import { describe, it, expect } from "vitest";
import { recommendMusic, type MusicRecommendation } from "@/lib/eeg-music";

// ── Helpers ──────────────────────────────────────────────────────────────────

function makeState(overrides: Partial<Parameters<typeof recommendMusic>[0]> = {}) {
  return {
    dominantBand: "alpha",
    arousal: 0.5,
    stress: 0.3,
    focus: 0.5,
    emotion: "neutral",
    ...overrides,
  };
}

function assertValidRecommendation(rec: MusicRecommendation) {
  expect(rec.category).toBeTruthy();
  expect(rec.reason).toBeTruthy();
  expect(rec.reason.length).toBeGreaterThan(0);
  expect(rec.spotifyQuery).toBeTruthy();
  expect(rec.spotifyQuery.length).toBeGreaterThan(0);
  expect(rec.colorAccent).toBeTruthy();
  if (rec.binauralFreq !== undefined) {
    expect(rec.binauralFreq).toBeGreaterThanOrEqual(1);
    expect(rec.binauralFreq).toBeLessThanOrEqual(40);
  }
}

// ── Category mapping tests ───────────────────────────────────────────────────

describe("recommendMusic", () => {
  it("recommends calm music when stress is high and arousal is high", () => {
    const rec = recommendMusic(makeState({ stress: 0.8, arousal: 0.7 }));
    expect(rec.category).toBe("calm");
    expect(rec.binauralFreq).toBe(10);
    assertValidRecommendation(rec);
  });

  it("recommends focus music when focus is high and stress is low", () => {
    // Beta-dominant state with active focus — not alpha-dominant (which triggers meditation)
    const rec = recommendMusic(makeState({ dominantBand: "beta", focus: 0.8, stress: 0.2 }));
    expect(rec.category).toBe("focus");
    expect(rec.binauralFreq).toBe(40);
    assertValidRecommendation(rec);
  });

  it("recommends energy music when arousal is low", () => {
    const rec = recommendMusic(makeState({ arousal: 0.2 }));
    expect(rec.category).toBe("energy");
    assertValidRecommendation(rec);
  });

  it("recommends sleep music when theta is dominant", () => {
    const rec = recommendMusic(makeState({ dominantBand: "theta", arousal: 0.2 }));
    expect(rec.category).toBe("sleep");
    expect(rec.binauralFreq).toBe(3);
    assertValidRecommendation(rec);
  });

  it("recommends meditation music when alpha is high and emotion is relaxed", () => {
    const rec = recommendMusic(
      makeState({ dominantBand: "alpha", arousal: 0.3, stress: 0.1, emotion: "neutral" })
    );
    expect(rec.category).toBe("meditation");
    expect(rec.binauralFreq).toBe(6);
    assertValidRecommendation(rec);
  });

  it("returns non-empty reason and spotifyQuery for every recommendation", () => {
    const states = [
      makeState({ stress: 0.9, arousal: 0.8 }),
      makeState({ focus: 0.9, stress: 0.1 }),
      makeState({ arousal: 0.1 }),
      makeState({ dominantBand: "theta", arousal: 0.2 }),
      makeState({ dominantBand: "alpha", arousal: 0.3, stress: 0.1, emotion: "neutral" }),
    ];
    for (const state of states) {
      const rec = recommendMusic(state);
      assertValidRecommendation(rec);
    }
  });

  it("keeps binaural frequencies in safe range (1-40 Hz)", () => {
    const states = [
      makeState({ stress: 0.9, arousal: 0.8 }),
      makeState({ focus: 0.9, stress: 0.1 }),
      makeState({ dominantBand: "theta", arousal: 0.2 }),
      makeState({ dominantBand: "alpha", arousal: 0.3, stress: 0.1, emotion: "neutral" }),
    ];
    for (const state of states) {
      const rec = recommendMusic(state);
      if (rec.binauralFreq !== undefined) {
        expect(rec.binauralFreq).toBeGreaterThanOrEqual(1);
        expect(rec.binauralFreq).toBeLessThanOrEqual(40);
      }
    }
  });

  it("all 5 categories are reachable", () => {
    const categories = new Set<string>();

    // calm: high stress + high arousal
    categories.add(recommendMusic(makeState({ stress: 0.8, arousal: 0.7 })).category);
    // focus: high focus + low stress + beta-dominant (not alpha, which triggers meditation)
    categories.add(recommendMusic(makeState({ dominantBand: "beta", focus: 0.8, stress: 0.2 })).category);
    // energy: low arousal
    categories.add(recommendMusic(makeState({ arousal: 0.1 })).category);
    // sleep: theta dominant
    categories.add(recommendMusic(makeState({ dominantBand: "theta", arousal: 0.2 })).category);
    // meditation: alpha dominant + relaxed
    categories.add(
      recommendMusic(makeState({ dominantBand: "alpha", arousal: 0.3, stress: 0.1, emotion: "neutral" })).category
    );

    expect(categories).toContain("calm");
    expect(categories).toContain("focus");
    expect(categories).toContain("energy");
    expect(categories).toContain("sleep");
    expect(categories).toContain("meditation");
    expect(categories.size).toBe(5);
  });
});

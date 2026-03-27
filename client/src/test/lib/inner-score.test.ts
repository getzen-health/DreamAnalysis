import { describe, it, expect } from "vitest";
import {
  normalizeStress,
  normalizeValence,
  normalizeEnergy,
  detectTier,
  computeScore,
  computeNarrative,
  getScoreLabel,
} from "@/lib/inner-score";

// ─── Normalization ───────────────────────────────────────────────────────────

describe("normalizeStress", () => {
  it("inverts stress 0.0 → 100", () => {
    expect(normalizeStress(0)).toBe(100);
  });
  it("inverts stress 1.0 → 0", () => {
    expect(normalizeStress(1)).toBe(0);
  });
  it("inverts stress 0.3 → 70", () => {
    expect(normalizeStress(0.3)).toBe(70);
  });
  it("clamps above 1", () => {
    expect(normalizeStress(1.5)).toBe(0);
  });
  it("clamps below 0", () => {
    expect(normalizeStress(-0.5)).toBe(100);
  });
});

describe("normalizeValence", () => {
  it("maps -1 → 0", () => {
    expect(normalizeValence(-1)).toBe(0);
  });
  it("maps 0 → 50", () => {
    expect(normalizeValence(0)).toBe(50);
  });
  it("maps 1 → 100", () => {
    expect(normalizeValence(1)).toBe(100);
  });
});

describe("normalizeEnergy", () => {
  it("maps arousal 0.0 → 0", () => {
    expect(normalizeEnergy({ arousal: 0 })).toBe(0);
  });
  it("maps arousal 1.0 → 100", () => {
    expect(normalizeEnergy({ arousal: 1 })).toBe(100);
  });
  it("maps mood log 1-5 scale: 3 → 60", () => {
    expect(normalizeEnergy({ moodEnergy: 3, moodScale: 5 })).toBe(60);
  });
  it("prefers arousal over mood log", () => {
    expect(normalizeEnergy({ arousal: 0.8, moodEnergy: 1, moodScale: 5 })).toBe(80);
  });
  it("returns 50 when no data", () => {
    expect(normalizeEnergy({})).toBe(50);
  });
});

// ─── Tier Detection ──────────────────────────────────────────────────────────

describe("detectTier", () => {
  it("returns null when no data", () => {
    expect(detectTier({})).toBeNull();
  });
  it("returns voice when only stress present", () => {
    expect(detectTier({ stress: 0.4 })).toBe("voice");
  });
  it("returns voice when only valence present", () => {
    expect(detectTier({ valence: 0.2 })).toBe("voice");
  });
  it("returns health_voice when sleep + stress present", () => {
    expect(detectTier({ sleepQuality: 80, stress: 0.3 })).toBe("health_voice");
  });
  it("returns eeg_health_voice when brainHealth + sleep present", () => {
    expect(detectTier({ brainHealth: 70, sleepQuality: 85, stress: 0.2 })).toBe("eeg_health_voice");
  });
  it("falls back to health_voice when brainHealth missing but sleep present", () => {
    expect(detectTier({ sleepQuality: 80, stress: 0.5 })).toBe("health_voice");
  });
  it("falls back to voice when sleep missing", () => {
    expect(detectTier({ stress: 0.4, valence: 0.1 })).toBe("voice");
  });
});

// ─── Score Computation ───────────────────────────────────────────────────────

describe("computeScore", () => {
  it("computes Tier 1 (voice only)", () => {
    const result = computeScore({ stress: 0.3, valence: 0.4 });
    expect(result.tier).toBe("voice");
    expect(result.score).toBeGreaterThanOrEqual(0);
    expect(result.score).toBeLessThanOrEqual(100);
    expect(result.factors).toHaveProperty("stress_inverse");
    expect(result.factors).toHaveProperty("valence");
  });

  it("computes Tier 2 (health + voice)", () => {
    const result = computeScore({ stress: 0.3, valence: 0.4, sleepQuality: 80, hrvTrend: 60, activity: 50 });
    expect(result.tier).toBe("health_voice");
    expect(result.factors).toHaveProperty("sleep_quality");
  });

  it("computes Tier 3 (EEG + health + voice)", () => {
    const result = computeScore({ stress: 0.2, valence: 0.5, sleepQuality: 85, hrvTrend: 70, activity: 60, brainHealth: 75 });
    expect(result.tier).toBe("eeg_health_voice");
    expect(result.factors).toHaveProperty("brain_health");
  });

  it("returns building state when no data", () => {
    const result = computeScore({});
    expect(result.score).toBeNull();
    expect(result.label).toBe("Building");
  });

  it("redistributes weights when optional factors missing in Tier 2", () => {
    const withoutHrv = computeScore({ stress: 0.3, valence: 0.4, sleepQuality: 80 });
    expect(withoutHrv.tier).toBe("health_voice");
    expect(withoutHrv.score).toBeGreaterThan(0);
  });

  it("labels Thriving for score >= 80", () => {
    const result = computeScore({ stress: 0.05, valence: 0.9, sleepQuality: 95, hrvTrend: 90, activity: 85 });
    expect(result.label).toBe("Thriving");
  });

  it("labels Low for score < 40", () => {
    const result = computeScore({ stress: 0.9, valence: -0.8 });
    expect(result.label).toBe("Low");
  });

  it("score is always 0-100", () => {
    const high = computeScore({ stress: 0, valence: 1, sleepQuality: 100, hrvTrend: 100, activity: 100, brainHealth: 100 });
    const low = computeScore({ stress: 1, valence: -1 });
    expect(high.score).toBeLessThanOrEqual(100);
    expect(low.score).toBeGreaterThanOrEqual(0);
  });
});

// ─── Score Labels ────────────────────────────────────────────────────────────

describe("getScoreLabel", () => {
  it("returns Thriving for 80+", () => expect(getScoreLabel(80)).toBe("Thriving"));
  it("returns Good for 60-79", () => expect(getScoreLabel(65)).toBe("Good"));
  it("returns Steady for 40-59", () => expect(getScoreLabel(50)).toBe("Steady"));
  it("returns Low for 0-39", () => expect(getScoreLabel(20)).toBe("Low"));
  it("returns Thriving for 100", () => expect(getScoreLabel(100)).toBe("Thriving"));
  it("returns Low for 0", () => expect(getScoreLabel(0)).toBe("Low"));
});

// ─── Narrative ───────────────────────────────────────────────────────────────

describe("computeNarrative", () => {
  it("mentions highest and lowest factor", () => {
    const narrative = computeNarrative({ sleep_quality: 90, stress_inverse: 40, valence: 60 }, 63, null);
    expect(narrative).toContain("sleep");
    expect(narrative).toContain("stress");
  });
  it("says well-balanced when all within 10 points", () => {
    const narrative = computeNarrative({ sleep_quality: 70, stress_inverse: 65, valence: 68 }, 68, null);
    expect(narrative).toContain("well-balanced");
  });
  it("mentions improvement when delta > 10", () => {
    const narrative = computeNarrative({ sleep_quality: 80 }, 80, 15);
    expect(narrative).toContain("improvement");
  });
  it("mentions dip when delta < -10", () => {
    const narrative = computeNarrative({ sleep_quality: 50 }, 50, -12);
    expect(narrative).toContain("Dip");
  });
  it("returns empty string for empty factors", () => {
    expect(computeNarrative({}, 50, null)).toBe("");
  });
});

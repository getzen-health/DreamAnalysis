import { describe, it, expect } from "vitest";
import { calculateEmotionConfidence } from "@/lib/confidence-calculator";

describe("calculateEmotionConfidence", () => {
  // 1. High model confidence + good signal → high overall
  it("returns high confidence when model confidence and signal quality are both high", () => {
    const result = calculateEmotionConfidence({
      modelConfidence: 0.9,
      signalQuality: 0.95,
    });
    expect(result.confidence).toBeGreaterThan(0.7);
    expect(result.label).toBe("High confidence");
    expect(result.showEmotion).toBe(true);
  });

  // 2. High model confidence + poor signal → reduced
  it("reduces confidence when signal quality is poor despite high model confidence", () => {
    const highSignal = calculateEmotionConfidence({
      modelConfidence: 0.9,
      signalQuality: 0.95,
    });
    const poorSignal = calculateEmotionConfidence({
      modelConfidence: 0.9,
      signalQuality: 0.3,
    });
    expect(poorSignal.confidence).toBeLessThan(highSignal.confidence);
  });

  // 3. Short session (<30s) → reduced
  it("reduces confidence for sessions shorter than 30 seconds", () => {
    const longSession = calculateEmotionConfidence({
      modelConfidence: 0.8,
      sessionDuration: 120,
    });
    const shortSession = calculateEmotionConfidence({
      modelConfidence: 0.8,
      sessionDuration: 10,
    });
    expect(shortSession.confidence).toBeLessThan(longSession.confidence);
  });

  // 4. High artifact percentage → reduced
  it("reduces confidence when artifact percentage is high", () => {
    const clean = calculateEmotionConfidence({
      modelConfidence: 0.8,
      artifactPercentage: 0.05,
    });
    const noisy = calculateEmotionConfidence({
      modelConfidence: 0.8,
      artifactPercentage: 0.6,
    });
    expect(noisy.confidence).toBeLessThan(clean.confidence);
  });

  // 5. All factors good → showEmotion = true
  it("sets showEmotion to true when all factors produce high confidence", () => {
    const result = calculateEmotionConfidence({
      modelConfidence: 0.85,
      signalQuality: 0.9,
      sessionDuration: 120,
      artifactPercentage: 0.05,
      agreementScore: 0.9,
    });
    expect(result.showEmotion).toBe(true);
    expect(result.confidence).toBeGreaterThan(0.5);
  });

  // 6. Very low confidence (<0.3) → showEmotion = false
  it("sets showEmotion to false when confidence falls below 0.3", () => {
    const result = calculateEmotionConfidence({
      modelConfidence: 0.25,
      signalQuality: 0.3,
    });
    expect(result.showEmotion).toBe(false);
  });

  // 7. Label matches confidence range (high/moderate/low)
  it("returns 'High confidence' label for confidence > 0.7", () => {
    const result = calculateEmotionConfidence({ modelConfidence: 0.95 });
    expect(result.label).toBe("High confidence");
  });

  it("returns 'Moderate confidence' label for confidence between 0.5 and 0.7", () => {
    const result = calculateEmotionConfidence({
      modelConfidence: 0.65,
      signalQuality: 0.85,
    });
    expect(result.confidence).toBeGreaterThanOrEqual(0.5);
    expect(result.confidence).toBeLessThanOrEqual(0.7);
    expect(result.label).toBe("Moderate confidence");
  });

  it("returns 'Low confidence' label for confidence between 0.3 and 0.5", () => {
    const result = calculateEmotionConfidence({
      modelConfidence: 0.45,
      signalQuality: 0.8,
    });
    expect(result.label).toBe("Low confidence");
  });

  it("returns 'Not enough data' label for confidence below 0.3", () => {
    const result = calculateEmotionConfidence({
      modelConfidence: 0.2,
      signalQuality: 0.3,
    });
    expect(result.label).toBe("Not enough data");
  });

  // 8. Missing factors use neutral defaults (don't crash)
  it("handles missing optional factors without crashing", () => {
    const result = calculateEmotionConfidence({ modelConfidence: 0.7 });
    expect(result).toBeDefined();
    expect(result.confidence).toBeGreaterThan(0);
    expect(typeof result.label).toBe("string");
    expect(typeof result.showEmotion).toBe("boolean");
  });

  it("handles all factors being undefined except modelConfidence", () => {
    const result = calculateEmotionConfidence({
      modelConfidence: 0.5,
      signalQuality: undefined,
      sessionDuration: undefined,
      artifactPercentage: undefined,
      agreementScore: undefined,
    });
    expect(result.confidence).toBeCloseTo(0.5, 1);
    expect(result.showEmotion).toBe(true);
  });

  // Edge: agreement score multiplied in
  it("reduces confidence when agreement score is low", () => {
    const highAgreement = calculateEmotionConfidence({
      modelConfidence: 0.8,
      agreementScore: 1.0,
    });
    const lowAgreement = calculateEmotionConfidence({
      modelConfidence: 0.8,
      agreementScore: 0.33,
    });
    expect(lowAgreement.confidence).toBeLessThan(highAgreement.confidence);
  });

  // Edge: session boost past 60s
  it("boosts confidence for sessions longer than 60 seconds", () => {
    const short = calculateEmotionConfidence({
      modelConfidence: 0.6,
      sessionDuration: 30,
    });
    const long = calculateEmotionConfidence({
      modelConfidence: 0.6,
      sessionDuration: 120,
    });
    expect(long.confidence).toBeGreaterThan(short.confidence);
  });

  // Edge: confidence is always clamped to [0, 1]
  it("clamps confidence to the 0-1 range", () => {
    const result = calculateEmotionConfidence({
      modelConfidence: 1.0,
      signalQuality: 1.0,
      sessionDuration: 300,
      artifactPercentage: 0,
      agreementScore: 1.0,
    });
    expect(result.confidence).toBeLessThanOrEqual(1);
    expect(result.confidence).toBeGreaterThanOrEqual(0);
  });
});

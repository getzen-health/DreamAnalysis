import { describe, it, expect } from "vitest";
import {
  estimateSleepFromPhone,
  type PhoneSleepInput,
  type PhoneSleepEstimate,
} from "./phone-sleep-detector";

// ── Helper ──────────────────────────────────────────────────────────────────

function estimate(overrides: Partial<PhoneSleepInput>): PhoneSleepEstimate {
  const defaults: PhoneSleepInput = {
    accelerometerVariance: 0,
    microphoneLevel: 0,
    timeOfNight: 2,
  };
  return estimateSleepFromPhone({ ...defaults, ...overrides });
}

// ── Wake detection ──────────────────────────────────────────────────────────

describe("Wake detection (high movement)", () => {
  it("returns Wake when accelerometer variance is high", () => {
    const result = estimate({ accelerometerVariance: 0.8 });
    expect(result.likelyStage).toBe("Wake");
    expect(result.dreamLikelihood).toBeLessThan(0.1);
  });

  it("returns Wake at the 0.5 boundary (just above)", () => {
    const result = estimate({ accelerometerVariance: 0.51 });
    expect(result.likelyStage).toBe("Wake");
  });

  it("has higher confidence with more movement", () => {
    const low = estimate({ accelerometerVariance: 0.55 });
    const high = estimate({ accelerometerVariance: 0.95 });
    expect(high.confidence).toBeGreaterThan(low.confidence);
  });

  it("confidence is between 0 and 1", () => {
    const result = estimate({ accelerometerVariance: 1.0 });
    expect(result.confidence).toBeGreaterThanOrEqual(0);
    expect(result.confidence).toBeLessThanOrEqual(1);
  });
});

// ── Light sleep detection ───────────────────────────────────────────────────

describe("Light sleep detection (moderate movement/noise)", () => {
  it("returns Light when accelerometer is moderate (0.2-0.5)", () => {
    const result = estimate({ accelerometerVariance: 0.35, microphoneLevel: 0.2 });
    expect(result.likelyStage).toBe("Light");
  });

  it("returns Light when microphone is loud even if still", () => {
    const result = estimate({ accelerometerVariance: 0.05, microphoneLevel: 0.7 });
    expect(result.likelyStage).toBe("Light");
  });

  it("has low but nonzero dream likelihood", () => {
    const result = estimate({ accelerometerVariance: 0.3, microphoneLevel: 0.3 });
    expect(result.dreamLikelihood).toBeGreaterThan(0);
    expect(result.dreamLikelihood).toBeLessThan(0.5);
  });
});

// ── Deep sleep detection (time-of-night) ────────────────────────────────────

describe("Deep sleep detection (still + quiet + early night)", () => {
  it("returns Deep at 1-3 hours into sleep when still and quiet", () => {
    const result = estimate({
      accelerometerVariance: 0.05,
      microphoneLevel: 0.1,
      timeOfNight: 1.5,
    });
    expect(result.likelyStage).toBe("Deep");
    expect(result.dreamLikelihood).toBeLessThanOrEqual(0.15);
  });

  it("returns Deep at 2.5 hours (mid-range of deep window)", () => {
    const result = estimate({
      accelerometerVariance: 0.02,
      microphoneLevel: 0.05,
      timeOfNight: 2.5,
    });
    expect(result.likelyStage).toBe("Deep");
  });
});

// ── REM detection (time-of-night) ───────────────────────────────────────────

describe("REM detection (still + quiet + later in night)", () => {
  it("returns REM at 4-7 hours into sleep when still and quiet", () => {
    const result = estimate({
      accelerometerVariance: 0.05,
      microphoneLevel: 0.1,
      timeOfNight: 5.0,
    });
    expect(result.likelyStage).toBe("REM");
    expect(result.dreamLikelihood).toBeGreaterThanOrEqual(0.4);
  });

  it("returns REM at 6 hours with high dream likelihood", () => {
    const result = estimate({
      accelerometerVariance: 0.01,
      microphoneLevel: 0.0,
      timeOfNight: 6.0,
    });
    expect(result.likelyStage).toBe("REM");
    expect(result.dreamLikelihood).toBeGreaterThanOrEqual(0.5);
  });
});

// ── Early sleep (< 0.5h) ───────────────────────────────────────────────────

describe("Early sleep (< 0.5h)", () => {
  it("returns Light in the first 30 minutes when still", () => {
    const result = estimate({
      accelerometerVariance: 0.05,
      microphoneLevel: 0.1,
      timeOfNight: 0.2,
    });
    expect(result.likelyStage).toBe("Light");
  });
});

// ── Late sleep (7h+) ───────────────────────────────────────────────────────

describe("Late sleep (7h+)", () => {
  it("returns Light at 8 hours with moderate dream likelihood", () => {
    const result = estimate({
      accelerometerVariance: 0.05,
      microphoneLevel: 0.1,
      timeOfNight: 8.0,
    });
    expect(result.likelyStage).toBe("Light");
    expect(result.dreamLikelihood).toBeGreaterThan(0.2);
  });
});

// ── Transition zone (3-4h) ──────────────────────────────────────────────────

describe("Transition zone (3-4h)", () => {
  it("returns Light at 3.5 hours (transition period)", () => {
    const result = estimate({
      accelerometerVariance: 0.05,
      microphoneLevel: 0.1,
      timeOfNight: 3.5,
    });
    expect(result.likelyStage).toBe("Light");
    expect(result.dreamLikelihood).toBeGreaterThan(0.1);
  });
});

// ── Heart rate: Deep sleep detection ────────────────────────────────────────

describe("Heart rate: Deep sleep (low HR + still)", () => {
  it("returns Deep when HR is low and body is still", () => {
    const result = estimate({
      accelerometerVariance: 0.05,
      microphoneLevel: 0.1,
      timeOfNight: 2.0,
      heartRate: 52,
    });
    expect(result.likelyStage).toBe("Deep");
    expect(result.dreamLikelihood).toBeLessThanOrEqual(0.15);
  });

  it("returns Deep even at 5h into night if HR is very low", () => {
    const result = estimate({
      accelerometerVariance: 0.05,
      microphoneLevel: 0.1,
      timeOfNight: 5.0,
      heartRate: 50,
    });
    expect(result.likelyStage).toBe("Deep");
  });
});

// ── Heart rate: REM detection ───────────────────────────────────────────────

describe("Heart rate: REM (elevated HR + still)", () => {
  it("returns REM when HR is elevated and body is still", () => {
    const result = estimate({
      accelerometerVariance: 0.05,
      microphoneLevel: 0.1,
      timeOfNight: 5.0,
      heartRate: 72,
    });
    expect(result.likelyStage).toBe("REM");
    expect(result.dreamLikelihood).toBeGreaterThan(0.3);
  });

  it("has higher dream likelihood later in the night", () => {
    const early = estimate({
      accelerometerVariance: 0.05,
      microphoneLevel: 0.1,
      timeOfNight: 1.0,
      heartRate: 70,
    });
    const late = estimate({
      accelerometerVariance: 0.05,
      microphoneLevel: 0.1,
      timeOfNight: 5.5,
      heartRate: 70,
    });
    expect(late.dreamLikelihood).toBeGreaterThan(early.dreamLikelihood);
  });
});

// ── Heart rate: mid-range falls through to time-based ───────────────────────

describe("Heart rate: mid-range (60-64 BPM)", () => {
  it("falls through to time-based estimate when HR is mid-range", () => {
    const withHR = estimate({
      accelerometerVariance: 0.05,
      microphoneLevel: 0.1,
      timeOfNight: 5.0,
      heartRate: 62,
    });
    const withoutHR = estimate({
      accelerometerVariance: 0.05,
      microphoneLevel: 0.1,
      timeOfNight: 5.0,
    });
    // Mid-range HR should produce the same result as no HR
    expect(withHR.likelyStage).toBe(withoutHR.likelyStage);
  });
});

// ── Output validation ───────────────────────────────────────────────────────

describe("Output validation", () => {
  const scenarios: PhoneSleepInput[] = [
    { accelerometerVariance: 0, microphoneLevel: 0, timeOfNight: 0 },
    { accelerometerVariance: 1, microphoneLevel: 1, timeOfNight: 10 },
    { accelerometerVariance: 0.5, microphoneLevel: 0.5, timeOfNight: 5 },
    { accelerometerVariance: 0.1, microphoneLevel: 0.1, timeOfNight: 2, heartRate: 55 },
    { accelerometerVariance: 0.1, microphoneLevel: 0.1, timeOfNight: 5, heartRate: 75 },
  ];

  it.each(scenarios)(
    "confidence is 0-1 for input %o",
    (input) => {
      const result = estimateSleepFromPhone(input);
      expect(result.confidence).toBeGreaterThanOrEqual(0);
      expect(result.confidence).toBeLessThanOrEqual(1);
    },
  );

  it.each(scenarios)(
    "dreamLikelihood is 0-1 for input %o",
    (input) => {
      const result = estimateSleepFromPhone(input);
      expect(result.dreamLikelihood).toBeGreaterThanOrEqual(0);
      expect(result.dreamLikelihood).toBeLessThanOrEqual(1);
    },
  );

  it.each(scenarios)(
    "likelyStage is a valid stage for input %o",
    (input) => {
      const result = estimateSleepFromPhone(input);
      expect(["Wake", "Light", "Deep", "REM"]).toContain(result.likelyStage);
    },
  );

  it.each(scenarios)(
    "reasoning is a non-empty string for input %o",
    (input) => {
      const result = estimateSleepFromPhone(input);
      expect(typeof result.reasoning).toBe("string");
      expect(result.reasoning.length).toBeGreaterThan(0);
    },
  );
});

// ── Edge cases ──────────────────────────────────────────────────────────────

describe("Edge cases", () => {
  it("handles zero inputs gracefully", () => {
    const result = estimateSleepFromPhone({
      accelerometerVariance: 0,
      microphoneLevel: 0,
      timeOfNight: 0,
    });
    expect(result.likelyStage).toBeDefined();
  });

  it("clamps out-of-range accelerometer values", () => {
    const result = estimateSleepFromPhone({
      accelerometerVariance: 5.0, // way above 1
      microphoneLevel: 0,
      timeOfNight: 2,
    });
    expect(result.likelyStage).toBe("Wake");
  });

  it("handles negative timeOfNight as 0", () => {
    const result = estimateSleepFromPhone({
      accelerometerVariance: 0.05,
      microphoneLevel: 0.1,
      timeOfNight: -1,
    });
    // time < 0.5 → Light
    expect(result.likelyStage).toBe("Light");
  });

  it("handles very high heart rate without crashing", () => {
    const result = estimateSleepFromPhone({
      accelerometerVariance: 0.05,
      microphoneLevel: 0.1,
      timeOfNight: 5,
      heartRate: 180,
    });
    expect(result.likelyStage).toBe("REM"); // elevated HR + still
  });

  it("handles heartRate of 0 as low HR", () => {
    const result = estimateSleepFromPhone({
      accelerometerVariance: 0.05,
      microphoneLevel: 0.1,
      timeOfNight: 2,
      heartRate: 0,
    });
    expect(result.likelyStage).toBe("Deep"); // HR < 60 + still
  });

  it("movement overrides HR — Wake even with sleep-like HR", () => {
    const result = estimateSleepFromPhone({
      accelerometerVariance: 0.8,
      microphoneLevel: 0,
      timeOfNight: 5,
      heartRate: 55,
    });
    expect(result.likelyStage).toBe("Wake");
  });
});

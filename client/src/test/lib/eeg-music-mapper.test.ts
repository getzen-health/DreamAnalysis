import { describe, it, expect } from "vitest";
import {
  mapEegToMusic,
  intervalToFreq,
  getScaleIntervals,
  pickNote,
  EegSonifier,
  type EegBandPowers,
  type SonificationParams,
} from "@/lib/eeg-music-mapper";

// ── Helpers ──────────────────────────────────────────────────────────────────

function makeBands(overrides: Partial<EegBandPowers> = {}): EegBandPowers {
  return {
    delta: 0.3,
    theta: 0.3,
    alpha: 0.5,
    beta: 0.3,
    ...overrides,
  };
}

function assertValidParams(params: SonificationParams) {
  expect(params.noteDuration).toBeGreaterThanOrEqual(200);
  expect(params.noteDuration).toBeLessThanOrEqual(1500);
  expect(params.volume).toBeGreaterThanOrEqual(0.1);
  expect(params.volume).toBeLessThanOrEqual(0.8);
  expect(params.overtones).toBeGreaterThanOrEqual(1);
  expect(params.overtones).toBeLessThanOrEqual(6);
  expect(params.bassFreq).toBeGreaterThanOrEqual(55);
  expect(params.bassFreq).toBeLessThanOrEqual(110);
  expect(["major", "minor", "pentatonic"]).toContain(params.scale);
  expect(params.baseFreq).toBeCloseTo(261.63, 1);
}

// ── mapEegToMusic ───────────────────────────────────────────────────────────

describe("mapEegToMusic", () => {
  it("returns valid params for default bands", () => {
    const params = mapEegToMusic(makeBands());
    assertValidParams(params);
  });

  it("high alpha produces slower notes (longer duration)", () => {
    const lowAlpha = mapEegToMusic(makeBands({ alpha: 0.1 }));
    const highAlpha = mapEegToMusic(makeBands({ alpha: 0.9 }));
    expect(highAlpha.noteDuration).toBeGreaterThan(lowAlpha.noteDuration);
  });

  it("high beta produces louder volume", () => {
    const lowBeta = mapEegToMusic(makeBands({ beta: 0.1 }));
    const highBeta = mapEegToMusic(makeBands({ beta: 0.9 }));
    expect(highBeta.volume).toBeGreaterThan(lowBeta.volume);
  });

  it("high theta produces more overtones", () => {
    const lowTheta = mapEegToMusic(makeBands({ theta: 0.0 }));
    const highTheta = mapEegToMusic(makeBands({ theta: 1.0 }));
    expect(highTheta.overtones).toBeGreaterThan(lowTheta.overtones);
  });

  it("high delta produces lower bass frequency", () => {
    const lowDelta = mapEegToMusic(makeBands({ delta: 0.1 }));
    const highDelta = mapEegToMusic(makeBands({ delta: 0.9 }));
    expect(highDelta.bassFreq).toBeLessThan(lowDelta.bassFreq);
  });

  it("positive valence selects major scale", () => {
    const params = mapEegToMusic(makeBands(), 0.5);
    expect(params.scale).toBe("major");
  });

  it("negative valence selects minor scale", () => {
    const params = mapEegToMusic(makeBands(), -0.5);
    expect(params.scale).toBe("minor");
  });

  it("neutral valence selects pentatonic scale", () => {
    const params = mapEegToMusic(makeBands(), 0.0);
    expect(params.scale).toBe("pentatonic");
  });

  it("clamps extreme band values", () => {
    const params = mapEegToMusic({
      delta: 5.0,
      theta: -1.0,
      alpha: 10.0,
      beta: -5.0,
    });
    assertValidParams(params);
  });
});

// ── intervalToFreq ──────────────────────────────────────────────────────────

describe("intervalToFreq", () => {
  it("returns base freq for 0 semitones", () => {
    expect(intervalToFreq(440, 0)).toBeCloseTo(440, 1);
  });

  it("returns octave for 12 semitones", () => {
    expect(intervalToFreq(440, 12)).toBeCloseTo(880, 1);
  });

  it("returns correct frequency for 7 semitones (perfect fifth)", () => {
    expect(intervalToFreq(440, 7)).toBeCloseTo(659.26, 0);
  });
});

// ── getScaleIntervals ───────────────────────────────────────────────────────

describe("getScaleIntervals", () => {
  it("major scale has 8 notes", () => {
    expect(getScaleIntervals("major")).toHaveLength(8);
  });

  it("minor scale has 8 notes", () => {
    expect(getScaleIntervals("minor")).toHaveLength(8);
  });

  it("pentatonic scale has 6 notes", () => {
    expect(getScaleIntervals("pentatonic")).toHaveLength(6);
  });

  it("scales start at 0 and end at 12", () => {
    for (const scale of ["major", "minor", "pentatonic"] as const) {
      const intervals = getScaleIntervals(scale);
      expect(intervals[0]).toBe(0);
      expect(intervals[intervals.length - 1]).toBe(12);
    }
  });

  it("returns a copy, not the original array", () => {
    const a = getScaleIntervals("major");
    const b = getScaleIntervals("major");
    expect(a).not.toBe(b);
    expect(a).toEqual(b);
  });
});

// ── pickNote ────────────────────────────────────────────────────────────────

describe("pickNote", () => {
  it("returns a positive frequency", () => {
    for (let i = 0; i < 20; i++) {
      const freq = pickNote(261.63, "major");
      expect(freq).toBeGreaterThan(0);
    }
  });

  it("frequency is within one octave of base", () => {
    for (let i = 0; i < 50; i++) {
      const freq = pickNote(261.63, "major");
      expect(freq).toBeGreaterThanOrEqual(261.63 * 0.99); // allow rounding
      expect(freq).toBeLessThanOrEqual(261.63 * 2.01);
    }
  });
});

// ── EegSonifier ─────────────────────────────────────────────────────────────

describe("EegSonifier", () => {
  it("initializes without playing", () => {
    const sonifier = new EegSonifier();
    expect(sonifier.isPlaying).toBe(false);
  });

  it("currentParams returns valid defaults", () => {
    const sonifier = new EegSonifier();
    assertValidParams(sonifier.currentParams);
  });

  it("stop is safe to call when not playing", () => {
    const sonifier = new EegSonifier();
    expect(() => sonifier.stop()).not.toThrow();
  });

  it("update changes params without starting", () => {
    const sonifier = new EegSonifier();
    sonifier.update(makeBands({ alpha: 0.9 }), 0.5);
    expect(sonifier.currentParams.scale).toBe("major");
    expect(sonifier.isPlaying).toBe(false);
  });
});

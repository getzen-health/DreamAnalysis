/**
 * Tests for feature-level fusion of HR/HRV data (#492).
 *
 * These test the enhanced readHealthSource() that incorporates
 * heart rate and HRV into stress/arousal estimation.
 */

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import {
  _readHealthSource as readHealthSource,
  _fuse as fuse,
  type _SourceReading as SourceReading,
} from "@/lib/data-fusion";

beforeEach(() => {
  localStorage.clear();
});

afterEach(() => {
  localStorage.clear();
});

describe("readHealthSource with HR/HRV fusion (#492)", () => {
  it("returns base values when no HR or HRV present", () => {
    localStorage.setItem("ndw_health_emotion", JSON.stringify({
      emotion: "neutral",
      stress: 0.4,
      arousal: 0.5,
      timestamp: Date.now(),
    }));

    const result = readHealthSource();
    expect(result).not.toBeNull();
    expect(result!.stress).toBeCloseTo(0.4, 1);
    expect(result!.arousal).toBeCloseTo(0.5, 1);
  });

  it("increases stress when HR > 100 bpm", () => {
    localStorage.setItem("ndw_health_emotion", JSON.stringify({
      emotion: "neutral",
      stress: 0.3,
      arousal: 0.4,
      heart_rate: 120,
      timestamp: Date.now(),
    }));

    const result = readHealthSource();
    expect(result).not.toBeNull();
    // Stress should be higher than base 0.3
    expect(result!.stress).toBeGreaterThan(0.3);
    // Arousal should also increase
    expect(result!.arousal).toBeGreaterThan(0.4);
  });

  it("does not change stress when HR <= 100 bpm", () => {
    localStorage.setItem("ndw_health_emotion", JSON.stringify({
      emotion: "neutral",
      stress: 0.3,
      arousal: 0.4,
      heart_rate: 80,
      timestamp: Date.now(),
    }));

    const result = readHealthSource();
    expect(result).not.toBeNull();
    expect(result!.stress).toBeCloseTo(0.3, 1);
  });

  it("increases stress when HRV is low (< 40ms)", () => {
    localStorage.setItem("ndw_health_emotion", JSON.stringify({
      emotion: "neutral",
      stress: 0.3,
      hrv: 20,
      timestamp: Date.now(),
    }));

    const result = readHealthSource();
    expect(result).not.toBeNull();
    expect(result!.stress).toBeGreaterThan(0.3);
  });

  it("decreases stress when HRV is high (> 60ms)", () => {
    localStorage.setItem("ndw_health_emotion", JSON.stringify({
      emotion: "neutral",
      stress: 0.5,
      hrv: 80,
      timestamp: Date.now(),
    }));

    const result = readHealthSource();
    expect(result).not.toBeNull();
    expect(result!.stress).toBeLessThan(0.5);
  });

  it("does not change stress when HRV is in normal range (40-60ms)", () => {
    localStorage.setItem("ndw_health_emotion", JSON.stringify({
      emotion: "neutral",
      stress: 0.4,
      hrv: 50,
      timestamp: Date.now(),
    }));

    const result = readHealthSource();
    expect(result).not.toBeNull();
    expect(result!.stress).toBeCloseTo(0.4, 1);
  });

  it("clamps stress to [0, 1] with extreme HR", () => {
    localStorage.setItem("ndw_health_emotion", JSON.stringify({
      emotion: "neutral",
      stress: 0.9,
      heart_rate: 200,
      timestamp: Date.now(),
    }));

    const result = readHealthSource();
    expect(result).not.toBeNull();
    expect(result!.stress).toBeLessThanOrEqual(1);
    expect(result!.stress).toBeGreaterThanOrEqual(0);
  });

  it("combines HR and HRV effects", () => {
    // High HR + Low HRV = maximum stress bump
    localStorage.setItem("ndw_health_emotion", JSON.stringify({
      emotion: "neutral",
      stress: 0.3,
      heart_rate: 130,
      hrv: 15,
      timestamp: Date.now(),
    }));

    const result = readHealthSource();
    expect(result).not.toBeNull();
    // Should be notably higher than 0.3
    expect(result!.stress).toBeGreaterThan(0.45);
  });

  it("reads heartRate as alternate key", () => {
    localStorage.setItem("ndw_health_emotion", JSON.stringify({
      emotion: "neutral",
      stress: 0.3,
      heartRate: 120,
      timestamp: Date.now(),
    }));

    const result = readHealthSource();
    expect(result).not.toBeNull();
    expect(result!.stress).toBeGreaterThan(0.3);
  });

  it("reads heart_rate_variability as alternate key", () => {
    localStorage.setItem("ndw_health_emotion", JSON.stringify({
      emotion: "neutral",
      stress: 0.3,
      heart_rate_variability: 20,
      timestamp: Date.now(),
    }));

    const result = readHealthSource();
    expect(result).not.toBeNull();
    expect(result!.stress).toBeGreaterThan(0.3);
  });
});

describe("fuse with HR/HRV-enhanced health source (#492)", () => {
  it("health HR/HRV data influences fused stress when present", () => {
    const now = Date.now();

    // Health source with high HR will have elevated stress
    localStorage.setItem("ndw_health_emotion", JSON.stringify({
      emotion: "neutral",
      stress: 0.3,
      heart_rate: 130,
      confidence: 0.5,
      timestamp: now,
    }));

    const health = readHealthSource();
    expect(health).not.toBeNull();

    // Without HR, base stress would be 0.3.
    // With HR > 100, it should be higher.
    const eeg: SourceReading = {
      stress: 0.2, focus: 0.8, valence: 0.5, arousal: 0.5,
      emotion: "happy", confidence: 0.9, timestamp: now,
    };

    const result = fuse([
      { source: "eeg", reading: eeg },
      { source: "health", reading: health! },
    ]);

    // The fused stress should be influenced by the health's elevated stress
    // But EEG dominates (weight 0.50 vs health 0.15), so it should still be lowish
    expect(result.stress).toBeGreaterThan(0.15);
    expect(result.stress).toBeLessThan(0.8);
  });
});

import { describe, it, expect, beforeEach, vi } from "vitest";
import { DeviationDetector } from "@/lib/insight-engine/deviation-detector";
import { BaselineStore } from "@/lib/insight-engine/baseline-store";

beforeEach(() => localStorage.clear());

describe("DeviationDetector.detect", () => {
  it("returns no events when reading is within baseline", () => {
    const store = new BaselineStore();
    const detector = new DeviationDetector(store);
    // stress at population mean — z-score ~0
    const events = detector.detect({ stress: 0.40, focus: 0.55, valence: 0.55, arousal: 0.50 });
    expect(events).toHaveLength(0);
  });

  it("fires DeviationEvent when |zScore| > 1.5", () => {
    const store = new BaselineStore();
    const detector = new DeviationDetector(store);
    // stress = 0.70, population mean = 0.40, std = 0.15 → z = 2.0
    const events = detector.detect({ stress: 0.70, focus: 0.55, valence: 0.55, arousal: 0.50 });
    expect(events.length).toBeGreaterThanOrEqual(1);
    const stressEvent = events.find(e => e.metric === "stress");
    expect(stressEvent).toBeDefined();
    expect(stressEvent!.zScore).toBeGreaterThan(1.5);
    expect(stressEvent!.direction).toBe("high");
  });

  it("sets direction=low for below-baseline reading", () => {
    const store = new BaselineStore();
    const detector = new DeviationDetector(store);
    // focus = 0.10, population mean = 0.55, std = 0.18 → z ≈ -2.5
    const events = detector.detect({ stress: 0.40, focus: 0.10, valence: 0.55, arousal: 0.50 });
    const focusEvent = events.find(e => e.metric === "focus");
    expect(focusEvent!.direction).toBe("low");
  });

  it("starts timer on first deviation and populates durationMinutes", () => {
    vi.useFakeTimers();
    const now = Date.now();
    vi.setSystemTime(now - 5 * 60 * 1000); // 5 min ago
    const store = new BaselineStore();
    const detector = new DeviationDetector(store);
    // First detection starts timer
    detector.detect({ stress: 0.70, focus: 0.55, valence: 0.55, arousal: 0.50 });

    vi.setSystemTime(now); // now, 5 min later
    const events = detector.detect({ stress: 0.70, focus: 0.55, valence: 0.55, arousal: 0.50 });
    const stressEvent = events.find(e => e.metric === "stress");
    expect(stressEvent!.durationMinutes).toBeGreaterThanOrEqual(4.9);
    vi.useRealTimers();
  });

  it("clears timer when deviation recovers (|z| <= 1.0)", () => {
    const store = new BaselineStore();
    const detector = new DeviationDetector(store);
    // Start deviation
    detector.detect({ stress: 0.70, focus: 0.55, valence: 0.55, arousal: 0.50 });
    // Recover
    detector.detect({ stress: 0.42, focus: 0.55, valence: 0.55, arousal: 0.50 });
    // Next detection should not have a timer
    const events = detector.detect({ stress: 0.70, focus: 0.55, valence: 0.55, arousal: 0.50 });
    const stressEvent = events.find(e => e.metric === "stress");
    expect(stressEvent!.durationMinutes).toBeLessThan(0.1); // fresh timer
  });
});

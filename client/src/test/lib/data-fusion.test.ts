import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import {
  dataFusionBus,
  _fuse as fuse,
  _readEEGSource as readEEGSource,
  _readVoiceSource as readVoiceSource,
  _readHealthSource as readHealthSource,
  type FusedState,
  type _SourceReading as SourceReading,
} from "@/lib/data-fusion";

beforeEach(() => {
  localStorage.clear();
});

afterEach(() => {
  localStorage.clear();
  dataFusionBus.destroy();
});

// ── Source readers ──────────────────────────────────────────────────────────

describe("readEEGSource", () => {
  it("returns null when no EEG data in localStorage", () => {
    expect(readEEGSource()).toBeNull();
  });

  it("reads EEG emotion from localStorage", () => {
    localStorage.setItem("ndw_last_eeg_emotion", JSON.stringify({
      emotion: "happy",
      valence: 0.6,
      arousal: 0.7,
      stress_index: 0.2,
      focus_index: 0.8,
      confidence: 0.75,
      timestamp: 1000,
    }));

    const result = readEEGSource();
    expect(result).not.toBeNull();
    expect(result!.emotion).toBe("happy");
    expect(result!.valence).toBe(0.6);
    expect(result!.arousal).toBe(0.7);
    expect(result!.stress).toBe(0.2);
    expect(result!.focus).toBe(0.8);
    expect(result!.confidence).toBe(0.75);
  });

  it("returns null for invalid JSON", () => {
    localStorage.setItem("ndw_last_eeg_emotion", "not-json");
    expect(readEEGSource()).toBeNull();
  });

  it("returns null when emotion field is missing", () => {
    localStorage.setItem("ndw_last_eeg_emotion", JSON.stringify({
      valence: 0.5,
    }));
    expect(readEEGSource()).toBeNull();
  });
});

describe("readVoiceSource", () => {
  it("returns null when no voice data", () => {
    expect(readVoiceSource()).toBeNull();
  });

  it("reads voice data with result wrapper", () => {
    localStorage.setItem("ndw_last_emotion", JSON.stringify({
      result: {
        emotion: "sad",
        valence: -0.3,
        arousal: 0.4,
        stress_index: 0.6,
        focus_index: 0.3,
        confidence: 0.65,
      },
      timestamp: 2000,
    }));

    const result = readVoiceSource();
    expect(result).not.toBeNull();
    expect(result!.emotion).toBe("sad");
    expect(result!.valence).toBe(-0.3);
    expect(result!.timestamp).toBe(2000);
  });

  it("reads voice data without result wrapper", () => {
    localStorage.setItem("ndw_last_emotion", JSON.stringify({
      emotion: "neutral",
      valence: 0.0,
      arousal: 0.5,
      confidence: 0.5,
    }));

    const result = readVoiceSource();
    expect(result).not.toBeNull();
    expect(result!.emotion).toBe("neutral");
  });
});

describe("readHealthSource", () => {
  it("returns null when no health data", () => {
    expect(readHealthSource()).toBeNull();
  });

  it("reads health emotion from localStorage", () => {
    localStorage.setItem("ndw_health_emotion", JSON.stringify({
      emotion: "calm",
      valence: 0.3,
      arousal: 0.3,
      stress: 0.2,
      focus: 0.5,
      confidence: 0.4,
      timestamp: 3000,
    }));

    const result = readHealthSource();
    expect(result).not.toBeNull();
    expect(result!.emotion).toBe("calm");
    expect(result!.stress).toBe(0.2);
  });
});

// ── Fusion logic ───────────────────────────────────────────────────────────

describe("fuse", () => {
  it("returns neutral defaults when no sources", () => {
    const result = fuse([]);
    expect(result.emotion).toBe("neutral");
    expect(result.confidence).toBe(0);
    expect(result.source).toBe("fused");
  });

  it("passes through single source directly", () => {
    const reading: SourceReading = {
      stress: 0.3,
      focus: 0.7,
      valence: 0.5,
      arousal: 0.6,
      emotion: "happy",
      confidence: 0.8,
      timestamp: Date.now(),
    };

    const result = fuse([{ source: "eeg", reading }]);
    expect(result.source).toBe("eeg");
    expect(result.emotion).toBe("happy");
    expect(result.stress).toBe(0.3);
    expect(result.focus).toBe(0.7);
    expect(result.mood).toBeCloseTo(0.75, 1); // (0.5 + 1) / 2
  });

  it("fuses two sources with weighted average", () => {
    const now = Date.now();
    const eeg: SourceReading = {
      stress: 0.2, focus: 0.8, valence: 0.6, arousal: 0.7,
      emotion: "happy", confidence: 0.9, timestamp: now,
    };
    const voice: SourceReading = {
      stress: 0.4, focus: 0.6, valence: 0.2, arousal: 0.5,
      emotion: "neutral", confidence: 0.7, timestamp: now,
    };

    const result = fuse([
      { source: "eeg", reading: eeg },
      { source: "voice", reading: voice },
    ]);

    expect(result.source).toBe("fused");
    // EEG has higher weight (0.50 * 0.9 = 0.45) vs voice (0.35 * 0.7 = 0.245)
    // So emotion should be from EEG (highest weight)
    expect(result.emotion).toBe("happy");
    // Stress should be closer to EEG's 0.2 than voice's 0.4
    expect(result.stress).toBeLessThan(0.35);
    expect(result.confidence).toBeGreaterThan(0);
  });

  it("EEG dominates when all three sources present", () => {
    const now = Date.now();
    const sources = [
      { source: "eeg", reading: { stress: 0.1, focus: 0.9, valence: 0.8, arousal: 0.8, emotion: "happy", confidence: 0.9, timestamp: now } },
      { source: "voice", reading: { stress: 0.5, focus: 0.5, valence: 0.0, arousal: 0.5, emotion: "neutral", confidence: 0.7, timestamp: now } },
      { source: "health", reading: { stress: 0.3, focus: 0.6, valence: 0.2, arousal: 0.4, emotion: "calm", confidence: 0.4, timestamp: now } },
    ];

    const result = fuse(sources);
    expect(result.emotion).toBe("happy"); // EEG has highest effective weight
    expect(result.source).toBe("fused");
  });

  it("discounts stale readings", () => {
    const now = Date.now();
    const stale = now - 10 * 60 * 1000; // 10 minutes ago

    const fresh: SourceReading = {
      stress: 0.2, focus: 0.8, valence: 0.6, arousal: 0.7,
      emotion: "happy", confidence: 0.8, timestamp: now,
    };
    const old: SourceReading = {
      stress: 0.8, focus: 0.2, valence: -0.5, arousal: 0.3,
      emotion: "sad", confidence: 0.8, timestamp: stale,
    };

    const result = fuse([
      { source: "voice", reading: fresh },
      { source: "health", reading: old },
    ]);

    // Fresh voice should dominate over stale health
    expect(result.emotion).toBe("happy");
    expect(result.stress).toBeLessThan(0.5);
  });

  it("clamps values to valid ranges", () => {
    const result = fuse([{
      source: "eeg",
      reading: {
        stress: 1.5, focus: -0.3, valence: 2.0, arousal: -1.0,
        emotion: "happy", confidence: 1.2, timestamp: Date.now(),
      },
    }]);

    expect(result.stress).toBeLessThanOrEqual(1);
    expect(result.stress).toBeGreaterThanOrEqual(0);
    expect(result.focus).toBeGreaterThanOrEqual(0);
    expect(result.valence).toBeLessThanOrEqual(1);
    expect(result.arousal).toBeGreaterThanOrEqual(0);
    expect(result.mood).toBeGreaterThanOrEqual(0);
    expect(result.mood).toBeLessThanOrEqual(1);
  });
});

// ── DataFusionBus ──────────────────────────────────────────────────────────

describe("DataFusionBus", () => {
  it("returns null state before any data", () => {
    dataFusionBus.initialize();
    expect(dataFusionBus.getState()).toBeNull();
  });

  it("computes state when EEG data is present", () => {
    localStorage.setItem("ndw_last_eeg_emotion", JSON.stringify({
      emotion: "happy", valence: 0.5, arousal: 0.6,
      stress_index: 0.2, focus_index: 0.8, confidence: 0.9,
      timestamp: Date.now(),
    }));

    dataFusionBus.initialize();
    dataFusionBus.recompute();

    const state = dataFusionBus.getState();
    expect(state).not.toBeNull();
    expect(state!.emotion).toBe("happy");
    expect(state!.source).toBe("eeg");
  });

  it("notifies subscribers on recompute", () => {
    const listener = vi.fn();
    dataFusionBus.initialize();
    dataFusionBus.subscribe(listener);

    localStorage.setItem("ndw_last_eeg_emotion", JSON.stringify({
      emotion: "sad", valence: -0.3, arousal: 0.4,
      stress_index: 0.6, focus_index: 0.3, confidence: 0.7,
      timestamp: Date.now(),
    }));

    dataFusionBus.recompute();

    expect(listener).toHaveBeenCalled();
    const state: FusedState = listener.mock.calls[listener.mock.calls.length - 1][0];
    expect(state.emotion).toBe("sad");
  });

  it("unsubscribe stops notifications", () => {
    const listener = vi.fn();
    dataFusionBus.initialize();
    const unsub = dataFusionBus.subscribe(listener);
    listener.mockClear();

    unsub();

    localStorage.setItem("ndw_last_eeg_emotion", JSON.stringify({
      emotion: "happy", valence: 0.5, arousal: 0.6,
      confidence: 0.9, timestamp: Date.now(),
    }));
    dataFusionBus.recompute();

    expect(listener).not.toHaveBeenCalled();
  });

  it("sends current state to new subscriber immediately", () => {
    localStorage.setItem("ndw_last_eeg_emotion", JSON.stringify({
      emotion: "neutral", valence: 0.0, arousal: 0.5,
      confidence: 0.6, timestamp: Date.now(),
    }));

    dataFusionBus.initialize();
    dataFusionBus.recompute();

    const listener = vi.fn();
    dataFusionBus.subscribe(listener);

    expect(listener).toHaveBeenCalledTimes(1);
    expect(listener.mock.calls[0][0].emotion).toBe("neutral");
  });

  it("responds to window events", () => {
    localStorage.setItem("ndw_last_emotion", JSON.stringify({
      result: { emotion: "happy", valence: 0.5, arousal: 0.6, confidence: 0.8 },
      timestamp: Date.now(),
    }));

    dataFusionBus.initialize();
    window.dispatchEvent(new Event("ndw-voice-updated"));

    const state = dataFusionBus.getState();
    expect(state).not.toBeNull();
    expect(state!.emotion).toBe("happy");
  });

  it("destroy removes event listeners", () => {
    const listener = vi.fn();
    dataFusionBus.initialize();
    dataFusionBus.subscribe(listener);
    listener.mockClear();

    dataFusionBus.destroy();

    localStorage.setItem("ndw_last_eeg_emotion", JSON.stringify({
      emotion: "angry", valence: -0.5, arousal: 0.8,
      confidence: 0.7, timestamp: Date.now(),
    }));
    window.dispatchEvent(new Event("ndw-eeg-updated"));

    // Listener should NOT have been called because bus is destroyed
    expect(listener).not.toHaveBeenCalled();
  });
});

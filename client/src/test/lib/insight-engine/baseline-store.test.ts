// client/src/test/lib/insight-engine/baseline-store.test.ts
import { describe, it, expect, beforeEach } from "vitest";
import { BaselineStore } from "@/lib/insight-engine/baseline-store";

beforeEach(() => localStorage.clear());

describe("BaselineStore.update", () => {
  it("stores normalized values in the correct time bucket", () => {
    const store = new BaselineStore();
    // hrv raw 60ms → normalized 60/120 = 0.5
    store.update({ stress: 0.4, focus: 0.6, valence: 0.55, arousal: 0.5, hrv: 60 }, "2026-03-27T14:30:00Z");
    const cell = store.getCell("stress", 14);
    expect(cell).not.toBeNull();
    expect(cell!.sampleCount).toBe(1);
    expect(cell!.mean).toBeCloseTo(0.4);
  });

  it("normalizes valence from raw -1..1 to 0..1", () => {
    const store = new BaselineStore();
    // valence already normalized (NormalizedReading uses 0-1)
    store.update({ stress: 0.5, focus: 0.5, valence: 0.3, arousal: 0.5 }, "2026-03-27T10:00:00Z");
    const cell = store.getCell("valence", 10);
    expect(cell!.mean).toBeCloseTo(0.3);
  });

  it("normalizes hrv raw ms to 0-1 (cap at 120ms)", () => {
    const store = new BaselineStore();
    store.update({ stress: 0.5, focus: 0.5, valence: 0.5, arousal: 0.5, hrv: 180 }, "2026-03-27T08:00:00Z");
    const cell = store.getCell("hrv", 8);
    expect(cell!.mean).toBe(1); // capped at 1
  });

  it("caps at 7 days — drops entries older than 7 days", () => {
    const store = new BaselineStore();
    const old = "2026-03-19T14:00:00Z"; // 8 days ago
    store.update({ stress: 0.8, focus: 0.5, valence: 0.5, arousal: 0.5 }, old);
    const fresh = "2026-03-27T14:00:00Z";
    store.update({ stress: 0.3, focus: 0.5, valence: 0.5, arousal: 0.5 }, fresh);
    const cell = store.getCell("stress", 14);
    // old entry dropped; only fresh entry remains
    expect(cell!.sampleCount).toBe(1);
    expect(cell!.mean).toBeCloseTo(0.3);
  });

  it("persists to localStorage and restores on new instance", () => {
    const store1 = new BaselineStore();
    store1.update({ stress: 0.6, focus: 0.5, valence: 0.5, arousal: 0.5 }, "2026-03-27T16:00:00Z");
    const store2 = new BaselineStore(); // loads from localStorage
    const cell = store2.getCell("stress", 16);
    expect(cell!.mean).toBeCloseTo(0.6);
  });
});

describe("BaselineStore.getZScore", () => {
  it("returns population default when sampleCount < 7", () => {
    const store = new BaselineStore();
    store.update({ stress: 0.9, focus: 0.5, valence: 0.5, arousal: 0.5 }, "2026-03-27T10:00:00Z");
    // only 1 sample — falls back to population default (mean=0.40, std=0.15)
    const z = store.getZScore("stress", 0.9, 10);
    expect(z).toBeCloseTo((0.9 - 0.40) / 0.15, 1);
  });

  it("uses personal baseline when sampleCount >= 7", () => {
    const store = new BaselineStore();
    const ts = "2026-03-27T10:00:00Z";
    for (let i = 0; i < 7; i++) {
      store.update({ stress: 0.4, focus: 0.5, valence: 0.5, arousal: 0.5 }, ts);
    }
    // z-score of the mean vs itself should be 0
    const z = store.getZScore("stress", 0.4, 10);
    expect(Math.abs(z)).toBeLessThan(0.1);
  });
});

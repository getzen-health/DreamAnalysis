import { describe, it, expect, beforeEach } from "vitest";
import {
  loadPersonalAdapter,
  savePersonalAdapter,
  applyAdapter,
  updateFromCorrection,
  updateAfterSession,
  getPersonalizationStats,
  resetPersonalAdapter,
  type PersonalAdapter,
} from "@/lib/personal-adapter";

// ── Clear localStorage before each test ────────────────────────────────────

beforeEach(() => {
  localStorage.clear();
});

// ── Helper: create a fresh adapter ─────────────────────────────────────────

function freshAdapter(): PersonalAdapter {
  return loadPersonalAdapter();
}

// ── 1. Fresh adapter has zero biases and unit multipliers ──────────────────

describe("fresh adapter", () => {
  it("has zero biases and unit multipliers", () => {
    const adapter = freshAdapter();
    expect(adapter.biases).toEqual([0, 0, 0, 0, 0, 0]);
    expect(adapter.multipliers).toEqual([1, 1, 1, 1, 1, 1]);
    expect(adapter.classCounts).toEqual([0, 0, 0, 0, 0, 0]);
    expect(adapter.correctionCounts).toEqual([0, 0, 0, 0, 0, 0]);
    expect(adapter.totalSessions).toBe(0);
    expect(adapter.totalCorrections).toBe(0);
    expect(adapter.lastUpdated).toBeTruthy();
  });
});

// ── 2. applyAdapter with zero biases returns input probabilities unchanged ─

describe("applyAdapter", () => {
  it("with zero biases returns input probabilities unchanged", () => {
    const adapter = freshAdapter();
    const probs = [0.3, 0.1, 0.1, 0.05, 0.35, 0.1];
    const result = applyAdapter(probs, adapter);

    for (let i = 0; i < probs.length; i++) {
      expect(result[i]).toBeCloseTo(probs[i], 4);
    }
  });

  // ── 3. Positive bias for class 0 increases class 0 probability ───────────

  it("with positive bias for class 0 increases class 0 probability", () => {
    const adapter = freshAdapter();
    adapter.biases[0] = 0.5;
    const probs = [0.2, 0.2, 0.15, 0.15, 0.15, 0.15];

    const result = applyAdapter(probs, adapter);
    expect(result[0]).toBeGreaterThan(probs[0]);
  });

  // ── 13. Output probabilities still sum to ~1.0 after adaptation ──────────

  it("output probabilities sum to ~1.0 after adaptation", () => {
    const adapter = freshAdapter();
    adapter.biases[0] = 0.3;
    adapter.biases[2] = -0.2;
    adapter.multipliers[1] = 1.3;
    adapter.multipliers[4] = 0.7;

    const probs = [0.2, 0.15, 0.1, 0.15, 0.25, 0.15];
    const result = applyAdapter(probs, adapter);

    const sum = result.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1.0, 4);
  });

  // ── 14. Learning rate of 0 produces no change ────────────────────────────

  it("with negative bias reduces that class probability", () => {
    const adapter = freshAdapter();
    adapter.biases[3] = -0.5;
    const probs = [0.15, 0.15, 0.15, 0.25, 0.15, 0.15];

    const result = applyAdapter(probs, adapter);
    expect(result[3]).toBeLessThan(probs[3]);
  });
});

// ── 4. Correction increases correct class bias, decreases wrong class bias ─

describe("updateFromCorrection", () => {
  it("increases correct class bias and decreases wrong class bias", () => {
    const adapter = freshAdapter();
    const updated = updateFromCorrection(adapter, 2, 0); // predicted angry, correct happy

    expect(updated.biases[0]).toBeGreaterThan(0); // correct class boosted
    expect(updated.biases[2]).toBeLessThan(0);    // wrong class reduced
    expect(updated.totalCorrections).toBe(1);
    expect(updated.correctionCounts[0]).toBe(1);
  });

  // ── 5. Multiple corrections for same class accumulate ────────────────────

  it("multiple corrections for same class accumulate", () => {
    let adapter = freshAdapter();
    adapter = updateFromCorrection(adapter, 2, 0);
    adapter = updateFromCorrection(adapter, 3, 0);
    adapter = updateFromCorrection(adapter, 1, 0);

    expect(adapter.biases[0]).toBeGreaterThan(0.1); // accumulated 3x boost
    expect(adapter.totalCorrections).toBe(3);
    expect(adapter.correctionCounts[0]).toBe(3);
  });

  // ── 6. Biases stay clamped to [-1, 1] ────────────────────────────────────

  it("biases stay clamped to [-1, 1]", () => {
    let adapter = freshAdapter();
    // Apply many corrections to push bias beyond bounds
    for (let i = 0; i < 30; i++) {
      adapter = updateFromCorrection(adapter, 1, 0);
    }
    expect(adapter.biases[0]).toBeLessThanOrEqual(1);
    expect(adapter.biases[0]).toBeGreaterThanOrEqual(-1);
    expect(adapter.biases[1]).toBeLessThanOrEqual(1);
    expect(adapter.biases[1]).toBeGreaterThanOrEqual(-1);
  });

  // ── 7. Multipliers stay clamped to [0.5, 1.5] ────────────────────────────

  it("multipliers stay clamped to [0.5, 1.5]", () => {
    let adapter = freshAdapter();
    // Push multipliers to extremes
    for (let i = 0; i < 50; i++) {
      adapter = updateFromCorrection(adapter, 1, 0);
    }
    expect(adapter.multipliers[0]).toBeLessThanOrEqual(1.5);
    expect(adapter.multipliers[0]).toBeGreaterThanOrEqual(0.5);
    expect(adapter.multipliers[1]).toBeLessThanOrEqual(1.5);
    expect(adapter.multipliers[1]).toBeGreaterThanOrEqual(0.5);
  });

  // ── 14. Learning rate of 0 produces no change ────────────────────────────

  it("learning rate of 0 produces no change to biases", () => {
    const adapter = freshAdapter();
    const updated = updateFromCorrection(adapter, 2, 0, 0);

    expect(updated.biases[0]).toBe(0);
    expect(updated.biases[2]).toBe(0);
    // Still counts as a correction event
    expect(updated.totalCorrections).toBe(1);
  });
});

// ── 8. Anti-collapse: dominant class (>40% frequency) gets bias reduced ────

describe("updateAfterSession", () => {
  it("reduces bias of dominant class (>40% frequency)", () => {
    const adapter = freshAdapter();
    // Simulate 10 sessions, 8 of which predicted class 0 (80% frequency)
    adapter.totalSessions = 10;
    adapter.classCounts = [8, 1, 0, 0, 1, 0];

    const updated = updateAfterSession(adapter, 0); // another class 0 prediction

    // Class 0 is dominant (>40%), its bias should be reduced
    expect(updated.biases[0]).toBeLessThan(0);
  });

  // ── 9. Anti-collapse: rare class (<5% frequency) gets bias boosted ────────

  it("boosts bias of rare class (<5% frequency)", () => {
    const adapter = freshAdapter();
    // Simulate 100 sessions, class 3 only appeared 2 times (2% frequency)
    adapter.totalSessions = 100;
    adapter.classCounts = [30, 25, 20, 2, 15, 8];

    const updated = updateAfterSession(adapter, 1); // predict class 1

    // Class 3 is rare (<5%), its bias should be boosted
    expect(updated.biases[3]).toBeGreaterThan(0);
  });

  it("increments totalSessions and classCounts", () => {
    const adapter = freshAdapter();
    const updated = updateAfterSession(adapter, 2);

    expect(updated.totalSessions).toBe(1);
    expect(updated.classCounts[2]).toBe(1);
  });
});

// ── 10. Save/load roundtrip preserves all fields ───────────────────────────

describe("save/load roundtrip", () => {
  it("preserves all fields", () => {
    const adapter = freshAdapter();
    adapter.biases = [0.1, -0.05, 0, 0.2, -0.1, 0.05];
    adapter.multipliers = [1.1, 0.9, 1.0, 1.2, 0.8, 1.05];
    adapter.classCounts = [10, 5, 3, 2, 8, 4];
    adapter.correctionCounts = [2, 1, 0, 0, 3, 1];
    adapter.totalSessions = 32;
    adapter.totalCorrections = 7;
    adapter.lastUpdated = "2026-03-20T12:00:00.000Z";

    savePersonalAdapter(adapter);
    const loaded = loadPersonalAdapter();

    expect(loaded.biases).toEqual(adapter.biases);
    expect(loaded.multipliers).toEqual(adapter.multipliers);
    expect(loaded.classCounts).toEqual(adapter.classCounts);
    expect(loaded.correctionCounts).toEqual(adapter.correctionCounts);
    expect(loaded.totalSessions).toBe(adapter.totalSessions);
    expect(loaded.totalCorrections).toBe(adapter.totalCorrections);
    expect(loaded.lastUpdated).toBe(adapter.lastUpdated);
  });
});

// ── 11. Reset produces fresh adapter ───────────────────────────────────────

describe("resetPersonalAdapter", () => {
  it("produces fresh adapter after customization", () => {
    let adapter = freshAdapter();
    adapter = updateFromCorrection(adapter, 1, 0);
    adapter = updateAfterSession(adapter, 3);
    savePersonalAdapter(adapter);

    // Verify it was stored with non-zero values
    const before = loadPersonalAdapter();
    expect(before.totalCorrections).toBe(1);

    // Reset
    const reset = resetPersonalAdapter();
    expect(reset.biases).toEqual([0, 0, 0, 0, 0, 0]);
    expect(reset.multipliers).toEqual([1, 1, 1, 1, 1, 1]);
    expect(reset.totalSessions).toBe(0);
    expect(reset.totalCorrections).toBe(0);

    // Also clears localStorage
    const loaded = loadPersonalAdapter();
    expect(loaded.totalSessions).toBe(0);
  });
});

// ── 12. Stats show correct confidence level based on session count ─────────

describe("getPersonalizationStats", () => {
  it("shows 'learning' for < 10 sessions", () => {
    const adapter = freshAdapter();
    adapter.totalSessions = 5;
    const stats = getPersonalizationStats(adapter);
    expect(stats.confidenceLevel).toBe("learning");
    expect(stats.sessionsProcessed).toBe(5);
  });

  it("shows 'calibrating' for 10-49 sessions", () => {
    const adapter = freshAdapter();
    adapter.totalSessions = 25;
    const stats = getPersonalizationStats(adapter);
    expect(stats.confidenceLevel).toBe("calibrating");
  });

  it("shows 'personalized' for 50+ sessions", () => {
    const adapter = freshAdapter();
    adapter.totalSessions = 52;
    const stats = getPersonalizationStats(adapter);
    expect(stats.confidenceLevel).toBe("personalized");
  });

  it("reports corrections applied", () => {
    const adapter = freshAdapter();
    adapter.totalCorrections = 7;
    const stats = getPersonalizationStats(adapter);
    expect(stats.correctionsApplied).toBe(7);
  });

  it("reports dominant adjustment for suppressed class", () => {
    const adapter = freshAdapter();
    adapter.biases = [0, 0, 0, -0.3, 0, 0]; // fearful suppressed
    const stats = getPersonalizationStats(adapter);
    expect(stats.dominantAdjustment).toContain("fearful");
  });
});

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

import {
  AdaptiveSampler,
  LowPowerManager,
  type AdaptiveConfig,
  LOW_POWER_STORAGE_KEY,
} from "@/lib/adaptive-sampling";

// ── Setup / Teardown ────────────────────────────────────────────────────────

beforeEach(() => {
  localStorage.clear();
  vi.useFakeTimers();
});

afterEach(() => {
  localStorage.clear();
  vi.useRealTimers();
});

// ── AdaptiveSampler ─────────────────────────────────────────────────────────

describe("AdaptiveSampler", () => {
  it("starts at full sampling rate", () => {
    const sampler = new AdaptiveSampler({ baseSampleRateHz: 256 });
    expect(sampler.getCurrentRate()).toBe(256);
  });

  it("reduces rate when signal variance is below idle threshold for sustained period", () => {
    const sampler = new AdaptiveSampler({
      baseSampleRateHz: 256,
      idleVarianceThreshold: 1.0,
      idleDurationMs: 3000,
      reducedRateHz: 64,
    });

    // Feed low-variance data for 4 seconds
    for (let i = 0; i < 20; i++) {
      sampler.reportVariance(0.5, Date.now());
      vi.advanceTimersByTime(200);
    }

    expect(sampler.getCurrentRate()).toBe(64);
  });

  it("returns to full rate when signal variance exceeds idle threshold", () => {
    const sampler = new AdaptiveSampler({
      baseSampleRateHz: 256,
      idleVarianceThreshold: 1.0,
      idleDurationMs: 1000,
      reducedRateHz: 64,
    });

    // Go idle
    for (let i = 0; i < 10; i++) {
      sampler.reportVariance(0.5, Date.now());
      vi.advanceTimersByTime(200);
    }
    expect(sampler.getCurrentRate()).toBe(64);

    // Active signal
    sampler.reportVariance(5.0, Date.now());
    expect(sampler.getCurrentRate()).toBe(256);
  });

  it("stays at full rate if variance fluctuates above threshold", () => {
    const sampler = new AdaptiveSampler({
      baseSampleRateHz: 256,
      idleVarianceThreshold: 1.0,
      idleDurationMs: 2000,
      reducedRateHz: 64,
    });

    // Alternating high/low variance — should never trigger idle
    for (let i = 0; i < 20; i++) {
      const variance = i % 2 === 0 ? 0.5 : 2.0;
      sampler.reportVariance(variance, Date.now());
      vi.advanceTimersByTime(200);
    }

    expect(sampler.getCurrentRate()).toBe(256);
  });

  it("reports whether currently in idle mode", () => {
    const sampler = new AdaptiveSampler({
      baseSampleRateHz: 256,
      idleVarianceThreshold: 1.0,
      idleDurationMs: 1000,
      reducedRateHz: 64,
    });

    expect(sampler.isIdle()).toBe(false);

    for (let i = 0; i < 10; i++) {
      sampler.reportVariance(0.3, Date.now());
      vi.advanceTimersByTime(200);
    }

    expect(sampler.isIdle()).toBe(true);
  });

  it("shouldProcess respects the current rate", () => {
    const sampler = new AdaptiveSampler({
      baseSampleRateHz: 256,
      idleVarianceThreshold: 1.0,
      idleDurationMs: 500,
      reducedRateHz: 64,
    });

    // Force idle
    for (let i = 0; i < 10; i++) {
      sampler.reportVariance(0.3, Date.now());
      vi.advanceTimersByTime(100);
    }
    expect(sampler.isIdle()).toBe(true);

    // At 64Hz reduced, we process 1 out of every 4 samples (256/64=4)
    // shouldProcess returns true for every Nth sample
    let processCount = 0;
    for (let i = 0; i < 256; i++) {
      if (sampler.shouldProcess(i)) processCount++;
    }
    expect(processCount).toBe(64);
  });

  it("shouldProcess processes all samples at full rate", () => {
    const sampler = new AdaptiveSampler({ baseSampleRateHz: 256 });
    let processCount = 0;
    for (let i = 0; i < 256; i++) {
      if (sampler.shouldProcess(i)) processCount++;
    }
    expect(processCount).toBe(256);
  });
});

// ── LowPowerManager ────────────────────────────────────────────────────────

describe("LowPowerManager", () => {
  it("defaults to disabled", () => {
    const manager = new LowPowerManager();
    expect(manager.isEnabled()).toBe(false);
  });

  it("can be enabled and persists to localStorage", () => {
    const manager = new LowPowerManager();
    manager.setEnabled(true);
    expect(manager.isEnabled()).toBe(true);

    const stored = localStorage.getItem(LOW_POWER_STORAGE_KEY);
    expect(stored).toBe("true");
  });

  it("can be disabled and persists to localStorage", () => {
    localStorage.setItem(LOW_POWER_STORAGE_KEY, "true");
    const manager = new LowPowerManager();
    expect(manager.isEnabled()).toBe(true);

    manager.setEnabled(false);
    expect(manager.isEnabled()).toBe(false);
    expect(localStorage.getItem(LOW_POWER_STORAGE_KEY)).toBe("false");
  });

  it("loads saved state from localStorage on construction", () => {
    localStorage.setItem(LOW_POWER_STORAGE_KEY, "true");
    const manager = new LowPowerManager();
    expect(manager.isEnabled()).toBe(true);
  });

  it("returns correct sync interval when enabled vs disabled", () => {
    const manager = new LowPowerManager();

    // Disabled: real-time (0 means no delay / immediate)
    expect(manager.getSyncIntervalMs()).toBe(0);

    // Enabled: 5 minutes
    manager.setEnabled(true);
    expect(manager.getSyncIntervalMs()).toBe(5 * 60 * 1000);
  });

  it("reports whether background processing should be active", () => {
    const manager = new LowPowerManager();

    // Disabled: background processing on
    expect(manager.shouldProcessInBackground()).toBe(true);

    // Enabled: background processing off
    manager.setEnabled(true);
    expect(manager.shouldProcessInBackground()).toBe(false);
  });
});

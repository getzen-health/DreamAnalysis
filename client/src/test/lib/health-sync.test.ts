/**
 * Tests for health-sync.ts — HealthSyncManager logic.
 *
 * Covers:
 *  1. buildSyncSummary returns correct data counts from a BiometricPayload
 *  2. buildSyncSummary returns empty summary for empty payload
 *  3. buildSyncSummary counts only present (non-null/non-zero) fields
 *  4. formatSyncSummary produces human-readable string
 *  5. getEmptyDataGuidance returns Withings guidance for Android
 *  6. getEmptyDataGuidance returns Apple Health guidance for iOS
 *  7. getEmptyDataGuidance returns generic guidance for web
 */

import { describe, it, expect } from "vitest";
import {
  buildSyncSummary,
  formatSyncSummary,
  getEmptyDataGuidance,
  type SyncSummary,
} from "@/lib/health-sync";

// ── buildSyncSummary ──────────────────────────────────────────────────────

describe("buildSyncSummary", () => {
  it("counts all present fields from a full payload", () => {
    const payload = {
      user_id: "test",
      current_heart_rate: 72,
      resting_heart_rate: 62,
      hrv_sdnn: 45,
      steps_today: 5234,
      sleep_total_hours: 7.2,
      active_energy_kcal: 320,
      weight_kg: 70,
    };
    const summary = buildSyncSummary(payload);
    expect(summary.heartRateReadings).toBe(1);
    expect(summary.restingHr).toBe(62);
    expect(summary.steps).toBe(5234);
    expect(summary.sleepHours).toBeCloseTo(7.2);
    expect(summary.activeCalories).toBe(320);
    expect(summary.hasData).toBe(true);
  });

  it("returns empty summary for payload with only user_id", () => {
    const payload = { user_id: "test" };
    const summary = buildSyncSummary(payload);
    expect(summary.heartRateReadings).toBe(0);
    expect(summary.steps).toBe(0);
    expect(summary.sleepHours).toBeNull();
    expect(summary.activeCalories).toBe(0);
    expect(summary.hasData).toBe(false);
  });

  it("does not count zero-value fields as present", () => {
    const payload = {
      user_id: "test",
      current_heart_rate: 0,
      steps_today: 0,
      active_energy_kcal: 0,
    };
    const summary = buildSyncSummary(payload);
    expect(summary.heartRateReadings).toBe(0);
    expect(summary.steps).toBe(0);
    expect(summary.hasData).toBe(false);
  });

  it("counts heart rate even without resting HR", () => {
    const payload = {
      user_id: "test",
      current_heart_rate: 80,
    };
    const summary = buildSyncSummary(payload);
    expect(summary.heartRateReadings).toBe(1);
    expect(summary.restingHr).toBeNull();
    expect(summary.hasData).toBe(true);
  });
});

// ── formatSyncSummary ─────────────────────────────────────────────────────

describe("formatSyncSummary", () => {
  it("formats a summary with steps and heart rate", () => {
    const summary: SyncSummary = {
      heartRateReadings: 1,
      restingHr: 62,
      steps: 5234,
      sleepHours: null,
      activeCalories: 0,
      hasData: true,
    };
    const text = formatSyncSummary(summary);
    expect(text).toContain("5,234 steps");
    expect(text).toContain("62 bpm resting HR");
  });

  it("includes sleep hours when present", () => {
    const summary: SyncSummary = {
      heartRateReadings: 0,
      restingHr: null,
      steps: 0,
      sleepHours: 7.5,
      activeCalories: 0,
      hasData: true,
    };
    const text = formatSyncSummary(summary);
    expect(text).toContain("7.5h sleep");
  });

  it("returns 'No data synced' for empty summary", () => {
    const summary: SyncSummary = {
      heartRateReadings: 0,
      restingHr: null,
      steps: 0,
      sleepHours: null,
      activeCalories: 0,
      hasData: false,
    };
    const text = formatSyncSummary(summary);
    expect(text).toBe("No data synced");
  });

  it("includes active calories when present", () => {
    const summary: SyncSummary = {
      heartRateReadings: 0,
      restingHr: null,
      steps: 0,
      sleepHours: null,
      activeCalories: 450,
      hasData: true,
    };
    const text = formatSyncSummary(summary);
    expect(text).toContain("450 kcal");
  });
});

// ── getEmptyDataGuidance ──────────────────────────────────────────────────

describe("getEmptyDataGuidance", () => {
  it("returns Withings guidance for android", () => {
    const guidance = getEmptyDataGuidance("android");
    expect(guidance).toContain("Health Connect");
    expect(guidance).toContain("Withings");
  });

  it("returns Apple Health guidance for ios", () => {
    const guidance = getEmptyDataGuidance("ios");
    expect(guidance).toContain("Apple Health");
  });

  it("returns generic guidance for web", () => {
    const guidance = getEmptyDataGuidance("web");
    expect(guidance).toContain("mobile");
  });
});

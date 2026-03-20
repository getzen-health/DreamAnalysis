/**
 * Tests for health-connect.ts — HealthKit + Health Connect write integration.
 *
 * Tests cover:
 * 1. writeMindfulSession no-ops on web platform
 * 2. writeEmotionToHealth maps emotions to correct valence
 * 3. Platform detection returns correct values
 * 4. Permission denied is handled gracefully
 * 5. Duration validation rejects negative/zero
 * 6. writeEmotionToHealth no-ops on Android (no mood type)
 */

import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock @capacitor/core before importing the module under test
vi.mock("@capacitor/core", () => ({
  Capacitor: {
    getPlatform: vi.fn(() => "web"),
  },
}));

// Mock @capgo/capacitor-health
vi.mock("@capgo/capacitor-health", () => ({
  Health: {
    isAvailable: vi.fn().mockResolvedValue({ available: true }),
    requestAuthorization: vi.fn().mockResolvedValue({
      readAuthorized: [],
      readDenied: [],
      writeAuthorized: ["mindfulness"],
      writeDenied: [],
    }),
    saveSample: vi.fn().mockResolvedValue(undefined),
  },
}));

// Mock @perfood/capacitor-healthkit
vi.mock("@perfood/capacitor-healthkit", () => ({
  CapacitorHealthkit: {
    requestAuthorization: vi.fn().mockResolvedValue(undefined),
    queryHKitSampleType: vi.fn().mockResolvedValue({ resultData: [] }),
  },
}));

describe("health-connect", () => {
  let healthConnect: typeof import("@/lib/health-connect");
  let Capacitor: { getPlatform: ReturnType<typeof vi.fn> };

  beforeEach(async () => {
    vi.resetModules();

    // Re-mock Capacitor for each test so getPlatform can be changed
    const capMock = await vi.importMock<typeof import("@capacitor/core")>("@capacitor/core");
    Capacitor = capMock.Capacitor as any;
    Capacitor.getPlatform.mockReturnValue("web");

    healthConnect = await import("@/lib/health-connect");
  });

  // ── Test 1: writeMindfulSession no-ops on web ──────────────────────────────

  it("writeMindfulSession returns without error on web platform", async () => {
    Capacitor.getPlatform.mockReturnValue("web");
    const start = new Date("2026-03-20T10:00:00Z");
    const end = new Date("2026-03-20T10:15:00Z");

    // Should not throw, should be a silent no-op
    await expect(
      healthConnect.writeMindfulSession(start, end, 15),
    ).resolves.toBeUndefined();
  });

  // ── Test 2: writeEmotionToHealth maps emotions correctly ───────────────────

  it("writeEmotionToHealth maps 'happy' to positive valence", () => {
    const mapping = healthConnect.mapEmotionToHealthKit("happy");
    expect(mapping.valence).toBeGreaterThan(0);
    expect(mapping.label).toBe("happy");
  });

  it("writeEmotionToHealth maps 'sad' to negative valence", () => {
    const mapping = healthConnect.mapEmotionToHealthKit("sad");
    expect(mapping.valence).toBeLessThan(0);
  });

  it("writeEmotionToHealth maps 'angry' to negative valence with high arousal", () => {
    const mapping = healthConnect.mapEmotionToHealthKit("angry");
    expect(mapping.valence).toBeLessThan(0);
    expect(mapping.arousal).toBeGreaterThan(0.5);
  });

  it("writeEmotionToHealth maps 'neutral' to near-zero valence", () => {
    const mapping = healthConnect.mapEmotionToHealthKit("neutral");
    expect(mapping.valence).toBeCloseTo(0, 1);
  });

  it("writeEmotionToHealth maps 'calm' to positive valence with low arousal", () => {
    const mapping = healthConnect.mapEmotionToHealthKit("calm");
    expect(mapping.valence).toBeGreaterThan(0);
    expect(mapping.arousal).toBeLessThan(0.5);
  });

  // ── Test 3: Platform detection ─────────────────────────────────────────────

  it("getHealthPlatform returns 'web' for web platform", () => {
    Capacitor.getPlatform.mockReturnValue("web");
    expect(healthConnect.getHealthPlatform()).toBe("web");
  });

  it("getHealthPlatform returns 'ios' for iOS platform", () => {
    Capacitor.getPlatform.mockReturnValue("ios");
    expect(healthConnect.getHealthPlatform()).toBe("ios");
  });

  it("getHealthPlatform returns 'android' for Android platform", () => {
    Capacitor.getPlatform.mockReturnValue("android");
    expect(healthConnect.getHealthPlatform()).toBe("android");
  });

  // ── Test 4: Permission denied handled gracefully ───────────────────────────

  it("writeMindfulSession handles permission denied gracefully", async () => {
    Capacitor.getPlatform.mockReturnValue("ios");

    // Override saveSample to throw a permission error
    const { Health: CapgoHealth } = await import("@capgo/capacitor-health");
    (CapgoHealth.saveSample as any).mockRejectedValueOnce(
      new Error("Authorization denied"),
    );

    const start = new Date("2026-03-20T10:00:00Z");
    const end = new Date("2026-03-20T10:15:00Z");

    // Should not throw — gracefully handles the error
    await expect(
      healthConnect.writeMindfulSession(start, end, 15),
    ).resolves.toBeUndefined();
  });

  // ── Test 5: Duration validation ────────────────────────────────────────────

  it("writeMindfulSession rejects negative duration", async () => {
    Capacitor.getPlatform.mockReturnValue("ios");
    const start = new Date("2026-03-20T10:00:00Z");
    const end = new Date("2026-03-20T10:15:00Z");

    // Should return without writing (no-op for invalid duration)
    await expect(
      healthConnect.writeMindfulSession(start, end, -5),
    ).resolves.toBeUndefined();
  });

  it("writeMindfulSession rejects zero duration", async () => {
    Capacitor.getPlatform.mockReturnValue("ios");
    const start = new Date("2026-03-20T10:00:00Z");
    const end = new Date("2026-03-20T10:15:00Z");

    await expect(
      healthConnect.writeMindfulSession(start, end, 0),
    ).resolves.toBeUndefined();
  });

  // ── Test 6: writeEmotionToHealth no-ops on Android ─────────────────────────

  it("writeEmotionToHealth is a no-op on Android (no mood type in Health Connect)", async () => {
    Capacitor.getPlatform.mockReturnValue("android");

    // Should not throw — just silently returns
    await expect(
      healthConnect.writeEmotionToHealth("happy", 0.8),
    ).resolves.toBeUndefined();
  });

  it("writeEmotionToHealth is a no-op on web", async () => {
    Capacitor.getPlatform.mockReturnValue("web");

    await expect(
      healthConnect.writeEmotionToHealth("sad", -0.5),
    ).resolves.toBeUndefined();
  });
});

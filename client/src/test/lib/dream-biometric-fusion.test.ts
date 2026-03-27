import { describe, it, expect } from "vitest";
import {
  fuseDreamBiometrics,
  type DreamEntry,
  type OvernightBiometrics,
} from "@/lib/dream-biometric-fusion";

// ── Fixtures ─────────────────────────────────────────────────────────────────

const baseDream: DreamEntry = {
  dreamText: "I was running through a dark forest, chased by shadows.",
  emotions: ["fear", "anxiety"],
  lucidityScore: 0.2,
  sleepQuality: 60,
  timestamp: "2026-03-27T07:00:00Z",
};

const baseBio: OvernightBiometrics = {
  avgHeartRate: 72,
  minHeartRate: 55,
  hrvSdnn: 28,
  sleepDuration: 6.2,
  sleepEfficiency: 78,
  deepSleepPct: 18,
  remSleepPct: 22,
};

// ── Tests ────────────────────────────────────────────────────────────────────

describe("fuseDreamBiometrics", () => {
  it("returns null when no dream text", () => {
    const empty: DreamEntry = { ...baseDream, dreamText: "" };
    expect(fuseDreamBiometrics(empty, baseBio)).toBeNull();
  });

  it("returns null when dream text is only whitespace", () => {
    const ws: DreamEntry = { ...baseDream, dreamText: "   " };
    expect(fuseDreamBiometrics(ws, baseBio)).toBeNull();
  });

  it("generates anxious headline for negative emotions + elevated HR", () => {
    const dream: DreamEntry = {
      ...baseDream,
      emotions: ["anxiety", "fear"],
    };
    const bio: OvernightBiometrics = {
      ...baseBio,
      avgHeartRate: 75, // elevated above 65 baseline + 5
    };
    const result = fuseDreamBiometrics(dream, bio);
    expect(result).not.toBeNull();
    expect(result!.headline).toBe("Anxious dream, restless body");
  });

  it("generates peaceful headline for positive emotions + good HRV", () => {
    const dream: DreamEntry = {
      ...baseDream,
      emotions: ["joy", "calm"],
    };
    const bio: OvernightBiometrics = {
      ...baseBio,
      avgHeartRate: 62,
      hrvSdnn: 55,
    };
    const result = fuseDreamBiometrics(dream, bio);
    expect(result).not.toBeNull();
    expect(result!.headline).toBe("Peaceful dream, recovered body");
  });

  it("generates lucid headline for high lucidity + high REM%", () => {
    const dream: DreamEntry = {
      ...baseDream,
      emotions: ["wonder"],
      lucidityScore: 0.8,
    };
    const bio: OvernightBiometrics = {
      ...baseBio,
      remSleepPct: 30,
    };
    const result = fuseDreamBiometrics(dream, bio);
    expect(result).not.toBeNull();
    expect(result!.headline).toMatch(/lucid dream/i);
  });

  it("generates disrupted headline for poor sleep quality + negative emotions", () => {
    const dream: DreamEntry = {
      ...baseDream,
      emotions: ["fear"],
      sleepQuality: 30,
    };
    const bio: OvernightBiometrics = {
      ...baseBio,
      avgHeartRate: 62, // not elevated — so the disrupted path is hit
    };
    const result = fuseDreamBiometrics(dream, bio);
    expect(result).not.toBeNull();
    expect(result!.headline).toMatch(/disrupted night/i);
  });

  it("includes biometric highlights with correct status labels", () => {
    const bio: OvernightBiometrics = {
      avgHeartRate: 75, // elevated
      hrvSdnn: 25,      // low
      deepSleepPct: 22, // normal
      remSleepPct: 27,  // elevated
    };
    const result = fuseDreamBiometrics(baseDream, bio);
    expect(result).not.toBeNull();
    const hl = result!.biometricHighlights;
    expect(hl.length).toBeGreaterThanOrEqual(3);

    const hrHl = hl.find((h) => h.label === "HR");
    expect(hrHl?.status).toBe("elevated");

    const hrvHl = hl.find((h) => h.label === "HRV");
    expect(hrvHl?.status).toBe("low");

    const deepHl = hl.find((h) => h.label === "Deep");
    expect(deepHl?.status).toBe("normal");

    const remHl = hl.find((h) => h.label === "REM");
    expect(remHl?.status).toBe("elevated");
  });

  it("includes sleep context string", () => {
    const result = fuseDreamBiometrics(baseDream, baseBio);
    expect(result).not.toBeNull();
    expect(result!.sleepContext).toContain("6.2h sleep");
    expect(result!.sleepContext).toContain("18% deep");
    expect(result!.sleepContext).toContain("22% REM");
  });

  it("body text contains HR and HRV values", () => {
    const result = fuseDreamBiometrics(baseDream, baseBio);
    expect(result).not.toBeNull();
    expect(result!.body).toContain("72 bpm");
    expect(result!.body).toContain("28ms");
  });

  it("handles missing biometrics gracefully", () => {
    const emptyBio: OvernightBiometrics = {};
    const result = fuseDreamBiometrics(baseDream, emptyBio);
    expect(result).not.toBeNull();
    expect(result!.biometricHighlights).toHaveLength(0);
    expect(result!.sleepContext).toBe("No sleep data");
    expect(result!.body).toBe("No biometric data available for this night.");
  });

  it("uses provided baselineHr for comparison", () => {
    const bio: OvernightBiometrics = { avgHeartRate: 72 };
    // With baseline 60, 72 bpm is elevated (>60+5=65)
    const result = fuseDreamBiometrics(
      { ...baseDream, emotions: ["fear"] },
      bio,
      60,
    );
    expect(result).not.toBeNull();
    expect(result!.headline).toBe("Anxious dream, restless body");
    expect(result!.biometricHighlights[0].status).toBe("elevated");
  });

  it("returns default headline when emotions are neutral", () => {
    const dream: DreamEntry = {
      ...baseDream,
      emotions: ["confusion", "neutral"],
    };
    const result = fuseDreamBiometrics(dream, baseBio);
    expect(result).not.toBeNull();
    expect(result!.headline).toContain("dream");
  });

  it("preserves dreamEmotions in output", () => {
    const result = fuseDreamBiometrics(baseDream, baseBio);
    expect(result).not.toBeNull();
    expect(result!.dreamEmotions).toEqual(["fear", "anxiety"]);
  });
});

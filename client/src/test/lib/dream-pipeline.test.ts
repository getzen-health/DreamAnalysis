import { describe, it, expect } from "vitest";
import {
  detectDeviceTier,
  getDreamPipelineConfig,
  tierLabel,
  type DeviceCapabilities,
  type DeviceTier,
} from "@/lib/dream-pipeline";

// ── detectDeviceTier ─────────────────────────────────────────────────────────

describe("detectDeviceTier", () => {
  describe("eeg_full devices", () => {
    it.each([
      "muse_2",
      "muse_s",
      "openbci_cyton",
      "openbci_cyton_daisy",
      "openbci_ganglion",
      "emotiv_epoc",
      "emotiv_epoc_plus",
      "emotiv_insight",
      "brainbit",
      "crown",
      "enobio",
      "synthetic",
    ])("classifies %s as eeg_full", (deviceType) => {
      const caps = detectDeviceTier(deviceType);
      expect(caps.tier).toBe("eeg_full");
      expect(caps.hasEEG).toBe(true);
      expect(caps.channels).toBeGreaterThanOrEqual(4);
    });
  });

  describe("eeg_basic devices", () => {
    it.each([
      "neurosky_mindwave",
      "neurosky_mindwave_mobile",
      "fp1_single",
    ])("classifies %s as eeg_basic", (deviceType) => {
      const caps = detectDeviceTier(deviceType);
      expect(caps.tier).toBe("eeg_basic");
      expect(caps.hasEEG).toBe(true);
      expect(caps.channels).toBe(1);
    });
  });

  describe("phone_only devices", () => {
    it.each([
      "phone_sensors",
      "phone_only",
      "watch_sensors",
    ])("classifies %s as phone_only", (deviceType) => {
      const caps = detectDeviceTier(deviceType);
      expect(caps.tier).toBe("phone_only");
      expect(caps.hasEEG).toBe(false);
      expect(caps.channels).toBe(0);
    });
  });

  describe("none tier", () => {
    it("returns none when deviceType is null", () => {
      const caps = detectDeviceTier(null);
      expect(caps.tier).toBe("none");
      expect(caps.hasEEG).toBe(false);
      expect(caps.channels).toBe(0);
      expect(caps.deviceName).toBe("No device");
    });

    it("returns none for unknown non-EEG device", () => {
      const caps = detectDeviceTier("unknown_gadget_xyz");
      expect(caps.tier).toBe("none");
    });
  });

  describe("device specs", () => {
    it("returns correct Muse 2 specs", () => {
      const caps = detectDeviceTier("muse_2");
      expect(caps.channels).toBe(4);
      expect(caps.sampleRate).toBe(256);
      expect(caps.hasPPG).toBe(true);
      expect(caps.hasAccelerometer).toBe(true);
      expect(caps.deviceName).toBe("Muse 2");
    });

    it("returns correct OpenBCI Cyton+Daisy specs", () => {
      const caps = detectDeviceTier("openbci_cyton_daisy");
      expect(caps.channels).toBe(16);
      expect(caps.sampleRate).toBe(125);
      expect(caps.hasPPG).toBe(false);
    });

    it("returns correct Emotiv EPOC specs", () => {
      const caps = detectDeviceTier("emotiv_epoc");
      expect(caps.channels).toBe(14);
      expect(caps.sampleRate).toBe(128);
      expect(caps.hasAccelerometer).toBe(true);
    });

    it("returns correct NeuroSky specs", () => {
      const caps = detectDeviceTier("neurosky_mindwave");
      expect(caps.channels).toBe(1);
      expect(caps.sampleRate).toBe(512);
    });

    it("returns correct synthetic specs", () => {
      const caps = detectDeviceTier("synthetic");
      expect(caps.tier).toBe("eeg_full");
      expect(caps.channels).toBe(4);
      expect(caps.sampleRate).toBe(256);
      expect(caps.deviceName).toBe("Synthetic");
    });
  });

  describe("case insensitivity", () => {
    it("normalizes device type to lowercase", () => {
      const caps = detectDeviceTier("Muse_2");
      expect(caps.tier).toBe("eeg_full");
    });

    it("trims whitespace from device type", () => {
      const caps = detectDeviceTier("  muse_s  ");
      expect(caps.tier).toBe("eeg_full");
    });
  });
});

// ── getDreamPipelineConfig ───────────────────────────────────────────────────

describe("getDreamPipelineConfig", () => {
  function makeCapabilities(tier: DeviceTier, overrides?: Partial<DeviceCapabilities>): DeviceCapabilities {
    const defaults: Record<DeviceTier, DeviceCapabilities> = {
      eeg_full:   { tier: "eeg_full",   channels: 4, sampleRate: 256, hasEEG: true,  hasPPG: true,  hasAccelerometer: true,  deviceName: "Muse 2" },
      eeg_basic:  { tier: "eeg_basic",  channels: 1, sampleRate: 512, hasEEG: true,  hasPPG: false, hasAccelerometer: false, deviceName: "NeuroSky" },
      phone_only: { tier: "phone_only", channels: 0, sampleRate: 0,   hasEEG: false, hasPPG: false, hasAccelerometer: true,  deviceName: "Phone" },
      none:       { tier: "none",       channels: 0, sampleRate: 0,   hasEEG: false, hasPPG: false, hasAccelerometer: false, deviceName: "No device" },
    };
    return { ...defaults[tier], ...overrides };
  }

  describe("eeg_full config", () => {
    it("enables all EEG features", () => {
      const config = getDreamPipelineConfig(makeCapabilities("eeg_full"));
      expect(config.enableDreamDetection).toBe(true);
      expect(config.enableSleepStaging).toBe(true);
      expect(config.enableEmotionTracking).toBe(true);
      expect(config.enableFAA).toBe(true);
      expect(config.enableDreamJournal).toBe(true);
    });

    it("enables movement tracking when accelerometer is available", () => {
      const config = getDreamPipelineConfig(makeCapabilities("eeg_full", { hasAccelerometer: true }));
      expect(config.enableMovementTracking).toBe(true);
    });

    it("disables movement tracking when no accelerometer", () => {
      const config = getDreamPipelineConfig(makeCapabilities("eeg_full", { hasAccelerometer: false }));
      expect(config.enableMovementTracking).toBe(false);
    });

    it("disables audio heuristics for EEG devices", () => {
      const config = getDreamPipelineConfig(makeCapabilities("eeg_full"));
      expect(config.enableAudioHeuristics).toBe(false);
    });
  });

  describe("eeg_basic config", () => {
    it("enables basic staging but not dream detection or emotion tracking", () => {
      const config = getDreamPipelineConfig(makeCapabilities("eeg_basic"));
      expect(config.enableSleepStaging).toBe(true);
      expect(config.enableDreamDetection).toBe(false);
      expect(config.enableEmotionTracking).toBe(false);
      expect(config.enableFAA).toBe(false);
      expect(config.enableDreamJournal).toBe(true);
    });
  });

  describe("phone_only config", () => {
    it("enables only movement + audio heuristics + journal", () => {
      const config = getDreamPipelineConfig(makeCapabilities("phone_only"));
      expect(config.enableMovementTracking).toBe(true);
      expect(config.enableAudioHeuristics).toBe(true);
      expect(config.enableDreamJournal).toBe(true);
      expect(config.enableSleepStaging).toBe(false);
      expect(config.enableDreamDetection).toBe(false);
      expect(config.enableEmotionTracking).toBe(false);
      expect(config.enableFAA).toBe(false);
    });
  });

  describe("none config", () => {
    it("enables only dream journal", () => {
      const config = getDreamPipelineConfig(makeCapabilities("none"));
      expect(config.enableDreamJournal).toBe(true);
      expect(config.enableSleepStaging).toBe(false);
      expect(config.enableDreamDetection).toBe(false);
      expect(config.enableEmotionTracking).toBe(false);
      expect(config.enableFAA).toBe(false);
      expect(config.enableMovementTracking).toBe(false);
      expect(config.enableAudioHeuristics).toBe(false);
    });
  });

  describe("config always includes device reference", () => {
    it("passes through the device capabilities", () => {
      const caps = makeCapabilities("eeg_full");
      const config = getDreamPipelineConfig(caps);
      expect(config.device).toBe(caps);
    });
  });
});

// ── tierLabel ────────────────────────────────────────────────────────────────

describe("tierLabel", () => {
  it.each<[DeviceTier, string]>([
    ["eeg_full",   "Full EEG"],
    ["eeg_basic",  "Basic EEG"],
    ["phone_only", "Phone Sensors"],
    ["none",       "Journal Only"],
  ])("returns %s for tier %s", (tier, expected) => {
    expect(tierLabel(tier)).toBe(expected);
  });
});

// ── Integration: detectDeviceTier → getDreamPipelineConfig ───────────────────

describe("end-to-end: detect → config", () => {
  it("Muse 2 gets full pipeline", () => {
    const caps = detectDeviceTier("muse_2");
    const config = getDreamPipelineConfig(caps);
    expect(config.enableDreamDetection).toBe(true);
    expect(config.enableSleepStaging).toBe(true);
    expect(config.enableEmotionTracking).toBe(true);
    expect(config.enableFAA).toBe(true);
  });

  it("NeuroSky gets basic pipeline", () => {
    const caps = detectDeviceTier("neurosky_mindwave");
    const config = getDreamPipelineConfig(caps);
    expect(config.enableSleepStaging).toBe(true);
    expect(config.enableDreamDetection).toBe(false);
    expect(config.enableFAA).toBe(false);
  });

  it("phone_only gets movement + audio only", () => {
    const caps = detectDeviceTier("phone_sensors");
    const config = getDreamPipelineConfig(caps);
    expect(config.enableMovementTracking).toBe(true);
    expect(config.enableAudioHeuristics).toBe(true);
    expect(config.enableSleepStaging).toBe(false);
  });

  it("null device gets journal only", () => {
    const caps = detectDeviceTier(null);
    const config = getDreamPipelineConfig(caps);
    expect(config.enableDreamJournal).toBe(true);
    expect(config.enableSleepStaging).toBe(false);
    expect(config.enableDreamDetection).toBe(false);
  });
});

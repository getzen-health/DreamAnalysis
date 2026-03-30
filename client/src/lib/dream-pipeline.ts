/**
 * dream-pipeline.ts — Device-agnostic dream/sleep pipeline abstraction.
 *
 * Maps any BrainFlow-supported device (or phone-only/no-device fallback)
 * to a capability tier, then generates the appropriate pipeline config
 * that enables or disables features based on what the hardware supports.
 *
 * Tier hierarchy:
 *   eeg_full   — 4+ EEG channels, full sleep staging + FAA + dream detection
 *   eeg_basic  — 1 channel, limited staging, no FAA (frontal asymmetry needs 2+ frontal)
 *   phone_only — accelerometer + mic heuristics (no EEG)
 *   none       — manual dream journaling only
 */

// ── Types ────────────────────────────────────────────────────────────────────

export type DeviceTier = "eeg_full" | "eeg_basic" | "phone_only" | "none";

export interface DeviceCapabilities {
  tier: DeviceTier;
  channels: number;
  sampleRate: number;
  hasEEG: boolean;
  hasPPG: boolean;
  hasAccelerometer: boolean;
  deviceName: string;
}

export interface DreamPipelineConfig {
  device: DeviceCapabilities;
  /** Full REM-based dream detection (requires 4+ ch EEG) */
  enableDreamDetection: boolean;
  /** 5-stage sleep staging: Wake/N1/N2/N3/REM */
  enableSleepStaging: boolean;
  /** FAA-based emotion tracking (requires 2+ frontal channels) */
  enableEmotionTracking: boolean;
  /** Frontal Alpha Asymmetry computation */
  enableFAA: boolean;
  /** Accelerometer-based movement/position tracking */
  enableMovementTracking: boolean;
  /** Mic-based snoring/breathing heuristics */
  enableAudioHeuristics: boolean;
  /** Manual dream journal entry (always available) */
  enableDreamJournal: boolean;
}

// ── Device → Tier Mapping ────────────────────────────────────────────────────

/**
 * Known device types grouped by tier. Keys are the `device_type` strings
 * used by the ML backend's `/devices` endpoint and BrainFlow board IDs.
 */
const EEG_FULL_DEVICES = new Set([
  "muse_2",
  "muse_s",
  "openbci_cyton",
  "openbci_cyton_daisy",
  "openbci_ganglion",
  "emotiv_epoc",
  "emotiv_epoc_plus",
  "emotiv_insight",
  "brainbit",
  "crown",              // Neurosity Crown
  "enobio",
]);

const EEG_BASIC_DEVICES = new Set([
  "neurosky_mindwave",
  "neurosky_mindwave_mobile",
  "fp1_single",         // generic single-channel
]);

const PHONE_ONLY_DEVICES = new Set([
  "phone_sensors",
  "phone_only",
  "watch_sensors",
]);

/** Device specs for known hardware. Channels / sample rate / sensor flags. */
const DEVICE_SPECS: Record<string, Omit<DeviceCapabilities, "tier" | "deviceName">> = {
  muse_2:                  { channels: 4,  sampleRate: 256,  hasEEG: true,  hasPPG: true,  hasAccelerometer: true },
  muse_s:                  { channels: 4,  sampleRate: 256,  hasEEG: true,  hasPPG: true,  hasAccelerometer: true },
  openbci_cyton:           { channels: 8,  sampleRate: 250,  hasEEG: true,  hasPPG: false, hasAccelerometer: false },
  openbci_cyton_daisy:     { channels: 16, sampleRate: 125,  hasEEG: true,  hasPPG: false, hasAccelerometer: false },
  openbci_ganglion:        { channels: 4,  sampleRate: 200,  hasEEG: true,  hasPPG: false, hasAccelerometer: true },
  emotiv_epoc:             { channels: 14, sampleRate: 128,  hasEEG: true,  hasPPG: false, hasAccelerometer: true },
  emotiv_epoc_plus:        { channels: 14, sampleRate: 256,  hasEEG: true,  hasPPG: false, hasAccelerometer: true },
  emotiv_insight:          { channels: 5,  sampleRate: 128,  hasEEG: true,  hasPPG: false, hasAccelerometer: true },
  brainbit:                { channels: 4,  sampleRate: 250,  hasEEG: true,  hasPPG: false, hasAccelerometer: false },
  crown:                   { channels: 8,  sampleRate: 256,  hasEEG: true,  hasPPG: false, hasAccelerometer: true },
  enobio:                  { channels: 8,  sampleRate: 500,  hasEEG: true,  hasPPG: false, hasAccelerometer: false },
  neurosky_mindwave:       { channels: 1,  sampleRate: 512,  hasEEG: true,  hasPPG: false, hasAccelerometer: false },
  neurosky_mindwave_mobile:{ channels: 1,  sampleRate: 512,  hasEEG: true,  hasPPG: false, hasAccelerometer: false },
  fp1_single:              { channels: 1,  sampleRate: 256,  hasEEG: true,  hasPPG: false, hasAccelerometer: false },
  phone_sensors:           { channels: 0,  sampleRate: 0,    hasEEG: false, hasPPG: false, hasAccelerometer: true },
  phone_only:              { channels: 0,  sampleRate: 0,    hasEEG: false, hasPPG: false, hasAccelerometer: true },
  watch_sensors:           { channels: 0,  sampleRate: 0,    hasEEG: false, hasPPG: true,  hasAccelerometer: true },
  synthetic:               { channels: 4,  sampleRate: 256,  hasEEG: true,  hasPPG: false, hasAccelerometer: false },
};

// ── Public API ───────────────────────────────────────────────────────────────

/**
 * Detect the capability tier and hardware specs for a given device type.
 *
 * @param deviceType — The `device_type` string from the ML backend, or null
 *   if no device is selected.
 * @returns Full DeviceCapabilities object for the device.
 */
export function detectDeviceTier(deviceType: string | null): DeviceCapabilities {
  if (!deviceType) {
    return {
      tier: "none",
      channels: 0,
      sampleRate: 0,
      hasEEG: false,
      hasPPG: false,
      hasAccelerometer: false,
      deviceName: "No device",
    };
  }

  const normalized = deviceType.toLowerCase().trim();

  // Look up known specs, fall back to reasonable defaults
  const specs = DEVICE_SPECS[normalized];

  // Determine tier
  let tier: DeviceTier;
  if (EEG_FULL_DEVICES.has(normalized) || normalized === "synthetic") {
    tier = "eeg_full";
  } else if (EEG_BASIC_DEVICES.has(normalized)) {
    tier = "eeg_basic";
  } else if (PHONE_ONLY_DEVICES.has(normalized)) {
    tier = "phone_only";
  } else if (specs?.hasEEG && specs.channels >= 4) {
    // Unknown device but has 4+ EEG channels — treat as full
    tier = "eeg_full";
  } else if (specs?.hasEEG) {
    // Unknown device with fewer than 4 channels
    tier = "eeg_basic";
  } else {
    // Completely unknown — fall back to none
    tier = "none";
  }

  // Build a human-readable device name
  const deviceName = formatDeviceName(normalized);

  if (specs) {
    return { tier, ...specs, deviceName };
  }

  // Unknown device, no specs in our table — return minimal info based on tier
  return {
    tier,
    channels: tier === "eeg_full" ? 4 : tier === "eeg_basic" ? 1 : 0,
    sampleRate: tier !== "none" && tier !== "phone_only" ? 256 : 0,
    hasEEG: tier === "eeg_full" || tier === "eeg_basic",
    hasPPG: false,
    hasAccelerometer: tier === "phone_only",
    deviceName,
  };
}

/**
 * Generate the pipeline configuration for a given set of device capabilities.
 *
 * Each tier enables/disables features based on what the hardware can support:
 *   - eeg_full:   all EEG features enabled (staging, dream detection, FAA, emotion)
 *   - eeg_basic:  basic staging (no FAA, limited dream detection)
 *   - phone_only: movement + audio heuristics only
 *   - none:       dream journal only
 */
export function getDreamPipelineConfig(capabilities: DeviceCapabilities): DreamPipelineConfig {
  const { tier } = capabilities;

  switch (tier) {
    case "eeg_full":
      return {
        device: capabilities,
        enableDreamDetection: true,
        enableSleepStaging: true,
        enableEmotionTracking: true,
        enableFAA: true,
        enableMovementTracking: capabilities.hasAccelerometer,
        enableAudioHeuristics: false,
        enableDreamJournal: true,
      };

    case "eeg_basic":
      return {
        device: capabilities,
        enableDreamDetection: false,   // needs multi-channel for reliable REM detection
        enableSleepStaging: true,      // basic staging from single channel (delta/theta/alpha)
        enableEmotionTracking: false,  // FAA needs 2+ frontal channels
        enableFAA: false,
        enableMovementTracking: capabilities.hasAccelerometer,
        enableAudioHeuristics: false,
        enableDreamJournal: true,
      };

    case "phone_only":
      return {
        device: capabilities,
        enableDreamDetection: false,
        enableSleepStaging: false,     // no EEG — can only estimate from movement
        enableEmotionTracking: false,
        enableFAA: false,
        enableMovementTracking: true,
        enableAudioHeuristics: true,   // mic for snoring/breathing
        enableDreamJournal: true,
      };

    case "none":
    default:
      return {
        device: capabilities,
        enableDreamDetection: false,
        enableSleepStaging: false,
        enableEmotionTracking: false,
        enableFAA: false,
        enableMovementTracking: false,
        enableAudioHeuristics: false,
        enableDreamJournal: true,
      };
  }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Human-readable label for a tier. */
export function tierLabel(tier: DeviceTier): string {
  switch (tier) {
    case "eeg_full":   return "Full EEG";
    case "eeg_basic":  return "Basic EEG";
    case "phone_only": return "Phone Sensors";
    case "none":       return "Journal Only";
  }
}

/** Format a device_type string into a human-readable name. */
function formatDeviceName(deviceType: string): string {
  const NAMES: Record<string, string> = {
    muse_2: "Muse 2",
    muse_s: "Muse S",
    openbci_cyton: "OpenBCI Cyton",
    openbci_cyton_daisy: "OpenBCI Cyton+Daisy",
    openbci_ganglion: "OpenBCI Ganglion",
    emotiv_epoc: "Emotiv EPOC",
    emotiv_epoc_plus: "Emotiv EPOC+",
    emotiv_insight: "Emotiv Insight",
    brainbit: "BrainBit",
    crown: "Neurosity Crown",
    enobio: "Enobio",
    neurosky_mindwave: "NeuroSky MindWave",
    neurosky_mindwave_mobile: "NeuroSky MindWave Mobile",
    fp1_single: "Single-Channel (Fp1)",
    phone_sensors: "Phone Sensors",
    phone_only: "Phone Only",
    watch_sensors: "Smartwatch",
    synthetic: "Synthetic",
  };
  return NAMES[deviceType] || deviceType.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

/**
 * health-connect.ts — Write Mindful Minutes and mood data to Apple HealthKit / Google Health Connect.
 *
 * Uses @capgo/capacitor-health (unified iOS + Android plugin with saveSample()).
 *
 * Platform behavior:
 *   iOS     → Writes HKCategoryTypeIdentifierMindfulSession + (future) HKStateOfMind
 *   Android → Writes MindfulnessSession via Health Connect
 *   Web     → All functions are silent no-ops
 *
 * Error handling:
 *   All functions catch errors internally and log to console.
 *   They NEVER throw — the app must work fine without health integration.
 *   Permission denial, plugin failures, and unsupported platforms are all no-ops.
 *
 * Usage:
 *   import { writeMindfulSession, writeEmotionToHealth, requestHealthWritePermissions } from "@/lib/health-connect";
 *
 *   // After a neurofeedback session completes:
 *   await writeMindfulSession(sessionStart, sessionEnd, durationMinutes);
 *
 *   // After voice check-in detects emotion (iOS only — Android no-op):
 *   await writeEmotionToHealth("happy", 0.8);
 */

import { Capacitor } from "@capacitor/core";

// ── Platform detection ──────────────────────────────────────────────────────

export type HealthPlatform = "ios" | "android" | "web";

export function getHealthPlatform(): HealthPlatform {
  const platform = Capacitor.getPlatform();
  if (platform === "ios") return "ios";
  if (platform === "android") return "android";
  return "web";
}

// ── Emotion → HealthKit mapping ─────────────────────────────────────────────

export interface EmotionHealthMapping {
  label: string;
  /** Valence: -1 (very unpleasant) to +1 (very pleasant) */
  valence: number;
  /** Arousal: 0 (very calm) to 1 (very energetic) */
  arousal: number;
}

/**
 * Maps our emotion labels to Apple HealthKit State of Mind parameters.
 * Based on Russell's Circumplex Model of Affect.
 *
 * Apple's HKStateOfMind (iOS 18+) uses:
 *   - kind: .momentaryEmotion
 *   - valence: -1 to +1
 *   - labels: one of 38 predefined labels
 *
 * Since the current @capgo/capacitor-health plugin doesn't expose
 * HKStateOfMind writing directly, this mapping is prepared for when
 * native support is added. For now, the mapping is used for validation
 * and the actual write is a no-op on all platforms for mood data.
 */
export function mapEmotionToHealthKit(emotion: string): EmotionHealthMapping {
  const lower = emotion.toLowerCase();

  switch (lower) {
    case "happy":
    case "excited":
      return { label: "happy", valence: 0.7, arousal: 0.7 };
    case "sad":
    case "lonely":
      return { label: "sad", valence: -0.6, arousal: 0.2 };
    case "angry":
    case "frustrated":
      return { label: "angry", valence: -0.5, arousal: 0.8 };
    case "fear":
    case "fearful":
    case "anxious":
    case "nervous":
      return { label: "fearful", valence: -0.4, arousal: 0.7 };
    case "surprise":
    case "surprised":
      return { label: "surprised", valence: 0.2, arousal: 0.8 };
    case "calm":
    case "relaxed":
    case "peaceful":
      return { label: "calm", valence: 0.5, arousal: 0.2 };
    case "neutral":
    default:
      return { label: "neutral", valence: 0, arousal: 0.4 };
  }
}

// ── Permission request ──────────────────────────────────────────────────────

/**
 * Request write permissions for mindfulness sessions.
 * Safe to call on web (no-op). Safe to call multiple times.
 * Returns true if permissions were granted, false otherwise.
 */
export async function requestHealthWritePermissions(): Promise<boolean> {
  const platform = getHealthPlatform();
  if (platform === "web") return false;

  try {
    const { Health: CapgoHealth } = await import("@capgo/capacitor-health");
    const result = await CapgoHealth.requestAuthorization({
      write: ["mindfulness"],
    });
    return result.writeAuthorized.includes("mindfulness");
  } catch (e) {
    console.warn("[health-connect] Failed to request write permissions:", e);
    return false;
  }
}

// ── Write mindful session ───────────────────────────────────────────────────

/**
 * Write a mindfulness/meditation session to the native health store.
 *
 * On iOS: Creates an HKCategoryTypeIdentifierMindfulSession in Apple Health.
 * On Android: Creates a MindfulnessSession in Health Connect.
 * On Web: Silent no-op.
 *
 * @param startDate - When the session started
 * @param endDate - When the session ended
 * @param durationMinutes - Duration of the session in minutes (must be > 0)
 */
export async function writeMindfulSession(
  startDate: Date,
  endDate: Date,
  durationMinutes: number,
): Promise<void> {
  // Validate duration
  if (durationMinutes <= 0) {
    console.warn("[health-connect] writeMindfulSession: invalid duration", durationMinutes);
    return;
  }

  const platform = getHealthPlatform();
  if (platform === "web") return;

  try {
    const { Health: CapgoHealth } = await import("@capgo/capacitor-health");
    await CapgoHealth.saveSample({
      dataType: "mindfulness",
      value: durationMinutes,
      unit: "minute",
      startDate: startDate.toISOString(),
      endDate: endDate.toISOString(),
    });
    console.log("[health-connect] Wrote mindful session:", durationMinutes, "min");
  } catch (e) {
    // Permission denied, plugin error, or unsupported — all gracefully handled
    console.warn("[health-connect] Failed to write mindful session:", e);
  }
}

// ── Write emotion to health ─────────────────────────────────────────────────

/**
 * Write emotion/mood data to the native health store.
 *
 * iOS only: Maps voice emotion to Apple HealthKit State of Mind parameters.
 * Android: No-op (Health Connect has no mood/emotion data type).
 * Web: No-op.
 *
 * Note: HKStateOfMind (iOS 18+) requires native Swift code to write.
 * The @capgo/capacitor-health plugin doesn't support it yet.
 * This function is prepared for future integration — currently logs
 * the mapped emotion but does not write to HealthKit.
 *
 * @param emotion - Detected emotion label (e.g., "happy", "sad", "angry")
 * @param valence - Valence from the detection model (-1 to +1)
 */
export async function writeEmotionToHealth(
  emotion: string,
  valence: number,
): Promise<void> {
  const platform = getHealthPlatform();

  // Only iOS has mood/emotion data types (HKStateOfMind, iOS 18+)
  // Android Health Connect has no equivalent — always no-op
  if (platform !== "ios") return;

  try {
    const mapping = mapEmotionToHealthKit(emotion);
    // Log for now — actual HKStateOfMind writing requires native Swift bridge
    // that @capgo/capacitor-health doesn't expose yet.
    console.log("[health-connect] Emotion mapped for HealthKit:", {
      emotion,
      detectedValence: valence,
      mappedValence: mapping.valence,
      mappedArousal: mapping.arousal,
      healthKitLabel: mapping.label,
    });
    // Future: when plugin supports HKStateOfMind, write here:
    // await CapgoHealth.saveStateOfMind({ ... });
  } catch (e) {
    console.warn("[health-connect] Failed to write emotion:", e);
  }
}

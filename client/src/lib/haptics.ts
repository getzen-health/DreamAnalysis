/**
 * haptics.ts — thin wrapper around @capacitor/haptics
 *
 * On desktop/browser: all calls are silent no-ops.
 * On iOS/Android via Capacitor: triggers native haptic feedback.
 *
 * Usage:
 *   import { hapticLight, hapticMedium, hapticHeavy, hapticSuccess } from "@/lib/haptics";
 *   hapticLight();   // inhale start / UI tap
 *   hapticMedium();  // exhale start / phase transition
 *   hapticHeavy();   // hold end / important event
 *   hapticSuccess(); // session complete / dream detected
 */

import { Haptics, ImpactStyle, NotificationType } from "@capacitor/haptics";

// Check once at load time whether we're running inside Capacitor
function isNative(): boolean {
  try {
    // Capacitor injects window.Capacitor when running as a native app
    return !!(window as unknown as { Capacitor?: { isNativePlatform?: () => boolean } })
      .Capacitor?.isNativePlatform?.();
  } catch {
    return false;
  }
}

const NATIVE = isNative();

/** Short tap — use on inhale start, UI button taps */
export async function hapticLight(): Promise<void> {
  if (!NATIVE) return;
  try {
    await Haptics.impact({ style: ImpactStyle.Light });
  } catch { /* ignore — haptics unavailable */ }
}

/** Medium tap — use on phase transitions (exhale start, hold start) */
export async function hapticMedium(): Promise<void> {
  if (!NATIVE) return;
  try {
    await Haptics.impact({ style: ImpactStyle.Medium });
  } catch { /* ignore */ }
}

/** Heavy tap — use on session end, strong alerts */
export async function hapticHeavy(): Promise<void> {
  if (!NATIVE) return;
  try {
    await Haptics.impact({ style: ImpactStyle.Heavy });
  } catch { /* ignore */ }
}

/** Success notification — use on session complete, dream detected */
export async function hapticSuccess(): Promise<void> {
  if (!NATIVE) return;
  try {
    await Haptics.notification({ type: NotificationType.Success });
  } catch { /* ignore */ }
}

/** Warning notification — use on high stress alert */
export async function hapticWarning(): Promise<void> {
  if (!NATIVE) return;
  try {
    await Haptics.notification({ type: NotificationType.Warning });
  } catch { /* ignore */ }
}

/** Error notification — use on connection failure */
export async function hapticError(): Promise<void> {
  if (!NATIVE) return;
  try {
    await Haptics.notification({ type: NotificationType.Error });
  } catch { /* ignore */ }
}

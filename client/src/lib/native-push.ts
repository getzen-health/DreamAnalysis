/**
 * native-push.ts — Capacitor Push Notifications for iOS (APNs) and Android (FCM).
 *
 * Complements the existing web-push VAPID path (which works in PWA/browser).
 * This module activates only when running inside the Capacitor native shell.
 *
 * Flow:
 *   1. Request permission (shows iOS/Android native dialog)
 *   2. Receive APNs/FCM registration token
 *   3. POST token to Express endpoint /api/notifications/native-token
 *   4. Server stores token per user → sends push when stress is high
 *
 * Background push delivery:
 *   When the app is backgrounded and the server calls FCM/APNs,
 *   the OS delivers the notification without opening the app.
 *   If the user taps it, Capacitor fires the pushNotificationActionPerformed
 *   event which this module handles → navigates to the correct page.
 *
 * Server side (FCM):
 *   Set FIREBASE_SERVICE_ACCOUNT_KEY (JSON string) in environment.
 *   The /api/notifications/send-native endpoint uses firebase-admin to deliver.
 *
 * Set up requirements:
 *   iOS:  Add push notifications capability in Xcode, upload APNs key to Firebase
 *   Android: google-services.json in android/app/
 */

import { Capacitor } from "@capacitor/core";
import { getParticipantId } from "./participant";
import { apiRequest } from "./queryClient";

// ── Types ─────────────────────────────────────────────────────────────────────

export interface NativePushToken {
  token: string;
  platform: "ios" | "android";
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function isNative(): boolean {
  return Capacitor.isNativePlatform();
}

function getPlatform(): "ios" | "android" {
  return Capacitor.getPlatform() === "ios" ? "ios" : "android";
}

async function registerTokenWithServer(token: string): Promise<void> {
  const userId = getParticipantId();
  await apiRequest("POST", "/api/notifications/native-token", {
    userId,
    token,
    platform: getPlatform(),
  });
}

// ── Main registration function ────────────────────────────────────────────────

let _registered = false;

/**
 * Request push notification permission and register the device token.
 * Safe to call multiple times — only registers once per session.
 * No-op on web (falls back to VAPID web push).
 */
export async function registerNativePush(): Promise<NativePushToken | null> {
  if (!isNative()) return null;
  if (_registered) return null;

  // Skip on Android until google-services.json / Firebase is configured —
  // calling PushNotifications.register() without Firebase crashes the native process.
  if (Capacitor.getPlatform() === "android") {
    console.info("[native-push] Skipping on Android — Firebase not configured");
    return null;
  }

  const { PushNotifications } = await import("@capacitor/push-notifications");

  // Wrap everything in try-catch — PushNotifications.register() crashes on
  // Android if google-services.json / Firebase is not configured.
  try {
    let permStatus = await PushNotifications.checkPermissions();
    if (permStatus.receive === "prompt") {
      permStatus = await PushNotifications.requestPermissions();
    }
    if (permStatus.receive !== "granted") {
      console.warn("[native-push] Permission not granted");
      return null;
    }

    return new Promise((resolve) => {
      PushNotifications.addListener("registration", async (token) => {
        _registered = true;
        const payload: NativePushToken = { token: token.value, platform: getPlatform() };
        try {
          await registerTokenWithServer(token.value);
        } catch (e) {
          console.warn("[native-push] Failed to register token with server:", e);
        }
        resolve(payload);
      });

      PushNotifications.addListener("registrationError", (err) => {
        console.error("[native-push] Registration error:", err);
        resolve(null);
      });

      PushNotifications.addListener("pushNotificationReceived", (notification) => {
        window.dispatchEvent(
          new CustomEvent("native-push-received", { detail: notification })
        );
      });

      PushNotifications.addListener("pushNotificationActionPerformed", (action) => {
        const data = action.notification.data as Record<string, string> | undefined;
        const route = data?.route ?? "/brain-report";
        window.dispatchEvent(
          new CustomEvent("native-push-navigate", { detail: { route } })
        );
      });

      // This crashes natively if Firebase is not set up — the registrationError
      // listener above will catch the JS-side error, but the native crash needs
      // the outer try-catch + a timeout to resolve the promise.
      PushNotifications.register();

      // Safety timeout: if neither registration nor error fires within 10s, resolve
      setTimeout(() => resolve(null), 10_000);
    });
  } catch (e) {
    console.warn("[native-push] Registration failed (Firebase not configured?):", e);
    return null;
  }
}

// ── Push status helper ────────────────────────────────────────────────────────

export interface PushStatusResult {
  available: boolean;
  reason?: string;
}

/**
 * Returns the current push notification availability and an explanation
 * if unavailable. Use this in Settings to show honest status to the user
 * instead of a broken toggle.
 */
export async function getPushStatus(): Promise<PushStatusResult> {
  // Web browser
  if (!isNative()) {
    const supported =
      typeof window !== "undefined" &&
      "serviceWorker" in navigator &&
      "Notification" in window;
    if (!supported) {
      return { available: false, reason: "Push notifications are not supported in this browser." };
    }
    // Web push is supported — availability depends on permission
    const perm = Notification.permission;
    if (perm === "denied") {
      return { available: false, reason: "Notifications are blocked. Enable them in your browser settings." };
    }
    return { available: true };
  }

  // Native: Android — Firebase not configured
  if (Capacitor.getPlatform() === "android") {
    return {
      available: false,
      reason: "Push notifications require Firebase setup, which is not configured yet.",
    };
  }

  // Native: iOS — check actual permission status
  try {
    const { PushNotifications } = await import("@capacitor/push-notifications");
    const permStatus = await PushNotifications.checkPermissions();
    if (permStatus.receive === "granted") {
      return { available: true };
    }
    if (permStatus.receive === "denied") {
      return {
        available: false,
        reason: "Notifications are blocked. Enable them in Settings > Notifications.",
      };
    }
    // "prompt" — user hasn't decided yet, but it's available
    return { available: true };
  } catch {
    return {
      available: false,
      reason: "Could not check notification permissions.",
    };
  }
}

/**
 * Clear the stored notification badge count (iOS).
 * Call this when the user opens a notification-triggered page.
 */
export async function clearBadge(): Promise<void> {
  if (!isNative()) return;
  try {
    const { PushNotifications } = await import("@capacitor/push-notifications");
    await PushNotifications.removeAllDeliveredNotifications();
  } catch { /* ignore */ }
}

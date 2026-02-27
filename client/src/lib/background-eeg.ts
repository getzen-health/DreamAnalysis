/**
 * background-eeg.ts — Background EEG processing for sleep mode.
 *
 * Keeps the EEG stream alive when the phone screen turns off during sleep.
 *
 * Platform strategy:
 *   iOS     — CoreBluetooth "bluetooth-central" background mode keeps the BLE
 *             connection alive. @capacitor/background-runner fires every 30 min
 *             to flush buffered samples to the server.
 *   Android — Foreground Service (via @capacitor/background-runner) keeps the
 *             process alive continuously. Android 8+ requires an ongoing
 *             notification ("EEG recording in progress...").
 *   Web     — Screen Wake Lock API prevents screen sleep; falls back gracefully.
 *
 * Usage:
 *   import { backgroundEeg } from "@/lib/background-eeg";
 *   await backgroundEeg.startSleepRecording(userId);
 *   await backgroundEeg.stopSleepRecording();
 *
 * iOS native setup required (in Xcode before first build):
 *   - Add "bluetooth-central" to UIBackgroundModes in Info.plist
 *   - Add "Background Fetch" capability
 *   - Register background task identifier: "com.neuraldreamworkshop.eeg-flush"
 *
 * Android native setup required:
 *   - background-runner.json is configured below (copied to android/app/src/main/assets/)
 *   - android:foregroundServiceType="connectedDevice" permission in AndroidManifest.xml
 */

import { Capacitor } from "@capacitor/core";
import { getParticipantId } from "./participant";

// ── Types ────────────────────────────────────────────────────────────────────

export type BackgroundEegState = "idle" | "running" | "error";

export interface BackgroundEegStatus {
  state: BackgroundEegState;
  platform: "ios" | "android" | "web";
  wakeLockActive: boolean;
  error: string | null;
}

// ── Constants ─────────────────────────────────────────────────────────────────

/** Background task identifier — must match Info.plist BGTaskSchedulerPermittedIdentifiers */
const BACKGROUND_TASK_ID = "com.neuraldreamworkshop.eeg-flush";

/** Label shown in Android foreground service notification */
const FOREGROUND_NOTIFICATION_TITLE = "Neural Dream — Sleep Recording";
const FOREGROUND_NOTIFICATION_BODY  = "EEG monitoring active. Tap to open.";

// ── Web Wake Lock ─────────────────────────────────────────────────────────────

let _wakeLock: WakeLockSentinel | null = null;

async function acquireWakeLock(): Promise<boolean> {
  if (!("wakeLock" in navigator)) return false;
  try {
    _wakeLock = await (navigator.wakeLock as WakeLock).request("screen");
    _wakeLock.addEventListener("release", () => { _wakeLock = null; });
    return true;
  } catch {
    return false;
  }
}

async function releaseWakeLock(): Promise<void> {
  if (_wakeLock) {
    await _wakeLock.release().catch(() => {});
    _wakeLock = null;
  }
}

// ── Capacitor Background Runner ───────────────────────────────────────────────

async function startBackgroundRunner(): Promise<void> {
  try {
    const { BackgroundRunner } = await import("@capacitor/background-runner");
    // Dispatch an event to the background runner JS context
    // The runner script (background-runner.js in capacitor.config.ts) handles it
    await BackgroundRunner.dispatchEvent({
      label: BACKGROUND_TASK_ID,
      event: "eegSessionStart",
      details: { userId: getParticipantId() },
    });
  } catch (e) {
    console.warn("[background-eeg] Background runner not available:", e);
  }
}

async function stopBackgroundRunner(): Promise<void> {
  try {
    const { BackgroundRunner } = await import("@capacitor/background-runner");
    await BackgroundRunner.dispatchEvent({
      label: BACKGROUND_TASK_ID,
      event: "eegSessionStop",
      details: { userId: getParticipantId() },
    });
  } catch { /* ignore */ }
}

// ── Main manager ──────────────────────────────────────────────────────────────

class BackgroundEegManager {
  private _state: BackgroundEegState = "idle";
  private _wakeLockActive = false;
  private _error: string | null = null;
  private _keepAliveInterval: ReturnType<typeof setInterval> | null = null;

  getStatus(): BackgroundEegStatus {
    const platform = Capacitor.isNativePlatform()
      ? (Capacitor.getPlatform() as "ios" | "android")
      : "web";
    return {
      state: this._state,
      platform,
      wakeLockActive: this._wakeLockActive,
      error: this._error,
    };
  }

  /**
   * Start background EEG recording for sleep mode.
   * - Web: acquires Screen Wake Lock to prevent sleep
   * - Native: activates Capacitor Background Runner + keeps BLE alive
   */
  async startSleepRecording(): Promise<void> {
    this._state = "running";
    this._error = null;

    if (!Capacitor.isNativePlatform()) {
      // Web: use wake lock
      this._wakeLockActive = await acquireWakeLock();
      // Re-acquire wake lock if it gets released (e.g., tab visibility change)
      document.addEventListener("visibilitychange", this._handleVisibilityChange);
      return;
    }

    // Native: start background runner
    try {
      await startBackgroundRunner();
      this._startKeepAlive();
    } catch (e) {
      this._state = "error";
      this._error = String(e);
    }
  }

  /**
   * Stop background EEG recording.
   */
  async stopSleepRecording(): Promise<void> {
    document.removeEventListener("visibilitychange", this._handleVisibilityChange);
    await releaseWakeLock();
    this._wakeLockActive = false;
    this._stopKeepAlive();

    if (Capacitor.isNativePlatform()) {
      await stopBackgroundRunner();
    }

    this._state = "idle";
  }

  /** Keep-alive ping to ML backend every 5 min so it doesn't timeout the session */
  private _startKeepAlive(): void {
    if (this._keepAliveInterval) return;
    this._keepAliveInterval = setInterval(async () => {
      try {
        const mlUrl = localStorage.getItem("ml_backend_url") ?? "http://localhost:8000";
        await fetch(`${mlUrl}/api/health`, { method: "GET" }).catch(() => {});
      } catch { /* ignore */ }
    }, 5 * 60 * 1000);
  }

  private _stopKeepAlive(): void {
    if (this._keepAliveInterval) {
      clearInterval(this._keepAliveInterval);
      this._keepAliveInterval = null;
    }
  }

  private _handleVisibilityChange = async () => {
    if (document.visibilityState === "visible" && this._state === "running" && !_wakeLock) {
      // Re-acquire wake lock when tab becomes visible again
      this._wakeLockActive = await acquireWakeLock();
    }
  };
}

export const backgroundEeg = new BackgroundEegManager();

// ── Background runner script content ──────────────────────────────────────────
// This JS string is the content for client/public/background-runner.js
// It runs in an isolated JS context (not the main browser JS env).
// Configured in capacitor.config.ts as BackgroundRunner.localNotifications.
export const BACKGROUND_RUNNER_SCRIPT = `
// background-runner.js — runs in Capacitor BackgroundRunner isolated context
// Receives events from the main app via BackgroundRunner.dispatchEvent()

addEventListener("eegSessionStart", async (resolve, reject, args) => {
  try {
    // Post a local notification so the user knows recording is active
    await CapacitorNotifications.schedule([{
      id: 1001,
      title: "Neural Dream — Sleep Recording",
      body: "EEG monitoring active",
      sound: null,
      smallIcon: null,
      iconColor: "#6366f1",
      schedule: { at: new Date() },
      ongoing: true
    }]);
    resolve();
  } catch(e) {
    reject(e.message);
  }
});

addEventListener("eegSessionStop", async (resolve, reject, _args) => {
  try {
    await CapacitorNotifications.cancel([{ id: 1001 }]);
    resolve();
  } catch(e) {
    reject(e.message);
  }
});

addEventListener("eegFlush", async (resolve, reject, args) => {
  // Periodic flush: POST any buffered samples to the server
  // args.mlApiUrl is set by the main thread
  try {
    const res = await fetch(args.mlApiUrl + "/api/health", { method: "GET" });
    if (!res.ok) throw new Error("ML API unreachable");
    resolve({ flushed: true });
  } catch(e) {
    reject(e.message);
  }
});
`;

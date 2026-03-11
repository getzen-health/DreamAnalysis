/**
 * health-sync.ts — Apple HealthKit (iOS) and Google Health Connect (Android) integration.
 *
 * Pulls biometric data from the OS health APIs and posts it to the ML backend's
 * POST /biometrics/update endpoint so MultimodalEmotionFusion gets enriched with
 * real wearable signals.
 *
 * Platform routing:
 *   iOS    → @perfood/capacitor-healthkit (heart rate, HRV proxy, respiratory rate,
 *             SpO2, body temp, sleep stages, steps, active energy)
 *   Android → capacitor-health (heart rate workouts, steps, active calories, mindfulness)
 *   Web     → no-op (silently skipped)
 *
 * Usage:
 *   import { healthSync } from "@/lib/health-sync";
 *   await healthSync.initialize();   // request permissions
 *   await healthSync.syncNow();      // pull + post to backend
 *
 *   // Or use the useHealthSync() hook which auto-syncs every 15 min
 */

import { Capacitor } from "@capacitor/core";
import { getParticipantId } from "./participant";
import { getMLApiUrl } from "./ml-api";

// ── Types ────────────────────────────────────────────────────────────────────

/** Shape matching POST /biometrics/update in ml/api/routes/biometrics.py */
export interface BiometricPayload {
  user_id: string;
  hrv_sdnn?: number;
  hrv_rmssd?: number;
  hrv_lf_hf_ratio?: number;
  resting_heart_rate?: number;
  current_heart_rate?: number;
  respiratory_rate?: number;
  spo2?: number;
  skin_temperature_deviation?: number;
  sleep_total_hours?: number;
  sleep_rem_hours?: number;
  sleep_deep_hours?: number;
  sleep_efficiency?: number;
  hours_since_wake?: number;
  steps_today?: number;
  active_energy_kcal?: number;
  exercise_minutes_today?: number;
  minutes_since_last_meal?: number;
}

export type HealthSyncStatus =
  | "unavailable"  // not on native platform
  | "unauthorized" // permissions denied
  | "idle"         // ready, not syncing
  | "syncing"      // sync in progress
  | "ok"           // last sync succeeded
  | "error";       // last sync failed

export interface HealthSyncState {
  status: HealthSyncStatus;
  lastSyncAt: Date | null;
  latestPayload: BiometricPayload | null;
  error: string | null;
}

// ── Platform detection ────────────────────────────────────────────────────────

function getOS(): "ios" | "android" | "web" {
  const platform = Capacitor.getPlatform();
  if (platform === "ios") return "ios";
  if (platform === "android") return "android";
  return "web";
}

// ── iOS HealthKit data pull ──────────────────────────────────────────────────

async function pullAppleHealth(userId: string): Promise<BiometricPayload> {
  const { CapacitorHealthkit } = await import("@perfood/capacitor-healthkit");

  const now = new Date();
  const yesterday = new Date(now.getTime() - 24 * 60 * 60 * 1000);
  const todayStart = new Date(now);
  todayStart.setHours(0, 0, 0, 0);
  const fmt = (d: Date) => d.toISOString();

  const payload: BiometricPayload = { user_id: userId };

  // ── Heart rate (last 30 min → current HR) ──
  try {
    const hrResult = await CapacitorHealthkit.queryHKitSampleType<{ value: number }>({
      sampleName: "heartRate",
      startDate: fmt(new Date(now.getTime() - 30 * 60 * 1000)),
      endDate: fmt(now),
      limit: 20,
    });
    if (hrResult.resultData.length > 0) {
      const samples = hrResult.resultData.map((d) => d.value).filter((v) => v > 0);
      if (samples.length > 0) {
        payload.current_heart_rate = samples[samples.length - 1];
      }
    }
  } catch { /* permission not granted or data unavailable */ }

  // ── Resting heart rate (today) ──
  try {
    const rhrResult = await CapacitorHealthkit.queryHKitSampleType<{ value: number }>({
      sampleName: "restingHeartRate",
      startDate: fmt(todayStart),
      endDate: fmt(now),
      limit: 3,
    });
    if (rhrResult.resultData.length > 0) {
      payload.resting_heart_rate = rhrResult.resultData[0].value;
    }
  } catch { /* ok */ }

  // ── HRV SDNN (from Heart Rate Variability samples, last 8 hours) ──
  // Apple Health stores HRV as RMSSD (standard nighttime HRV measurement).
  // We expose it as hrv_sdnn (the ML backend treats both similarly).
  try {
    const hrvResult = await CapacitorHealthkit.queryHKitSampleType<{ value: number }>({
      sampleName: "heartRate", // HealthKit stores per-beat HR; derive HRV from variance
      startDate: fmt(new Date(now.getTime() - 8 * 60 * 60 * 1000)),
      endDate: fmt(now),
      limit: 200,
    });
    if (hrvResult.resultData.length >= 10) {
      const beats = hrvResult.resultData.map((d) => d.value).filter((v) => v > 30 && v < 200);
      if (beats.length >= 2) {
        // Convert BPM to RR intervals (ms), then compute SDNN proxy
        const rr = beats.map((bpm) => 60000 / bpm);
        const mean = rr.reduce((a, b) => a + b, 0) / rr.length;
        const sdnn = Math.sqrt(rr.reduce((a, v) => a + (v - mean) ** 2, 0) / rr.length);
        payload.hrv_sdnn = sdnn;
      }
    }
  } catch { /* ok */ }

  // ── Respiratory rate ──
  try {
    const rrResult = await CapacitorHealthkit.queryHKitSampleType<{ value: number }>({
      sampleName: "respiratoryRate",
      startDate: fmt(new Date(now.getTime() - 60 * 60 * 1000)),
      endDate: fmt(now),
      limit: 5,
    });
    if (rrResult.resultData.length > 0) {
      payload.respiratory_rate = rrResult.resultData[0].value;
    }
  } catch { /* ok */ }

  // ── SpO2 ──
  try {
    const spo2Result = await CapacitorHealthkit.queryHKitSampleType<{ value: number }>({
      sampleName: "oxygenSaturation",
      startDate: fmt(new Date(now.getTime() - 4 * 60 * 60 * 1000)),
      endDate: fmt(now),
      limit: 5,
    });
    if (spo2Result.resultData.length > 0) {
      payload.spo2 = spo2Result.resultData[0].value * 100; // convert 0-1 → 0-100 if needed
    }
  } catch { /* ok */ }

  // ── Body temperature ──
  try {
    const tempResult = await CapacitorHealthkit.queryHKitSampleType<{ value: number }>({
      sampleName: "bodyTemperature",
      startDate: fmt(new Date(now.getTime() - 12 * 60 * 60 * 1000)),
      endDate: fmt(now),
      limit: 3,
    });
    if (tempResult.resultData.length > 0) {
      // Express as deviation from 37°C (normal body temp)
      const temp = tempResult.resultData[0].value;
      payload.skin_temperature_deviation = temp - 37.0;
    }
  } catch { /* ok */ }

  // ── Sleep (last night) ──
  try {
    const sleepStart = new Date(now);
    sleepStart.setDate(sleepStart.getDate() - 1);
    sleepStart.setHours(20, 0, 0, 0);
    const sleepEnd = new Date(todayStart);
    sleepEnd.setHours(12, 0, 0, 0);

    const sleepResult = await CapacitorHealthkit.queryHKitSampleType<{
      sleepState: string;
      duration: number;
    }>({
      sampleName: "sleepAnalysis",
      startDate: fmt(sleepStart),
      endDate: fmt(sleepEnd),
      limit: 100,
    });

    if (sleepResult.resultData.length > 0) {
      let totalMin = 0, remMin = 0, deepMin = 0;
      for (const s of sleepResult.resultData) {
        const durMin = s.duration / 60;
        const state = s.sleepState?.toLowerCase() ?? "";
        // HealthKit sleep states: inBed, awake, core/light, deep, rem
        if (state !== "awake" && state !== "inbed") totalMin += durMin;
        if (state === "rem") remMin += durMin;
        if (state === "deep") deepMin += durMin;
      }
      payload.sleep_total_hours = totalMin / 60;
      payload.sleep_rem_hours   = remMin / 60;
      payload.sleep_deep_hours  = deepMin / 60;
      if (totalMin > 0) {
        payload.sleep_efficiency = Math.min(1, totalMin / 480) * 100; // 8h = 100%
      }
    }
  } catch { /* ok */ }

  // ── Steps today ──
  try {
    const stepsResult = await CapacitorHealthkit.queryHKitSampleType<{ value: number }>({
      sampleName: "stepCount",
      startDate: fmt(todayStart),
      endDate: fmt(now),
      limit: 200,
    });
    if (stepsResult.resultData.length > 0) {
      payload.steps_today = stepsResult.resultData.reduce((sum, s) => sum + s.value, 0);
    }
  } catch { /* ok */ }

  // ── Active energy today ──
  try {
    const energyResult = await CapacitorHealthkit.queryHKitSampleType<{ value: number }>({
      sampleName: "activeEnergyBurned",
      startDate: fmt(todayStart),
      endDate: fmt(now),
      limit: 200,
    });
    if (energyResult.resultData.length > 0) {
      payload.active_energy_kcal = energyResult.resultData.reduce((sum, s) => sum + s.value, 0);
    }
  } catch { /* ok */ }

  // ── Hours since wake (derived from sleep end time) ──
  if (payload.sleep_total_hours !== undefined) {
    // Assume woke up 6-8 hours before current time if we have sleep data
    payload.hours_since_wake = Math.max(0, (now.getHours() - 7)); // rough proxy
  }

  return payload;
}

// ── Android Google Health Connect data pull ───────────────────────────────────

async function pullAndroidHealth(userId: string): Promise<BiometricPayload> {
  const { Health } = await import("capacitor-health");

  const payload: BiometricPayload = { user_id: userId };
  const now = new Date();
  const todayStart = new Date(now);
  todayStart.setHours(0, 0, 0, 0);
  const fmt = (d: Date) => d.toISOString();

  // ── Heart rate from workouts ──
  try {
    const workouts = await Health.queryWorkouts({
      startDate: fmt(new Date(now.getTime() - 24 * 60 * 60 * 1000)),
      endDate: fmt(now),
      includeHeartRate: true,
      includeRoute: false,
      includeSteps: false,
    });
    if (workouts.workouts.length > 0) {
      const latest = workouts.workouts[workouts.workouts.length - 1];
      if (latest.heartRate && latest.heartRate.length > 0) {
        const hrSamples = latest.heartRate.map((h) => h.bpm).filter((v) => v > 0);
        if (hrSamples.length > 0) {
          payload.current_heart_rate = hrSamples[hrSamples.length - 1];
        }
      }
    }
  } catch { /* ok */ }

  // ── Steps today ──
  try {
    const steps = await Health.queryAggregated({
      startDate: fmt(todayStart),
      endDate: fmt(now),
      dataType: "steps",
      bucket: "DAY",
    });
    if (steps.aggregatedData.length > 0) {
      payload.steps_today = steps.aggregatedData.reduce((s, d) => s + d.value, 0);
    }
  } catch { /* ok */ }

  // ── Active calories today ──
  try {
    const cals = await Health.queryAggregated({
      startDate: fmt(todayStart),
      endDate: fmt(now),
      dataType: "active-calories",
      bucket: "DAY",
    });
    if (cals.aggregatedData.length > 0) {
      payload.active_energy_kcal = cals.aggregatedData.reduce((s, d) => s + d.value, 0);
    }
  } catch { /* ok */ }

  // ── Mindfulness (exercise minutes proxy) ──
  try {
    const mind = await Health.queryAggregated({
      startDate: fmt(todayStart),
      endDate: fmt(now),
      dataType: "mindfulness",
      bucket: "DAY",
    });
    if (mind.aggregatedData.length > 0) {
      payload.exercise_minutes_today = mind.aggregatedData.reduce((s, d) => s + d.value, 0);
    }
  } catch { /* ok */ }

  return payload;
}

// ── Permission request ────────────────────────────────────────────────────────

async function requestPermissionsIos(): Promise<void> {
  const { CapacitorHealthkit } = await import("@perfood/capacitor-healthkit");
  await CapacitorHealthkit.requestAuthorization({
    all: [],
    read: [
      "heartRate",
      "restingHeartRate",
      "respiratoryRate",
      "oxygenSaturation",
      "bodyTemperature",
      "sleepAnalysis",
      "stepCount",
      "activeEnergyBurned",
    ],
    write: [],
  });
}

async function requestPermissionsAndroid(): Promise<void> {
  const { Health } = await import("capacitor-health");
  await Health.requestHealthPermissions({
    permissions: [
      "READ_STEPS",
      "READ_HEART_RATE",
      "READ_ACTIVE_CALORIES",
      "READ_WORKOUTS",
      "READ_MINDFULNESS",
    ],
  });
}

// ── Post to ML backend ────────────────────────────────────────────────────────

async function postToBackend(payload: BiometricPayload): Promise<void> {
  const url = getMLApiUrl();
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (url.includes("ngrok")) headers["ngrok-skip-browser-warning"] = "true";

  const res = await fetch(`${url}/api/biometrics/update`, {
    method: "POST",
    headers,
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`/biometrics/update → ${res.status}`);
}

// ── HealthSyncManager ─────────────────────────────────────────────────────────

class HealthSyncManager {
  private state: HealthSyncState = {
    status: "unavailable",
    lastSyncAt: null,
    latestPayload: null,
    error: null,
  };
  private listeners: Set<(s: HealthSyncState) => void> = new Set();
  private syncTimer: ReturnType<typeof setInterval> | null = null;
  private initialized = false;

  getState(): HealthSyncState {
    return { ...this.state };
  }

  subscribe(listener: (s: HealthSyncState) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private emit(): void {
    const snap = this.getState();
    this.listeners.forEach((l) => l(snap));
  }

  private set(updates: Partial<HealthSyncState>): void {
    this.state = { ...this.state, ...updates };
    this.emit();
  }

  /** Request permissions and mark manager as ready. Call once on app start. */
  async initialize(): Promise<void> {
    if (this.initialized) return;
    this.initialized = true;

    const os = getOS();
    if (os === "web") {
      this.set({ status: "unavailable" });
      return;
    }

    try {
      if (os === "ios") {
        await requestPermissionsIos();
      } else {
        const { Health } = await import("capacitor-health");
        const available = await Health.isHealthAvailable();
        if (!available.available) {
          this.set({ status: "unavailable", error: "Google Health Connect not installed" });
          return;
        }
        await requestPermissionsAndroid();
      }
      this.set({ status: "idle", error: null });
    } catch (e) {
      this.set({
        status: "unauthorized",
        error: `Health permissions denied: ${String(e)}`,
      });
    }
  }

  /** Pull latest health data and push to ML backend. */
  async syncNow(): Promise<void> {
    const os = getOS();
    if (os === "web" || this.state.status === "unavailable" || this.state.status === "unauthorized") {
      return;
    }

    this.set({ status: "syncing" });
    const userId = getParticipantId();

    try {
      let payload: BiometricPayload;
      if (os === "ios") {
        payload = await pullAppleHealth(userId);
      } else {
        payload = await pullAndroidHealth(userId);
      }

      await postToBackend(payload);

      this.set({
        status: "ok",
        lastSyncAt: new Date(),
        latestPayload: payload,
        error: null,
      });
    } catch (e) {
      this.set({
        status: "error",
        error: String(e),
      });
    }
  }

  /** Start background sync every intervalMs (default: 15 minutes). */
  startAutoSync(intervalMs = 15 * 60 * 1000): void {
    if (this.syncTimer) return;
    // First sync immediately
    this.syncNow().catch(() => {});
    this.syncTimer = setInterval(() => {
      this.syncNow().catch(() => {});
    }, intervalMs);
  }

  stopAutoSync(): void {
    if (this.syncTimer) {
      clearInterval(this.syncTimer);
      this.syncTimer = null;
    }
  }
}

// ── Singleton ────────────────────────────────────────────────────────────────

export const healthSync = new HealthSyncManager();

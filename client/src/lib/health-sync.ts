/**
 * health-sync.ts — Apple HealthKit (iOS) and Google Health Connect (Android) integration.
 *
 * Pulls biometric data from the OS health APIs and posts it to:
 *   1. ML backend's POST /biometrics/update (MultimodalEmotionFusion enrichment)
 *   2. Supabase Edge Function ingest-health-data (normalized health_samples pipeline)
 *
 * Platform routing:
 *   iOS    → @perfood/capacitor-healthkit (heart rate, HRV proxy, respiratory rate,
 *             SpO2, body temp, sleep stages, steps, active energy, weight, body fat,
 *             lean mass, height, VO2 max, workouts)
 *   Android → capacitor-health (heart rate workouts, steps, active calories, mindfulness,
 *             weight, body fat)
 *   Web     → no-op (silently skipped)
 *
 * Usage:
 *   import { healthSync } from "@/lib/health-sync";
 *   await healthSync.initialize();   // request permissions
 *   await healthSync.syncNow();      // pull + post to backend + Supabase
 *
 *   // Or use the useHealthSync() hook which auto-syncs every 15 min
 */

import { Capacitor } from "@capacitor/core";
import { getParticipantId } from "./participant";
import { getMLApiUrl } from "./ml-api";
import { apiRequest } from "./queryClient";
import { sbGetSetting, sbSaveGeneric } from "./supabase-store";

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
  active_calories?: number;
  exercise_minutes_today?: number;
  minutes_since_last_meal?: number;
  // Body composition
  weight_kg?: number;
  body_fat_pct?: number;
  lean_mass_kg?: number;
  height_cm?: number;
  vo2_max?: number;
  // Extended metrics (Withings-style)
  walking_distance_km?: number;
  flights_climbed?: number;
  standing_hours?: number;
  blood_pressure_systolic?: number;
  blood_pressure_diastolic?: number;
  body_temperature_c?: number;
  ecg_classification?: string;
  water_intake_ml?: number;
}

/** Shape for Supabase ingest-health-data Edge Function */
export interface SupabaseHealthSample {
  source: string;
  metric: string;
  value: number;
  unit: string;
  recorded_at: string;
  metadata?: Record<string, unknown>;
}

/** Workout data from Apple HealthKit */
export interface WorkoutData {
  workoutType: string;
  durationMinutes: number;
  caloriesBurned: number;
  averageHeartRate?: number;
  startDate: string;
  endDate: string;
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

interface PullResult {
  payload: BiometricPayload;
  workouts: WorkoutData[];
}

async function pullAppleHealth(userId: string): Promise<PullResult> {
  const { CapacitorHealthkit } = await import("@perfood/capacitor-healthkit");

  const now = new Date();
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

  // ── Weight (last 24 hours) ──
  try {
    const weightResult = await CapacitorHealthkit.queryHKitSampleType<{ value: number; startDate: string }>({
      sampleName: "bodyMass",
      startDate: fmt(new Date(now.getTime() - 24 * 60 * 60 * 1000)),
      endDate: fmt(now),
      limit: 5,
    });
    if (weightResult.resultData.length > 0) {
      payload.weight_kg = weightResult.resultData[weightResult.resultData.length - 1].value;
    }
  } catch { /* ok */ }

  // ── Body fat percentage (last 24 hours) ──
  try {
    const bfResult = await CapacitorHealthkit.queryHKitSampleType<{ value: number; startDate: string }>({
      sampleName: "bodyFatPercentage",
      startDate: fmt(new Date(now.getTime() - 24 * 60 * 60 * 1000)),
      endDate: fmt(now),
      limit: 5,
    });
    if (bfResult.resultData.length > 0) {
      // HealthKit stores as 0-1 fraction, convert to percentage
      payload.body_fat_pct = bfResult.resultData[bfResult.resultData.length - 1].value * 100;
    }
  } catch { /* ok */ }

  // ── Lean body mass (last 24 hours) ──
  try {
    const lbmResult = await CapacitorHealthkit.queryHKitSampleType<{ value: number; startDate: string }>({
      sampleName: "leanBodyMass",
      startDate: fmt(new Date(now.getTime() - 24 * 60 * 60 * 1000)),
      endDate: fmt(now),
      limit: 5,
    });
    if (lbmResult.resultData.length > 0) {
      payload.lean_mass_kg = lbmResult.resultData[lbmResult.resultData.length - 1].value;
    }
  } catch { /* ok */ }

  // ── Height (last recorded value, wider window) ──
  try {
    const heightResult = await CapacitorHealthkit.queryHKitSampleType<{ value: number }>({
      sampleName: "height",
      startDate: fmt(new Date(now.getTime() - 365 * 24 * 60 * 60 * 1000)),
      endDate: fmt(now),
      limit: 3,
    });
    if (heightResult.resultData.length > 0) {
      // HealthKit stores height in meters, convert to cm
      payload.height_cm = heightResult.resultData[heightResult.resultData.length - 1].value * 100;
    }
  } catch { /* ok */ }

  // ── VO2 Max (last 24 hours) ──
  try {
    const vo2Result = await CapacitorHealthkit.queryHKitSampleType<{ value: number }>({
      sampleName: "vo2Max",
      startDate: fmt(new Date(now.getTime() - 24 * 60 * 60 * 1000)),
      endDate: fmt(now),
      limit: 5,
    });
    if (vo2Result.resultData.length > 0) {
      payload.vo2_max = vo2Result.resultData[vo2Result.resultData.length - 1].value;
    }
  } catch { /* ok */ }

  // ── Walking distance today (meters → km) ──
  try {
    const distResult = await CapacitorHealthkit.queryHKitSampleType<{ value: number }>({
      sampleName: "distanceWalkingRunning",
      startDate: fmt(todayStart),
      endDate: fmt(now),
      limit: 200,
    });
    if (distResult.resultData.length > 0) {
      const totalMeters = distResult.resultData.reduce((sum, s) => sum + s.value, 0);
      payload.walking_distance_km = totalMeters / 1000;
    }
  } catch { /* ok */ }

  // ── Flights climbed today ──
  try {
    const flightsResult = await CapacitorHealthkit.queryHKitSampleType<{ value: number }>({
      sampleName: "flightsClimbed",
      startDate: fmt(todayStart),
      endDate: fmt(now),
      limit: 200,
    });
    if (flightsResult.resultData.length > 0) {
      payload.flights_climbed = flightsResult.resultData.reduce((sum, s) => sum + s.value, 0);
    }
  } catch { /* ok */ }

  // ── Standing hours today (Apple Stand Hours) ──
  try {
    const standResult = await CapacitorHealthkit.queryHKitSampleType<{ value: number }>({
      sampleName: "appleStandHour",
      startDate: fmt(todayStart),
      endDate: fmt(now),
      limit: 24,
    });
    if (standResult.resultData.length > 0) {
      payload.standing_hours = standResult.resultData.reduce((sum, s) => sum + s.value, 0);
    }
  } catch { /* ok */ }

  // ── Blood pressure (last 24 hours) ──
  try {
    const bpSystolic = await CapacitorHealthkit.queryHKitSampleType<{ value: number }>({
      sampleName: "bloodPressureSystolic",
      startDate: fmt(new Date(now.getTime() - 24 * 60 * 60 * 1000)),
      endDate: fmt(now),
      limit: 5,
    });
    if (bpSystolic.resultData.length > 0) {
      payload.blood_pressure_systolic = bpSystolic.resultData[bpSystolic.resultData.length - 1].value;
    }
    const bpDiastolic = await CapacitorHealthkit.queryHKitSampleType<{ value: number }>({
      sampleName: "bloodPressureDiastolic",
      startDate: fmt(new Date(now.getTime() - 24 * 60 * 60 * 1000)),
      endDate: fmt(now),
      limit: 5,
    });
    if (bpDiastolic.resultData.length > 0) {
      payload.blood_pressure_diastolic = bpDiastolic.resultData[bpDiastolic.resultData.length - 1].value;
    }
  } catch { /* ok */ }

  // ── Body temperature (absolute value, not deviation) ──
  try {
    const btResult = await CapacitorHealthkit.queryHKitSampleType<{ value: number }>({
      sampleName: "bodyTemperature",
      startDate: fmt(new Date(now.getTime() - 12 * 60 * 60 * 1000)),
      endDate: fmt(now),
      limit: 3,
    });
    if (btResult.resultData.length > 0) {
      payload.body_temperature_c = btResult.resultData[0].value;
    }
  } catch { /* ok */ }

  // ── ECG / AFib classification (if available) ──
  try {
    const ecgResult = await CapacitorHealthkit.queryHKitSampleType<{ value: number; classification: string }>({
      sampleName: "electrocardiogram",
      startDate: fmt(new Date(now.getTime() - 24 * 60 * 60 * 1000)),
      endDate: fmt(now),
      limit: 3,
    });
    if (ecgResult.resultData.length > 0) {
      const latest = ecgResult.resultData[ecgResult.resultData.length - 1];
      payload.ecg_classification = latest.classification ?? "unknown";
    }
  } catch { /* ok */ }

  // ── Water intake (dietary water, last 24 hours) ──
  try {
    const waterResult = await CapacitorHealthkit.queryHKitSampleType<{ value: number }>({
      sampleName: "dietaryWater",
      startDate: fmt(todayStart),
      endDate: fmt(now),
      limit: 50,
    });
    if (waterResult.resultData.length > 0) {
      // HealthKit stores dietary water in liters; convert to ml
      const totalLiters = waterResult.resultData.reduce((sum, s) => sum + s.value, 0);
      payload.water_intake_ml = totalLiters * 1000;
    }
  } catch { /* ok */ }

  // ── Workouts (last 24 hours) ──
  const workouts: WorkoutData[] = [];
  try {
    const workoutResult = await CapacitorHealthkit.queryHKitSampleType<{
      workoutActivityType: string;
      duration: number;
      totalEnergyBurned: number;
      startDate: string;
      endDate: string;
    }>({
      sampleName: "workout",
      startDate: fmt(new Date(now.getTime() - 24 * 60 * 60 * 1000)),
      endDate: fmt(now),
      limit: 20,
    });

    let totalExerciseMinutes = 0;

    for (const w of workoutResult.resultData) {
      const durationMin = (w.duration || 0) / 60;
      totalExerciseMinutes += durationMin;

      const workout: WorkoutData = {
        workoutType: w.workoutActivityType || "unknown",
        durationMinutes: durationMin,
        caloriesBurned: w.totalEnergyBurned || 0,
        startDate: w.startDate,
        endDate: w.endDate,
      };

      // Try to get average HR during the workout window
      try {
        const workoutHR = await CapacitorHealthkit.queryHKitSampleType<{ value: number }>({
          sampleName: "heartRate",
          startDate: w.startDate,
          endDate: w.endDate,
          limit: 100,
        });
        if (workoutHR.resultData.length > 0) {
          const hrValues = workoutHR.resultData.map((h) => h.value).filter((v) => v > 0);
          if (hrValues.length > 0) {
            workout.averageHeartRate = hrValues.reduce((a, b) => a + b, 0) / hrValues.length;
          }
        }
      } catch { /* ok */ }

      workouts.push(workout);
    }

    if (totalExerciseMinutes > 0) {
      payload.exercise_minutes_today = totalExerciseMinutes;
    }
  } catch { /* ok */ }

  // ── Hours since wake (derived from sleep end time) ──
  if (payload.sleep_total_hours !== undefined) {
    // Assume woke up 6-8 hours before current time if we have sleep data
    payload.hours_since_wake = Math.max(0, (now.getHours() - 7)); // rough proxy
  }

  return { payload, workouts };
}

// ── Android Google Health Connect data pull ───────────────────────────────────

async function pullAndroidHealth(userId: string): Promise<PullResult> {
  // Use @capgo/capacitor-health which supports heartRate, sleep, HRV, weight, etc.
  const capgoModule = await import("@capgo/capacitor-health");
  const HC = capgoModule.Health;

  const payload: BiometricPayload = { user_id: userId };
  const diagnostics: string[] = [];
  const now = new Date();
  const todayStart = new Date(now);
  todayStart.setHours(0, 0, 0, 0);
  const fmt = (d: Date) => d.toISOString();

  // Check if Health Connect is available
  try {
    const avail = await HC.isAvailable();
    diagnostics.push(`Health Connect available: ${JSON.stringify(avail)}`);
    if (!avail?.available) {
      diagnostics.push("Health Connect NOT available — install from Play Store");
      (payload as any)._diagnostics = diagnostics;
      return { payload, workouts: [] };
    }
  } catch (e) {
    diagnostics.push(`isAvailable check failed: ${String(e).slice(0, 100)}`);
  }

  // Request permissions
  try {
    await HC.requestAuthorization({
      read: ["heartRate", "restingHeartRate", "heartRateVariability", "steps", "calories", "distance", "sleep", "oxygenSaturation", "weight", "bodyFat", "respiratoryRate", "bloodPressure"],
      write: [],
    });
    diagnostics.push("Permissions requested OK");
  } catch (e) {
    diagnostics.push(`Permission request failed: ${String(e).slice(0, 100)}`);
  }

  // Helper: read latest samples for a data type
  async function readLatest(dataType: string, hoursBack: number = 24): Promise<number | null> {
    try {
      const result = await HC.readSamples({
        dataType: dataType as any,
        startDate: fmt(new Date(now.getTime() - hoursBack * 3600000)),
        endDate: fmt(now),
        limit: 10,
      });
      const samples = result?.samples ?? [];
      if (samples.length > 0) {
        const latest = samples[samples.length - 1];
        const val = latest.value ?? 0;
        diagnostics.push(`${dataType}: ${samples.length} samples, latest=${val}`);
        return val;
      }
      diagnostics.push(`${dataType}: 0 samples`);
      return null;
    } catch (e) {
      diagnostics.push(`${dataType}: FAILED (${String(e).slice(0, 80)})`);
      return null;
    }
  }

  // Heart rate — try readSamples first, then queryAggregated
  let hr = await readLatest("heartRate", 4);
  if (!hr) {
    // Fallback: try queryAggregated
    try {
      const agg = await HC.queryAggregated({
        dataType: "heartRate" as any,
        startDate: fmt(new Date(now.getTime() - 4 * 3600000)),
        endDate: fmt(now),
        bucket: "hour",
      });
      if (agg?.samples?.length > 0) {
        const latest = agg.samples[agg.samples.length - 1];
        if (latest.value > 0) { hr = latest.value; diagnostics.push(`heartRate(agg): ${latest.value}`); }
      }
    } catch (e) { diagnostics.push(`heartRate(agg): FAILED (${String(e).slice(0, 60)})`); }
  }
  if (hr && hr > 0) payload.current_heart_rate = Math.round(hr);

  // Resting heart rate
  const rhr = await readLatest("restingHeartRate", 48);
  if (rhr && rhr > 0) payload.resting_heart_rate = Math.round(rhr);

  // HRV
  const hrv = await readLatest("heartRateVariability", 48);
  if (hrv && hrv > 0) payload.hrv_rmssd = hrv;

  // Sleep
  const sleep = await readLatest("sleep", 24);
  if (sleep && sleep > 0) payload.sleep_total_hours = sleep / 60;

  // Steps (use queryAggregated for steps — it's supported)
  try {
    const stepsResult = await HC.queryAggregated({
      dataType: "steps" as any,
      startDate: fmt(todayStart),
      endDate: fmt(now),
      bucket: "day",
    });
    if (stepsResult?.samples?.length > 0 && stepsResult.samples[0].value > 0) {
      payload.steps_today = stepsResult.samples[0].value;
      diagnostics.push(`steps: ${payload.steps_today}`);
    } else {
      diagnostics.push("steps: 0");
    }
  } catch (e) { diagnostics.push(`steps: FAILED (${String(e).slice(0, 80)})`); }

  // Active calories
  try {
    const calsResult = await HC.queryAggregated({
      dataType: "calories" as any,
      startDate: fmt(todayStart),
      endDate: fmt(now),
      bucket: "day",
    });
    if (calsResult?.samples?.length > 0 && calsResult.samples[0].value > 0) {
      payload.active_calories = Math.round(calsResult.samples[0].value);
      diagnostics.push(`calories: ${payload.active_calories}`);
    }
  } catch (e) { diagnostics.push(`calories: FAILED (${String(e).slice(0, 80)})`); }

  // SpO2
  const spo2 = await readLatest("oxygenSaturation", 24);
  if (spo2 && spo2 > 0) payload.spo2 = spo2;

  // Weight
  const weight = await readLatest("weight", 168); // last week
  if (weight && weight > 0) payload.weight_kg = weight;

  // Body fat
  const bodyFat = await readLatest("bodyFat", 168);
  if (bodyFat && bodyFat > 0) payload.body_fat_pct = bodyFat;

  // Distance
  const dist = await readLatest("distance", 24);
  if (dist && dist > 0) payload.walking_distance_km = dist / 1000;

  // Respiratory rate
  const rr = await readLatest("respiratoryRate", 24);
  if (rr && rr > 0) payload.respiratory_rate = rr;

  // Blood pressure
  const bpSys = await readLatest("bloodPressure", 168);
  if (bpSys && bpSys > 0) payload.blood_pressure_systolic = bpSys;

  if (payload.active_calories) payload.active_energy_kcal = payload.active_calories;

  // Summary: count how many data types returned data
  const dataKeys = Object.keys(payload).filter(k => k !== "user_id" && !k.startsWith("_"));
  diagnostics.push(`--- Summary: ${dataKeys.length} data types with values: ${dataKeys.join(", ") || "NONE"}`);
  if (dataKeys.length === 0) {
    diagnostics.push("NO DATA returned from Health Connect. Possible causes:");
    diagnostics.push("  1. Withings app: Settings > Health Connect > enable ALL data types");
    diagnostics.push("  2. Withings app: force sync (pull down to refresh in Withings app)");
    diagnostics.push("  3. Health Connect app: check Withings has read permissions granted");
    diagnostics.push("  4. No wearable data recorded in the last 24-48 hours");
  }

  // Store diagnostics so UI can show what happened
  (payload as any)._diagnostics = diagnostics;
  // Also persist diagnostics for debugging
  try { localStorage.setItem("ndw_health_diagnostics", JSON.stringify(diagnostics)); } catch { /* ok */ }

  return { payload, workouts: [] };
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
      "bodyMass",
      "bodyFatPercentage",
      "leanBodyMass",
      "height",
      "vo2Max",
      "workout",
      "distanceWalkingRunning",
      "flightsClimbed",
      "appleStandHour",
      "bloodPressureSystolic",
      "bloodPressureDiastolic",
      "electrocardiogram",
      "dietaryWater",
    ],
    write: [],
  });
}

async function requestPermissionsAndroid(): Promise<void> {
  const capgoModule = await import("@capgo/capacitor-health");
  await capgoModule.Health.requestAuthorization({
    read: [
      "heartRate",
      "restingHeartRate",
      "heartRateVariability",
      "steps",
      "calories",
      "distance",
      "sleep",
      "oxygenSaturation",
      "weight",
      "bodyFat",
      "respiratoryRate",
      "bloodPressure",
    ],
    write: [],
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

// ── Post to Supabase Edge Function ───────────────────────────────────────────

function buildSupabaseSamples(
  payload: BiometricPayload,
  workouts: WorkoutData[],
  source: "apple_health" | "google_fit",
): SupabaseHealthSample[] {
  const now = new Date().toISOString();
  const samples: SupabaseHealthSample[] = [];

  const add = (metric: string, value: number | undefined, unit: string, recordedAt?: string) => {
    if (value !== undefined && !isNaN(value)) {
      samples.push({ source, metric, value, unit, recorded_at: recordedAt || now });
    }
  };

  // Existing biometric fields
  add("heart_rate", payload.current_heart_rate, "bpm");
  add("resting_hr", payload.resting_heart_rate, "bpm");
  add("hrv_rmssd", payload.hrv_sdnn, "ms"); // SDNN proxy stored as hrv_rmssd
  add("respiratory_rate", payload.respiratory_rate, "breaths/min");
  add("spo2", payload.spo2, "%");
  add("skin_temp", payload.skin_temperature_deviation !== undefined
    ? 37.0 + payload.skin_temperature_deviation : undefined, "degC");
  add("sleep_deep_min", payload.sleep_deep_hours !== undefined
    ? payload.sleep_deep_hours * 60 : undefined, "min");
  add("sleep_rem_min", payload.sleep_rem_hours !== undefined
    ? payload.sleep_rem_hours * 60 : undefined, "min");
  add("sleep_efficiency", payload.sleep_efficiency, "%");
  add("steps", payload.steps_today, "count");
  add("active_calories", payload.active_energy_kcal, "kcal");
  add("exercise_minutes", payload.exercise_minutes_today, "min");

  // Body composition (new)
  add("weight_kg", payload.weight_kg, "kg");
  add("body_fat_pct", payload.body_fat_pct, "%");
  add("lean_mass_kg", payload.lean_mass_kg, "kg");
  add("height_cm", payload.height_cm, "cm");
  add("vo2_max", payload.vo2_max, "ml/kg/min");

  // Extended metrics
  add("walking_distance_km", payload.walking_distance_km, "km");
  add("flights_climbed", payload.flights_climbed, "count");
  add("standing_hours", payload.standing_hours, "hours");
  add("blood_pressure_systolic", payload.blood_pressure_systolic, "mmHg");
  add("blood_pressure_diastolic", payload.blood_pressure_diastolic, "mmHg");
  add("body_temperature", payload.body_temperature_c, "degC");
  add("water_intake_ml", payload.water_intake_ml, "ml");

  // Workouts → workout_strain per workout (with metadata)
  for (const w of workouts) {
    if (w.durationMinutes > 0) {
      samples.push({
        source,
        metric: "workout_strain",
        value: w.caloriesBurned,
        unit: "kcal",
        recorded_at: w.startDate || now,
        metadata: {
          workout_type: w.workoutType,
          duration_minutes: w.durationMinutes,
          average_heart_rate: w.averageHeartRate,
        },
      });
    }
  }

  return samples;
}

async function postToSupabase(userId: string, samples: SupabaseHealthSample[]): Promise<void> {
  const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
  if (!supabaseUrl || samples.length === 0) return;

  try {
    const res = await fetch(`${supabaseUrl}/functions/v1/ingest-health-data`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId, samples }),
    });
    if (!res.ok) {
      console.warn(`[health-sync] Supabase ingest failed: ${res.status}`);
    }
  } catch (e) {
    // Supabase ingest is best-effort; don't fail the sync
    console.warn("[health-sync] Supabase ingest error:", e);
  }
}

// ── HealthSyncManager ─────────────────────────────────────────────────────────

class HealthSyncManager {
  private state: HealthSyncState = {
    status: "unavailable",
    lastSyncAt: null,
    // Load cached health data on startup so UI has data immediately
    latestPayload: (() => {
      try {
        const saved = sbGetSetting("ndw_health_payload");
        return saved ? JSON.parse(saved) : null;
      } catch { return null; }
    })(),
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
        const capgoModule = await import("@capgo/capacitor-health");
        const Health = capgoModule.Health;
        let available: { available: boolean };
        try {
          available = await Health.isAvailable();
        } catch {
          this.set({
            status: "unavailable",
            error: "Google Health Connect is not installed. Install it from the Play Store to sync health data.",
          });
          return;
        }
        if (!available.available) {
          this.set({
            status: "unavailable",
            error: "Google Health Connect is not installed. Install it from the Play Store to sync health data.",
          });
          return;
        }
        await requestPermissionsAndroid();
      }
      this.set({ status: "idle", error: null });
    } catch (e) {
      const msg = String(e);
      const isInstallIssue = msg.includes("not installed") || msg.includes("not found") || msg.includes("ActivityNotFoundException");
      this.set({
        status: isInstallIssue ? "unavailable" : "unauthorized",
        error: isInstallIssue
          ? "Google Health Connect is not installed. Install it from the Play Store to sync health data."
          : `Health permissions denied: ${msg}`,
      });
    }
  }

  /** Pull latest health data and push to ML backend + Supabase. */
  async syncNow(): Promise<void> {
    const os = getOS();
    if (os === "web") {
      // On web, still try loading cached data
      if (!this.state.latestPayload) {
        try {
          const saved = sbGetSetting("ndw_health_payload");
          if (saved) this.set({ status: "ok", latestPayload: JSON.parse(saved), error: null });
        } catch { /* ok */ }
      }
      return;
    }
    if (this.state.status === "unavailable" || this.state.status === "unauthorized") {
      // Health Connect unavailable — still load cached data so UI isn't empty
      if (!this.state.latestPayload) {
        try {
          const saved = sbGetSetting("ndw_health_payload");
          if (saved) this.set({ status: "ok", latestPayload: JSON.parse(saved), error: null });
        } catch { /* ok */ }
      }
      return;
    }

    this.set({ status: "syncing" });
    const userId = getParticipantId();

    try {
      let result: PullResult;
      if (os === "ios") {
        result = await pullAppleHealth(userId);
      } else {
        result = await pullAndroidHealth(userId);
      }

      const { payload, workouts } = result;

      // Set latestPayload IMMEDIATELY so UI has data even if backends fail
      this.set({
        status: "ok",
        lastSyncAt: new Date(),
        latestPayload: payload,
        error: null,
      });

      // Also save to localStorage so data persists across page navigations
      try { sbSaveGeneric("ndw_health_payload", payload); } catch { /* ok */ }

      // Post to ML backend (best-effort — not available on native APK)
      try { await postToBackend(payload); } catch { /* ML backend unavailable — ok */ }

      // Post to Supabase health pipeline (best-effort)
      const source = os === "ios" ? "apple_health" as const : "google_fit" as const;
      const samples = buildSupabaseSamples(payload, workouts, source);
      try { await postToSupabase(userId, samples); } catch { /* Supabase unavailable — ok */ }

      // Persist body metrics to body_metrics table (best-effort)
      if (payload.weight_kg || payload.body_fat_pct) {
        try {
          await apiRequest("POST", "/api/body-metrics", {
            weightKg: payload.weight_kg ?? null,
            bodyFatPct: payload.body_fat_pct ?? null,
            heightCm: payload.height_cm ?? null,
            source,
            recordedAt: new Date().toISOString(),
          });
        } catch (e) {
          console.warn("[health-sync] body-metrics persist failed:", e);
        }
      }

      // Persist workouts to workouts table (best-effort)
      for (const w of workouts) {
        try {
          await apiRequest("POST", "/api/workouts", {
            name: w.workoutType || "Workout",
            workoutType: w.workoutType || "mixed",
            startedAt: w.startDate,
            endedAt: w.endDate,
            durationMin: w.durationMinutes,
            caloriesBurned: w.caloriesBurned,
            avgHr: w.averageHeartRate ?? null,
            source,
          });
        } catch (e) {
          console.warn("[health-sync] workout persist failed:", e);
        }
      }

      // latestPayload already set above — just update status if backends succeeded
    } catch (e) {
      // Even on error, try to preserve any previously pulled data
      const cached = this.state.latestPayload;
      if (!cached) {
        // No data at all — try loading from localStorage
        try {
          const saved = sbGetSetting("ndw_health_payload");
          if (saved) {
            this.set({ status: "ok", latestPayload: JSON.parse(saved), error: null });
            return;
          }
        } catch { /* ok */ }
      }
      this.set({
        status: cached ? "ok" : "error",
        error: cached ? null : String(e),
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

// ── Sync Summary Utilities ──────────────────────────────────────────────────

export interface SyncSummary {
  heartRateReadings: number;
  restingHr: number | null;
  steps: number;
  sleepHours: number | null;
  activeCalories: number;
  hasData: boolean;
}

/**
 * Build a human-readable summary of what data is present in a BiometricPayload.
 * Used by the Health page to show "Last synced: X steps, Y bpm resting HR" etc.
 */
export function buildSyncSummary(payload: BiometricPayload | null): SyncSummary {
  if (!payload) {
    return {
      heartRateReadings: 0,
      restingHr: null,
      steps: 0,
      sleepHours: null,
      activeCalories: 0,
      hasData: false,
    };
  }

  const hr = payload.current_heart_rate;
  const rhr = payload.resting_heart_rate;
  const steps = payload.steps_today ?? 0;
  const sleep = payload.sleep_total_hours ?? null;
  const cal = payload.active_energy_kcal ?? 0;

  const heartRateReadings = (hr != null && hr > 0) ? 1 : 0;
  const restingHr = (rhr != null && rhr > 0) ? rhr : null;

  const hasData =
    heartRateReadings > 0 ||
    restingHr !== null ||
    steps > 0 ||
    sleep !== null ||
    cal > 0;

  return {
    heartRateReadings,
    restingHr: restingHr !== null ? Math.round(restingHr) : null,
    steps,
    sleepHours: sleep,
    activeCalories: Math.round(cal),
    hasData,
  };
}

/**
 * Format a SyncSummary into a single-line human-readable string.
 * e.g. "5,234 steps, 62 bpm resting HR, 7.5h sleep"
 */
export function formatSyncSummary(summary: SyncSummary): string {
  if (!summary.hasData) return "No data synced";

  const parts: string[] = [];

  if (summary.steps > 0) {
    parts.push(`${summary.steps.toLocaleString()} steps`);
  }
  if (summary.restingHr !== null) {
    parts.push(`${summary.restingHr} bpm resting HR`);
  }
  if (summary.heartRateReadings > 0 && summary.restingHr === null) {
    parts.push("HR reading");
  }
  if (summary.sleepHours !== null) {
    parts.push(`${summary.sleepHours.toFixed(1)}h sleep`);
  }
  if (summary.activeCalories > 0) {
    parts.push(`${summary.activeCalories} kcal`);
  }

  return parts.join(", ");
}

/**
 * Return guidance text for users when health data is empty.
 * Platform-specific instructions for connecting health data sources.
 */
export function getEmptyDataGuidance(platform: "ios" | "android" | "web"): string {
  if (platform === "android") {
    return "No data from Health Connect \u2014 open the Withings app and enable Health Connect sync in the Share tab. Then return here and tap Sync Now.";
  }
  if (platform === "ios") {
    return "No data from Apple Health \u2014 open the Health app, go to Sharing, and ensure AntarAI has permission to read your health data.";
  }
  return "Health sync is only available on mobile. Open this app on your phone to sync health data.";
}

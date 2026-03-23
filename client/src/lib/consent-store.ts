/**
 * Biometric consent state management.
 *
 * GDPR / HIPAA compliance: per-modality consent toggles for biometric data.
 * All modalities default to OFF — no pre-checked boxes.
 * Consent records include timestamps for audit trail.
 *
 * Storage: localStorage (instant) + Supabase (persistent, fire-and-forget).
 */

import { getSupabase } from "./supabase-browser";

// ── Types ────────────────────────────────────────────────────────────────

export type ConsentModality = "eeg" | "voice" | "health" | "nutrition" | "location";

export interface BiometricConsentState {
  eeg: boolean;
  voice: boolean;
  health: boolean;
  nutrition: boolean;
  location: boolean;
}

export interface ConsentModalityInfo {
  id: ConsentModality;
  label: string;
  description: string;
}

// ── Constants ────────────────────────────────────────────────────────────

const STORAGE_KEY = "ndw_biometric_consent";

/**
 * All modalities default to OFF for new users.
 * No pre-checked boxes — explicit opt-in required.
 */
export const DEFAULT_CONSENT_STATE: BiometricConsentState = {
  eeg: false,
  voice: false,
  health: false,
  nutrition: false,
  location: false,
};

/** Modality definitions with human-readable labels and descriptions. */
export const CONSENT_MODALITIES: ConsentModalityInfo[] = [
  {
    id: "eeg",
    label: "EEG Brain Data",
    description: "Brainwave recordings from Muse or compatible BCI headsets",
  },
  {
    id: "voice",
    label: "Voice Analysis",
    description: "Voice recordings processed for emotion and stress analysis",
  },
  {
    id: "health",
    label: "Health Data Sync",
    description: "Heart rate, HRV, sleep, and activity from connected wearables",
  },
  {
    id: "nutrition",
    label: "Nutrition Tracking",
    description: "Meal logs, supplements, and food-emotion correlations",
  },
  {
    id: "location",
    label: "Location",
    description: "Location data for environmental context (if enabled)",
  },
];

// ── Persistence (localStorage) ───────────────────────────────────────────

export function getConsentState(): BiometricConsentState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULT_CONSENT_STATE };
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object" || typeof parsed.eeg !== "boolean") {
      return { ...DEFAULT_CONSENT_STATE };
    }
    return {
      eeg: parsed.eeg ?? false,
      voice: parsed.voice ?? false,
      health: parsed.health ?? false,
      nutrition: parsed.nutrition ?? false,
      location: parsed.location ?? false,
    };
  } catch {
    return { ...DEFAULT_CONSENT_STATE };
  }
}

export function saveConsentState(state: BiometricConsentState): void {
  try {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        ...state,
        updated_at: Date.now(),
      })
    );
  } catch {
    // localStorage full or unavailable
  }

  // Persist to Supabase (fire-and-forget)
  syncConsentToSupabase(state).catch(() => {});
}

// ── Supabase sync ────────────────────────────────────────────────────────

async function syncConsentToSupabase(state: BiometricConsentState): Promise<void> {
  // Block Supabase sync when Privacy Mode is active (Issue #493)
  try {
    if (localStorage.getItem("ndw_privacy_mode") === "true") return;
  } catch { /* proceed if localStorage unavailable */ }

  const sb = await getSupabase();
  if (!sb) return;

  try {
    // Upsert a single consent_records row per user
    await sb.from("consent_records").upsert(
      {
        user_id: "local",
        eeg_consent: state.eeg,
        voice_consent: state.voice,
        health_consent: state.health,
        nutrition_consent: state.nutrition,
        location_consent: state.location,
        updated_at: new Date().toISOString(),
      },
      { onConflict: "user_id" }
    );
  } catch (err) {
    console.warn("[consent-store] Supabase sync failed:", err);
  }
}

// ── Consent check helper ─────────────────────────────────────────────────

/**
 * Check if consent is granted for a specific modality.
 * Call this before collecting any biometric data.
 */
export function isConsentGranted(modality: ConsentModality): boolean {
  const state = getConsentState();
  return state[modality] === true;
}

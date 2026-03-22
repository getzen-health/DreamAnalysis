/**
 * Tests for the useConsent hook and underlying consent storage.
 *
 * Requirements:
 * - All modalities default to OFF for new users (no pre-checked boxes)
 * - Consent records include timestamps
 * - Consent state persists via localStorage + Supabase
 * - Accept All / Reject All set all toggles at once
 * - Individual modality toggles work independently
 */
import { describe, it, expect, beforeEach, vi } from "vitest";
import {
  getConsentState,
  saveConsentState,
  type BiometricConsentState,
  type ConsentModality,
  CONSENT_MODALITIES,
  DEFAULT_CONSENT_STATE,
} from "@/lib/consent-store";

// ── Mock localStorage ─────────────────────────────────────────────────────

const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: vi.fn((key: string) => store[key] ?? null),
    setItem: vi.fn((key: string, value: string) => { store[key] = value; }),
    removeItem: vi.fn((key: string) => { delete store[key]; }),
    clear: vi.fn(() => { store = {}; }),
  };
})();
Object.defineProperty(globalThis, "localStorage", { value: localStorageMock });

beforeEach(() => {
  localStorageMock.clear();
  vi.clearAllMocks();
});

// ── Modality definitions ─────────────────────────────────────────────────

describe("CONSENT_MODALITIES", () => {
  it("includes eeg, voice, health, nutrition, and location", () => {
    const ids = CONSENT_MODALITIES.map((m) => m.id);
    expect(ids).toContain("eeg");
    expect(ids).toContain("voice");
    expect(ids).toContain("health");
    expect(ids).toContain("nutrition");
    expect(ids).toContain("location");
  });

  it("each modality has id, label, and description", () => {
    for (const m of CONSENT_MODALITIES) {
      expect(typeof m.id).toBe("string");
      expect(typeof m.label).toBe("string");
      expect(typeof m.description).toBe("string");
      expect(m.label.length).toBeGreaterThan(0);
    }
  });
});

// ── Default state ────────────────────────────────────────────────────────

describe("DEFAULT_CONSENT_STATE", () => {
  it("all modalities are OFF by default (no pre-checked boxes)", () => {
    expect(DEFAULT_CONSENT_STATE.eeg).toBe(false);
    expect(DEFAULT_CONSENT_STATE.voice).toBe(false);
    expect(DEFAULT_CONSENT_STATE.health).toBe(false);
    expect(DEFAULT_CONSENT_STATE.nutrition).toBe(false);
    expect(DEFAULT_CONSENT_STATE.location).toBe(false);
  });
});

// ── Persistence ──────────────────────────────────────────────────────────

describe("getConsentState / saveConsentState", () => {
  it("returns default state when nothing is stored", () => {
    const state = getConsentState();
    expect(state).toEqual(DEFAULT_CONSENT_STATE);
  });

  it("roundtrips save and load correctly", () => {
    const custom: BiometricConsentState = {
      eeg: true,
      voice: false,
      health: true,
      nutrition: false,
      location: false,
    };
    saveConsentState(custom);
    const loaded = getConsentState();
    expect(loaded).toEqual(custom);
  });

  it("returns default state for corrupted localStorage", () => {
    localStorage.setItem("ndw_biometric_consent", "not-valid-json{{{");
    const state = getConsentState();
    expect(state).toEqual(DEFAULT_CONSENT_STATE);
  });

  it("returns default state for non-object localStorage", () => {
    localStorage.setItem("ndw_biometric_consent", '"just a string"');
    const state = getConsentState();
    expect(state).toEqual(DEFAULT_CONSENT_STATE);
  });

  it("stores updated_at timestamp on save", () => {
    const before = Date.now();
    saveConsentState({ ...DEFAULT_CONSENT_STATE, eeg: true });
    const raw = localStorage.getItem("ndw_biometric_consent");
    const parsed = JSON.parse(raw!);
    expect(parsed.updated_at).toBeGreaterThanOrEqual(before);
    expect(parsed.updated_at).toBeLessThanOrEqual(Date.now());
  });
});

// ── Accept All / Reject All helpers ──────────────────────────────────────

describe("Accept All / Reject All logic", () => {
  it("accept all sets every modality to true", () => {
    const allOn: BiometricConsentState = {
      eeg: true,
      voice: true,
      health: true,
      nutrition: true,
      location: true,
    };
    saveConsentState(allOn);
    const loaded = getConsentState();
    for (const key of Object.keys(loaded) as ConsentModality[]) {
      expect(loaded[key]).toBe(true);
    }
  });

  it("reject all sets every modality to false", () => {
    // First enable everything
    saveConsentState({
      eeg: true, voice: true, health: true, nutrition: true, location: true,
    });
    // Then reject all
    saveConsentState(DEFAULT_CONSENT_STATE);
    const loaded = getConsentState();
    for (const key of Object.keys(loaded) as ConsentModality[]) {
      expect(loaded[key]).toBe(false);
    }
  });
});

// ── isConsentGranted helper ──────────────────────────────────────────────

describe("consent check before data collection", () => {
  it("getConsentState returns false for eeg when not granted", () => {
    const state = getConsentState();
    expect(state.eeg).toBe(false);
  });

  it("getConsentState returns true for eeg after explicit consent", () => {
    saveConsentState({ ...DEFAULT_CONSENT_STATE, eeg: true });
    const state = getConsentState();
    expect(state.eeg).toBe(true);
  });

  it("individual toggles do not affect other modalities", () => {
    saveConsentState({ ...DEFAULT_CONSENT_STATE, voice: true });
    const state = getConsentState();
    expect(state.voice).toBe(true);
    expect(state.eeg).toBe(false);
    expect(state.health).toBe(false);
    expect(state.nutrition).toBe(false);
    expect(state.location).toBe(false);
  });
});

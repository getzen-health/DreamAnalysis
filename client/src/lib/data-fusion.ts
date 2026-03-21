/**
 * data-fusion.ts — Shared data layer that fuses EEG, voice, and health signals
 * into a unified state that any page can subscribe to.
 *
 * Sources:
 *   - EEG: real-time frames from use-device hook (via "ndw-eeg-updated" events + localStorage)
 *   - Voice: latest voice analysis (via "ndw-voice-updated" events + localStorage)
 *   - Health: health sync data (via "ndw-health-updated" events + localStorage)
 *
 * When EEG is streaming, it dominates (highest weight, real-time).
 * When only voice or health is available, those are used.
 * When multiple sources are present, confidence-weighted fusion is applied.
 */

import { saveEmotionHistory as sbSaveEmotionHistory } from "./supabase-store";

// ── Types ──────────────────────────────────────────────────────────────────

export type FusionSource = "eeg" | "voice" | "health" | "fused";

export interface FusedState {
  stress: number;       // 0-1
  focus: number;        // 0-1
  mood: number;         // 0-1 (mapped from valence)
  valence: number;      // -1 to 1
  arousal: number;      // 0-1
  emotion: string;      // primary emotion label
  source: FusionSource; // which source(s) contributed
  confidence: number;   // 0-1
  timestamp: number;    // epoch ms
}

export type FusionListener = (state: FusedState) => void;

// ── Source data readers ────────────────────────────────────────────────────

interface SourceReading {
  stress: number;
  focus: number;
  valence: number;
  arousal: number;
  emotion: string;
  confidence: number;
  timestamp: number;
}

function readEEGSource(): SourceReading | null {
  try {
    const raw = localStorage.getItem("ndw_last_eeg_emotion");
    if (!raw) return null;
    const data = JSON.parse(raw);
    if (!data?.emotion) return null;
    return {
      stress: data.stress_index ?? data.stress ?? 0.5,
      focus: data.focus_index ?? data.focus ?? 0.5,
      valence: data.valence ?? 0,
      arousal: data.arousal ?? 0.5,
      emotion: data.emotion,
      confidence: data.confidence ?? 0.5,
      timestamp: data.timestamp ?? Date.now(),
    };
  } catch {
    return null;
  }
}

function readVoiceSource(): SourceReading | null {
  try {
    const raw = localStorage.getItem("ndw_last_emotion");
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    const data = parsed?.result ?? parsed;
    if (!data?.emotion) return null;
    return {
      stress: data.stress_index ?? data.stress ?? 0.5,
      focus: data.focus_index ?? data.focus ?? 0.5,
      valence: data.valence ?? 0,
      arousal: data.arousal ?? 0.5,
      emotion: data.emotion,
      confidence: data.confidence ?? 0.5,
      timestamp: parsed?.timestamp ?? data?.timestamp ?? Date.now(),
    };
  } catch {
    return null;
  }
}

function readHealthSource(): SourceReading | null {
  try {
    const raw = localStorage.getItem("ndw_health_emotion");
    if (!raw) return null;
    const data = JSON.parse(raw);
    if (!data) return null;
    return {
      stress: data.stress ?? data.stress_index ?? 0.5,
      focus: data.focus ?? data.focus_index ?? 0.5,
      valence: data.valence ?? 0,
      arousal: data.arousal ?? 0.5,
      emotion: data.emotion ?? "neutral",
      confidence: data.confidence ?? 0.3,
      timestamp: data.timestamp ?? Date.now(),
    };
  } catch {
    return null;
  }
}

// ── Fusion weights ─────────────────────────────────────────────────────────
// EEG gets highest weight when available (real-time physiological signal)

const SOURCE_WEIGHTS: Record<string, number> = {
  eeg: 0.50,
  voice: 0.35,
  health: 0.15,
};

// Staleness threshold: if a reading is older than this, discount it
const STALE_MS = 5 * 60 * 1000; // 5 minutes

function clip(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function fuse(sources: Array<{ source: string; reading: SourceReading }>): FusedState {
  if (sources.length === 0) {
    return {
      stress: 0, focus: 0, mood: 0.5, valence: 0, arousal: 0.5,
      emotion: "neutral", source: "fused", confidence: 0, timestamp: Date.now(),
    };
  }

  if (sources.length === 1) {
    const { source, reading } = sources[0];
    const v = clip(reading.valence, -1, 1);
    return {
      stress: clip(reading.stress, 0, 1),
      focus: clip(reading.focus, 0, 1),
      mood: clip((v + 1) / 2, 0, 1),
      valence: v,
      arousal: clip(reading.arousal, 0, 1),
      emotion: reading.emotion,
      source: source as FusionSource,
      confidence: clip(reading.confidence, 0, 1),
      timestamp: reading.timestamp,
    };
  }

  // Multiple sources: confidence-weighted fusion
  const now = Date.now();
  let totalWeight = 0;
  let fusedStress = 0;
  let fusedFocus = 0;
  let fusedValence = 0;
  let fusedArousal = 0;
  let fusedConfidence = 0;
  let bestEmotion = "neutral";
  let bestEmotionWeight = 0;

  for (const { source, reading } of sources) {
    const baseWeight = SOURCE_WEIGHTS[source] ?? 0.2;
    const age = now - reading.timestamp;
    const freshnessMultiplier = age > STALE_MS ? 0.5 : 1.0;
    const weight = baseWeight * reading.confidence * freshnessMultiplier;

    fusedStress += reading.stress * weight;
    fusedFocus += reading.focus * weight;
    fusedValence += reading.valence * weight;
    fusedArousal += reading.arousal * weight;
    fusedConfidence += reading.confidence * weight;
    totalWeight += weight;

    if (weight > bestEmotionWeight) {
      bestEmotionWeight = weight;
      bestEmotion = reading.emotion;
    }
  }

  if (totalWeight === 0) totalWeight = 1;

  const valence = clip(fusedValence / totalWeight, -1, 1);

  return {
    stress: clip(fusedStress / totalWeight, 0, 1),
    focus: clip(fusedFocus / totalWeight, 0, 1),
    mood: clip((valence + 1) / 2, 0, 1),
    valence,
    arousal: clip(fusedArousal / totalWeight, 0, 1),
    emotion: bestEmotion,
    source: "fused",
    confidence: clip(fusedConfidence / totalWeight, 0, 1),
    timestamp: now,
  };
}

// ── Singleton event bus ────────────────────────────────────────────────────

class DataFusionBus {
  private listeners: Set<FusionListener> = new Set();
  private currentState: FusedState | null = null;
  private boundHandler: (() => void) | null = null;

  /** Start listening for source updates. */
  initialize(): void {
    if (this.boundHandler) return; // already initialized

    this.boundHandler = () => this.recompute();

    // Listen for all source update events
    window.addEventListener("ndw-eeg-updated", this.boundHandler);
    window.addEventListener("ndw-voice-updated", this.boundHandler);
    window.addEventListener("ndw-emotion-update", this.boundHandler);
    window.addEventListener("ndw-health-updated", this.boundHandler);
    window.addEventListener("storage", this.boundHandler);

    // Compute initial state
    this.recompute();
  }

  /** Stop listening and reset state. */
  destroy(): void {
    if (this.boundHandler) {
      window.removeEventListener("ndw-eeg-updated", this.boundHandler);
      window.removeEventListener("ndw-voice-updated", this.boundHandler);
      window.removeEventListener("ndw-emotion-update", this.boundHandler);
      window.removeEventListener("ndw-health-updated", this.boundHandler);
      window.removeEventListener("storage", this.boundHandler);
      this.boundHandler = null;
    }
    this.currentState = null;
    this.listeners.clear();
  }

  /** Subscribe to fused state updates. Returns unsubscribe function. */
  subscribe(listener: FusionListener): () => void {
    this.listeners.add(listener);
    // Send current state immediately if available
    if (this.currentState) {
      listener(this.currentState);
    }
    return () => { this.listeners.delete(listener); };
  }

  /** Get current fused state without subscribing. */
  getState(): FusedState | null {
    return this.currentState;
  }

  /** Force recompute from all sources. */
  recompute(): void {
    const sources: Array<{ source: string; reading: SourceReading }> = [];

    const eeg = readEEGSource();
    if (eeg) sources.push({ source: "eeg", reading: eeg });

    const voice = readVoiceSource();
    if (voice) sources.push({ source: "voice", reading: voice });

    const health = readHealthSource();
    if (health) sources.push({ source: "health", reading: health });

    if (sources.length === 0) {
      this.currentState = null;
      return;
    }

    const newState = fuse(sources);
    this.currentState = newState;

    // Persist fused emotion reading to Supabase (fire-and-forget, throttled)
    sbSaveEmotionHistory("local", {
      stress: newState.stress,
      focus: newState.focus,
      mood: newState.mood,
      source: newState.source,
      dominantEmotion: newState.emotion,
    }).catch(() => {});

    this.listeners.forEach((listener) => {
      try {
        listener(newState);
      } catch {
        // Listener error should not break the bus
      }
    });
  }
}

/** Singleton data fusion bus instance. */
export const dataFusionBus = new DataFusionBus();

// ── Exported helpers for testing ───────────────────────────────────────────

export { fuse as _fuse, readEEGSource as _readEEGSource, readVoiceSource as _readVoiceSource, readHealthSource as _readHealthSource };
export type { SourceReading as _SourceReading };

/**
 * eeg-music-mapper.ts
 *
 * EEG-to-music sonification tool using Web Audio API.
 * Maps EEG band powers to musical parameters and generates
 * simple tones in real time based on current brain state.
 *
 * Mappings:
 *   alpha power  -> melody pace (note duration)
 *   beta power   -> rhythm intensity (volume envelope)
 *   theta power  -> harmony complexity (number of overtones)
 *   delta power  -> bass drone frequency
 *   valence (FAA) -> scale selection (major/minor)
 *
 * This is a sonification tool, not full music generation.
 * It creates ambient soundscapes that reflect EEG state in real time.
 *
 * References:
 *   Miranda & Brouse (2005) — Brain-Computer Music Interface
 *   Mealla et al. (2014) — Sonification of EEG signals
 */

// ── Types ──────────────────────────────────────────────────────────────────

export interface EegBandPowers {
  delta: number;   // 0.5-4 Hz power (0-1 normalized)
  theta: number;   // 4-8 Hz power
  alpha: number;   // 8-12 Hz power
  beta: number;    // 12-30 Hz power
  gamma?: number;  // 30-100 Hz power (optional, usually EMG noise on Muse)
}

export interface SonificationParams {
  /** Note duration in ms (derived from alpha) */
  noteDuration: number;
  /** Volume 0-1 (derived from beta) */
  volume: number;
  /** Number of overtones 1-6 (derived from theta) */
  overtones: number;
  /** Bass frequency in Hz (derived from delta) */
  bassFreq: number;
  /** Scale type (derived from valence) */
  scale: "major" | "minor" | "pentatonic";
  /** Base frequency for melody in Hz */
  baseFreq: number;
}

export interface SonificationState {
  isPlaying: boolean;
  currentParams: SonificationParams;
}

// ── Constants ──────────────────────────────────────────────────────────────

// Musical scales (frequencies relative to base note)
const MAJOR_INTERVALS = [0, 2, 4, 5, 7, 9, 11, 12];     // C major
const MINOR_INTERVALS = [0, 2, 3, 5, 7, 8, 10, 12];     // C minor
const PENTATONIC_INTERVALS = [0, 2, 4, 7, 9, 12];        // C pentatonic

// Note duration range (ms)
const MIN_NOTE_DURATION = 200;
const MAX_NOTE_DURATION = 1500;

// Bass drone frequency range (Hz)
const MIN_BASS_FREQ = 55;    // A1
const MAX_BASS_FREQ = 110;   // A2

// Base melody frequency (C4 = 261.63 Hz)
const BASE_MELODY_FREQ = 261.63;

// ── Mapping Functions ──────────────────────────────────────────────────────

/**
 * Map EEG band powers to sonification parameters.
 *
 * @param bands - Current EEG band power values (0-1 normalized)
 * @param valence - Current emotional valence (-1 to 1, from FAA)
 * @returns Sonification parameters for audio generation
 */
export function mapEegToMusic(
  bands: EegBandPowers,
  valence: number = 0,
): SonificationParams {
  // Alpha -> melody pace (higher alpha = slower, more relaxed pace)
  const alpha = clamp(bands.alpha, 0, 1);
  const noteDuration = MIN_NOTE_DURATION + alpha * (MAX_NOTE_DURATION - MIN_NOTE_DURATION);

  // Beta -> rhythm intensity (higher beta = louder, more active)
  const beta = clamp(bands.beta, 0, 1);
  const volume = clamp(0.1 + beta * 0.7, 0.1, 0.8);

  // Theta -> harmony complexity (higher theta = more overtones)
  const theta = clamp(bands.theta, 0, 1);
  const overtones = Math.max(1, Math.min(6, Math.round(1 + theta * 5)));

  // Delta -> bass drone frequency (higher delta = lower bass)
  const delta = clamp(bands.delta, 0, 1);
  const bassFreq = MAX_BASS_FREQ - delta * (MAX_BASS_FREQ - MIN_BASS_FREQ);

  // Valence -> scale selection
  let scale: "major" | "minor" | "pentatonic";
  if (valence > 0.2) {
    scale = "major";
  } else if (valence < -0.2) {
    scale = "minor";
  } else {
    scale = "pentatonic";  // neutral = pentatonic (neither happy nor sad)
  }

  return {
    noteDuration: Math.round(noteDuration),
    volume: round2(volume),
    overtones,
    bassFreq: round2(bassFreq),
    scale,
    baseFreq: BASE_MELODY_FREQ,
  };
}

/**
 * Convert a semitone interval to frequency.
 */
export function intervalToFreq(baseFreq: number, semitones: number): number {
  return baseFreq * Math.pow(2, semitones / 12);
}

/**
 * Get the scale intervals for a given scale type.
 */
export function getScaleIntervals(scale: "major" | "minor" | "pentatonic"): number[] {
  switch (scale) {
    case "major": return [...MAJOR_INTERVALS];
    case "minor": return [...MINOR_INTERVALS];
    case "pentatonic": return [...PENTATONIC_INTERVALS];
  }
}

/**
 * Pick a random note from the current scale.
 */
export function pickNote(
  baseFreq: number,
  scale: "major" | "minor" | "pentatonic",
): number {
  const intervals = getScaleIntervals(scale);
  const idx = Math.floor(Math.random() * intervals.length);
  return intervalToFreq(baseFreq, intervals[idx]);
}

// ── Web Audio Sonification Engine ──────────────────────────────────────────

/**
 * EEG Sonification Engine using Web Audio API.
 *
 * Creates ambient tones that reflect current EEG state.
 * Call start() to begin, update() to change parameters,
 * and stop() to end.
 */
export class EegSonifier {
  private _ctx: AudioContext | null = null;
  private _masterGain: GainNode | null = null;
  private _isPlaying = false;
  private _params: SonificationParams;
  private _noteTimer: ReturnType<typeof setTimeout> | null = null;
  private _droneOsc: OscillatorNode | null = null;

  constructor() {
    this._params = mapEegToMusic(
      { delta: 0.3, theta: 0.3, alpha: 0.5, beta: 0.3 },
      0,
    );
  }

  get isPlaying(): boolean {
    return this._isPlaying;
  }

  get currentParams(): SonificationParams {
    return { ...this._params };
  }

  /**
   * Start sonification. Creates AudioContext and begins playing.
   */
  start(): void {
    if (this._isPlaying) return;
    if (typeof window === "undefined" || !window.AudioContext) return;

    this._ctx = new AudioContext();
    this._masterGain = this._ctx.createGain();
    this._masterGain.gain.value = this._params.volume;
    this._masterGain.connect(this._ctx.destination);

    this._isPlaying = true;
    this._startDrone();
    this._scheduleNote();
  }

  /**
   * Update sonification parameters from new EEG data.
   */
  update(bands: EegBandPowers, valence: number = 0): void {
    this._params = mapEegToMusic(bands, valence);
    if (this._masterGain) {
      this._masterGain.gain.setTargetAtTime(
        this._params.volume,
        this._ctx!.currentTime,
        0.1,
      );
    }
    if (this._droneOsc) {
      this._droneOsc.frequency.setTargetAtTime(
        this._params.bassFreq,
        this._ctx!.currentTime,
        0.5,
      );
    }
  }

  /**
   * Stop all audio and clean up.
   */
  stop(): void {
    this._isPlaying = false;
    if (this._noteTimer) {
      clearTimeout(this._noteTimer);
      this._noteTimer = null;
    }
    if (this._droneOsc) {
      try { this._droneOsc.stop(); } catch { /* already stopped */ }
      this._droneOsc = null;
    }
    if (this._ctx) {
      this._ctx.close();
      this._ctx = null;
    }
    this._masterGain = null;
  }

  private _startDrone(): void {
    if (!this._ctx || !this._masterGain) return;

    this._droneOsc = this._ctx.createOscillator();
    this._droneOsc.type = "sine";
    this._droneOsc.frequency.value = this._params.bassFreq;

    const droneGain = this._ctx.createGain();
    droneGain.gain.value = 0.15;  // Subtle background drone

    this._droneOsc.connect(droneGain);
    droneGain.connect(this._masterGain);
    this._droneOsc.start();
  }

  private _scheduleNote(): void {
    if (!this._isPlaying || !this._ctx || !this._masterGain) return;

    this._playNote();
    this._noteTimer = setTimeout(
      () => this._scheduleNote(),
      this._params.noteDuration,
    );
  }

  private _playNote(): void {
    if (!this._ctx || !this._masterGain) return;

    const freq = pickNote(this._params.baseFreq, this._params.scale);
    const now = this._ctx.currentTime;
    const duration = this._params.noteDuration / 1000;

    // Create oscillator with overtones
    for (let i = 0; i < this._params.overtones; i++) {
      const osc = this._ctx.createOscillator();
      const gain = this._ctx.createGain();

      osc.type = i === 0 ? "sine" : "triangle";
      osc.frequency.value = freq * (i + 1);  // Harmonic series

      // Volume decreases for higher overtones
      const overtoneVol = this._params.volume / (i + 1) / this._params.overtones;
      gain.gain.setValueAtTime(0, now);
      gain.gain.linearRampToValueAtTime(overtoneVol, now + 0.05);
      gain.gain.exponentialRampToValueAtTime(0.001, now + duration);

      osc.connect(gain);
      gain.connect(this._masterGain);

      osc.start(now);
      osc.stop(now + duration);
    }
  }
}

// ── Helpers ────────────────────────────────────────────────────────────────

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function round2(value: number): number {
  return Math.round(value * 100) / 100;
}

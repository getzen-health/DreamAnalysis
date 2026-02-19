/**
 * Mood-based music hook — Web Audio API binaural beats + ambient drones
 * that adapt to detected emotional states.
 *
 * Soundscapes crossfade over 2.5s and require 3 consistent emotion readings
 * before switching to prevent rapid flickering.
 */

import { useState, useEffect, useRef, useCallback } from "react";

export interface Soundscape {
  name: string;
  emotion: string;
  droneFreq: number;
  binauralBeat: number;
}

const SOUNDSCAPES: Record<string, Soundscape> = {
  happy:   { name: "Bright Horizons",  emotion: "happy",   droneFreq: 261, binauralBeat: 12 },
  relaxed: { name: "Alpha Waves",      emotion: "relaxed", droneFreq: 174, binauralBeat: 10 },
  focused: { name: "Deep Focus",       emotion: "focused", droneFreq: 196, binauralBeat: 15 },
  sad:     { name: "Gentle Comfort",   emotion: "sad",     droneFreq: 146, binauralBeat: 6  },
  fearful: { name: "Grounded Calm",    emotion: "fearful", droneFreq: 130, binauralBeat: 8  },
  angry:   { name: "Soothing Breath",  emotion: "angry",   droneFreq: 164, binauralBeat: 7  },
};

const DEFAULT_SOUNDSCAPE = SOUNDSCAPES.relaxed;
const CROSSFADE_SEC = 2.5;
const DEBOUNCE_COUNT = 3;
const BINAURAL_BASE_FREQ = 200;

export interface MoodMusicState {
  isEnabled: boolean;
  isPlaying: boolean;
  isMuted: boolean;
  volume: number;
  currentSoundscape: Soundscape;
  binauralActive: boolean;
  droneActive: boolean;
  enable: () => void;
  toggle: () => void;
  toggleMute: () => void;
  setVolume: (v: number) => void;
}

export function useMoodMusic(emotion?: string, isStreaming?: boolean): MoodMusicState {
  const [isEnabled, setIsEnabled] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [volume, setVolumeState] = useState(0.35);
  const [currentSoundscape, setCurrentSoundscape] = useState<Soundscape>(DEFAULT_SOUNDSCAPE);

  // Use refs for values accessed inside audio callbacks to avoid stale closures
  const volumeRef = useRef(0.35);
  const isMutedRef = useRef(false);

  // Audio context + nodes
  const ctxRef = useRef<AudioContext | null>(null);
  const masterGainRef = useRef<GainNode | null>(null);
  const droneOscRef = useRef<OscillatorNode | null>(null);
  const droneGainRef = useRef<GainNode | null>(null);
  const leftOscRef = useRef<OscillatorNode | null>(null);
  const rightOscRef = useRef<OscillatorNode | null>(null);
  const binauralGainRef = useRef<GainNode | null>(null);

  // Emotion debounce
  const emotionCountRef = useRef<Record<string, number>>({});
  const lastAppliedEmotionRef = useRef<string>("");

  const enable = useCallback(() => {
    if (ctxRef.current) return;

    const ctx = new AudioContext();
    ctxRef.current = ctx;

    // Master gain
    const master = ctx.createGain();
    master.gain.value = volumeRef.current;
    master.connect(ctx.destination);
    masterGainRef.current = master;

    // Drone oscillator
    const droneGain = ctx.createGain();
    droneGain.gain.value = 0.25;
    droneGain.connect(master);
    droneGainRef.current = droneGain;

    const drone = ctx.createOscillator();
    drone.type = "sine";
    drone.frequency.value = DEFAULT_SOUNDSCAPE.droneFreq;
    drone.connect(droneGain);
    drone.start();
    droneOscRef.current = drone;

    // Binaural beat — stereo panned oscillators
    const binauralGain = ctx.createGain();
    binauralGain.gain.value = 0.15;
    binauralGain.connect(master);
    binauralGainRef.current = binauralGain;

    const panL = ctx.createStereoPanner();
    panL.pan.value = -1;
    panL.connect(binauralGain);

    const panR = ctx.createStereoPanner();
    panR.pan.value = 1;
    panR.connect(binauralGain);

    const leftOsc = ctx.createOscillator();
    leftOsc.type = "sine";
    leftOsc.frequency.value = BINAURAL_BASE_FREQ;
    leftOsc.connect(panL);
    leftOsc.start();
    leftOscRef.current = leftOsc;

    const rightOsc = ctx.createOscillator();
    rightOsc.type = "sine";
    rightOsc.frequency.value = BINAURAL_BASE_FREQ + DEFAULT_SOUNDSCAPE.binauralBeat;
    rightOsc.connect(panR);
    rightOsc.start();
    rightOscRef.current = rightOsc;

    // Some browsers start AudioContext in suspended state — explicitly resume
    if (ctx.state === "suspended") {
      ctx.resume();
    }

    setIsEnabled(true);
    setIsPlaying(true);
  }, []);

  const toggle = useCallback(() => {
    const ctx = ctxRef.current;
    if (!ctx) return;
    if (ctx.state === "running") {
      ctx.suspend();
      setIsPlaying(false);
    } else {
      ctx.resume();
      setIsPlaying(true);
    }
  }, []);

  const toggleMute = useCallback(() => {
    const next = !isMutedRef.current;
    isMutedRef.current = next;
    setIsMuted(next);
    if (masterGainRef.current) {
      masterGainRef.current.gain.value = next ? 0 : volumeRef.current;
    }
  }, []);

  const setVolume = useCallback((v: number) => {
    volumeRef.current = v;
    setVolumeState(v);
    if (masterGainRef.current && !isMutedRef.current) {
      masterGainRef.current.gain.value = v;
    }
  }, []);

  // Crossfade to new soundscape
  const applySoundscape = useCallback((scape: Soundscape) => {
    const ctx = ctxRef.current;
    if (!ctx) return;
    const now = ctx.currentTime;

    if (droneOscRef.current) {
      droneOscRef.current.frequency.cancelScheduledValues(now);
      droneOscRef.current.frequency.setValueAtTime(droneOscRef.current.frequency.value, now);
      droneOscRef.current.frequency.linearRampToValueAtTime(scape.droneFreq, now + CROSSFADE_SEC);
    }

    if (leftOscRef.current) {
      leftOscRef.current.frequency.cancelScheduledValues(now);
      leftOscRef.current.frequency.setValueAtTime(leftOscRef.current.frequency.value, now);
      leftOscRef.current.frequency.linearRampToValueAtTime(BINAURAL_BASE_FREQ, now + CROSSFADE_SEC);
    }
    if (rightOscRef.current) {
      rightOscRef.current.frequency.cancelScheduledValues(now);
      rightOscRef.current.frequency.setValueAtTime(rightOscRef.current.frequency.value, now);
      rightOscRef.current.frequency.linearRampToValueAtTime(BINAURAL_BASE_FREQ + scape.binauralBeat, now + CROSSFADE_SEC);
    }

    setCurrentSoundscape(scape);
  }, []);

  // React to emotion changes with debounce
  useEffect(() => {
    if (!isEnabled || !emotion || !isStreaming) return;

    const key = emotion.toLowerCase();
    const scape = SOUNDSCAPES[key];
    if (!scape || key === lastAppliedEmotionRef.current) return;

    const counts = emotionCountRef.current;
    counts[key] = (counts[key] || 0) + 1;

    for (const k of Object.keys(counts)) {
      if (k !== key) counts[k] = 0;
    }

    if (counts[key] >= DEBOUNCE_COUNT) {
      lastAppliedEmotionRef.current = key;
      counts[key] = 0;
      applySoundscape(scape);
    }
  }, [emotion, isEnabled, isStreaming, applySoundscape]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      const ctx = ctxRef.current;
      if (ctx) {
        try { droneOscRef.current?.stop(); } catch { /* already stopped */ }
        try { leftOscRef.current?.stop(); } catch { /* already stopped */ }
        try { rightOscRef.current?.stop(); } catch { /* already stopped */ }
        ctx.close();
        ctxRef.current = null;
      }
    };
  }, []);

  return {
    isEnabled,
    isPlaying,
    isMuted,
    volume,
    currentSoundscape,
    binauralActive: isPlaying && !isMuted,
    droneActive: isPlaying && !isMuted,
    enable,
    toggle,
    toggleMute,
    setVolume,
  };
}

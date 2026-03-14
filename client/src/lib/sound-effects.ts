/**
 * sound-effects.ts — lightweight Web Audio API sound effects.
 *
 * No audio files needed — generates tones programmatically.
 * Silently no-ops if AudioContext is unavailable (e.g. SSR, old browsers).
 *
 * Usage:
 *   import { playStartBeep, playSuccessChime } from "@/lib/sound-effects";
 *   playStartBeep();     // recording start
 *   playSuccessChime();  // analysis complete / onboarding done
 */

let ctx: AudioContext | null = null;

function getContext(): AudioContext {
  if (!ctx) ctx = new AudioContext();
  return ctx;
}

/** Short upward beep — recording start */
export function playStartBeep() {
  try {
    const c = getContext();
    const osc = c.createOscillator();
    const gain = c.createGain();
    osc.connect(gain);
    gain.connect(c.destination);
    osc.frequency.setValueAtTime(600, c.currentTime);
    osc.frequency.linearRampToValueAtTime(900, c.currentTime + 0.15);
    gain.gain.setValueAtTime(0.15, c.currentTime);
    gain.gain.linearRampToValueAtTime(0, c.currentTime + 0.15);
    osc.start(c.currentTime);
    osc.stop(c.currentTime + 0.15);
  } catch {}
}

/** Two-tone success chime — analysis complete */
export function playSuccessChime() {
  try {
    const c = getContext();
    [523, 659].forEach((freq, i) => {
      const osc = c.createOscillator();
      const gain = c.createGain();
      osc.connect(gain);
      gain.connect(c.destination);
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0.12, c.currentTime + i * 0.12);
      gain.gain.linearRampToValueAtTime(0, c.currentTime + i * 0.12 + 0.2);
      osc.start(c.currentTime + i * 0.12);
      osc.stop(c.currentTime + i * 0.12 + 0.2);
    });
  } catch {}
}

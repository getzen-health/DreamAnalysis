/**
 * gamma-entrainment.ts — 40 Hz gamma binaural beat generator
 *
 * Generates a 40 Hz binaural beat using the Web Audio API:
 *   Left ear: 400 Hz tone
 *   Right ear: 440 Hz tone
 *   Perceived difference: 40 Hz (gamma band)
 *
 * Research:
 *   - 40 Hz gamma entrainment improves attention in ADHD adults
 *     (Jirakittayakorn & Wongsawat, 2017)
 *   - 40 Hz auditory stimulation reduces amyloid plaques in Alzheimer's
 *     mouse models (Iaccarino et al., Nature 2016)
 *   - Gamma oscillations (30-50 Hz) are associated with higher cognitive
 *     processing, attention, and memory binding (Herrmann et al., 2010)
 *
 * Future: overlay nature sounds (rain, ocean waves) for a more immersive
 * experience. Would require loading audio buffers and mixing into the
 * gain node chain.
 */

// ── Types ──────────────────────────────────────────────────────────────────

export interface GammaEntrainmentHandle {
  /** Start playback. Resumes AudioContext if suspended. */
  start: () => void;
  /** Stop playback and disconnect all nodes. */
  stop: () => void;
  /** Set volume (0.0 to 1.0). */
  setVolume: (volume: number) => void;
  /** Whether playback has been started. */
  isPlaying: () => boolean;
}

// ── Constants ──────────────────────────────────────────────────────────────

/** Left ear carrier frequency (Hz). */
export const LEFT_FREQ = 400;

/** Right ear carrier frequency (Hz). */
export const RIGHT_FREQ = 440;

/** Target binaural beat frequency: RIGHT_FREQ - LEFT_FREQ = 40 Hz (gamma). */
export const BEAT_FREQ = RIGHT_FREQ - LEFT_FREQ;

/** Default volume (0-1). Gentle to avoid discomfort with headphones. */
export const DEFAULT_VOLUME = 0.15;

// ── Main ───────────────────────────────────────────────────────────────────

/**
 * Create a 40 Hz gamma entrainment binaural beat generator.
 *
 * Requires headphones — the binaural effect only works when each ear
 * receives its own tone via stereo separation.
 *
 * @param audioContext An AudioContext to use. Caller is responsible for
 *   creating and managing the context lifecycle.
 * @param initialVolume Optional initial volume (0.0-1.0). Defaults to 0.15.
 * @returns A handle with start/stop/setVolume controls.
 */
export function createGammaEntrainment(
  audioContext: AudioContext,
  initialVolume: number = DEFAULT_VOLUME,
): GammaEntrainmentHandle {
  const ctx = audioContext;

  // Left ear oscillator: 400 Hz
  const oscLeft = ctx.createOscillator();
  oscLeft.type = "sine";
  oscLeft.frequency.value = LEFT_FREQ;

  // Right ear oscillator: 440 Hz
  const oscRight = ctx.createOscillator();
  oscRight.type = "sine";
  oscRight.frequency.value = RIGHT_FREQ;

  // Stereo panners for L/R isolation
  const panLeft = ctx.createStereoPanner();
  panLeft.pan.value = -1;

  const panRight = ctx.createStereoPanner();
  panRight.pan.value = 1;

  // Master gain for volume control
  const gain = ctx.createGain();
  gain.gain.value = Math.max(0, Math.min(1, initialVolume));

  // Wire: osc -> panner -> gain -> destination
  oscLeft.connect(panLeft);
  panLeft.connect(gain);

  oscRight.connect(panRight);
  panRight.connect(gain);

  gain.connect(ctx.destination);

  let started = false;
  let stopped = false;

  return {
    start() {
      if (started || stopped) return;
      started = true;
      if (ctx.state === "suspended") {
        ctx.resume();
      }
      oscLeft.start();
      oscRight.start();
    },

    stop() {
      if (stopped) return;
      stopped = true;
      started = false;
      try { oscLeft.stop(); } catch { /* already stopped */ }
      try { oscRight.stop(); } catch { /* already stopped */ }
      try { oscLeft.disconnect(); } catch { /* noop */ }
      try { oscRight.disconnect(); } catch { /* noop */ }
      try { panLeft.disconnect(); } catch { /* noop */ }
      try { panRight.disconnect(); } catch { /* noop */ }
      try { gain.disconnect(); } catch { /* noop */ }
    },

    setVolume(volume: number) {
      gain.gain.value = Math.max(0, Math.min(1, volume));
    },

    isPlaying() {
      return started && !stopped;
    },
  };
}

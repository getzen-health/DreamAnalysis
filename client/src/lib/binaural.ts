/**
 * binaural.ts
 *
 * Simple binaural beat generator using the Web Audio API.
 *
 * Creates two sine-wave oscillators: one in the left ear at baseFreq,
 * one in the right ear at baseFreq + beatFreq. The brain perceives
 * the difference frequency as a binaural beat.
 *
 * Research:
 * - 10 Hz (alpha): promotes relaxation (Klimesch, 1999)
 * - 40 Hz (gamma): improves attention in ADHD adults (Jirakittayakorn & Wongsawat, 2017)
 * - 6 Hz (theta): deepens meditation (Lagopoulos et al., 2009)
 * - 3 Hz (delta): aids sleep onset (Marshall et al., 2006)
 *
 * Beat frequency must be between 1 and 40 Hz (safety constraint).
 */

export interface BinauralHandle {
  start: () => void;
  stop: () => void;
}

/**
 * Create a binaural beat generator.
 *
 * @param baseFreq   Base frequency in Hz (e.g., 200 Hz)
 * @param beatFreq   Beat frequency in Hz (1-40 Hz). The brain perceives this as the binaural beat.
 * @param audioContext  An AudioContext instance to use
 * @returns  An object with start() and stop() methods
 * @throws   If beatFreq is outside the 1-40 Hz range
 */
export function createBinauralBeat(
  baseFreq: number,
  beatFreq: number,
  audioContext: AudioContext,
): BinauralHandle {
  if (beatFreq < 1 || beatFreq > 40) {
    throw new RangeError(
      `Beat frequency must be between 1 and 40 Hz, got ${beatFreq} Hz`
    );
  }

  const ctx = audioContext;

  // Left ear oscillator: baseFreq
  const oscLeft = ctx.createOscillator();
  oscLeft.type = "sine";
  oscLeft.frequency.value = baseFreq;

  // Right ear oscillator: baseFreq + beatFreq
  const oscRight = ctx.createOscillator();
  oscRight.type = "sine";
  oscRight.frequency.value = baseFreq + beatFreq;

  // Stereo panners to separate L/R
  const panLeft = ctx.createStereoPanner();
  panLeft.pan.value = -1;

  const panRight = ctx.createStereoPanner();
  panRight.pan.value = 1;

  // Gain node for volume control
  const gain = ctx.createGain();
  gain.gain.value = 0.15; // gentle volume

  // Wire: osc -> panner -> gain -> destination
  oscLeft.connect(panLeft);
  panLeft.connect(gain);

  oscRight.connect(panRight);
  panRight.connect(gain);

  gain.connect(ctx.destination);

  let started = false;

  return {
    start() {
      if (started) return;
      started = true;
      if (ctx.state === "suspended") {
        ctx.resume();
      }
      oscLeft.start();
      oscRight.start();
    },
    stop() {
      try { oscLeft.stop(); } catch { /* already stopped */ }
      try { oscRight.stop(); } catch { /* already stopped */ }
      try { oscLeft.disconnect(); } catch { /* noop */ }
      try { oscRight.disconnect(); } catch { /* noop */ }
      try { gain.disconnect(); } catch { /* noop */ }
    },
  };
}

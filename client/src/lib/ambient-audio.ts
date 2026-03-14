/**
 * ambient-audio.ts
 *
 * Procedural ambient sound generators using the Web Audio API.
 * Each generator creates an infinite-duration soundscape from noise buffers,
 * oscillators, and filters — no external audio files needed.
 *
 * Every generator returns { start, stop, gainNode, audioContext } so the
 * caller can fade volume via the GainNode or stop playback entirely.
 */

export type AmbientType = "ocean" | "rain" | "forest" | "night" | "stream" | "campfire";

export interface AmbientHandle {
  start(): void;
  stop(): void;
  gainNode: GainNode;
  audioContext: AudioContext;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Fill an AudioBuffer with white noise (uniform random in [-1, 1]). */
function fillWhiteNoise(buffer: AudioBuffer): void {
  for (let ch = 0; ch < buffer.numberOfChannels; ch++) {
    const data = buffer.getChannelData(ch);
    for (let i = 0; i < data.length; i++) {
      data[i] = Math.random() * 2 - 1;
    }
  }
}

/** Fill an AudioBuffer with pink noise (1/f spectrum). */
function fillPinkNoise(buffer: AudioBuffer): void {
  for (let ch = 0; ch < buffer.numberOfChannels; ch++) {
    const data = buffer.getChannelData(ch);
    // Voss-McCartney algorithm — fast 1/f approximation
    let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
    for (let i = 0; i < data.length; i++) {
      const white = Math.random() * 2 - 1;
      b0 = 0.99886 * b0 + white * 0.0555179;
      b1 = 0.99332 * b1 + white * 0.0750759;
      b2 = 0.96900 * b2 + white * 0.1538520;
      b3 = 0.86650 * b3 + white * 0.3104856;
      b4 = 0.55000 * b4 + white * 0.5329522;
      b5 = -0.7616 * b5 - white * 0.0168980;
      const pink = b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362;
      b6 = white * 0.115926;
      data[i] = pink * 0.11; // normalize amplitude
    }
  }
}

/** Fill an AudioBuffer with brown noise (1/f^2 spectrum). */
function fillBrownNoise(buffer: AudioBuffer): void {
  for (let ch = 0; ch < buffer.numberOfChannels; ch++) {
    const data = buffer.getChannelData(ch);
    let last = 0;
    for (let i = 0; i < data.length; i++) {
      const white = Math.random() * 2 - 1;
      last = (last + 0.02 * white) / 1.02;
      data[i] = last * 3.5; // normalize
    }
  }
}

/**
 * Create a looping noise source from a pre-filled buffer.
 * The buffer is 4 seconds long which, when looped, sounds continuous.
 */
function createNoiseSource(
  ctx: AudioContext,
  fill: (buf: AudioBuffer) => void,
): AudioBufferSourceNode {
  const bufferLength = ctx.sampleRate * 4; // 4 seconds
  const buffer = ctx.createBuffer(2, bufferLength, ctx.sampleRate);
  fill(buffer);

  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.loop = true;
  return source;
}

// ─── Generators ───────────────────────────────────────────────────────────────

/**
 * Ocean Waves — low-pass filtered white noise with slow amplitude modulation
 * (0.1 Hz sine) to simulate wave rhythm.
 */
function createOcean(ctx: AudioContext, master: GainNode): { start(): void; stop(): void } {
  const source = createNoiseSource(ctx, fillWhiteNoise);

  // Low-pass to remove harsh high frequencies — ocean is deep and rumbling
  const lpf = ctx.createBiquadFilter();
  lpf.type = "lowpass";
  lpf.frequency.value = 400;
  lpf.Q.value = 0.7;

  // Slow amplitude modulation — wave rhythm at ~0.1 Hz
  const lfo = ctx.createOscillator();
  lfo.type = "sine";
  lfo.frequency.value = 0.1;

  const lfoGain = ctx.createGain();
  lfoGain.gain.value = 0.3; // modulation depth

  const ampGain = ctx.createGain();
  ampGain.gain.value = 0.6;

  // LFO modulates the amplitude gain
  lfo.connect(lfoGain);
  lfoGain.connect(ampGain.gain);

  source.connect(lpf);
  lpf.connect(ampGain);
  ampGain.connect(master);

  return {
    start() {
      source.start();
      lfo.start();
    },
    stop() {
      try { source.stop(); } catch { /* already stopped */ }
      try { lfo.stop(); } catch { /* already stopped */ }
    },
  };
}

/**
 * Rain — band-pass filtered noise (200-4000 Hz) with random droplet impulses.
 */
function createRain(ctx: AudioContext, master: GainNode): { start(): void; stop(): void } {
  // Steady rain backdrop
  const source = createNoiseSource(ctx, fillWhiteNoise);

  const bpf = ctx.createBiquadFilter();
  bpf.type = "bandpass";
  bpf.frequency.value = 1200;
  bpf.Q.value = 0.5;

  const rainGain = ctx.createGain();
  rainGain.gain.value = 0.45;

  source.connect(bpf);
  bpf.connect(rainGain);
  rainGain.connect(master);

  // Random droplet layer — short filtered noise bursts
  const dropletSource = createNoiseSource(ctx, fillWhiteNoise);

  const dropBpf = ctx.createBiquadFilter();
  dropBpf.type = "bandpass";
  dropBpf.frequency.value = 3000;
  dropBpf.Q.value = 2;

  const dropGain = ctx.createGain();
  dropGain.gain.value = 0;

  dropletSource.connect(dropBpf);
  dropBpf.connect(dropGain);
  dropGain.connect(master);

  // Schedule random droplet bursts
  let dropletInterval: ReturnType<typeof setInterval> | null = null;

  return {
    start() {
      source.start();
      dropletSource.start();

      // Random droplets every 100-400ms
      dropletInterval = setInterval(() => {
        const now = ctx.currentTime;
        const intensity = 0.05 + Math.random() * 0.15;
        dropGain.gain.setValueAtTime(intensity, now);
        dropGain.gain.exponentialRampToValueAtTime(0.001, now + 0.03 + Math.random() * 0.05);
      }, 100 + Math.random() * 300);
    },
    stop() {
      if (dropletInterval) clearInterval(dropletInterval);
      try { source.stop(); } catch { /* already stopped */ }
      try { dropletSource.stop(); } catch { /* already stopped */ }
    },
  };
}

/**
 * Forest — gentle pink noise at low volume simulating wind through trees.
 */
function createForest(ctx: AudioContext, master: GainNode): { start(): void; stop(): void } {
  const source = createNoiseSource(ctx, fillPinkNoise);

  // Gentle low-pass — forest wind is soft
  const lpf = ctx.createBiquadFilter();
  lpf.type = "lowpass";
  lpf.frequency.value = 800;
  lpf.Q.value = 0.5;

  // Very slow modulation — wind gusts
  const lfo = ctx.createOscillator();
  lfo.type = "sine";
  lfo.frequency.value = 0.05;

  const lfoGain = ctx.createGain();
  lfoGain.gain.value = 0.15;

  const ampGain = ctx.createGain();
  ampGain.gain.value = 0.35;

  lfo.connect(lfoGain);
  lfoGain.connect(ampGain.gain);

  source.connect(lpf);
  lpf.connect(ampGain);
  ampGain.connect(master);

  return {
    start() {
      source.start();
      lfo.start();
    },
    stop() {
      try { source.stop(); } catch { /* already stopped */ }
      try { lfo.stop(); } catch { /* already stopped */ }
    },
  };
}

/**
 * Night Sky — near-silence with very faint pink noise (barely audible ambience).
 */
function createNight(ctx: AudioContext, master: GainNode): { start(): void; stop(): void } {
  const source = createNoiseSource(ctx, fillPinkNoise);

  const lpf = ctx.createBiquadFilter();
  lpf.type = "lowpass";
  lpf.frequency.value = 500;
  lpf.Q.value = 0.3;

  const ampGain = ctx.createGain();
  ampGain.gain.value = 0.12; // very quiet

  source.connect(lpf);
  lpf.connect(ampGain);
  ampGain.connect(master);

  return {
    start() {
      source.start();
    },
    stop() {
      try { source.stop(); } catch { /* already stopped */ }
    },
  };
}

/**
 * Stream — higher frequency white noise, band-passed (500-3000 Hz).
 */
function createStream(ctx: AudioContext, master: GainNode): { start(): void; stop(): void } {
  const source = createNoiseSource(ctx, fillWhiteNoise);

  const bpf = ctx.createBiquadFilter();
  bpf.type = "bandpass";
  bpf.frequency.value = 1500;
  bpf.Q.value = 0.8;

  // Gentle shimmer modulation
  const lfo = ctx.createOscillator();
  lfo.type = "sine";
  lfo.frequency.value = 0.15;

  const lfoGain = ctx.createGain();
  lfoGain.gain.value = 0.1;

  const ampGain = ctx.createGain();
  ampGain.gain.value = 0.4;

  lfo.connect(lfoGain);
  lfoGain.connect(ampGain.gain);

  source.connect(bpf);
  bpf.connect(ampGain);
  ampGain.connect(master);

  return {
    start() {
      source.start();
      lfo.start();
    },
    stop() {
      try { source.stop(); } catch { /* already stopped */ }
      try { lfo.stop(); } catch { /* already stopped */ }
    },
  };
}

/**
 * Campfire — brown noise (1/f^2) with random crackle pops.
 */
function createCampfire(ctx: AudioContext, master: GainNode): { start(): void; stop(): void } {
  // Base crackle — brown noise
  const source = createNoiseSource(ctx, fillBrownNoise);

  const lpf = ctx.createBiquadFilter();
  lpf.type = "lowpass";
  lpf.frequency.value = 600;
  lpf.Q.value = 0.5;

  const baseGain = ctx.createGain();
  baseGain.gain.value = 0.4;

  source.connect(lpf);
  lpf.connect(baseGain);
  baseGain.connect(master);

  // Pop/crackle layer — short white noise impulses
  const popSource = createNoiseSource(ctx, fillWhiteNoise);

  const popBpf = ctx.createBiquadFilter();
  popBpf.type = "highpass";
  popBpf.frequency.value = 1000;

  const popGain = ctx.createGain();
  popGain.gain.value = 0;

  popSource.connect(popBpf);
  popBpf.connect(popGain);
  popGain.connect(master);

  let popInterval: ReturnType<typeof setInterval> | null = null;

  return {
    start() {
      source.start();
      popSource.start();

      // Random pops — fire crackle
      popInterval = setInterval(() => {
        const now = ctx.currentTime;
        const intensity = 0.08 + Math.random() * 0.2;
        popGain.gain.setValueAtTime(intensity, now);
        popGain.gain.exponentialRampToValueAtTime(0.001, now + 0.01 + Math.random() * 0.03);
      }, 200 + Math.random() * 600);
    },
    stop() {
      if (popInterval) clearInterval(popInterval);
      try { source.stop(); } catch { /* already stopped */ }
      try { popSource.stop(); } catch { /* already stopped */ }
    },
  };
}

// ─── Public API ───────────────────────────────────────────────────────────────

const GENERATORS: Record<AmbientType, (ctx: AudioContext, master: GainNode) => { start(): void; stop(): void }> = {
  ocean: createOcean,
  rain: createRain,
  forest: createForest,
  night: createNight,
  stream: createStream,
  campfire: createCampfire,
};

/**
 * Create an ambient audio generator of the given type.
 *
 * Usage:
 *   const handle = createAmbientAudio("ocean");
 *   handle.start();
 *   // ... later
 *   handle.stop();
 *
 * The returned gainNode can be used for fade-out (see audio-fade.ts).
 */
export function createAmbientAudio(type: AmbientType): AmbientHandle {
  const ctx = new AudioContext();

  // Master gain node — controls overall volume and is the fade target
  const gainNode = ctx.createGain();
  gainNode.gain.value = 1.0;
  gainNode.connect(ctx.destination);

  const generator = GENERATORS[type](ctx, gainNode);

  let started = false;

  return {
    start() {
      if (started) return;
      started = true;
      // Resume context if suspended (autoplay policy)
      if (ctx.state === "suspended") {
        ctx.resume();
      }
      generator.start();
    },
    stop() {
      generator.stop();
      ctx.close();
    },
    gainNode,
    audioContext: ctx,
  };
}

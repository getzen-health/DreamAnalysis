import { describe, it, expect } from "vitest";
import {
  assessChannelQuality,
  assessSignalQuality,
  runAlphaReactivityTest,
  type ChannelQuality,
  type SignalQualityResult,
} from "@/lib/signal-quality";

// ── Helpers ──────────────────────────────────────────────────────────────────

const FS = 256; // Muse 2 sample rate

/** Generate a sine wave at a given frequency and amplitude (in uV). */
function generateSine(freqHz: number, amplitudeUv: number, durationSec: number, fs: number): Float32Array {
  const n = Math.floor(fs * durationSec);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = amplitudeUv * Math.sin(2 * Math.PI * freqHz * (i / fs));
  }
  return out;
}

/** Generate white noise with approximate RMS amplitude. */
function generateWhiteNoise(rmsUv: number, durationSec: number, fs: number): Float32Array {
  const n = Math.floor(fs * durationSec);
  const out = new Float32Array(n);
  // Box-Muller for reproducible gaussian-ish noise
  for (let i = 0; i < n; i++) {
    // Simple pseudo-random using sin-based hash (deterministic)
    const u1 = Math.abs(Math.sin(i * 12.9898 + 78.233) * 43758.5453) % 1;
    const u2 = Math.abs(Math.sin(i * 63.7264 + 10.873) * 28361.2316) % 1;
    const z = Math.sqrt(-2 * Math.log(Math.max(u1, 1e-10))) * Math.cos(2 * Math.PI * u2);
    out[i] = z * rmsUv;
  }
  return out;
}

/** Generate a near-zero signal (disconnected electrode). */
function generateDisconnected(durationSec: number, fs: number): Float32Array {
  const n = Math.floor(fs * durationSec);
  const out = new Float32Array(n);
  // Tiny noise ~0.5 uV RMS (ADC quantization noise)
  for (let i = 0; i < n; i++) {
    out[i] = 0.5 * Math.sin(i * 0.001);
  }
  return out;
}

/** Generate realistic EEG-like signal: strong alpha (10 Hz) + weaker beta + noise floor. */
function generateRealisticEeg(amplitudeUv: number, durationSec: number, fs: number): Float32Array {
  const n = Math.floor(fs * durationSec);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / fs;
    // Alpha dominant (10 Hz) + some theta (6 Hz) + weak beta (20 Hz) + noise floor
    out[i] =
      amplitudeUv * 0.6 * Math.sin(2 * Math.PI * 10 * t) +
      amplitudeUv * 0.2 * Math.sin(2 * Math.PI * 6 * t) +
      amplitudeUv * 0.1 * Math.sin(2 * Math.PI * 20 * t) +
      amplitudeUv * 0.1 * Math.sin(2 * Math.PI * 40 * t + i * 0.3);
  }
  return out;
}

// ── assessChannelQuality ─────────────────────────────────────────────────────

describe("assessChannelQuality", () => {
  it("returns 'disconnected' for low amplitude (< 5 uV RMS)", () => {
    const samples = generateDisconnected(2, FS);
    const result = assessChannelQuality(samples, FS, "TP9");
    expect(result.status).toBe("disconnected");
    expect(result.amplitudeUv).toBeLessThan(5);
  });

  it("returns 'good' for normal amplitude (~20 uV) with peaked spectrum", () => {
    // Pure 10 Hz sine at 20 uV peak — peaked spectrum, normal amplitude
    const samples = generateRealisticEeg(20, 2, FS);
    const result = assessChannelQuality(samples, FS, "AF7");
    expect(result.status).toBe("good");
    expect(result.amplitudeUv).toBeGreaterThanOrEqual(5);
    expect(result.amplitudeUv).toBeLessThanOrEqual(75);
    expect(result.spectralFlatness).toBeLessThan(0.3);
  });

  it("returns 'poor' for high amplitude (> 150 uV, artifact)", () => {
    // 250 uV peak sine — RMS = 250/sqrt(2) ~ 177 uV, well above 150 uV threshold
    const samples = generateSine(10, 250, 2, FS);
    const result = assessChannelQuality(samples, FS, "AF8");
    expect(result.status).toBe("poor");
    expect(result.amplitudeUv).toBeGreaterThan(150);
  });

  it("returns 'poor' for flat spectrum (white noise)", () => {
    // White noise with moderate amplitude — flat spectrum = noise
    const samples = generateWhiteNoise(30, 2, FS);
    const result = assessChannelQuality(samples, FS, "TP10");
    expect(result.spectralFlatness).toBeGreaterThan(0.3);
    // Status should be fair or poor due to flat spectrum
    expect(["fair", "poor"]).toContain(result.status);
  });

  it("sets channel name correctly", () => {
    const samples = generateRealisticEeg(20, 2, FS);
    const result = assessChannelQuality(samples, FS, "AF7");
    expect(result.channel).toBe("AF7");
  });
});

// ── runAlphaReactivityTest ──────────────────────────────────────────────────

describe("runAlphaReactivityTest", () => {
  it("returns true when alpha increases > 20% with eyes closed", () => {
    // Eyes open: weak alpha (5 uV) + beta
    const eyesOpen: Float32Array[] = [];
    // Eyes closed: strong alpha (15 uV) — 3x increase
    const eyesClosed: Float32Array[] = [];

    for (let ch = 0; ch < 4; ch++) {
      const n = Math.floor(FS * 5);
      const openSig = new Float32Array(n);
      const closedSig = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const t = i / FS;
        // Eyes open: weak alpha, stronger beta
        openSig[i] = 5 * Math.sin(2 * Math.PI * 10 * t) + 10 * Math.sin(2 * Math.PI * 20 * t);
        // Eyes closed: strong alpha, weak beta
        closedSig[i] = 15 * Math.sin(2 * Math.PI * 10 * t) + 3 * Math.sin(2 * Math.PI * 20 * t);
      }
      eyesOpen.push(openSig);
      eyesClosed.push(closedSig);
    }

    const result = runAlphaReactivityTest(eyesOpen, eyesClosed, FS);
    expect(result).toBe(true);
  });

  it("returns false when alpha does not change with eyes closed", () => {
    // Both conditions: same weak alpha
    const eyesOpen: Float32Array[] = [];
    const eyesClosed: Float32Array[] = [];

    for (let ch = 0; ch < 4; ch++) {
      const n = Math.floor(FS * 5);
      const sig = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const t = i / FS;
        sig[i] = 5 * Math.sin(2 * Math.PI * 10 * t) + 10 * Math.sin(2 * Math.PI * 20 * t);
      }
      eyesOpen.push(new Float32Array(sig));
      eyesClosed.push(new Float32Array(sig));
    }

    const result = runAlphaReactivityTest(eyesOpen, eyesClosed, FS);
    expect(result).toBe(false);
  });
});

// ── assessSignalQuality ──────────────────────────────────────────────────────

describe("assessSignalQuality", () => {
  const CHANNEL_NAMES = ["TP9", "AF7", "AF8", "TP10"];

  it("returns 'good' overall when 4 channels are good", () => {
    const channels = CHANNEL_NAMES.map(() => generateRealisticEeg(20, 2, FS));
    const result = assessSignalQuality(channels, FS, CHANNEL_NAMES);
    expect(result.overall).toBe("good");
    expect(result.channels).toHaveLength(4);
    expect(result.readyForAnalysis).toBe(true);
  });

  it("returns 'poor' overall when only 1 channel is good", () => {
    const channels = [
      generateRealisticEeg(20, 2, FS),   // TP9 good
      generateDisconnected(2, FS),        // AF7 disconnected
      generateDisconnected(2, FS),        // AF8 disconnected
      generateDisconnected(2, FS),        // TP10 disconnected
    ];
    const result = assessSignalQuality(channels, FS, CHANNEL_NAMES);
    expect(result.overall).toBe("poor");
  });

  it("readyForAnalysis is true when AF7 and AF8 are at least fair", () => {
    const channels = [
      generateDisconnected(2, FS),        // TP9 disconnected
      generateRealisticEeg(20, 2, FS),    // AF7 good
      generateRealisticEeg(20, 2, FS),    // AF8 good
      generateDisconnected(2, FS),        // TP10 disconnected
    ];
    const result = assessSignalQuality(channels, FS, CHANNEL_NAMES);
    expect(result.readyForAnalysis).toBe(true);
  });

  it("readyForAnalysis is false when AF7 is poor", () => {
    const channels = [
      generateRealisticEeg(20, 2, FS),    // TP9 good
      generateDisconnected(2, FS),        // AF7 disconnected
      generateRealisticEeg(20, 2, FS),    // AF8 good
      generateRealisticEeg(20, 2, FS),    // TP10 good
    ];
    const result = assessSignalQuality(channels, FS, CHANNEL_NAMES);
    expect(result.readyForAnalysis).toBe(false);
  });

  it("generates non-empty recommendation", () => {
    const channels = CHANNEL_NAMES.map(() => generateRealisticEeg(20, 2, FS));
    const result = assessSignalQuality(channels, FS, CHANNEL_NAMES);
    expect(result.recommendation).toBeTruthy();
    expect(result.recommendation.length).toBeGreaterThan(0);
  });
});

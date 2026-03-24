import { describe, it, expect } from "vitest";
import { computeAudioSNR } from "@/lib/voice-onnx";

// ─── Helpers ────────────────────────────────────────────────────────────────

/** Generate a sine wave at the given frequency and sample rate. */
function makeSineWave(sr: number, durationSec: number, freq = 440, amplitude = 1.0): Float32Array {
  const n = Math.floor(sr * durationSec);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = amplitude * Math.sin(2 * Math.PI * freq * i / sr);
  }
  return out;
}

/** Generate silence (all zeros). */
function makeSilence(sr: number, durationSec: number): Float32Array {
  return new Float32Array(Math.floor(sr * durationSec));
}

/** Generate white noise with given amplitude. */
function makeNoise(sr: number, durationSec: number, amplitude = 0.1): Float32Array {
  const n = Math.floor(sr * durationSec);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = (Math.random() * 2 - 1) * amplitude;
  }
  return out;
}

/**
 * Generate a speech-like signal: alternating voiced segments and silence,
 * with optional additive background noise.
 */
function makeSpeechWithNoise(
  sr: number,
  durationSec: number,
  speechAmplitude: number,
  noiseAmplitude: number,
): Float32Array {
  const n = Math.floor(sr * durationSec);
  const out = new Float32Array(n);
  const segmentLen = Math.floor(sr * 0.3); // 300ms segments

  for (let i = 0; i < n; i++) {
    // Background noise everywhere
    const noise = (Math.random() * 2 - 1) * noiseAmplitude;

    const segment = Math.floor(i / segmentLen);
    if (segment % 2 === 0) {
      // Voiced segment: speech + noise
      out[i] = speechAmplitude * Math.sin(2 * Math.PI * 200 * i / sr) + noise;
    } else {
      // Silence segment: noise only
      out[i] = noise;
    }
  }
  return out;
}

// ─── computeAudioSNR ───────────────────────────────────────────────────────

describe("computeAudioSNR", () => {
  it("returns a finite number", () => {
    const signal = makeSineWave(16000, 1.0);
    const snr = computeAudioSNR(signal, 16000);
    expect(Number.isFinite(snr)).toBe(true);
  });

  it("returns high SNR for clean speech (no noise)", () => {
    // Clean speech-like signal with no background noise
    const signal = makeSpeechWithNoise(16000, 2.0, 0.5, 0.0001);
    const snr = computeAudioSNR(signal, 16000);
    // Should be well above 15 dB for essentially clean signal
    expect(snr).toBeGreaterThan(20);
  });

  it("returns low SNR when noise is comparable to speech", () => {
    // Speech amplitude = noise amplitude (roughly 0 dB SNR)
    const signal = makeSpeechWithNoise(16000, 2.0, 0.1, 0.1);
    const snr = computeAudioSNR(signal, 16000);
    // With equal speech and noise energy, SNR should be low
    expect(snr).toBeLessThan(10);
  });

  it("returns 0 for pure silence", () => {
    const silence = makeSilence(16000, 1.0);
    const snr = computeAudioSNR(silence, 16000);
    expect(snr).toBe(0);
  });

  it("higher noise means lower SNR", () => {
    const cleanish = makeSpeechWithNoise(16000, 2.0, 0.5, 0.01);
    const noisy = makeSpeechWithNoise(16000, 2.0, 0.5, 0.15);

    const snrClean = computeAudioSNR(cleanish, 16000);
    const snrNoisy = computeAudioSNR(noisy, 16000);

    expect(snrClean).toBeGreaterThan(snrNoisy);
  });

  it("returns SNR in dB (non-negative)", () => {
    const signal = makeSineWave(16000, 1.0);
    const snr = computeAudioSNR(signal, 16000);
    // SNR should be non-negative (we clamp at 0)
    expect(snr).toBeGreaterThanOrEqual(0);
  });

  it("handles short audio without errors", () => {
    const short = makeSineWave(16000, 0.1);
    const snr = computeAudioSNR(short, 16000);
    expect(Number.isFinite(snr)).toBe(true);
    expect(snr).toBeGreaterThanOrEqual(0);
  });

  it("continuous tone returns high SNR (all frames are speech)", () => {
    // A continuous tone has all speech frames and no silence frames.
    // This should be treated as high SNR (no noise detected).
    const tone = makeSineWave(16000, 2.0, 440, 0.5);
    const snr = computeAudioSNR(tone, 16000);
    expect(snr).toBeGreaterThan(30);
  });
});

// ─── SNR confidence modulation ──────────────────────────────────────────────

describe("SNR confidence modulation logic", () => {
  it("confidence unchanged when SNR >= 15 dB", () => {
    const snr = 20;
    const rawConfidence = 0.8;
    const modulated = snr >= 15 ? rawConfidence : rawConfidence * (snr / 15);
    expect(modulated).toBe(0.8);
  });

  it("confidence halved when SNR = 7.5 dB", () => {
    const snr = 7.5;
    const rawConfidence = 0.8;
    const modulated = snr >= 15 ? rawConfidence : rawConfidence * (snr / 15);
    expect(modulated).toBeCloseTo(0.4, 5);
  });

  it("confidence zero when SNR = 0 dB", () => {
    const snr = 0;
    const rawConfidence = 0.8;
    const modulated = snr >= 15 ? rawConfidence : rawConfidence * (snr / 15);
    expect(modulated).toBe(0);
  });
});

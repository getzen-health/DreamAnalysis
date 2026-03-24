/**
 * Tests for extractEnergyEnvelope() in voice-onnx.ts.
 *
 * Verifies:
 *   1. Sine wave produces non-zero attack time
 *   2. Silent audio produces zero/default values
 *   3. sustainRatio is in [0, 1]
 *   4. All values are finite
 */
import { describe, it, expect } from "vitest";
import { extractEnergyEnvelope } from "@/lib/voice-onnx";

const SR = 16000;

/** Generate a sine wave at the given frequency and duration. */
function makeSine(freq: number, durationSec: number, sr: number): Float32Array {
  const n = Math.floor(sr * durationSec);
  const buf = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    buf[i] = Math.sin(2 * Math.PI * freq * i / sr);
  }
  return buf;
}

/** Generate silence. */
function makeSilence(durationSec: number, sr: number): Float32Array {
  return new Float32Array(Math.floor(sr * durationSec));
}

/** Generate audio with a clear onset: silence then sine. */
function makeOnsetSignal(
  silenceSec: number,
  toneSec: number,
  freq: number,
  sr: number,
): Float32Array {
  const silenceLen = Math.floor(sr * silenceSec);
  const toneLen = Math.floor(sr * toneSec);
  const buf = new Float32Array(silenceLen + toneLen);
  for (let i = 0; i < toneLen; i++) {
    buf[silenceLen + i] = 0.8 * Math.sin(2 * Math.PI * freq * i / sr);
  }
  return buf;
}

describe("extractEnergyEnvelope", () => {
  it("sine wave produces non-zero attack time", () => {
    // Silence (0.1s) then sine (0.5s) — attack time should be > 0
    const signal = makeOnsetSignal(0.1, 0.5, 440, SR);
    const envelope = extractEnergyEnvelope(signal, SR);
    expect(envelope.attackTimeMs).toBeGreaterThan(0);
  });

  it("silent audio produces zero/default values", () => {
    const silence = makeSilence(1.0, SR);
    const envelope = extractEnergyEnvelope(silence, SR);
    expect(envelope.attackTimeMs).toBe(0);
    expect(envelope.sustainRatio).toBe(0);
    expect(envelope.decayRateDb).toBe(0);
  });

  it("sustainRatio is in [0, 1]", () => {
    const signal = makeSine(440, 1.0, SR);
    const envelope = extractEnergyEnvelope(signal, SR);
    expect(envelope.sustainRatio).toBeGreaterThanOrEqual(0);
    expect(envelope.sustainRatio).toBeLessThanOrEqual(1);
  });

  it("all values are finite", () => {
    // Test with various signal types
    const signals = [
      makeSine(200, 0.5, SR),
      makeSilence(0.5, SR),
      makeOnsetSignal(0.2, 0.8, 300, SR),
    ];

    for (const signal of signals) {
      const envelope = extractEnergyEnvelope(signal, SR);
      expect(Number.isFinite(envelope.attackTimeMs)).toBe(true);
      expect(Number.isFinite(envelope.sustainRatio)).toBe(true);
      expect(Number.isFinite(envelope.decayRateDb)).toBe(true);
    }
  });

  it("sustained sine has high sustain ratio", () => {
    // Pure constant-amplitude sine should have sustainRatio near 1
    const signal = makeSine(440, 1.0, SR);
    const envelope = extractEnergyEnvelope(signal, SR);
    expect(envelope.sustainRatio).toBeGreaterThan(0.5);
  });

  it("returns defaults for very short audio", () => {
    // Audio shorter than one frame (25ms at 16kHz = 400 samples)
    const short = new Float32Array(100);
    short[50] = 1.0;
    const envelope = extractEnergyEnvelope(short, SR);
    expect(envelope.attackTimeMs).toBe(0);
    expect(envelope.sustainRatio).toBe(0);
    expect(envelope.decayRateDb).toBe(0);
  });
});

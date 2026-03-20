import { describe, it, expect } from "vitest";
import { resampleTo16k, prepareAudioInput } from "@/lib/voice-onnx";

describe("resampleTo16k", () => {
  it("returns same array if already 16kHz", () => {
    const input = new Float32Array([0.1, -0.2, 0.3]);
    const result = resampleTo16k(input, 16000);
    expect(result.length).toBe(3);
    expect(result[0]).toBeCloseTo(0.1);
  });

  it("downsamples 48kHz to 16kHz (3:1 ratio)", () => {
    // 48kHz signal with 48 samples = 1ms of audio
    const input = new Float32Array(48);
    for (let i = 0; i < 48; i++) input[i] = Math.sin(2 * Math.PI * i / 48);
    const result = resampleTo16k(input, 48000);
    // 48 samples at 48kHz -> 16 samples at 16kHz
    expect(result.length).toBe(16);
  });

  it("downsamples 44100Hz to 16kHz", () => {
    const input = new Float32Array(44100); // 1 second
    const result = resampleTo16k(input, 44100);
    expect(result.length).toBe(16000);
  });
});

describe("prepareAudioInput", () => {
  it("normalizes audio to [-1, 1] range", () => {
    const input = new Float32Array([0.5, -0.5, 2.0, -3.0]);
    const result = prepareAudioInput(input, 16000);
    expect(Math.max(...result)).toBeLessThanOrEqual(1.0);
    expect(Math.min(...result)).toBeGreaterThanOrEqual(-1.0);
  });

  it("resamples and normalizes in one call", () => {
    const input = new Float32Array(48000); // 1s at 48kHz
    for (let i = 0; i < input.length; i++) input[i] = Math.sin(2 * Math.PI * 440 * i / 48000) * 0.5;
    const result = prepareAudioInput(input, 48000);
    expect(result.length).toBe(16000);
    expect(Math.abs(result[0])).toBeLessThanOrEqual(1.0);
  });

  it("pads short audio to minimum 1 second", () => {
    const input = new Float32Array(8000); // 0.5s at 16kHz
    const result = prepareAudioInput(input, 16000);
    expect(result.length).toBeGreaterThanOrEqual(16000);
  });
});

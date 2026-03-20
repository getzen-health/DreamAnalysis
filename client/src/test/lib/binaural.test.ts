import { describe, it, expect, vi, beforeEach } from "vitest";
import { createBinauralBeat } from "@/lib/binaural";

// ── Mock Web Audio API ───────────────────────────────────────────────────────

function createMockAudioContext(): AudioContext {
  const mockOscillator = {
    type: "sine" as OscillatorType,
    frequency: { value: 0 },
    connect: vi.fn(),
    start: vi.fn(),
    stop: vi.fn(),
    disconnect: vi.fn(),
  };

  const mockPanner = {
    pan: { value: 0 },
    connect: vi.fn(),
  };

  const mockGain = {
    gain: { value: 1 },
    connect: vi.fn(),
  };

  return {
    createOscillator: vi.fn(() => ({ ...mockOscillator })),
    createStereoPanner: vi.fn(() => ({ ...mockPanner })),
    createGain: vi.fn(() => ({ ...mockGain })),
    destination: {},
    state: "running" as AudioContextState,
    resume: vi.fn(),
  } as unknown as AudioContext;
}

// ── Tests ────────────────────────────────────────────────────────────────────

describe("createBinauralBeat", () => {
  let mockCtx: AudioContext;

  beforeEach(() => {
    mockCtx = createMockAudioContext();
  });

  it("returns start and stop functions", () => {
    const beat = createBinauralBeat(200, 10, mockCtx);
    expect(typeof beat.start).toBe("function");
    expect(typeof beat.stop).toBe("function");
  });

  it("throws on beat frequency below 1 Hz", () => {
    expect(() => createBinauralBeat(200, 0.5, mockCtx)).toThrow();
  });

  it("throws on beat frequency above 40 Hz", () => {
    expect(() => createBinauralBeat(200, 41, mockCtx)).toThrow();
  });

  it("accepts beat frequency at boundary (1 Hz)", () => {
    const beat = createBinauralBeat(200, 1, mockCtx);
    expect(typeof beat.start).toBe("function");
  });

  it("accepts beat frequency at boundary (40 Hz)", () => {
    const beat = createBinauralBeat(200, 40, mockCtx);
    expect(typeof beat.start).toBe("function");
  });

  it("creates two oscillators for left and right ears", () => {
    createBinauralBeat(200, 10, mockCtx);
    // Should create exactly 2 oscillators (left ear, right ear)
    expect(mockCtx.createOscillator).toHaveBeenCalledTimes(2);
  });

  it("creates stereo panners to separate left and right channels", () => {
    createBinauralBeat(200, 10, mockCtx);
    expect(mockCtx.createStereoPanner).toHaveBeenCalledTimes(2);
  });
});

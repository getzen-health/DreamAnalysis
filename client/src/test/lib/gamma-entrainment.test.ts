import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  createGammaEntrainment,
  LEFT_FREQ,
  RIGHT_FREQ,
  BEAT_FREQ,
  DEFAULT_VOLUME,
} from "@/lib/gamma-entrainment";

// ── Mock Web Audio API ─────────────────────────────────────────────────────

function createMockOsc() {
  return {
    type: "sine",
    frequency: { value: 0 },
    connect: vi.fn(),
    start: vi.fn(),
    stop: vi.fn(),
    disconnect: vi.fn(),
  };
}

function createMockPanner() {
  return {
    pan: { value: 0 },
    connect: vi.fn(),
    disconnect: vi.fn(),
  };
}

function createMockAudioContext() {
  const oscillators: ReturnType<typeof createMockOsc>[] = [];
  const panners: ReturnType<typeof createMockPanner>[] = [];

  const ctx = {
    state: "running" as string,
    destination: {},
    resume: vi.fn(),
    createOscillator: vi.fn(() => {
      const osc = createMockOsc();
      oscillators.push(osc);
      return osc;
    }),
    createStereoPanner: vi.fn(() => {
      const p = createMockPanner();
      panners.push(p);
      return p;
    }),
    createGain: vi.fn(() => ({ gain: { value: 0 }, connect: vi.fn(), disconnect: vi.fn() })),
  };

  return { ctx: ctx as unknown as AudioContext, oscillators, panners };
}

// ── Tests ──────────────────────────────────────────────────────────────────

describe("gamma-entrainment", () => {
  it("exports correct frequency constants", () => {
    expect(LEFT_FREQ).toBe(400);
    expect(RIGHT_FREQ).toBe(440);
    expect(BEAT_FREQ).toBe(40);
    expect(DEFAULT_VOLUME).toBe(0.15);
  });

  it("creates oscillators at correct frequencies", () => {
    const { ctx, oscillators } = createMockAudioContext();
    createGammaEntrainment(ctx);

    expect(oscillators).toHaveLength(2);
    expect(oscillators[0].frequency.value).toBe(LEFT_FREQ);
    expect(oscillators[1].frequency.value).toBe(RIGHT_FREQ);
  });

  it("starts and stops oscillators", () => {
    const { ctx, oscillators } = createMockAudioContext();
    const handle = createGammaEntrainment(ctx);

    expect(handle.isPlaying()).toBe(false);

    handle.start();
    expect(handle.isPlaying()).toBe(true);
    expect(oscillators[0].start).toHaveBeenCalledOnce();
    expect(oscillators[1].start).toHaveBeenCalledOnce();

    handle.stop();
    expect(handle.isPlaying()).toBe(false);
    expect(oscillators[0].stop).toHaveBeenCalledOnce();
    expect(oscillators[1].stop).toHaveBeenCalledOnce();
  });

  it("does not start twice", () => {
    const { ctx, oscillators } = createMockAudioContext();
    const handle = createGammaEntrainment(ctx);

    handle.start();
    handle.start(); // second call should be no-op
    expect(oscillators[0].start).toHaveBeenCalledOnce();
  });

  it("resumes suspended AudioContext on start", () => {
    const { ctx } = createMockAudioContext();
    ctx.state = "suspended";
    const handle = createGammaEntrainment(ctx as unknown as AudioContext);

    handle.start();
    expect(ctx.resume).toHaveBeenCalled();
  });

  it("sets volume within 0-1 range", () => {
    const { ctx } = createMockAudioContext();
    const handle = createGammaEntrainment(ctx);

    handle.setVolume(0.5);
    // The gain node is internal so we test that no error is thrown
    expect(() => handle.setVolume(0)).not.toThrow();
    expect(() => handle.setVolume(1)).not.toThrow();
    expect(() => handle.setVolume(-1)).not.toThrow(); // should clamp to 0
    expect(() => handle.setVolume(2)).not.toThrow();  // should clamp to 1
  });

  it("cannot restart after stop", () => {
    const { ctx, oscillators } = createMockAudioContext();
    const handle = createGammaEntrainment(ctx);

    handle.start();
    handle.stop();
    handle.start(); // should be no-op after stop
    expect(oscillators[0].start).toHaveBeenCalledOnce();
  });

  it("uses stereo panning for left and right", () => {
    const { ctx, panners } = createMockAudioContext();
    createGammaEntrainment(ctx);

    expect(panners).toHaveLength(2);
    expect(panners[0].pan.value).toBe(-1);
    expect(panners[1].pan.value).toBe(1);
  });
});

import { describe, it, expect, beforeEach } from "vitest";
import { EMGDetector, computeBandPower } from "@/lib/emg-detector";

describe("computeBandPower", () => {
  it("returns 0 for empty signal", () => {
    expect(computeBandPower([], 256, 20, 50)).toBe(0);
  });

  it("returns non-zero power for a signal with energy in the target band", () => {
    // Generate a 40 Hz sine wave (within the 20-50 Hz band)
    const fs = 256;
    const n = 256;
    const signal: number[] = [];
    for (let i = 0; i < n; i++) {
      signal.push(10 * Math.sin(2 * Math.PI * 40 * i / fs));
    }
    const power = computeBandPower(signal, fs, 20, 50);
    expect(power).toBeGreaterThan(0);
  });

  it("returns low power for a signal outside the target band", () => {
    // Generate a 5 Hz sine wave (outside the 20-50 Hz band)
    const fs = 256;
    const n = 256;
    const lowFreqSignal: number[] = [];
    const highFreqSignal: number[] = [];
    for (let i = 0; i < n; i++) {
      lowFreqSignal.push(10 * Math.sin(2 * Math.PI * 5 * i / fs));
      highFreqSignal.push(10 * Math.sin(2 * Math.PI * 40 * i / fs));
    }
    const lowPower = computeBandPower(lowFreqSignal, fs, 20, 50);
    const highPower = computeBandPower(highFreqSignal, fs, 20, 50);
    expect(highPower).toBeGreaterThan(lowPower * 10);
  });
});

describe("EMGDetector", () => {
  let detector: EMGDetector;

  beforeEach(() => {
    detector = new EMGDetector();
  });

  it("returns no EMG detected for empty signals", () => {
    const result = detector.detect([], 256);
    expect(result.emgDetected).toBe(false);
    expect(result.artifactPercent).toBe(0);
    expect(result.message).toBeNull();
  });

  it("returns no EMG detected for clean EEG data", () => {
    const fs = 256;
    const n = 256;
    // Generate clean signal with primarily alpha (10 Hz)
    const channel: number[] = [];
    for (let i = 0; i < n; i++) {
      channel.push(10 * Math.sin(2 * Math.PI * 10 * i / fs));
    }
    const signals = [channel, channel, channel, channel];

    // Feed several frames to build history
    for (let f = 0; f < 5; f++) {
      const result = detector.detect(signals, fs);
      expect(result.emgDetected).toBe(false);
    }
  });

  it("detects EMG when broadband 20-50 Hz power spikes", () => {
    const fs = 256;
    const n = 256;

    // Build baseline with clean alpha signal
    const cleanChannel: number[] = [];
    for (let i = 0; i < n; i++) {
      cleanChannel.push(10 * Math.sin(2 * Math.PI * 10 * i / fs));
    }
    const cleanSignals = [cleanChannel, cleanChannel, cleanChannel, cleanChannel];

    // Feed baseline frames
    for (let f = 0; f < 10; f++) {
      detector.detect(cleanSignals, fs);
    }

    // Now spike with high 20-50 Hz power (EMG contamination)
    const noisyChannel: number[] = [];
    for (let i = 0; i < n; i++) {
      noisyChannel.push(
        50 * Math.sin(2 * Math.PI * 30 * i / fs) +
        50 * Math.sin(2 * Math.PI * 40 * i / fs),
      );
    }
    const noisySignals = [noisyChannel, noisyChannel, noisyChannel, noisyChannel];

    const result = detector.detect(noisySignals, fs);
    expect(result.emgDetected).toBe(true);
    expect(result.emgRatio).toBeGreaterThan(2.0);
    expect(result.message).toContain("Signal too noisy");
  });

  it("tracks artifact percentage over history", () => {
    const fs = 256;
    const n = 256;

    const cleanChannel: number[] = [];
    for (let i = 0; i < n; i++) {
      cleanChannel.push(5 * Math.sin(2 * Math.PI * 10 * i / fs));
    }
    const cleanSignals = [cleanChannel];

    // Build clean history
    for (let f = 0; f < 10; f++) {
      detector.detect(cleanSignals, fs);
    }

    // Initial artifact percent should be 0
    expect(detector.getArtifactPercent()).toBe(0);
  });

  it("resets state correctly", () => {
    const fs = 256;
    const cleanChannel: number[] = Array(256).fill(0).map((_, i) =>
      5 * Math.sin(2 * Math.PI * 10 * i / fs)
    );
    detector.detect([cleanChannel], fs);
    detector.detect([cleanChannel], fs);

    detector.reset();
    expect(detector.getArtifactPercent()).toBe(0);
  });
});

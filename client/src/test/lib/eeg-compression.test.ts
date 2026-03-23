import { describe, it, expect } from "vitest";

import {
  deltaEncode,
  deltaDecode,
  compressEEGFrame,
  decompressEEGFrame,
} from "@/lib/eeg-compression";

// ── deltaEncode / deltaDecode ───────────────────────────────────────────────

describe("deltaEncode", () => {
  it("returns differences between consecutive samples", () => {
    const samples = [100, 102, 101, 105, 103];
    const encoded = deltaEncode(samples);
    // First element is the initial value, rest are deltas
    expect(encoded[0]).toBe(100);
    expect(encoded[1]).toBe(2); // 102 - 100
    expect(encoded[2]).toBe(-1); // 101 - 102
    expect(encoded[3]).toBe(4); // 105 - 101
    expect(encoded[4]).toBe(-2); // 103 - 105
  });

  it("returns empty array for empty input", () => {
    expect(deltaEncode([])).toEqual([]);
  });

  it("returns single element unchanged", () => {
    expect(deltaEncode([42])).toEqual([42]);
  });

  it("handles constant signal (all zeros after first)", () => {
    const samples = [50, 50, 50, 50];
    const encoded = deltaEncode(samples);
    expect(encoded).toEqual([50, 0, 0, 0]);
  });
});

describe("deltaDecode", () => {
  it("reconstructs original samples from deltas", () => {
    const original = [100, 102, 101, 105, 103];
    const encoded = deltaEncode(original);
    const decoded = deltaDecode(encoded);
    expect(decoded).toEqual(original);
  });

  it("returns empty array for empty input", () => {
    expect(deltaDecode([])).toEqual([]);
  });

  it("returns single element unchanged", () => {
    expect(deltaDecode([42])).toEqual([42]);
  });

  it("roundtrips floating point data", () => {
    const original = [10.5, 11.2, 10.8, 12.0];
    const decoded = deltaDecode(deltaEncode(original));
    decoded.forEach((val, i) => {
      expect(val).toBeCloseTo(original[i], 10);
    });
  });
});

// ── compressEEGFrame / decompressEEGFrame ───────────────────────────────────

describe("compressEEGFrame", () => {
  it("compresses a frame with multiple channels", () => {
    const frame = {
      channels: [
        [100, 102, 101, 105],
        [200, 198, 199, 195],
      ],
      timestamp: 1700000000000,
    };

    const compressed = compressEEGFrame(frame);
    expect(compressed.timestamp).toBe(1700000000000);
    expect(compressed.channelCount).toBe(2);
    expect(compressed.samplesPerChannel).toBe(4);
    // Each channel should be delta-encoded
    expect(compressed.encodedChannels.length).toBe(2);
    expect(compressed.encodedChannels[0][0]).toBe(100); // first value preserved
    expect(compressed.encodedChannels[1][0]).toBe(200);
  });

  it("tracks compression ratio in metadata", () => {
    // A constant signal should compress well (all deltas = 0)
    const frame = {
      channels: [[50, 50, 50, 50, 50, 50, 50, 50]],
      timestamp: Date.now(),
    };

    const compressed = compressEEGFrame(frame);
    expect(compressed.compressionRatio).toBeGreaterThan(0);
    expect(typeof compressed.compressionRatio).toBe("number");
  });

  it("handles single-channel frame", () => {
    const frame = {
      channels: [[10, 12, 11, 15]],
      timestamp: Date.now(),
    };
    const compressed = compressEEGFrame(frame);
    expect(compressed.channelCount).toBe(1);
    expect(compressed.encodedChannels.length).toBe(1);
  });
});

describe("decompressEEGFrame", () => {
  it("roundtrips a multi-channel frame", () => {
    const frame = {
      channels: [
        [100, 102, 101, 105, 103],
        [200, 198, 199, 195, 201],
        [50, 52, 48, 55, 49],
        [80, 81, 79, 82, 78],
      ],
      timestamp: 1700000000000,
    };

    const compressed = compressEEGFrame(frame);
    const decompressed = decompressEEGFrame(compressed);

    expect(decompressed.timestamp).toBe(frame.timestamp);
    expect(decompressed.channels.length).toBe(4);
    decompressed.channels.forEach((ch, i) => {
      ch.forEach((val, j) => {
        expect(val).toBeCloseTo(frame.channels[i][j], 10);
      });
    });
  });

  it("roundtrips a frame with realistic EEG amplitude values", () => {
    // Simulate realistic EEG (20uV RMS, 256Hz, ~1 second)
    const sampleCount = 256;
    const channels: number[][] = [];
    for (let ch = 0; ch < 4; ch++) {
      const data: number[] = [];
      let value = Math.random() * 40 - 20;
      for (let i = 0; i < sampleCount; i++) {
        value += (Math.random() - 0.5) * 2; // small random walk
        data.push(value);
      }
      channels.push(data);
    }

    const frame = { channels, timestamp: Date.now() };
    const compressed = compressEEGFrame(frame);
    const decompressed = decompressEEGFrame(compressed);

    decompressed.channels.forEach((ch, i) => {
      ch.forEach((val, j) => {
        expect(val).toBeCloseTo(frame.channels[i][j], 10);
      });
    });
  });
});

import { describe, it, expect } from "vitest";
import { writeEdfPlus, type EdfConfig } from "@/lib/edf-writer";

// ── Helpers ──────────────────────────────────────────────────────────────────

const MUSE_CHANNELS = [
  { label: "EEG AF7", physicalMin: -500, physicalMax: 500, samplesPerSecond: 256 },
  { label: "EEG AF8", physicalMin: -500, physicalMax: 500, samplesPerSecond: 256 },
  { label: "EEG TP9", physicalMin: -500, physicalMax: 500, samplesPerSecond: 256 },
  { label: "EEG TP10", physicalMin: -500, physicalMax: 500, samplesPerSecond: 256 },
];

function makeConfig(overrides: Partial<EdfConfig> = {}): EdfConfig {
  return {
    patientId: "TestUser",
    startDate: new Date(2026, 2, 21, 14, 30, 0), // March 21, 2026 14:30:00
    channels: MUSE_CHANNELS,
    data: [
      new Float32Array(256),
      new Float32Array(256),
      new Float32Array(256),
      new Float32Array(256),
    ],
    recordDuration: 1,
    ...overrides,
  };
}

/** Read ASCII string from an ArrayBuffer at a given offset and length. */
function readAscii(buf: ArrayBuffer, offset: number, len: number): string {
  const bytes = new Uint8Array(buf, offset, len);
  return String.fromCharCode(...bytes);
}

// ── Tests ────────────────────────────────────────────────────────────────────

describe("writeEdfPlus", () => {
  it("produces a header of exactly 256 + (4 * 256) = 1280 bytes for 4 channels", () => {
    const config = makeConfig();
    const buf = writeEdfPlus(config);

    // Header bytes field is at offset 184, 8 bytes
    const headerBytesStr = readAscii(buf, 184, 8).trim();
    expect(Number(headerBytesStr)).toBe(1280);

    // Total file should be header + data
    // 1 record * (4 channels * 256 samples * 2 bytes) = 2048 bytes of data
    expect(buf.byteLength).toBe(1280 + 2048);
  });

  it("writes version field as '0' padded to 8 bytes", () => {
    const buf = writeEdfPlus(makeConfig());
    const version = readAscii(buf, 0, 8);
    expect(version).toBe("0       ");
  });

  it("writes the number of signals field matching channel count", () => {
    const config = makeConfig();
    const buf = writeEdfPlus(config);

    // Number of signals is at offset 252, 4 bytes
    const nSignals = readAscii(buf, 252, 4).trim();
    expect(Number(nSignals)).toBe(4);
  });

  it("writes data samples as int16 little-endian", () => {
    // Create a single channel with known values
    const data = new Float32Array(256);
    data[0] = 250; // half of physicalMax (500)

    const config = makeConfig({
      channels: [
        { label: "EEG AF7", physicalMin: -500, physicalMax: 500, samplesPerSecond: 256 },
      ],
      data: [data],
    });

    const buf = writeEdfPlus(config);
    const view = new DataView(buf);

    // Header for 1 channel = 256 + 256 = 512 bytes
    const headerSize = 512;
    // First sample at headerSize offset, int16 little-endian
    const firstSample = view.getInt16(headerSize, true);

    // 250 uV with range [-500, 500] -> digital:
    // (250 - (-500)) / (500 - (-500)) * (32767 - (-32768)) + (-32768)
    // = 750 / 1000 * 65535 + (-32768) = 49151.25 - 32768 = 16383.25 -> rounds to 16383
    expect(firstSample).toBe(16383);
  });

  it("converts physical-to-digital correctly: 0 uV maps to midpoint (~0 digital)", () => {
    const data = new Float32Array(256);
    data[0] = 0; // 0 uV should be digital midpoint

    const config = makeConfig({
      channels: [
        { label: "EEG AF7", physicalMin: -500, physicalMax: 500, samplesPerSecond: 256 },
      ],
      data: [data],
    });

    const buf = writeEdfPlus(config);
    const view = new DataView(buf);
    const headerSize = 512; // 256 + 1 * 256

    const firstSample = view.getInt16(headerSize, true);
    // 0 uV with symmetric range [-500, 500]:
    // (0 - (-500)) / 1000 * 65535 - 32768 = 500/1000 * 65535 - 32768
    // = 32767.5 - 32768 = -0.5 -> rounds to 0
    expect(firstSample).toBe(0);
  });

  it("writes 1-second record at 256 Hz = 256 samples * 4 channels * 2 bytes = 2048 bytes per record", () => {
    // 2 seconds of data -> 2 records
    const twoSecondsData = [
      new Float32Array(512),
      new Float32Array(512),
      new Float32Array(512),
      new Float32Array(512),
    ];
    const config = makeConfig({ data: twoSecondsData });
    const buf = writeEdfPlus(config);

    const headerSize = 1280; // 256 + 4 * 256
    const dataSize = buf.byteLength - headerSize;
    const bytesPerRecord = 256 * 4 * 2; // 2048

    expect(dataSize).toBe(2 * bytesPerRecord);
    expect(dataSize).toBe(4096);
  });

  it("produces a valid header-only file with empty data", () => {
    const config = makeConfig({
      data: [
        new Float32Array(0),
        new Float32Array(0),
        new Float32Array(0),
        new Float32Array(0),
      ],
    });

    const buf = writeEdfPlus(config);

    // Should be header only, no data records
    expect(buf.byteLength).toBe(1280);

    // Number of data records should be "0"
    const nRecords = readAscii(buf, 236, 8).trim();
    expect(Number(nRecords)).toBe(0);
  });

  it("writes the correct patient ID", () => {
    const config = makeConfig({ patientId: "Subject_42" });
    const buf = writeEdfPlus(config);
    const patientId = readAscii(buf, 8, 80).trim();
    expect(patientId).toBe("Subject_42");
  });

  it("writes correct date and time fields", () => {
    const config = makeConfig({
      startDate: new Date(2026, 2, 21, 14, 30, 45),
    });
    const buf = writeEdfPlus(config);

    // Date at offset 168, 8 bytes: DD.MM.YY
    const dateStr = readAscii(buf, 168, 8).trim();
    expect(dateStr).toBe("21.03.26");

    // Time at offset 176, 8 bytes: HH.MM.SS
    const timeStr = readAscii(buf, 176, 8).trim();
    expect(timeStr).toBe("14.30.45");
  });

  it("writes channel labels correctly in the per-signal header", () => {
    const buf = writeEdfPlus(makeConfig());

    // Channel labels start at offset 256 (after main header), 16 bytes each
    const label0 = readAscii(buf, 256, 16).trim();
    const label1 = readAscii(buf, 272, 16).trim();
    const label2 = readAscii(buf, 288, 16).trim();
    const label3 = readAscii(buf, 304, 16).trim();

    expect(label0).toBe("EEG AF7");
    expect(label1).toBe("EEG AF8");
    expect(label2).toBe("EEG TP9");
    expect(label3).toBe("EEG TP10");
  });

  it("clamps values outside physical range to digital min/max", () => {
    const data = new Float32Array(256);
    data[0] = 1000; // Way above physMax (500)
    data[1] = -1000; // Way below physMin (-500)

    const config = makeConfig({
      channels: [
        { label: "EEG AF7", physicalMin: -500, physicalMax: 500, samplesPerSecond: 256 },
      ],
      data: [data],
    });

    const buf = writeEdfPlus(config);
    const view = new DataView(buf);
    const headerSize = 512;

    const sample0 = view.getInt16(headerSize, true);
    const sample1 = view.getInt16(headerSize + 2, true);

    expect(sample0).toBe(32767);  // clamped to digMax
    expect(sample1).toBe(-32768); // clamped to digMin
  });
});

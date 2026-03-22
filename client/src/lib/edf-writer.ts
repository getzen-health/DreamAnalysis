/**
 * Minimal EDF+ writer for 4-channel Muse 2 EEG data.
 *
 * EDF+ is the de facto EEG standard (BIDS recommended). This writer produces
 * valid EDF+ files that can be opened in EDFbrowser, MNE-Python, EEGLAB, etc.
 *
 * Reference: https://www.edfplus.info/specs/edf.html
 */

export interface EdfChannel {
  label: string;            // e.g. "EEG AF7", "EEG AF8", "EEG TP9", "EEG TP10"
  physicalMin: number;      // e.g. -500 (uV)
  physicalMax: number;      // e.g.  500 (uV)
  samplesPerSecond: number; // e.g. 256
}

export interface EdfConfig {
  patientId: string;
  startDate: Date;
  channels: EdfChannel[];
  data: Float32Array[];     // one array per channel
  recordDuration: number;   // seconds per data record (typically 1)
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/** Pad or truncate a string to exactly `len` bytes, space-padded on the right. */
function asciiField(value: string, len: number): string {
  const trimmed = value.slice(0, len);
  return trimmed.padEnd(len, " ");
}

/** Format a date into EDF date string: DD.MM.YY */
function edfDate(d: Date): string {
  const dd = String(d.getDate()).padStart(2, "0");
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const yy = String(d.getFullYear() % 100).padStart(2, "0");
  return `${dd}.${mm}.${yy}`;
}

/** Format a date into EDF time string: HH.MM.SS */
function edfTime(d: Date): string {
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  const ss = String(d.getSeconds()).padStart(2, "0");
  return `${hh}.${mm}.${ss}`;
}

/**
 * Convert a physical value (uV) to a 16-bit digital value using linear scaling.
 *
 * digital = (physical - physMin) / (physMax - physMin) * (digMax - digMin) + digMin
 */
function physicalToDigital(
  value: number,
  physMin: number,
  physMax: number,
  digMin: number,
  digMax: number,
): number {
  const physRange = physMax - physMin;
  if (physRange === 0) return 0;
  const digRange = digMax - digMin;
  const digital = ((value - physMin) / physRange) * digRange + digMin;
  // Clamp to int16 range
  return Math.max(digMin, Math.min(digMax, Math.round(digital)));
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Write an EDF+ file from the given configuration.
 *
 * Returns an ArrayBuffer containing the complete EDF+ file that can be
 * saved directly to disk or triggered as a browser download.
 */
export function writeEdfPlus(config: EdfConfig): ArrayBuffer {
  const { patientId, startDate, channels, data, recordDuration } = config;
  const nChannels = channels.length;

  // Digital range: full int16
  const digMin = -32768;
  const digMax = 32767;

  // Compute number of data records
  // Total samples per channel / samples per record
  const samplesPerRecord = channels.map(
    (ch) => ch.samplesPerSecond * recordDuration,
  );
  const totalSamples = data.length > 0 ? data[0].length : 0;
  const samplesInOneRecord = channels.length > 0 ? samplesPerRecord[0] : 0;
  const nRecords =
    samplesInOneRecord > 0 ? Math.floor(totalSamples / samplesInOneRecord) : 0;

  // Header sizes
  const headerBytes = 256 + nChannels * 256;

  // Total file size = header + data records
  // Each record = sum of (samples_per_record[ch] * 2 bytes) for all channels
  const bytesPerRecord = samplesPerRecord.reduce(
    (sum, spr) => sum + spr * 2,
    0,
  );
  const totalSize = headerBytes + nRecords * bytesPerRecord;

  const buffer = new ArrayBuffer(totalSize);
  const view = new DataView(buffer);

  // ── Write main header (256 bytes) ──────────────────────────────────────────

  let offset = 0;

  function writeAscii(str: string): void {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset++, str.charCodeAt(i));
    }
  }

  // Version (8 bytes): "0       "
  writeAscii(asciiField("0", 8));

  // Patient ID (80 bytes)
  writeAscii(asciiField(patientId, 80));

  // Recording ID (80 bytes)
  writeAscii(asciiField("Startdate " + edfDate(startDate), 80));

  // Start date (8 bytes): DD.MM.YY
  writeAscii(asciiField(edfDate(startDate), 8));

  // Start time (8 bytes): HH.MM.SS
  writeAscii(asciiField(edfTime(startDate), 8));

  // Number of bytes in header (8 bytes)
  writeAscii(asciiField(String(headerBytes), 8));

  // Reserved (44 bytes): "EDF+C" for continuous EDF+
  writeAscii(asciiField("EDF+C", 44));

  // Number of data records (8 bytes)
  writeAscii(asciiField(String(nRecords), 8));

  // Duration of a data record in seconds (8 bytes)
  writeAscii(asciiField(String(recordDuration), 8));

  // Number of signals (4 bytes)
  writeAscii(asciiField(String(nChannels), 4));

  // ── Write per-channel headers (256 bytes each) ────────────────────────────

  // Labels (16 bytes each)
  for (const ch of channels) {
    writeAscii(asciiField(ch.label, 16));
  }

  // Transducer type (80 bytes each)
  for (let i = 0; i < nChannels; i++) {
    writeAscii(asciiField("AgCl electrode", 80));
  }

  // Physical dimension (8 bytes each)
  for (let i = 0; i < nChannels; i++) {
    writeAscii(asciiField("uV", 8));
  }

  // Physical minimum (8 bytes each)
  for (const ch of channels) {
    writeAscii(asciiField(String(ch.physicalMin), 8));
  }

  // Physical maximum (8 bytes each)
  for (const ch of channels) {
    writeAscii(asciiField(String(ch.physicalMax), 8));
  }

  // Digital minimum (8 bytes each)
  for (let i = 0; i < nChannels; i++) {
    writeAscii(asciiField(String(digMin), 8));
  }

  // Digital maximum (8 bytes each)
  for (let i = 0; i < nChannels; i++) {
    writeAscii(asciiField(String(digMax), 8));
  }

  // Prefiltering (80 bytes each)
  for (let i = 0; i < nChannels; i++) {
    writeAscii(asciiField("HP:1Hz LP:50Hz N:50/60Hz", 80));
  }

  // Number of samples per data record (8 bytes each)
  for (let i = 0; i < nChannels; i++) {
    writeAscii(asciiField(String(samplesPerRecord[i]), 8));
  }

  // Reserved per signal (32 bytes each)
  for (let i = 0; i < nChannels; i++) {
    writeAscii(asciiField("", 32));
  }

  // ── Write data records ─────────────────────────────────────────────────────

  for (let rec = 0; rec < nRecords; rec++) {
    for (let ch = 0; ch < nChannels; ch++) {
      const spr = samplesPerRecord[ch];
      const startSample = rec * spr;
      const channelData = data[ch];
      const physMin = channels[ch].physicalMin;
      const physMax = channels[ch].physicalMax;

      for (let s = 0; s < spr; s++) {
        const sampleIdx = startSample + s;
        const physValue =
          sampleIdx < channelData.length ? channelData[sampleIdx] : 0;
        const digital = physicalToDigital(
          physValue,
          physMin,
          physMax,
          digMin,
          digMax,
        );
        view.setInt16(offset, digital, true); // little-endian
        offset += 2;
      }
    }
  }

  return buffer;
}

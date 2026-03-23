/**
 * eeg-compression.ts — Delta encoding for EEG data compression.
 *
 * EEG samples between consecutive readings differ by small amounts (typically
 * < 2 uV at 256 Hz). Delta encoding stores only these differences, producing
 * values that cluster near zero and compress much better than raw amplitudes.
 *
 * Usage:
 *   import { compressEEGFrame, decompressEEGFrame } from "@/lib/eeg-compression";
 *   const compressed = compressEEGFrame({ channels, timestamp });
 *   const original   = decompressEEGFrame(compressed);
 */

// ── Delta Encoding ──────────────────────────────────────────────────────────

/**
 * Delta-encode an array of samples.
 * Output[0] = original first value (baseline).
 * Output[i] = samples[i] - samples[i-1] for i > 0.
 */
export function deltaEncode(samples: number[]): number[] {
  if (samples.length === 0) return [];
  const encoded: number[] = new Array(samples.length);
  encoded[0] = samples[0];
  for (let i = 1; i < samples.length; i++) {
    encoded[i] = samples[i] - samples[i - 1];
  }
  return encoded;
}

/**
 * Reverse delta encoding to reconstruct original samples.
 */
export function deltaDecode(deltas: number[]): number[] {
  if (deltas.length === 0) return [];
  const decoded: number[] = new Array(deltas.length);
  decoded[0] = deltas[0];
  for (let i = 1; i < deltas.length; i++) {
    decoded[i] = decoded[i - 1] + deltas[i];
  }
  return decoded;
}

// ── Frame Compression ───────────────────────────────────────────────────────

export interface EEGFrame {
  channels: number[][];
  timestamp: number;
}

export interface CompressedEEGFrame {
  encodedChannels: number[][];
  timestamp: number;
  channelCount: number;
  samplesPerChannel: number;
  compressionRatio: number;
}

/**
 * Compress an EEG frame by delta-encoding each channel.
 * The compression ratio is calculated as the reduction in representable range:
 * raw values need full dynamic range; deltas cluster near zero.
 */
export function compressEEGFrame(frame: EEGFrame): CompressedEEGFrame {
  const encodedChannels = frame.channels.map((ch) => deltaEncode(ch));

  // Estimate compression ratio: mean absolute value of raw vs deltas.
  // Smaller deltas = better compressibility.
  let rawSum = 0;
  let deltaSum = 0;
  let totalSamples = 0;

  for (let c = 0; c < frame.channels.length; c++) {
    for (let i = 0; i < frame.channels[c].length; i++) {
      rawSum += Math.abs(frame.channels[c][i]);
      deltaSum += Math.abs(encodedChannels[c][i]);
      totalSamples++;
    }
  }

  const avgRaw = totalSamples > 0 ? rawSum / totalSamples : 1;
  const avgDelta = totalSamples > 0 ? deltaSum / totalSamples : 1;
  const compressionRatio = avgRaw > 0 ? avgRaw / Math.max(avgDelta, 1e-10) : 1;

  return {
    encodedChannels,
    timestamp: frame.timestamp,
    channelCount: frame.channels.length,
    samplesPerChannel: frame.channels.length > 0 ? frame.channels[0].length : 0,
    compressionRatio,
  };
}

/**
 * Decompress a compressed EEG frame back to original channel data.
 */
export function decompressEEGFrame(compressed: CompressedEEGFrame): EEGFrame {
  const channels = compressed.encodedChannels.map((ch) => deltaDecode(ch));
  return {
    channels,
    timestamp: compressed.timestamp,
  };
}

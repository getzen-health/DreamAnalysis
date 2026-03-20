/** Resample audio to 16kHz using linear interpolation. */
export function resampleTo16k(samples: Float32Array, sourceSr: number): Float32Array {
  if (sourceSr === 16000) return samples;
  const ratio = sourceSr / 16000;
  const outLen = Math.round(samples.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const srcIdx = i * ratio;
    const lo = Math.floor(srcIdx);
    const hi = Math.min(lo + 1, samples.length - 1);
    const frac = srcIdx - lo;
    out[i] = samples[lo] * (1 - frac) + samples[hi] * frac;
  }
  return out;
}

/** Prepare raw PCM for Wav2Small: resample to 16kHz, normalize, pad to >=1s. */
export function prepareAudioInput(samples: Float32Array, sourceSr: number): Float32Array {
  let audio = resampleTo16k(samples, sourceSr);

  // Normalize to [-1, 1]
  let maxAbs = 0;
  for (let i = 0; i < audio.length; i++) {
    const abs = Math.abs(audio[i]);
    if (abs > maxAbs) maxAbs = abs;
  }
  if (maxAbs > 1.0) {
    const scale = 1.0 / maxAbs;
    audio = audio.map(s => s * scale);
  }

  // Pad to minimum 1 second (16000 samples)
  if (audio.length < 16000) {
    const padded = new Float32Array(16000);
    padded.set(audio);
    return padded;
  }
  return audio;
}

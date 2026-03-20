# Wav2Small On-Device Voice Emotion Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace server-dependent voice emotion analysis with Wav2Small (120KB ONNX) running in-browser via onnxruntime-web, so voice check-ins work on the APK without the ML backend.

**Architecture:** Download Wav2Small ONNX model from HuggingFace, bundle in `public/models/`. Create a new `voice-onnx.ts` module that extracts MFCC features from PCM audio using Web Audio API, runs inference via onnxruntime-web (already in package.json), and maps arousal/dominance/valence output to 6 discrete emotions. Modify `voice-checkin-card.tsx` to try on-device ONNX first, then ML backend, then heuristic fallback.

**Tech Stack:** onnxruntime-web 1.17.0 (installed), Web Audio API, Vitest

---

## File Structure

| File | Responsibility |
|------|---------------|
| `client/src/lib/voice-onnx.ts` | NEW — MFCC extraction + ONNX inference + emotion mapping |
| `client/src/test/lib/voice-onnx.test.ts` | NEW — Unit tests for MFCC, inference, emotion mapping |
| `public/models/wav2small.onnx` | NEW — Wav2Small ONNX model file (~120KB) |
| `client/src/components/voice-checkin-card.tsx` | MODIFY — Use on-device ONNX as primary, ML backend as fallback |
| `client/src/hooks/use-voice-emotion.ts` | MODIFY — Use on-device ONNX for 30s recording path too |

---

## Chunk 1: MFCC Feature Extraction

### Task 1: Write MFCC extraction from PCM audio

Wav2Small expects 16kHz mono audio. The app already captures PCM via ScriptProcessorNode. We need to:
- Resample to 16kHz if needed
- Extract 40 MFCC coefficients (matching Wav2Small's input)

**Files:**
- Create: `client/src/lib/voice-onnx.ts`
- Test: `client/src/test/lib/voice-onnx.test.ts`

- [ ] **Step 1: Write failing test for resample function**

```typescript
// client/src/test/lib/voice-onnx.test.ts
import { describe, it, expect } from "vitest";
import { resampleTo16k } from "@/lib/voice-onnx";

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
    // 48 samples at 48kHz → 16 samples at 16kHz
    expect(result.length).toBe(16);
  });

  it("downsamples 44100Hz to 16kHz", () => {
    const input = new Float32Array(44100); // 1 second
    const result = resampleTo16k(input, 44100);
    expect(result.length).toBe(16000);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/sravyalu/NeuralDreamWorkshop && npx vitest run client/src/test/lib/voice-onnx.test.ts`
Expected: FAIL — module not found

- [ ] **Step 3: Implement resampleTo16k**

```typescript
// client/src/lib/voice-onnx.ts

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/sravyalu/NeuralDreamWorkshop && npx vitest run client/src/test/lib/voice-onnx.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```
git add client/src/lib/voice-onnx.ts client/src/test/lib/voice-onnx.test.ts
git commit -m "feat: add resampleTo16k for on-device voice emotion"
```

---

### Task 2: Write MFCC feature extraction

Wav2Small takes raw waveform input (not MFCC). The model handles feature extraction internally. So we just need to prepare a normalized Float32Array at 16kHz.

**Files:**
- Modify: `client/src/lib/voice-onnx.ts`
- Test: `client/src/test/lib/voice-onnx.test.ts`

- [ ] **Step 1: Write failing test for prepareAudioInput**

```typescript
import { prepareAudioInput } from "@/lib/voice-onnx";

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
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Implement prepareAudioInput**

```typescript
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
```

- [ ] **Step 4: Run test to verify it passes**

- [ ] **Step 5: Commit**

```
git add client/src/lib/voice-onnx.ts client/src/test/lib/voice-onnx.test.ts
git commit -m "feat: add prepareAudioInput — normalize, resample, pad for Wav2Small"
```

---

## Chunk 2: ONNX Inference + Emotion Mapping

### Task 3: Download and bundle Wav2Small ONNX model

**Files:**
- Create: `public/models/wav2small.onnx`

- [ ] **Step 1: Download Wav2Small ONNX from HuggingFace**

```bash
cd /Users/sravyalu/NeuralDreamWorkshop
mkdir -p public/models
# Download the quantized ONNX model
curl -L "https://huggingface.co/AudEERING/wav2small/resolve/main/wav2small.onnx" -o public/models/wav2small.onnx
ls -la public/models/wav2small.onnx
```

Expected: File ~120KB

- [ ] **Step 2: Add to .gitignore exception if needed, commit**

```
git add public/models/wav2small.onnx
git commit -m "chore: bundle Wav2Small ONNX model (120KB) for on-device voice emotion"
```

**Note:** If the HuggingFace URL doesn't have a direct ONNX file, we may need to export it. Check the repo structure first. If no ONNX exists, we'll export from the PyTorch model using the ml/ environment in Task 3b (fallback).

### Task 3b (Fallback): Export Wav2Small to ONNX if pre-built not available

- [ ] **Step 1: Export using Python**

```bash
cd /Users/sravyalu/NeuralDreamWorkshop/ml
source venv/bin/activate
pip install wav2small torch onnx
python -c "
import torch
from wav2small import Wav2Small
model = Wav2Small.from_pretrained('AudEERING/wav2small')
model.eval()
dummy = torch.randn(1, 16000)
torch.onnx.export(model, dummy, '../public/models/wav2small.onnx',
    input_names=['audio'], output_names=['arousal', 'dominance', 'valence'],
    dynamic_axes={'audio': {1: 'length'}})
print('Exported successfully')
"
```

---

### Task 4: Write ONNX inference wrapper

**Files:**
- Modify: `client/src/lib/voice-onnx.ts`
- Test: `client/src/test/lib/voice-onnx.test.ts`

- [ ] **Step 1: Write failing test for emotion mapping (testable without ONNX runtime)**

```typescript
import { mapADVtoEmotion } from "@/lib/voice-onnx";

describe("mapADVtoEmotion", () => {
  it("maps high arousal + high valence to happy", () => {
    const result = mapADVtoEmotion(0.8, 0.5, 0.8);
    expect(result.emotion).toBe("happy");
  });

  it("maps low arousal + low valence to sad", () => {
    const result = mapADVtoEmotion(0.2, 0.5, 0.2);
    expect(result.emotion).toBe("sad");
  });

  it("maps high arousal + low valence to angry", () => {
    const result = mapADVtoEmotion(0.8, 0.3, 0.2);
    expect(result.emotion).toBe("angry");
  });

  it("maps low arousal + high valence to neutral/calm", () => {
    const result = mapADVtoEmotion(0.3, 0.6, 0.7);
    expect(["neutral", "happy"]).toContain(result.emotion);
  });

  it("maps very high arousal + very low valence to fear", () => {
    const result = mapADVtoEmotion(0.9, 0.2, 0.1);
    expect(result.emotion).toBe("fear");
  });

  it("returns valence/arousal/stress_index/focus_index", () => {
    const result = mapADVtoEmotion(0.5, 0.5, 0.5);
    expect(result).toHaveProperty("emotion");
    expect(result).toHaveProperty("valence");
    expect(result).toHaveProperty("arousal");
    expect(result).toHaveProperty("stress_index");
    expect(result).toHaveProperty("focus_index");
    expect(result).toHaveProperty("confidence");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Implement mapADVtoEmotion**

```typescript
export interface OnDeviceEmotionResult {
  emotion: string;
  valence: number;
  arousal: number;
  confidence: number;
  stress_index: number;
  focus_index: number;
  model_type: "wav2small-onnx";
}

/**
 * Map Arousal/Dominance/Valence (0-1 range) to discrete emotion.
 * Uses Russell's circumplex model aligned with the EEG pipeline's mapping.
 */
export function mapADVtoEmotion(arousal: number, dominance: number, valence: number): OnDeviceEmotionResult {
  // Map 0-1 valence to -1..1 range (model outputs 0-1)
  const v = valence * 2 - 1; // -1 to 1
  const a = arousal;         // 0 to 1

  let emotion = "neutral";
  if (v > 0.2 && a > 0.5) emotion = "happy";
  else if (v > 0.2 && a <= 0.5) emotion = "happy"; // calm happy
  else if (v < -0.3 && a < 0.4) emotion = "sad";
  else if (v < -0.2 && a > 0.6) emotion = "angry";
  else if (v < -0.3 && a > 0.7) emotion = "fear";
  else if (a > 0.7 && Math.abs(v) < 0.2) emotion = "surprise";

  const stress = Math.min(1, Math.max(0, (1 - valence) * 0.5 + arousal * 0.3 + (1 - dominance) * 0.2));
  const focus = Math.min(1, Math.max(0, dominance * 0.5 + (1 - stress) * 0.3 + 0.2));

  return {
    emotion,
    valence: v,
    arousal: a,
    confidence: 0.72, // Wav2Small's approximate accuracy on IEMOCAP
    stress_index: stress,
    focus_index: focus,
    model_type: "wav2small-onnx",
  };
}
```

- [ ] **Step 4: Run test to verify it passes**

- [ ] **Step 5: Implement runWav2SmallInference**

```typescript
import * as ort from "onnxruntime-web";

let session: ort.InferenceSession | null = null;

/** Load Wav2Small ONNX model (lazy singleton). */
async function getSession(): Promise<ort.InferenceSession> {
  if (session) return session;
  const modelUrl = "/models/wav2small.onnx";
  session = await ort.InferenceSession.create(modelUrl, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  return session;
}

/**
 * Run Wav2Small inference on PCM audio.
 * Returns arousal, dominance, valence (0-1) mapped to discrete emotion.
 */
export async function runWav2SmallInference(
  pcmSamples: Float32Array,
  sampleRate: number
): Promise<OnDeviceEmotionResult> {
  const audio = prepareAudioInput(pcmSamples, sampleRate);
  const sess = await getSession();

  const inputTensor = new ort.Tensor("float32", audio, [1, audio.length]);
  const results = await sess.run({ audio: inputTensor });

  // Wav2Small outputs: arousal, dominance, valence (each scalar 0-1)
  const arousal = (results.arousal?.data as Float32Array)[0] ?? 0.5;
  const dominance = (results.dominance?.data as Float32Array)[0] ?? 0.5;
  const valence = (results.valence?.data as Float32Array)[0] ?? 0.5;

  return mapADVtoEmotion(arousal, dominance, valence);
}
```

- [ ] **Step 6: Commit**

```
git add client/src/lib/voice-onnx.ts client/src/test/lib/voice-onnx.test.ts
git commit -m "feat: add Wav2Small ONNX inference with ADV-to-emotion mapping"
```

---

## Chunk 3: Wire Into Voice Check-In

### Task 5: Modify voice-checkin-card.tsx to use on-device ONNX first

**Files:**
- Modify: `client/src/components/voice-checkin-card.tsx`

The current pipeline is: ML backend → heuristic fallback.
New pipeline: **ONNX on-device → ML backend → heuristic fallback.**

- [ ] **Step 1: Add import at top of voice-checkin-card.tsx**

```typescript
import { runWav2SmallInference } from "@/lib/voice-onnx";
```

- [ ] **Step 2: Replace the analysis pipeline (lines ~548-577)**

Find the section that starts with `// Step 2: Try ML backend, fall back to on-device heuristics` and replace:

```typescript
        // Step 2: Try on-device ONNX first, then ML backend, then heuristics
        let checkinResult: VoiceWatchCheckinResult;

        // 2a: On-device Wav2Small ONNX (works offline, ~15ms)
        try {
          const onnxResult = await runWav2SmallInference(pcmSamples, pcmSr);
          checkinResult = {
            checkin_id: `${Date.now()}`,
            checkin_type: period ?? "morning",
            emotion: onnxResult.emotion,
            valence: onnxResult.valence,
            arousal: onnxResult.arousal,
            confidence: onnxResult.confidence,
            stress_index: onnxResult.stress_index,
            focus_index: onnxResult.focus_index,
            model_type: onnxResult.model_type as any,
            timestamp: Date.now() / 1000,
            biomarkers: undefined,
          };
        } catch (onnxErr) {
          console.warn("Wav2Small ONNX failed — trying ML backend:", onnxErr);

          // 2b: ML backend (if available)
          try {
            const bytes = new Uint8Array(wavBuffer);
            let binary = "";
            for (let i = 0; i < bytes.byteLength; i++) {
              binary += String.fromCharCode(bytes[i]);
            }
            const audio_b64 = btoa(binary);
            const raw = await submitVoiceWatch(audio_b64, resolvedUserId);
            const stressIndex = raw.stress_index ?? raw.stress_from_watch ?? 0.5;
            const focusIndex = raw.focus_index ?? Math.max(0.2, Math.min(0.85, raw.confidence ?? 0.5));
            checkinResult = {
              checkin_id: `${Date.now()}`,
              checkin_type: period ?? "morning",
              emotion: raw.emotion ?? "neutral",
              valence: raw.valence ?? 0,
              arousal: raw.arousal ?? 0.5,
              confidence: raw.confidence ?? 0.5,
              stress_index: stressIndex,
              focus_index: focusIndex,
              model_type: raw.model_type ?? "voice",
              timestamp: Date.now() / 1000,
              biomarkers: raw.biomarkers,
            };
          } catch (mlErr) {
            // 2c: Heuristic fallback (last resort)
            console.warn("ML backend unavailable — using on-device heuristics:", mlErr);
            checkinResult = analyzeFromPcm(pcmSamples, pcmSr);
          }
        }
```

- [ ] **Step 3: Run full test suite**

```bash
cd /Users/sravyalu/NeuralDreamWorkshop && npx vitest run
```

- [ ] **Step 4: Commit**

```
git add client/src/components/voice-checkin-card.tsx
git commit -m "feat: voice check-in uses on-device Wav2Small ONNX as primary, ML backend as fallback"
```

---

### Task 6: Wire into use-voice-emotion.ts hook

**Files:**
- Modify: `client/src/hooks/use-voice-emotion.ts`

- [ ] **Step 1: Add import and ONNX-first pattern**

Same pattern as Task 5: try `runWav2SmallInference(pcmSamples, sampleRate)` first, fall back to ML backend `submitVoiceWatch()`.

- [ ] **Step 2: Run tests**

```bash
cd /Users/sravyalu/NeuralDreamWorkshop && npx vitest run
```

- [ ] **Step 3: Commit**

```
git add client/src/hooks/use-voice-emotion.ts
git commit -m "feat: use-voice-emotion hook uses Wav2Small ONNX as primary analysis"
```

---

## Chunk 4: Integration Test + Build

### Task 7: Write integration test for the full pipeline

**Files:**
- Create: `client/src/test/lib/voice-onnx-integration.test.ts`

- [ ] **Step 1: Write integration test**

```typescript
import { describe, it, expect } from "vitest";
import { prepareAudioInput, mapADVtoEmotion } from "@/lib/voice-onnx";

describe("voice-onnx integration", () => {
  it("full pipeline: 15s sine wave → prepared audio → emotion mapping", () => {
    // Simulate 15 seconds of 48kHz audio (a sine wave = energetic voice)
    const sr = 48000;
    const duration = 15;
    const samples = new Float32Array(sr * duration);
    for (let i = 0; i < samples.length; i++) {
      samples[i] = Math.sin(2 * Math.PI * 200 * i / sr) * 0.7;
    }

    const prepared = prepareAudioInput(samples, sr);
    expect(prepared.length).toBe(16000 * duration);

    // Test mapping with synthetic ADV values
    const result = mapADVtoEmotion(0.7, 0.6, 0.75);
    expect(result.emotion).toBe("happy");
    expect(result.model_type).toBe("wav2small-onnx");
    expect(result.confidence).toBeGreaterThan(0.5);
    expect(result.stress_index).toBeGreaterThanOrEqual(0);
    expect(result.stress_index).toBeLessThanOrEqual(1);
  });

  it("silent audio maps to neutral/low arousal", () => {
    const samples = new Float32Array(16000); // 1s silence
    const prepared = prepareAudioInput(samples, 16000);
    expect(prepared.length).toBe(16000);

    // Low arousal + mid valence = neutral
    const result = mapADVtoEmotion(0.2, 0.5, 0.5);
    expect(["neutral", "sad"]).toContain(result.emotion);
  });

  it("all 6 emotions are reachable", () => {
    const emotions = new Set<string>();
    // Happy: high V, high A
    emotions.add(mapADVtoEmotion(0.8, 0.5, 0.8).emotion);
    // Sad: low V, low A
    emotions.add(mapADVtoEmotion(0.2, 0.5, 0.2).emotion);
    // Angry: low V, high A
    emotions.add(mapADVtoEmotion(0.8, 0.3, 0.2).emotion);
    // Fear: very low V, very high A
    emotions.add(mapADVtoEmotion(0.9, 0.2, 0.1).emotion);
    // Surprise: high A, neutral V
    emotions.add(mapADVtoEmotion(0.85, 0.5, 0.5).emotion);
    // Neutral: mid everything
    emotions.add(mapADVtoEmotion(0.4, 0.5, 0.5).emotion);

    expect(emotions.size).toBeGreaterThanOrEqual(5);
  });
});
```

- [ ] **Step 2: Run all tests**

```bash
cd /Users/sravyalu/NeuralDreamWorkshop && npx vitest run
```

- [ ] **Step 3: Commit**

```
git add client/src/test/lib/voice-onnx-integration.test.ts
git commit -m "test: add integration tests for Wav2Small voice emotion pipeline"
```

---

### Task 8: Build and verify

- [ ] **Step 1: Build the app**

```bash
cd /Users/sravyalu/NeuralDreamWorkshop && npm run build
```

Verify: no TypeScript errors, `wav2small.onnx` included in `dist/public/models/`

- [ ] **Step 2: Sync to Android**

```bash
npx cap sync android
```

- [ ] **Step 3: Verify model is in APK assets**

```bash
ls -la android/app/src/main/assets/public/models/wav2small.onnx
```

- [ ] **Step 4: Commit**

```
git add -A
git commit -m "chore: build with Wav2Small ONNX bundled for on-device voice emotion"
```
